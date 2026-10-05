//! Filter evaluated *before* broadcast.
//!
//! A `WHERE` predicate is element-wise: whether grid cell `(t, y, x)` is kept
//! depends only on its input columns at `(t, y, x)`. So instead of broadcasting
//! every column onto the full grid and then filtering (a plain `FilterExec`
//! above [`NdBroadcastExec`]), each conjunct is evaluated on the *minimal*
//! sub-grid its inputs span — its footprint — and the resulting boolean mask is
//! lifted to the target grid. The conjunct masks are combined (null → excluded)
//! into the set of retained target cells, which rides the nd batch as a
//! [`selection`](NdRecordBatch::selection).
//!
//! No column data moves here: the filter attaches an index array.
//! [`NdBroadcastExec`] then fuses the broadcast with the selection into one
//! gather per column, so the filtered-out cross-product is never materialized —
//! and every operator above the broadcast sees only the surviving rows.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanArray, UInt64Array};
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::metrics::{
    BaselineMetrics, Count, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet,
};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};
use futures::StreamExt;

use nd_arrow_array::batch::NdRecordBatch;
use nd_arrow_array::dimensions::Dimensions;
use nd_arrow_array::selection::Selection;

use super::expr_column::{NdExprColumn, ProjectMetrics};
use super::{NdExecutionPlan, SendableNdBatchStream, execute_flat, one_child, require_nd_input};
use crate::registry::NdNodeRegistry;

/// Per-partition counters recorded while filtering.
struct FilterMetrics {
    /// Target cells seen before filtering (∑ grid sizes, honoring any inbound
    /// selection).
    input_rows: Count,
    /// Cells removed by the predicate.
    rows_pruned: Count,
    /// Mask values read at retained cells by the multi-axis conjuncts.
    cells_evaluated: Count,
    /// Footprint evaluation work of the conjuncts (shared with projection).
    project: ProjectMetrics,
}

/// Applies a conjunction of element-wise predicates over un-broadcast nd
/// batches, recording the result as a grid selection instead of dropping
/// columns. Requires an nd-aware child and is itself nd-aware, so it slots
/// between [`NdSourceExec`](super::NdSourceExec) and [`NdBroadcastExec`].
#[derive(Debug, Clone)]
pub struct NdFilterExec {
    /// nd-aware child producing the input nd batches.
    input: Arc<dyn ExecutionPlan>,
    /// The nd side of `input`.
    nd_input: Arc<dyn NdExecutionPlan>,
    registry: Arc<NdNodeRegistry>,
    /// Predicate conjuncts, ANDed together (each must be boolean).
    predicates: Vec<Arc<dyn PhysicalExpr>>,
    /// Per-conjunct evaluation plan (derived from `predicates`).
    columns: Vec<NdExprColumn>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl NdFilterExec {
    /// The nd child is resolved through `registry`.
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        predicates: Vec<Arc<dyn PhysicalExpr>>,
        registry: Arc<NdNodeRegistry>,
    ) -> Result<Self> {
        let nd_input = require_nd_input("NdFilterExec", &input, &registry)?;
        if predicates.is_empty() {
            return Err(DataFusionError::Plan(
                "NdFilterExec requires at least one predicate".to_string(),
            ));
        }
        let input_schema = input.schema();
        let columns = predicates
            .iter()
            .map(|expr| NdExprColumn::build(&input_schema, expr))
            .collect::<Result<Vec<_>>>()?;

        // A filter preserves its input's columns and ordering; only row count
        // and statistics change, so reuse the child's plan properties.
        let properties = input.properties().clone();
        Ok(Self {
            input,
            nd_input,
            registry,
            predicates,
            columns,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    pub fn predicates(&self) -> &[Arc<dyn PhysicalExpr>] {
        &self.predicates
    }

    /// Filter one nd batch: compute the retained target cells and attach them as
    /// the batch's selection (intersected with any inbound selection).
    fn filter_batch(
        &self,
        batch: &NdRecordBatch,
        metrics: &FilterMetrics,
    ) -> Result<NdRecordBatch> {
        metrics.input_rows.add(batch.num_rows());
        let retained = retained_selection(
            &self.columns,
            batch,
            &metrics.project,
            &metrics.cells_evaluated,
        )?;
        let kept = retained.num_rows(batch.target());
        metrics
            .rows_pruned
            .add(batch.num_rows().saturating_sub(kept));
        Ok(batch.clone().with_selection(retained)?)
    }
}

/// Evaluate the conjuncts over one nd batch and return the retained target
/// cells in the coarsest selection state that holds them. Each conjunct is
/// evaluated on its footprint. A null predicate value excludes the cell.
///
/// The conjuncts on one target axis apply first: each keeps an index set of
/// its axis, so the selection stays a rectangle. Each other conjunct then
/// reads its footprint mask only at the cells that the selection still keeps,
/// so no mask of the full grid is built. `cells_evaluated` counts those mask
/// reads. The result starts from any selection a child already accumulated,
/// so it is always a subset of the batch's current rows.
fn retained_selection(
    columns: &[NdExprColumn],
    batch: &NdRecordBatch,
    metrics: &ProjectMetrics,
    cells_evaluated: &Count,
) -> Result<Selection> {
    let target = batch.target();
    let mut selection = batch.selection().clone();

    let mut residual = Vec::new();
    for column in columns {
        let masked = column.project(batch, target, metrics)?;
        match single_target_axis(masked.dims(), target) {
            Some(axis) => {
                let mask = as_boolean(masked.values())?;
                let indices: UInt64Array = (0..mask.len())
                    .filter(|&i| mask.is_valid(i) && mask.value(i))
                    .map(|i| i as u64)
                    .collect();
                let own = Selection::along_axis(target, axis, indices)?;
                selection = selection.intersect(&own, target)?;
            }
            None => residual.push(masked),
        }
    }
    if residual.is_empty() {
        return Ok(selection.coarsen(target));
    }

    // `keep[i]` is for the i-th retained cell, in row-major order.
    let cells = selection.cell_indices(target);
    let mut keep = vec![true; cells.len()];
    for masked in &residual {
        let at_cells = masked.broadcast_map(target)?.gather_indices_for(&selection);
        let values = masked.take_indices(&at_cells)?;
        let mask = as_boolean(&values)?;
        for (i, slot) in keep.iter_mut().enumerate() {
            *slot &= mask.is_valid(i) && mask.value(i);
        }
        cells_evaluated.add(cells.len());
    }
    let kept: UInt64Array = cells
        .values()
        .iter()
        .zip(&keep)
        .filter(|(_, keep)| **keep)
        .map(|(cell, _)| *cell)
        .collect();
    Ok(Selection::CellMask(kept).coarsen(target))
}

/// The target axis of a one-axis footprint, or `None` for any other
/// footprint. A size-1 axis that broadcasts onto a longer target axis does not
/// count.
fn single_target_axis(footprint: &Dimensions, target: &Dimensions) -> Option<usize> {
    let [dim] = footprint.iter().collect::<Vec<_>>()[..] else {
        return None;
    };
    let axis = target.position(dim.name())?;
    (target.get(axis).size() == dim.size()).then_some(axis)
}

fn as_boolean(array: &ArrayRef) -> Result<&BooleanArray> {
    array
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| {
            DataFusionError::Plan(
                "NdFilterExec predicate did not evaluate to a boolean".to_string(),
            )
        })
}

impl DisplayAs for NdFilterExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let preds: Vec<String> = self.predicates.iter().map(|p| p.to_string()).collect();
        write!(f, "NdFilterExec: predicate=[{}]", preds.join(" AND "))
    }
}

impl ExecutionPlan for NdFilterExec {
    fn name(&self) -> &str {
        "NdFilterExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let input = one_child("NdFilterExec", children)?;
        Ok(Arc::new(Self::try_new(
            input,
            self.predicates.clone(),
            self.registry.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // Like the other nd operators, the real output is the un-broadcast nd
        // stream from `execute_nd`; a standalone execution wraps this node in an
        // `NdBroadcastExec` to materialize. In a real plan an `NdBroadcastExec`
        // sits above and pulls `execute_nd` directly, so this path is unused.
        execute_flat(self, &self.registry, partition, context)
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
    /// The nd child must stay a direct child: a repartition between two nd
    /// nodes would break the nd side channel.
    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }
}

impl NdExecutionPlan for NdFilterExec {
    fn execute_nd(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableNdBatchStream> {
        let baseline = BaselineMetrics::new(&self.metrics, partition);
        let filter_metrics = FilterMetrics {
            input_rows: MetricBuilder::new(&self.metrics).counter("input_rows", partition),
            rows_pruned: MetricBuilder::new(&self.metrics).counter("rows_pruned", partition),
            cells_evaluated: MetricBuilder::new(&self.metrics)
                .counter("cells_evaluated", partition),
            project: ProjectMetrics::new(&self.metrics, partition),
        };
        let this = self.clone();
        let stream = self
            .nd_input
            .execute_nd(partition, context)?
            .map(move |item| {
                let _timer = baseline.elapsed_compute().timer();
                let batch = item?;
                let filtered = this.filter_batch(&batch, &filter_metrics)?;
                baseline.record_output(filtered.num_rows());
                Ok(filtered)
            });
        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{AsArray, Int32Array, UInt64Array};
    use arrow::datatypes::{DataType, Field, Int32Type, Schema, SchemaRef};
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::expressions::{binary, col, lit};

    use nd_arrow_array::array::NdArrowArray;
    use nd_arrow_array::dimensions::{Dimension, Dimensions};

    use super::*;

    fn dims(spec: &[(&str, usize)]) -> Dimensions {
        Dimensions::try_new(
            spec.iter()
                .map(|(name, size)| Dimension::new(*name, *size))
                .collect(),
        )
        .unwrap()
    }

    /// Grid (lat=3, lon=2): a `lat` coord, a `lon` coord, and a full-rank
    /// `temp{lat,lon}` data variable.
    fn test_batch() -> (SchemaRef, NdRecordBatch) {
        let schema: SchemaRef = Arc::new(Schema::new(vec![
            Field::new("lat", DataType::Int32, true),
            Field::new("lon", DataType::Int32, true),
            Field::new("temp", DataType::Int32, true),
        ]));
        let lat = NdArrowArray::try_new(
            Arc::new(Int32Array::from(vec![10, 20, 30])),
            dims(&[("lat", 3)]),
        )
        .unwrap();
        let lon =
            NdArrowArray::try_new(Arc::new(Int32Array::from(vec![1, 2])), dims(&[("lon", 2)]))
                .unwrap();
        let temp = NdArrowArray::try_new(
            Arc::new(Int32Array::from(vec![0, 1, 2, 3, 4, 5])),
            dims(&[("lat", 3), ("lon", 2)]),
        )
        .unwrap();
        let batch = NdRecordBatch::try_new(
            schema.clone(),
            vec![lat, lon, temp],
            dims(&[("lat", 3), ("lon", 2)]),
        )
        .unwrap();
        (schema, batch)
    }

    fn no_metrics() -> ProjectMetrics {
        ProjectMetrics::default()
    }

    /// Build the per-conjunct evaluation plan and compute the retained cells.
    fn select(
        schema: &SchemaRef,
        batch: &NdRecordBatch,
        preds: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Vec<u64> {
        let selection = selection_of(schema, batch, preds);
        selection.cell_indices(batch.target()).values().to_vec()
    }

    fn selection_of(
        schema: &SchemaRef,
        batch: &NdRecordBatch,
        preds: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Selection {
        let columns = preds
            .iter()
            .map(|expr| NdExprColumn::build(schema, expr))
            .collect::<Result<Vec<_>>>()
            .unwrap();
        retained_selection(&columns, batch, &no_metrics(), &Count::new()).unwrap()
    }

    fn pred(schema: &SchemaRef, name: &str, op: Operator, value: i32) -> Arc<dyn PhysicalExpr> {
        binary(col(name, schema).unwrap(), op, lit(value), schema).unwrap()
    }

    /// A one-axis filter returns axis indices, and the batch after the filter
    /// reports itself as a rectangle.
    #[test]
    fn a_one_axis_filter_keeps_a_rectangle() {
        let (schema, batch) = test_batch();
        let selection = selection_of(
            &schema,
            &batch,
            vec![pred(&schema, "lat", Operator::Gt, 15)],
        );
        assert_eq!(
            selection,
            Selection::AxisIndices(vec![Some(UInt64Array::from(vec![1u64, 2])), None])
        );
        let filtered = batch.with_selection(selection).unwrap();
        assert!(filtered.is_rectangle());
        assert_eq!(filtered.num_rows(), 4);
    }

    /// One-axis conjuncts on two axes intersect to one rectangle.
    #[test]
    fn one_axis_filters_on_two_axes_keep_a_rectangle() {
        let (schema, batch) = test_batch();
        let selection = selection_of(
            &schema,
            &batch,
            vec![
                pred(&schema, "lat", Operator::Lt, 25),
                pred(&schema, "lon", Operator::Eq, 2),
            ],
        );
        assert_eq!(
            selection,
            Selection::AxisIndices(vec![
                Some(UInt64Array::from(vec![0u64, 1])),
                Some(UInt64Array::from(vec![1u64])),
            ])
        );
        assert_eq!(selection.cell_indices(batch.target()).values(), &[1, 3]);
    }

    /// A monotone inner-axis predicate keeps a prefix of each profile.
    #[test]
    fn an_inner_axis_prefix_filter_is_ragged() {
        let schema: SchemaRef =
            Arc::new(Schema::new(vec![Field::new("PRES", DataType::Int32, true)]));
        let grid = dims(&[("N_PROF", 2), ("N_LEVELS", 3)]);
        let pres = NdArrowArray::try_new(
            Arc::new(Int32Array::from(vec![10, 20, 30, 10, 30, 40])),
            grid.clone(),
        )
        .unwrap();
        let batch = NdRecordBatch::try_new(schema.clone(), vec![pres], grid).unwrap();

        let selection = selection_of(
            &schema,
            &batch,
            vec![pred(&schema, "PRES", Operator::Lt, 25)],
        );
        assert_eq!(
            selection,
            Selection::Ragged {
                lengths: UInt64Array::from(vec![2u64, 1])
            }
        );
    }

    /// A multi-axis conjunct reads its mask only at the cells that the
    /// one-axis conjuncts keep.
    #[test]
    fn a_multi_axis_mask_is_read_only_at_retained_cells() {
        let (schema, batch) = test_batch();
        let columns = [
            pred(&schema, "lat", Operator::Gt, 15),
            pred(&schema, "temp", Operator::GtEq, 3),
        ]
        .iter()
        .map(|expr| NdExprColumn::build(&schema, expr))
        .collect::<Result<Vec<_>>>()
        .unwrap();
        let evaluated = Count::new();
        retained_selection(&columns, &batch, &no_metrics(), &evaluated).unwrap();
        // `lat > 15` keeps 4 of the 6 cells.
        assert_eq!(evaluated.value(), 4);
    }

    /// A full-rank conjunct makes the result a cell mask.
    #[test]
    fn a_full_rank_filter_gives_a_cell_mask() {
        let (schema, batch) = test_batch();
        let selection = selection_of(
            &schema,
            &batch,
            vec![
                pred(&schema, "lat", Operator::Gt, 15),
                pred(&schema, "temp", Operator::GtEq, 3),
            ],
        );
        assert_eq!(
            selection,
            Selection::CellMask(UInt64Array::from(vec![3u64, 4, 5]))
        );
        assert!(!selection.is_rectangle());
    }

    /// A single-axis predicate (`lat > 15`) selects whole lat-slices: cells
    /// where lat ∈ {20, 30}, i.e. target rows 2,3,4,5 of the C-order grid.
    #[test]
    fn single_axis_predicate_selects_slices() {
        let (schema, batch) = test_batch();
        let pred = pred(&schema, "lat", Operator::Gt, 15);
        assert_eq!(select(&schema, &batch, vec![pred]), vec![2, 3, 4, 5]);

        // Materializing the selected batch gathers exactly those cells.
        let out = batch
            .with_selection(Selection::CellMask(UInt64Array::from(vec![2u64, 3, 4, 5])))
            .unwrap()
            .materialize()
            .unwrap();
        assert_eq!(
            out.column(0).as_primitive::<Int32Type>().values(),
            &[20, 20, 30, 30]
        );
        assert_eq!(
            out.column(2).as_primitive::<Int32Type>().values(),
            &[2, 3, 4, 5]
        );
    }

    /// A cross-axis predicate (`lat + lon > 22`) selects an arbitrary,
    /// non-factorizable subset of the grid — handled the same way.
    #[test]
    fn cross_axis_predicate_selects_arbitrary_cells() {
        let (schema, batch) = test_batch();
        // Grid cells (lat,lon): (10,1)=11,(10,2)=12,(20,1)=21,(20,2)=22,
        // (30,1)=31,(30,2)=32. `>22` keeps cells 4,5 (lat=30).
        let pred = binary(
            binary(
                col("lat", &schema).unwrap(),
                Operator::Plus,
                col("lon", &schema).unwrap(),
                &schema,
            )
            .unwrap(),
            Operator::Gt,
            lit(22i32),
            &schema,
        )
        .unwrap();
        assert_eq!(select(&schema, &batch, vec![pred]), vec![4, 5]);
    }

    /// Two conjuncts intersect: `lat > 15 AND lon = 2` keeps cells 3 and 5.
    #[test]
    fn conjuncts_intersect() {
        let (schema, batch) = test_batch();
        let p1 = pred(&schema, "lat", Operator::Gt, 15);
        let p2 = pred(&schema, "lon", Operator::Eq, 2);
        assert_eq!(select(&schema, &batch, vec![p1, p2]), vec![3, 5]);
    }

    /// A filter over an already-selected batch intersects with the inbound
    /// selection rather than replacing it.
    #[test]
    fn intersects_inbound_selection() {
        let (schema, batch) = test_batch();
        let batch = batch
            .with_selection(Selection::CellMask(UInt64Array::from(vec![0u64, 2, 4])))
            .unwrap();
        // lat > 15 keeps target rows 2,3,4,5; intersect with {0,2,4} → {2,4}.
        let pred = pred(&schema, "lat", Operator::Gt, 15);
        assert_eq!(select(&schema, &batch, vec![pred]), vec![2, 4]);
    }

    /// Two conjuncts on the *same* axis form a range: `lat > 10 AND lat < 30`
    /// keeps lat = 20, i.e. cells 2 and 3.
    #[test]
    fn same_axis_conjuncts_form_a_range() {
        let (schema, batch) = test_batch();
        let lo = pred(&schema, "lat", Operator::Gt, 10);
        let hi = pred(&schema, "lat", Operator::Lt, 30);
        assert_eq!(select(&schema, &batch, vec![lo, hi]), vec![2, 3]);
    }

    /// A predicate on the *inner* axis only (`lon = 2`) tiles across lat: cells
    /// 1, 3, 5.
    #[test]
    fn inner_axis_predicate_tiles() {
        let (schema, batch) = test_batch();
        let pred = pred(&schema, "lon", Operator::Eq, 2);
        assert_eq!(select(&schema, &batch, vec![pred]), vec![1, 3, 5]);
    }

    /// A predicate on a full-rank data variable (`temp >= 3`) selects an
    /// arbitrary subset directly on the grid: cells 3, 4, 5.
    #[test]
    fn data_variable_predicate_selects_cells() {
        let (schema, batch) = test_batch();
        let pred = pred(&schema, "temp", Operator::GtEq, 3);
        assert_eq!(select(&schema, &batch, vec![pred]), vec![3, 4, 5]);
    }

    /// A single conjunct containing an `OR` (not split) unions its branches:
    /// `lat < 15 OR lon = 2` keeps cells 0,1 (lat=10) ∪ 1,3,5 (lon=2) = {0,1,3,5}.
    #[test]
    fn or_predicate_unions_branches() {
        let (schema, batch) = test_batch();
        let pred = binary(
            pred(&schema, "lat", Operator::Lt, 15),
            Operator::Or,
            pred(&schema, "lon", Operator::Eq, 2),
            &schema,
        )
        .unwrap();
        assert_eq!(select(&schema, &batch, vec![pred]), vec![0, 1, 3, 5]);
    }

    /// A cross-axis conjunct intersected with a single-axis one:
    /// `lat + lon > 22 AND lon = 1` keeps only cell 4.
    #[test]
    fn cross_axis_and_single_axis_intersect() {
        let (schema, batch) = test_batch();
        let cross = binary(
            binary(
                col("lat", &schema).unwrap(),
                Operator::Plus,
                col("lon", &schema).unwrap(),
                &schema,
            )
            .unwrap(),
            Operator::Gt,
            lit(22i32),
            &schema,
        )
        .unwrap();
        let inner = pred(&schema, "lon", Operator::Eq, 1);
        assert_eq!(select(&schema, &batch, vec![cross, inner]), vec![4]);
    }

    /// A predicate no cell satisfies yields an empty selection.
    #[test]
    fn predicate_selecting_nothing_is_empty() {
        let (schema, batch) = test_batch();
        let pred = pred(&schema, "lat", Operator::Gt, 100);
        assert_eq!(select(&schema, &batch, vec![pred]), Vec::<u64>::new());
    }

    /// A predicate every cell satisfies retains the whole grid, in order.
    #[test]
    fn predicate_selecting_everything_keeps_all() {
        let (schema, batch) = test_batch();
        let pred = pred(&schema, "lat", Operator::GtEq, 10);
        assert_eq!(select(&schema, &batch, vec![pred]), vec![0, 1, 2, 3, 4, 5]);
    }

    /// A cell where the predicate evaluates to NULL is excluded (SQL `WHERE`
    /// keeps only TRUE). With `lat = [10, null, 30]`, `lat > 15` drops the null
    /// slice and keeps only the lat=30 cells.
    #[test]
    fn null_predicate_value_excludes_cell() {
        let schema: SchemaRef = Arc::new(Schema::new(vec![
            Field::new("lat", DataType::Int32, true),
            Field::new("lon", DataType::Int32, true),
        ]));
        let lat = NdArrowArray::try_new(
            Arc::new(Int32Array::from(vec![Some(10), None, Some(30)])),
            dims(&[("lat", 3)]),
        )
        .unwrap();
        let lon =
            NdArrowArray::try_new(Arc::new(Int32Array::from(vec![1, 2])), dims(&[("lon", 2)]))
                .unwrap();
        let batch = NdRecordBatch::try_new(
            schema.clone(),
            vec![lat, lon],
            dims(&[("lat", 3), ("lon", 2)]),
        )
        .unwrap();

        let pred = pred(&schema, "lat", Operator::Gt, 15);
        assert_eq!(select(&schema, &batch, vec![pred]), vec![4, 5]);
    }
}
