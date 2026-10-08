//! Decoder node: nd-encoded `RecordBatch`es from a child plan → nd batches.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::compute::SortOptions;
use arrow::datatypes::Schema;
use datafusion::common::config::ConfigOptions;
use datafusion::error::Result;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr, PhysicalSortExpr};
use datafusion::physical_plan::filter_pushdown::{
    ChildPushdownResult, FilterDescription, FilterPushdownPhase, FilterPushdownPropagation,
};
use datafusion::physical_plan::metrics::{
    BaselineMetrics, Count, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet,
};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};
use futures::{StreamExt, TryStreamExt};
use nd_arrow_array::encoding::{decode_nd_record_batch_row, logical_schema, nd_batch_count};
use nd_arrow_array::{AxisOrder, NdGridAxes, SelectionKind};

use super::{NdBroadcastExec, NdExecutionPlan, SendableNdBatchStream, one_child};
use crate::registry::NdNodeRegistry;

/// Leaf of the nd pipeline: decodes the `nd.array`-encoded `RecordBatch`es
/// produced by a child plan (typically a `DataSourceExec` whose file opener
/// emits encoded batches) into un-broadcast [`NdRecordBatch`]es.
///
/// Its own output schema is the *logical* (decoded) schema; broadcasting to
/// flat Arrow happens in [`NdBroadcastExec`] above it.
#[derive(Debug, Clone)]
pub struct NdSourceExec {
    /// Child plan producing nd-encoded `RecordBatch`es.
    input: Arc<dyn ExecutionPlan>,
    /// The grid axes that the format declares, if any.
    grid_axes: Option<Arc<NdGridAxes>>,
    /// True when each partition yields its chunks in the order of the outer axis.
    ordered_chunks: bool,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl NdSourceExec {
    /// Wrap a child plan whose output columns are `nd.array`-encoded structs.
    pub fn try_new(input: Arc<dyn ExecutionPlan>) -> Result<Self> {
        Self::build(input, None, false)
    }

    /// Declare the grid axes of the scan. Each chunk then lies on these axes,
    /// in this order, and each axis has a coordinate column with its name. A
    /// grid sink needs this declaration.
    pub fn with_grid_axes(self, axes: NdGridAxes) -> Result<Self> {
        Self::build(self.input, Some(Arc::new(axes)), self.ordered_chunks)
    }

    /// Report the sort order of the grid.
    ///
    /// A format calls this when each partition yields its chunks split only
    /// along the outer axis, in the order of that axis, and every chunk holds
    /// every column. The node then reports a lexicographic order over the
    /// outer declared grid axes, see [`with_grid_axes`](Self::with_grid_axes).
    /// The order stops at the first `Unordered` axis, or at an axis whose
    /// coordinate column is not in the schema.
    pub fn with_ordered_chunks(self) -> Result<Self> {
        Self::build(self.input, self.grid_axes, true)
    }

    fn build(
        input: Arc<dyn ExecutionPlan>,
        grid_axes: Option<Arc<NdGridAxes>>,
        ordered_chunks: bool,
    ) -> Result<Self> {
        let logical_schema = logical_schema(&input.schema())?;
        let ordering = match (&grid_axes, ordered_chunks) {
            (Some(axes), true) => axis_ordering(axes, &logical_schema),
            _ => vec![],
        };
        let eq_properties = if ordering.is_empty() {
            EquivalenceProperties::new(logical_schema)
        } else {
            EquivalenceProperties::new_with_orderings(logical_schema, [ordering])
        };
        // Same partitioning/emission/boundedness as the child; only the schema
        // changes (encoded structs → logical value types).
        let properties = Arc::new(
            input
                .properties()
                .as_ref()
                .clone()
                .with_eq_properties(eq_properties),
        );
        Ok(Self {
            input,
            grid_axes,
            ordered_chunks,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }
}

/// The lexicographic order of the outer declared axes whose coordinate column
/// is in `logical`.
fn axis_ordering(axes: &NdGridAxes, logical: &Schema) -> Vec<PhysicalSortExpr> {
    let mut ordering = Vec::new();
    for (axis, order) in axes.axes() {
        let descending = match order {
            AxisOrder::Ascending => false,
            AxisOrder::Descending => true,
            AxisOrder::Unordered => break,
        };
        let Ok(index) = logical.index_of(axis) else {
            break;
        };
        // No coordinate value is null. The null placement is the SQL default,
        // so a plain `ORDER BY` matches.
        ordering.push(PhysicalSortExpr::new(
            Arc::new(Column::new(axis, index)),
            SortOptions::new(descending, descending),
        ));
    }
    ordering
}

impl DisplayAs for NdSourceExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.grid_axes {
            Some(axes) => {
                let names: Vec<&str> = axes.axes().iter().map(|(name, _)| name.as_str()).collect();
                write!(f, "NdSourceExec: grid_axes=[{}]", names.join(", "))
            }
            None => write!(f, "NdSourceExec"),
        }
    }
}

impl ExecutionPlan for NdSourceExec {
    fn name(&self) -> &str {
        "NdSourceExec"
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
        let input = one_child("NdSourceExec", children)?;
        Ok(Arc::new(Self::build(
            input,
            self.grid_axes.clone(),
            self.ordered_chunks,
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // The source only produces nd batches (`execute_nd`); flattening to
        // Arrow lives in `NdBroadcastExec`. When executed as a standalone plan,
        // borrow that broadcast behaviour rather than duplicating it.
        NdBroadcastExec::try_new(Arc::new(self.clone()), NdNodeRegistry::shared_default())?
            .execute(partition, context)
    }

    /// Hand the filters to the file source below, which prunes on them.
    fn gather_filters_for_pushdown(
        &self,
        _phase: FilterPushdownPhase,
        parent_filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> Result<FilterDescription> {
        super::offer_filters_to_child(parent_filters, &self.input)
    }

    /// A decoder applies no predicate, so the `FilterExec` above stays.
    fn handle_child_pushdown_result(
        &self,
        _phase: FilterPushdownPhase,
        child_pushdown_result: ChildPushdownResult,
        _config: &ConfigOptions,
    ) -> Result<FilterPushdownPropagation<Arc<dyn ExecutionPlan>>> {
        Ok(super::filters_stay_above(child_pushdown_result))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

impl NdExecutionPlan for NdSourceExec {
    fn grid_axes(&self) -> Option<Arc<NdGridAxes>> {
        self.grid_axes.clone()
    }

    /// A decoded batch keeps every cell.
    fn max_output_selection(&self) -> SelectionKind {
        SelectionKind::Full
    }

    fn execute_nd(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableNdBatchStream> {
        let baseline = BaselineMetrics::new(&self.metrics, partition);
        let nd_batches: Count = MetricBuilder::new(&self.metrics).counter("nd_batches", partition);

        let input = self.input.execute(partition, context)?;
        let stream = input
            .map(move |item| {
                let _timer = baseline.elapsed_compute().timer();
                match item {
                    // An empty encoded batch carries no nd array — skip it.
                    Ok(batch) if batch.num_rows() == 0 => Ok(Vec::new()),
                    Ok(batch) => {
                        // One nd array per encoded row: a batch that has been
                        // through a coalescing operator (RoundRobinBatch
                        // repartitioning, `CoalesceBatchesExec`, …) carries
                        // several, and decoding only row 0 would silently drop
                        // the rest.
                        let count = nd_batch_count(&batch);
                        let mut decoded = Vec::with_capacity(count);
                        for row in 0..count {
                            let nd = decode_nd_record_batch_row(&batch, row)?;
                            nd_batches.add(1);
                            baseline.record_output(nd.num_rows());
                            decoded.push(nd);
                        }
                        Ok(decoded)
                    }
                    Err(e) => Err(e),
                }
            })
            .map_ok(|decoded| futures::stream::iter(decoded.into_iter().map(Ok)))
            .try_flatten();
        Ok(Box::pin(stream))
    }
}
