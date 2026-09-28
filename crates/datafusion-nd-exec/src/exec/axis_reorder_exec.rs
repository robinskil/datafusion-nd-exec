//! Permute the target axes of nd batches.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::array::UInt64Array;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};
use futures::StreamExt;
use nd_arrow_array::{Dimensions, NdRecordBatch, Selection};

use super::{NdBroadcastExec, NdExecutionPlan, SendableNdBatchStream, require_nd_input};
use crate::registry::NdNodeRegistry;

/// Put the named axes first, in the given order, and keep the other target
/// axes after them in their current order. A name that a batch does not have
/// is ignored.
///
/// No column data moves: the broadcast aligns the columns by axis name, so
/// the strides absorb the new order. The rows then come out in the new
/// row-major order.
#[derive(Debug, Clone)]
pub struct NdAxisReorderExec {
    input: Arc<dyn ExecutionPlan>,
    nd_input: Arc<dyn NdExecutionPlan>,
    registry: Arc<NdNodeRegistry>,
    axes: Vec<String>,
    properties: Arc<PlanProperties>,
}

impl NdAxisReorderExec {
    pub fn try_new(input: Arc<dyn ExecutionPlan>, axes: Vec<String>) -> Result<Self> {
        Self::try_new_with_registry(input, axes, NdNodeRegistry::shared_default())
    }

    /// Like [`try_new`](Self::try_new), but resolves the nd child through
    /// `registry`.
    pub fn try_new_with_registry(
        input: Arc<dyn ExecutionPlan>,
        axes: Vec<String>,
        registry: Arc<NdNodeRegistry>,
    ) -> Result<Self> {
        let nd_input = require_nd_input("NdAxisReorderExec", &input, &registry)?;
        // The new row order drops any order of the input.
        let properties = Arc::new(
            input
                .properties()
                .as_ref()
                .clone()
                .with_eq_properties(EquivalenceProperties::new(input.schema())),
        );
        Ok(Self {
            input,
            nd_input,
            registry,
            axes,
            properties,
        })
    }

    pub fn axes(&self) -> &[String] {
        &self.axes
    }
}

/// The positions of the old target axes in the new order.
fn permutation(target: &Dimensions, axes: &[String]) -> Vec<usize> {
    let mut order: Vec<usize> = Vec::with_capacity(target.rank());
    for axis in axes.iter().filter_map(|a| target.position(a)) {
        if !order.contains(&axis) {
            order.push(axis);
        }
    }
    let rest: Vec<usize> = (0..target.rank())
        .filter(|axis| !order.contains(axis))
        .collect();
    order.extend(rest);
    order
}

/// Reorder the target of one batch.
fn reorder(batch: NdRecordBatch, axes: &[String]) -> Result<NdRecordBatch> {
    let old = batch.target().clone();
    let order = permutation(&old, axes);
    if order.iter().enumerate().all(|(i, &axis)| i == axis) {
        return Ok(batch);
    }
    let new = Dimensions::try_new(order.iter().map(|&axis| old.get(axis).clone()).collect())?;
    let selection = match batch.selection() {
        Selection::Full => Selection::Full,
        Selection::AxisIndices(indices) => {
            Selection::AxisIndices(order.iter().map(|&axis| indices[axis].clone()).collect())
        }
        other => {
            // Map each kept cell to its index in the new row-major order.
            let old_strides = old.c_strides();
            let new_strides = new.c_strides();
            let shape = old.shape();
            let mut cells: Vec<u64> = other
                .cell_indices(&old)
                .values()
                .iter()
                .map(|&cell| {
                    order
                        .iter()
                        .enumerate()
                        .map(|(new_axis, &old_axis)| {
                            let coord = (cell as usize / old_strides[old_axis]) % shape[old_axis];
                            (coord * new_strides[new_axis]) as u64
                        })
                        .sum()
                })
                .collect();
            cells.sort_unstable();
            Selection::CellMask(UInt64Array::from(cells)).coarsen(&new)
        }
    };
    Ok(
        NdRecordBatch::try_new(batch.schema().clone(), batch.columns().to_vec(), new)?
            .with_selection(selection)?,
    )
}

impl DisplayAs for NdAxisReorderExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NdAxisReorderExec: axes=[{}]", self.axes.join(", "))
    }
}

impl ExecutionPlan for NdAxisReorderExec {
    fn name(&self) -> &str {
        "NdAxisReorderExec"
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
        let [input] = <[_; 1]>::try_from(children).map_err(|_| {
            DataFusionError::Internal("NdAxisReorderExec expects exactly one child".to_string())
        })?;
        Ok(Arc::new(Self::try_new_with_registry(
            input,
            self.axes.clone(),
            self.registry.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        NdBroadcastExec::try_new_with_registry(Arc::new(self.clone()), self.registry.clone())?
            .execute(partition, context)
    }

    /// The nd child must stay a direct child.
    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }
}

impl NdExecutionPlan for NdAxisReorderExec {
    fn execute_nd(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableNdBatchStream> {
        let axes = self.axes.clone();
        let stream = self
            .nd_input
            .execute_nd(partition, context)?
            .map(move |item| reorder(item?, &axes));
        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::expressions::{binary, col, lit};
    use datafusion::physical_plan::collect;
    use futures::TryStreamExt;

    use super::*;
    use crate::exec::NdFilterExec;
    use crate::testing::{grid_table, sorted_rows};

    fn source() -> Arc<dyn ExecutionPlan> {
        let scan = grid_table()
            .unwrap()
            .nd_scan(None, NdNodeRegistry::shared_default())
            .unwrap();
        scan.children()[0].clone()
    }

    async fn rows(plan: Arc<dyn ExecutionPlan>) -> Vec<String> {
        let batches = collect(plan, Arc::new(TaskContext::default()))
            .await
            .unwrap();
        sorted_rows(&batches).unwrap()
    }

    #[tokio::test]
    async fn the_named_axes_come_first() {
        let reorder = NdAxisReorderExec::try_new(source(), vec!["lon".to_string()]).unwrap();
        let batches: Vec<_> = reorder
            .execute_nd(0, Arc::new(TaskContext::default()))
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let names: Vec<&str> = batches[0].target().iter().map(|d| d.name()).collect();
        assert_eq!(names, ["lon", "time", "lat"]);
    }

    #[tokio::test]
    async fn a_reorder_keeps_the_rows() {
        let reorder = Arc::new(
            NdAxisReorderExec::try_new(source(), vec!["lon".to_string(), "lat".to_string()])
                .unwrap(),
        );
        assert_eq!(rows(reorder).await, rows(source()).await);
    }

    #[tokio::test]
    async fn a_reorder_keeps_the_selection() {
        let schema = source().schema();
        let predicate = |name: &str, op, value: f64| {
            binary(col(name, &schema).unwrap(), op, lit(value), &schema).unwrap()
        };
        // A rectangle and a cell mask.
        for predicates in [
            vec![predicate("lat", Operator::Gt, -1.0)],
            vec![predicate("sst", Operator::Gt, 2.0)],
        ] {
            let filter: Arc<dyn ExecutionPlan> =
                Arc::new(NdFilterExec::try_new(source(), predicates).unwrap());
            let reorder = Arc::new(
                NdAxisReorderExec::try_new(filter.clone(), vec!["lon".to_string()]).unwrap(),
            );
            assert_eq!(rows(reorder).await, rows(filter).await);
        }
    }
}
