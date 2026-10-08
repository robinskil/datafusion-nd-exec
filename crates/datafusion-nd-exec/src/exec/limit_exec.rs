//! Limit over nd batches.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use datafusion::error::Result;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};
use futures::StreamExt;
use nd_arrow_array::NdGridAxes;

use super::{NdExecutionPlan, SendableNdBatchStream, execute_flat, one_child, require_nd_input};
use crate::registry::NdNodeRegistry;

/// Skip the first `skip` retained cells and keep the next `fetch`, per
/// partition, in stream order. The cut is a [`Selection::slice`], so no column
/// data moves. A cut on outer-axis boundaries keeps a rectangle.
///
/// [`Selection::slice`]: nd_arrow_array::Selection::slice
#[derive(Debug, Clone)]
pub struct NdLimitExec {
    input: Arc<dyn ExecutionPlan>,
    nd_input: Arc<dyn NdExecutionPlan>,
    registry: Arc<NdNodeRegistry>,
    skip: usize,
    fetch: Option<usize>,
    metrics: ExecutionPlanMetricsSet,
}

impl NdLimitExec {
    /// The nd child is resolved through `registry`.
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        skip: usize,
        fetch: Option<usize>,
        registry: Arc<NdNodeRegistry>,
    ) -> Result<Self> {
        let nd_input = require_nd_input("NdLimitExec", &input, &registry)?;
        Ok(Self {
            input,
            nd_input,
            registry,
            skip,
            fetch,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    pub fn skip(&self) -> usize {
        self.skip
    }
}

impl DisplayAs for NdLimitExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fetch = self.fetch.map_or("None".to_string(), |f| f.to_string());
        write!(f, "NdLimitExec: skip={}, fetch={fetch}", self.skip)
    }
}

impl ExecutionPlan for NdLimitExec {
    fn name(&self) -> &str {
        "NdLimitExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let input = one_child("NdLimitExec", children)?;
        Ok(Arc::new(Self::try_new(
            input,
            self.skip,
            self.fetch,
            self.registry.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        execute_flat(self, &self.registry, partition, context)
    }

    fn fetch(&self) -> Option<usize> {
        self.fetch
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    /// The nd child must stay a direct child.
    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }
}

impl NdExecutionPlan for NdLimitExec {
    fn grid_axes(&self) -> Option<Arc<NdGridAxes>> {
        self.nd_input.grid_axes()
    }

    fn execute_nd(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableNdBatchStream> {
        let baseline = BaselineMetrics::new(&self.metrics, partition);
        let state = (self.skip, self.fetch.unwrap_or(usize::MAX));
        let stream = self
            .nd_input
            .execute_nd(partition, context)?
            .scan(state, move |(skip, fetch), item| {
                // End the stream once the fetch is spent.
                if *fetch == 0 {
                    return futures::future::ready(None);
                }
                let cut = (|| {
                    let batch = item?;
                    let rows = batch.num_rows();
                    let skipped = (*skip).min(rows);
                    let take = (rows - skipped).min(*fetch);
                    *skip -= skipped;
                    *fetch -= take;
                    if take == 0 {
                        return Ok(None);
                    }
                    let selection = batch.selection().slice(batch.target(), skipped, take);
                    baseline.record_output(take);
                    Ok(Some(batch.with_selection(selection)?))
                })();
                futures::future::ready(Some(cut.transpose()))
            })
            .filter_map(futures::future::ready);
        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use datafusion::execution::TaskContext;
    use futures::TryStreamExt;
    use nd_arrow_array::NdRecordBatch;

    use super::*;
    use crate::testing::grid_table;

    /// The nd batches of partition 0 after a limit over the grid scan.
    async fn limited(skip: usize, fetch: Option<usize>) -> Vec<NdRecordBatch> {
        let scan = grid_table()
            .unwrap()
            .nd_scan(None, NdNodeRegistry::shared_default())
            .unwrap();
        // The nd child of the scan boundary.
        let source = scan.children()[0].clone();
        NdLimitExec::try_new(source, skip, fetch, NdNodeRegistry::shared_default())
            .unwrap()
            .execute_nd(0, Arc::new(TaskContext::default()))
            .unwrap()
            .try_collect()
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn a_cut_on_the_outer_axis_keeps_a_rectangle() {
        // Partition 0 holds one chunk of 2 x 3 x 2 cells. Skip the first time step.
        let batches = limited(6, Some(6)).await;
        assert_eq!(batches.len(), 1);
        assert!(batches[0].is_rectangle());
        assert_eq!(batches[0].num_rows(), 6);
    }

    #[tokio::test]
    async fn a_cut_stops_at_the_end_of_the_data() {
        let batches = limited(10, Some(5)).await;
        assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 2);
        assert!(limited(12, Some(5)).await.is_empty());
    }
}
