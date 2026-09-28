//! Merge the partitions of an nd input.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PlanProperties, SendableRecordBatchStream,
};
use nd_arrow_array::SelectionKind;

use super::{NdBroadcastExec, NdExecutionPlan, SendableNdBatchStream, require_nd_input};
use crate::registry::NdNodeRegistry;

/// Merge all partitions of an nd input into one partition. The batches come
/// in the order in which the input partitions yield them.
#[derive(Debug, Clone)]
pub struct NdCoalescePartitionsExec {
    input: Arc<dyn ExecutionPlan>,
    nd_input: Arc<dyn NdExecutionPlan>,
    registry: Arc<NdNodeRegistry>,
    properties: Arc<PlanProperties>,
}

impl NdCoalescePartitionsExec {
    pub fn try_new(input: Arc<dyn ExecutionPlan>) -> Result<Self> {
        Self::try_new_with_registry(input, NdNodeRegistry::shared_default())
    }

    /// Like [`try_new`](Self::try_new), but resolves the nd child through
    /// `registry`.
    pub fn try_new_with_registry(
        input: Arc<dyn ExecutionPlan>,
        registry: Arc<NdNodeRegistry>,
    ) -> Result<Self> {
        let nd_input = require_nd_input("NdCoalescePartitionsExec", &input, &registry)?;
        // The merge interleaves the partitions, so no order survives.
        let properties = Arc::new(
            input
                .properties()
                .as_ref()
                .clone()
                .with_eq_properties(EquivalenceProperties::new(input.schema()))
                .with_partitioning(Partitioning::UnknownPartitioning(1)),
        );
        Ok(Self {
            input,
            nd_input,
            registry,
            properties,
        })
    }
}

/// All partitions of `input` as one nd stream.
pub(crate) fn merge_partitions(
    input: &Arc<dyn ExecutionPlan>,
    nd_input: &Arc<dyn NdExecutionPlan>,
    context: Arc<TaskContext>,
) -> Result<SendableNdBatchStream> {
    let streams = (0..input.output_partitioning().partition_count())
        .map(|partition| nd_input.execute_nd(partition, context.clone()))
        .collect::<Result<Vec<_>>>()?;
    Ok(Box::pin(futures::stream::select_all(streams)))
}

impl DisplayAs for NdCoalescePartitionsExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NdCoalescePartitionsExec")
    }
}

impl ExecutionPlan for NdCoalescePartitionsExec {
    fn name(&self) -> &str {
        "NdCoalescePartitionsExec"
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
            DataFusionError::Internal(
                "NdCoalescePartitionsExec expects exactly one child".to_string(),
            )
        })?;
        Ok(Arc::new(Self::try_new_with_registry(
            input,
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

impl NdExecutionPlan for NdCoalescePartitionsExec {
    fn execute_nd(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableNdBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "NdCoalescePartitionsExec has one partition, got partition {partition}"
            )));
        }
        merge_partitions(&self.input, &self.nd_input, context)
    }

    fn max_output_selection(&self) -> SelectionKind {
        self.nd_input.max_output_selection()
    }
}
