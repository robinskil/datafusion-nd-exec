//! An nd leaf with no data.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};
use nd_arrow_array::SelectionKind;

use super::{NdBroadcastExec, NdExecutionPlan, SendableNdBatchStream};

/// An nd leaf that yields no batches in each of its partitions. A sink or an
/// optimizer uses it where an nd input has no data.
#[derive(Debug, Clone)]
pub struct NdEmptyExec {
    properties: Arc<PlanProperties>,
}

impl NdEmptyExec {
    pub fn new(schema: SchemaRef, partitions: usize) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(partitions),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Self {
            properties: Arc::new(properties),
        }
    }
}

impl DisplayAs for NdEmptyExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NdEmptyExec")
    }
}

impl ExecutionPlan for NdEmptyExec {
    fn name(&self) -> &str {
        "NdEmptyExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(DataFusionError::Internal(
                "NdEmptyExec has no children".to_string(),
            ));
        }
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        NdBroadcastExec::try_new(Arc::new(self.clone()))?.execute(partition, context)
    }
}

impl NdExecutionPlan for NdEmptyExec {
    fn execute_nd(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableNdBatchStream> {
        let partitions = self.properties.output_partitioning().partition_count();
        if partition >= partitions {
            return Err(DataFusionError::Internal(format!(
                "NdEmptyExec has {partitions} partitions, got partition {partition}"
            )));
        }
        Ok(Box::pin(futures::stream::empty()))
    }

    fn max_output_selection(&self) -> SelectionKind {
        SelectionKind::Full
    }
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::physical_plan::collect;

    use super::*;

    #[tokio::test]
    async fn an_empty_leaf_yields_no_rows() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "sst",
            DataType::Float64,
            true,
        )]));
        let empty = Arc::new(NdEmptyExec::new(schema.clone(), 2));
        let broadcast = Arc::new(NdBroadcastExec::try_new(empty).unwrap());
        assert_eq!(broadcast.schema(), schema);
        let batches = collect(broadcast, Arc::new(TaskContext::default()))
            .await
            .unwrap();
        assert!(batches.is_empty());
    }
}
