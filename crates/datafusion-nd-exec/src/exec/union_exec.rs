//! Union of nd inputs.

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

/// The partitions of all nd inputs in sequence. Each chunk keeps its own
/// grid, so inputs with different axis sizes union without a broadcast. A
/// scan over many profile files unions the files this way.
#[derive(Debug, Clone)]
pub struct NdUnionExec {
    inputs: Vec<Arc<dyn ExecutionPlan>>,
    /// The nd side of each input.
    nd_inputs: Vec<Arc<dyn NdExecutionPlan>>,
    registry: Arc<NdNodeRegistry>,
    properties: Arc<PlanProperties>,
}

impl NdUnionExec {
    /// A union of `inputs`. All inputs must have the same schema.
    pub fn try_new(inputs: Vec<Arc<dyn ExecutionPlan>>) -> Result<Self> {
        Self::try_new_with_registry(inputs, NdNodeRegistry::shared_default())
    }

    /// Like [`try_new`](Self::try_new), but resolves the nd inputs through
    /// `registry`.
    pub fn try_new_with_registry(
        inputs: Vec<Arc<dyn ExecutionPlan>>,
        registry: Arc<NdNodeRegistry>,
    ) -> Result<Self> {
        let Some(first) = inputs.first() else {
            return Err(DataFusionError::Plan(
                "NdUnionExec requires at least one input".to_string(),
            ));
        };
        let schema = first.schema();
        if let Some(other) = inputs.iter().find(|input| input.schema() != schema) {
            return Err(DataFusionError::Plan(format!(
                "NdUnionExec inputs must have one schema, got {schema:?} and {:?}",
                other.schema()
            )));
        }
        let nd_inputs = inputs
            .iter()
            .map(|input| require_nd_input("NdUnionExec", input, &registry))
            .collect::<Result<Vec<_>>>()?;
        let partitions = inputs
            .iter()
            .map(|input| input.output_partitioning().partition_count())
            .sum();
        let properties = Arc::new(
            first
                .properties()
                .as_ref()
                .clone()
                .with_eq_properties(EquivalenceProperties::new(schema))
                .with_partitioning(Partitioning::UnknownPartitioning(partitions)),
        );
        Ok(Self {
            inputs,
            nd_inputs,
            registry,
            properties,
        })
    }

    pub fn inputs(&self) -> &[Arc<dyn ExecutionPlan>] {
        &self.inputs
    }
}

impl DisplayAs for NdUnionExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NdUnionExec")
    }
}

impl ExecutionPlan for NdUnionExec {
    fn name(&self) -> &str {
        "NdUnionExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.inputs.iter().collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::try_new_with_registry(
            children,
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

    /// The nd inputs must stay direct children.
    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false; self.inputs.len()]
    }
}

impl NdExecutionPlan for NdUnionExec {
    fn execute_nd(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableNdBatchStream> {
        let mut rest = partition;
        for (input, nd_input) in self.inputs.iter().zip(&self.nd_inputs) {
            let count = input.output_partitioning().partition_count();
            if rest < count {
                return nd_input.execute_nd(rest, context);
            }
            rest -= count;
        }
        Err(DataFusionError::Internal(format!(
            "NdUnionExec has no partition {partition}"
        )))
    }

    fn max_output_selection(&self) -> SelectionKind {
        self.nd_inputs
            .iter()
            .map(|input| input.max_output_selection())
            .max()
            .unwrap_or(SelectionKind::Full)
    }
}
