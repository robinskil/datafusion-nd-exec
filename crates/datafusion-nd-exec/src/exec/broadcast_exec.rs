//! Terminal materializer of the nd pipeline.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use datafusion::common::config::ConfigOptions;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::filter_pushdown::{
    ChildPushdownResult, FilterDescription, FilterPushdownPhase, FilterPushdownPropagation,
};
use datafusion::physical_plan::metrics::{
    BaselineMetrics, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet,
};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};

use super::{NdExecutionPlan, materialize_nd_stream, require_nd_input};
use crate::registry::NdNodeRegistry;

/// Boundary between the nd pipeline and the rest of the plan: pulls nd
/// batches from its child and emits flat, fully-broadcast `RecordBatch`es.
/// Broadcast and any accumulated selection compose into a single gather per
/// column, so the unfiltered cross-product never exists in memory.
#[derive(Debug, Clone)]
pub struct NdBroadcastExec {
    input: Arc<dyn ExecutionPlan>,
    /// The nd side of `input`.
    nd_input: Arc<dyn NdExecutionPlan>,
    registry: Arc<NdNodeRegistry>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl NdBroadcastExec {
    /// Build a broadcast over `input`, resolved through the default registry.
    pub fn try_new(input: Arc<dyn ExecutionPlan>) -> Result<Self> {
        Self::try_new_with_registry(input, NdNodeRegistry::shared_default())
    }

    /// Build a broadcast over `input`, resolved through `registry`.
    pub fn try_new_with_registry(
        input: Arc<dyn ExecutionPlan>,
        registry: Arc<NdNodeRegistry>,
    ) -> Result<Self> {
        let nd_input = require_nd_input("NdBroadcastExec", &input, &registry)?;
        let properties = input.properties().clone();
        Ok(Self {
            input,
            nd_input,
            registry,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    /// The registry that resolves the nd child.
    pub fn registry(&self) -> &Arc<NdNodeRegistry> {
        &self.registry
    }

    /// The nd side of the child.
    pub fn nd_input(&self) -> &Arc<dyn NdExecutionPlan> {
        &self.nd_input
    }

    /// The names of the nd nodes below this boundary, top down.
    pub fn region(&self) -> Vec<String> {
        let mut names = Vec::new();
        let mut node = self.input.clone();
        loop {
            names.push(node.name().to_string());
            let next = match node.children()[..] {
                [child] if self.registry.as_nd_plan(child).is_some() => child.clone(),
                _ => break,
            };
            node = next;
        }
        names
    }

    /// The nd-aware child whose batches this node materializes.
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }
}

impl DisplayAs for NdBroadcastExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NdBroadcastExec: region=[{}]", self.region().join(", "))
    }
}

impl ExecutionPlan for NdBroadcastExec {
    fn name(&self) -> &str {
        "NdBroadcastExec"
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
            DataFusionError::Internal("NdBroadcastExec expects exactly one child".to_string())
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
        let stream = self.nd_input.execute_nd(partition, context)?;
        let baseline = BaselineMetrics::new(&self.metrics, partition);
        // Per-column materialization outcome: a real broadcast gather, or a
        // zero-copy pass-through of an already-full-rank column.
        let broadcasts =
            MetricBuilder::new(&self.metrics).counter("implicit_broadcasts", partition);
        let passthroughs =
            MetricBuilder::new(&self.metrics).counter("passthrough_columns", partition);
        Ok(materialize_nd_stream(
            self.input.schema(),
            stream,
            baseline,
            broadcasts,
            passthroughs,
        ))
    }

    /// Pass the filters down the nd pipeline, towards the file source that can
    /// prune on them.
    ///
    /// The broadcast is where the flat schema begins, so this is the node a
    /// `FilterExec` offers its predicate to. Below it the columns are the
    /// encoded form of the same columns, named and ordered the same, so the
    /// predicate travels unchanged.
    fn gather_filters_for_pushdown(
        &self,
        _phase: FilterPushdownPhase,
        parent_filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> Result<FilterDescription> {
        super::offer_filters_to_child(parent_filters, &self.input)
    }

    /// The broadcast applies no predicate, so the `FilterExec` above stays.
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
    /// The nd child must stay a direct child: a repartition between two nd
    /// nodes would break the nd side channel.
    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    /// A sort must not move below this node: below it the rows are nd
    /// batches or encoded chunks. The order still reaches the plan through the
    /// equivalence properties.
    fn maintains_input_order(&self) -> Vec<bool> {
        vec![false]
    }
}
