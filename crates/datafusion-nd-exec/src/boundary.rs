//! One optimizer rule that moves nodes below the nd boundary.
//!
//! An nd region is the part of a plan below an [`NdBroadcastExec`]. The
//! broadcast is the boundary: above it the rows are flat. The rule looks at
//! the node above each boundary and asks the [`NdSinker`]s of the registry if
//! the node operates on grids. If one does, the rule puts its nd replacement
//! below the boundary and looks at the next node up. If none does, the region
//! stops there.
//!
//! A round-robin `RepartitionExec` between the node and the boundary does not
//! stop the rule. Element-wise nodes commute with it, so the rule keeps it
//! above the new boundary.
//!
//! [`NdSinker`]: crate::registry::NdSinker

use std::sync::Arc;

use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::error::Result;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::{ExecutionPlan, Partitioning};

use nd_arrow_array::SelectionKind;

use crate::exec::NdBroadcastExec;
use crate::registry::NdNodeRegistry;

/// Moves nodes that operate on grids below the nd boundary.
#[derive(Debug)]
pub struct NdBoundaryRule {
    registry: Arc<NdNodeRegistry>,
}

impl NdBoundaryRule {
    pub fn new(registry: Arc<NdNodeRegistry>) -> Self {
        Self { registry }
    }

    /// Sink `node` below the boundaries under it, if a sinker accepts it.
    /// Every child of `node` must be a boundary.
    fn sink(&self, node: Arc<dyn ExecutionPlan>) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
        let mut belows: Vec<Arc<dyn ExecutionPlan>> =
            node.children().into_iter().cloned().collect();
        if belows.is_empty() {
            return Ok(Transformed::no(node));
        }
        // The round-robin repartitions between a one-child `node` and the
        // boundary, top down.
        let mut passthrough = Vec::new();
        if let [below] = &mut belows[..] {
            while is_round_robin(below) {
                passthrough.push(below.clone());
                let next = below.children()[0].clone();
                *below = next;
            }
        }
        let mut nd_children = Vec::with_capacity(belows.len());
        let mut child_selection = SelectionKind::Full;
        for below in &belows {
            let Some(boundary) = below.as_any().downcast_ref::<NdBroadcastExec>() else {
                return Ok(Transformed::no(node));
            };
            nd_children.push(boundary.input().clone());
            child_selection = child_selection.max(boundary.nd_input().max_output_selection());
        }

        for sinker in self.registry.sinkers() {
            let Some(sunk) = sinker.try_sink(&node, &nd_children, &self.registry)? else {
                continue;
            };
            // The new node must accept every selection its child can output.
            if child_selection > sunk.nd.accepts_selection() {
                continue;
            }
            let mut rebuilt: Arc<dyn ExecutionPlan> = Arc::new(
                NdBroadcastExec::try_new_with_registry(sunk.nd, self.registry.clone())?,
            );
            for repartition in passthrough.iter().rev() {
                rebuilt = repartition.clone().with_new_children(vec![rebuilt])?;
            }
            if let Some(residual) = sunk.residual {
                rebuilt = residual.with_new_children(vec![rebuilt])?;
            }
            return Ok(Transformed::yes(rebuilt));
        }
        Ok(Transformed::no(node))
    }
}

fn is_round_robin(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.as_any()
        .downcast_ref::<RepartitionExec>()
        .is_some_and(|r| matches!(r.partitioning(), Partitioning::RoundRobinBatch(_)))
}

impl PhysicalOptimizerRule for NdBoundaryRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Bottom up: after a node sinks, its parent sees the new boundary next.
        plan.transform_up(|node| self.sink(node)).map(|t| t.data)
    }

    fn name(&self) -> &str {
        "NdBoundaryRule"
    }

    fn schema_check(&self) -> bool {
        true
    }
}
