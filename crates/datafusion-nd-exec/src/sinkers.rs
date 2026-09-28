//! The built-in sink checks.

use std::sync::Arc;

use datafusion::error::Result;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{PhysicalExpr, conjunction, split_conjunction};
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::filter::{FilterExec, FilterExecBuilder};
use datafusion::physical_plan::limit::{GlobalLimitExec, LocalLimitExec};
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties, Partitioning};

use crate::exec::{
    NdCoalescePartitionsExec, NdExecutionPlan, NdFilterExec, NdLimitExec, NdProjectionExec,
    NdRepartitionExec, NdUnionExec,
};
use crate::optimizer::is_pushable_expr;
use crate::registry::{NdNodeRegistry, NdSinker, Sunk};

/// Sinks the element-wise conjuncts of a `FilterExec` into an
/// [`NdFilterExec`].
///
/// ```text
/// FilterExec[a AND b AND c]          FilterExec[c]            (residual, only if any)
///   NdBroadcastExec           ->       NdBroadcastExec
///     nd-child                           NdFilterExec[a, b]   (element-wise conjuncts)
///                                          nd-child
/// ```
///
/// `a` and `b` are element-wise ([`is_pushable_expr`]). `c` is not, for
/// example a volatile function. The embedded projection and limit of the
/// filter go below the boundary as an [`NdProjectionExec`] and an
/// [`NdLimitExec`] when no residual stays, else they stay on the residual.
#[derive(Debug, Default)]
pub struct FilterSinker;

impl NdSinker for FilterSinker {
    fn try_sink(
        &self,
        parent: &Arc<dyn ExecutionPlan>,
        children: &[Arc<dyn ExecutionPlan>],
        registry: &Arc<NdNodeRegistry>,
    ) -> Result<Option<Sunk>> {
        let [child] = children else {
            return Ok(None);
        };
        let Some(filter) = parent.as_any().downcast_ref::<FilterExec>() else {
            return Ok(None);
        };
        let (push, keep): (Vec<_>, Vec<_>) = split_conjunction(filter.predicate())
            .into_iter()
            .cloned()
            .partition(is_pushable_expr);
        if push.is_empty() {
            return Ok(None);
        }

        let nd_filter: Arc<dyn NdExecutionPlan> = Arc::new(NdFilterExec::try_new_with_registry(
            child.clone(),
            push,
            registry.clone(),
        )?);
        let projection = filter.projection().as_ref().map(|p| p.to_vec());

        if keep.is_empty() {
            let nd: Arc<dyn NdExecutionPlan> = match projection {
                None => nd_filter,
                Some(indices) => {
                    let schema = child.schema();
                    let exprs = indices
                        .iter()
                        .map(|&i| {
                            let name = schema.field(i).name().clone();
                            let column: Arc<dyn PhysicalExpr> = Arc::new(Column::new(&name, i));
                            (column, name)
                        })
                        .collect();
                    Arc::new(NdProjectionExec::try_new_with_registry(
                        nd_filter,
                        exprs,
                        Some(filter.schema()),
                        registry.clone(),
                    )?)
                }
            };
            let nd: Arc<dyn NdExecutionPlan> = match filter.fetch() {
                None => nd,
                Some(fetch) => Arc::new(NdLimitExec::try_new_with_registry(
                    nd,
                    0,
                    Some(fetch),
                    registry.clone(),
                )?),
            };
            return Ok(Some(Sunk::Below { nd, residual: None }));
        }

        // The residual keeps the old input as a placeholder: the boundary rule
        // replaces it with the new boundary.
        let residual = FilterExecBuilder::new(conjunction(keep), filter.input().clone())
            .apply_projection(projection)?
            .with_default_selectivity(filter.default_selectivity())
            .with_fetch(filter.fetch())
            .build()?;
        Ok(Some(Sunk::Below {
            nd: nd_filter,
            residual: Some(Arc::new(residual)),
        }))
    }
}

/// Sinks a `ProjectionExec` whose expressions are all element-wise into an
/// [`NdProjectionExec`].
///
/// ```text
/// ProjectionExec[exprs]              NdBroadcastExec
///   NdBroadcastExec           ->       NdProjectionExec[exprs]
///     nd-child                           nd-child
/// ```
#[derive(Debug, Default)]
pub struct ProjectionSinker;

impl NdSinker for ProjectionSinker {
    fn try_sink(
        &self,
        parent: &Arc<dyn ExecutionPlan>,
        children: &[Arc<dyn ExecutionPlan>],
        registry: &Arc<NdNodeRegistry>,
    ) -> Result<Option<Sunk>> {
        let [child] = children else {
            return Ok(None);
        };
        let Some(projection) = parent.as_any().downcast_ref::<ProjectionExec>() else {
            return Ok(None);
        };
        if !projection
            .expr()
            .iter()
            .all(|pe| is_pushable_expr(&pe.expr))
        {
            return Ok(None);
        }
        let exprs = projection
            .expr()
            .iter()
            .map(|pe| (pe.expr.clone(), pe.alias.clone()))
            .collect();
        // Keep the exact output schema, so the rewrite passes the schema check.
        let nd = NdProjectionExec::try_new_with_registry(
            child.clone(),
            exprs,
            Some(projection.schema()),
            registry.clone(),
        )?;
        Ok(Some(Sunk::Below {
            nd: Arc::new(nd),
            residual: None,
        }))
    }
}

/// Sinks a `UnionExec` whose inputs are all boundaries into an
/// [`NdUnionExec`].
///
/// ```text
/// UnionExec                          NdBroadcastExec
///   NdBroadcastExec           ->       NdUnionExec
///     nd-child-a                         nd-child-a
///   NdBroadcastExec                      nd-child-b
///     nd-child-b
/// ```
///
/// Each input must have exactly the schema of the union, because each nd batch
/// carries the schema of its own input.
#[derive(Debug, Default)]
pub struct UnionSinker;

impl NdSinker for UnionSinker {
    fn try_sink(
        &self,
        parent: &Arc<dyn ExecutionPlan>,
        children: &[Arc<dyn ExecutionPlan>],
        registry: &Arc<NdNodeRegistry>,
    ) -> Result<Option<Sunk>> {
        if !parent.as_any().is::<UnionExec>() {
            return Ok(None);
        }
        let schema = parent.schema();
        if children.iter().any(|child| child.schema() != schema) {
            return Ok(None);
        }
        let nd = NdUnionExec::try_new_with_registry(children.to_vec(), registry.clone())?;
        Ok(Some(Sunk::Below {
            nd: Arc::new(nd),
            residual: None,
        }))
    }
}

/// Sinks a `LocalLimitExec`, or a `GlobalLimitExec` over one partition, into
/// an [`NdLimitExec`].
///
/// ```text
/// LocalLimitExec[fetch]              NdBroadcastExec
///   NdBroadcastExec           ->       NdLimitExec[fetch]
///     nd-child                           nd-child
/// ```
#[derive(Debug, Default)]
pub struct LimitSinker;

impl NdSinker for LimitSinker {
    fn try_sink(
        &self,
        parent: &Arc<dyn ExecutionPlan>,
        children: &[Arc<dyn ExecutionPlan>],
        registry: &Arc<NdNodeRegistry>,
    ) -> Result<Option<Sunk>> {
        let [child] = children else {
            return Ok(None);
        };
        let any = parent.as_any();
        let (skip, fetch) = if let Some(local) = any.downcast_ref::<LocalLimitExec>() {
            (0, Some(local.fetch()))
        } else if let Some(global) = any.downcast_ref::<GlobalLimitExec>() {
            // A global limit counts over all rows, so it needs one partition.
            if child.output_partitioning().partition_count() != 1 {
                return Ok(None);
            }
            (global.skip(), global.fetch())
        } else {
            return Ok(None);
        };
        let nd = NdLimitExec::try_new_with_registry(child.clone(), skip, fetch, registry.clone())?;
        Ok(Some(Sunk::Below {
            nd: Arc::new(nd),
            residual: None,
        }))
    }
}

/// Sinks a round-robin `RepartitionExec` into an [`NdRepartitionExec`].
///
/// ```text
/// RepartitionExec[RoundRobin(n)]     NdBroadcastExec
///   NdBroadcastExec           ->       NdRepartitionExec[n]
///     nd-child                           nd-child
/// ```
///
/// An order-preserving repartition stays above the boundary.
#[derive(Debug, Default)]
pub struct RepartitionSinker;

impl NdSinker for RepartitionSinker {
    fn try_sink(
        &self,
        parent: &Arc<dyn ExecutionPlan>,
        children: &[Arc<dyn ExecutionPlan>],
        registry: &Arc<NdNodeRegistry>,
    ) -> Result<Option<Sunk>> {
        let [child] = children else {
            return Ok(None);
        };
        let Some(repartition) = parent.as_any().downcast_ref::<RepartitionExec>() else {
            return Ok(None);
        };
        let Partitioning::RoundRobinBatch(partitions) = repartition.partitioning() else {
            return Ok(None);
        };
        if repartition.preserve_order() {
            return Ok(None);
        }
        let nd =
            NdRepartitionExec::try_new_with_registry(child.clone(), *partitions, registry.clone())?;
        Ok(Some(Sunk::Below {
            nd: Arc::new(nd),
            residual: None,
        }))
    }
}

/// Sinks a `CoalescePartitionsExec` into an [`NdCoalescePartitionsExec`]. A
/// `fetch` on it becomes an [`NdLimitExec`] on top.
///
/// ```text
/// CoalescePartitionsExec             NdBroadcastExec
///   NdBroadcastExec           ->       NdCoalescePartitionsExec
///     nd-child                           nd-child
/// ```
#[derive(Debug, Default)]
pub struct CoalesceSinker;

impl NdSinker for CoalesceSinker {
    fn try_sink(
        &self,
        parent: &Arc<dyn ExecutionPlan>,
        children: &[Arc<dyn ExecutionPlan>],
        registry: &Arc<NdNodeRegistry>,
    ) -> Result<Option<Sunk>> {
        let [child] = children else {
            return Ok(None);
        };
        let Some(coalesce) = parent.as_any().downcast_ref::<CoalescePartitionsExec>() else {
            return Ok(None);
        };
        let merged: Arc<dyn NdExecutionPlan> = Arc::new(
            NdCoalescePartitionsExec::try_new_with_registry(child.clone(), registry.clone())?,
        );
        let nd: Arc<dyn NdExecutionPlan> = match coalesce.fetch() {
            None => merged,
            Some(fetch) => Arc::new(NdLimitExec::try_new_with_registry(
                merged,
                0,
                Some(fetch),
                registry.clone(),
            )?),
        };
        Ok(Some(Sunk::Below { nd, residual: None }))
    }
}
