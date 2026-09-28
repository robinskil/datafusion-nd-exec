//! The built-in sink checks: filter and projection.

use std::sync::Arc;

use datafusion::error::Result;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{PhysicalExpr, conjunction, split_conjunction};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::filter::{FilterExec, FilterExecBuilder};
use datafusion::physical_plan::projection::ProjectionExec;

use crate::exec::{NdExecutionPlan, NdFilterExec, NdProjectionExec};
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
/// example a volatile function. The embedded projection of the filter goes
/// below the boundary as an [`NdProjectionExec`] when no residual stays, else
/// it stays on the residual.
#[derive(Debug, Default)]
pub struct FilterSinker;

impl NdSinker for FilterSinker {
    fn try_sink(
        &self,
        parent: &Arc<dyn ExecutionPlan>,
        child: &Arc<dyn ExecutionPlan>,
        registry: &Arc<NdNodeRegistry>,
    ) -> Result<Option<Sunk>> {
        let Some(filter) = parent.as_any().downcast_ref::<FilterExec>() else {
            return Ok(None);
        };
        // A limit on the filter would be lost below the boundary.
        if filter.fetch().is_some() {
            return Ok(None);
        }

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
            let nd = match projection {
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
            return Ok(Some(Sunk { nd, residual: None }));
        }

        // The residual keeps the old input as a placeholder: the boundary rule
        // replaces it with the new boundary.
        let residual = FilterExecBuilder::new(conjunction(keep), filter.input().clone())
            .apply_projection(projection)?
            .with_default_selectivity(filter.default_selectivity())
            .build()?;
        Ok(Some(Sunk {
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
        child: &Arc<dyn ExecutionPlan>,
        registry: &Arc<NdNodeRegistry>,
    ) -> Result<Option<Sunk>> {
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
        Ok(Some(Sunk {
            nd: Arc::new(nd),
            residual: None,
        }))
    }
}
