//! Add the coordinate columns of a grid write to its query.

use std::fmt;
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result};
use datafusion::config::ConfigOptions;
use datafusion::logical_expr::dml::CopyTo;
use datafusion::logical_expr::{Expr, LogicalPlan, Projection};
use datafusion::optimizer::AnalyzerRule;
use nd_arrow_array::encoding::{nd_logical_metadata, schema_coordinates};

/// Add the coordinate column of each output axis of `plan` that the output
/// does not hold, so a grid sink can place each chunk by its coordinates.
///
/// The coordinate columns come from the schemas of the table scans, with
/// [`schema_coordinates`]. The axes of the output come from the field
/// metadata of its columns, so a computed column adds no axes. Each missing
/// column goes at the end of the top projection, under its own name. Without
/// such a projection, the plan does not change.
///
/// Use it on a logical plan before the optimizer, where a projection still
/// sees all columns of its input.
pub fn add_grid_coordinates(plan: LogicalPlan) -> Result<LogicalPlan> {
    let mut coordinates: Vec<(String, String)> = Vec::new();
    plan.apply(|node| {
        if let LogicalPlan::TableScan(scan) = node {
            for pair in schema_coordinates(&scan.source.schema()) {
                if !coordinates.contains(&pair) {
                    coordinates.push(pair);
                }
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })?;

    let output = plan.schema();
    let mut axes: Vec<String> = Vec::new();
    for field in output.fields() {
        let dims = nd_logical_metadata(field).and_then(|m| m.dims);
        for axis in dims.into_iter().flatten() {
            if !axes.contains(&axis) {
                axes.push(axis);
            }
        }
    }
    let missing: Vec<String> = axes
        .iter()
        .filter_map(|axis| coordinates.iter().find(|(a, _)| a == axis))
        .map(|(_, column)| column.clone())
        .filter(|column| !output.fields().iter().any(|f| f.name() == column))
        .collect();
    if missing.is_empty() {
        return Ok(plan);
    }
    Ok(extend(&plan, &missing)?.unwrap_or(plan))
}

/// `plan` with `missing` added to its top projection, or `None`.
fn extend(plan: &LogicalPlan, missing: &[String]) -> Result<Option<LogicalPlan>> {
    match plan {
        LogicalPlan::Projection(projection) => {
            let input = projection.input.schema();
            let mut exprs = projection.expr.clone();
            for name in missing {
                if let Ok((qualifier, _)) = input.qualified_field_with_unqualified_name(name) {
                    exprs.push(Expr::Column(Column::new(qualifier.cloned(), name)));
                }
            }
            if exprs.len() == projection.expr.len() {
                return Ok(None);
            }
            let projection = Projection::try_new(exprs, projection.input.clone())?;
            Ok(Some(LogicalPlan::Projection(projection)))
        }
        LogicalPlan::Sort(_)
        | LogicalPlan::Limit(_)
        | LogicalPlan::Filter(_)
        | LogicalPlan::SubqueryAlias(_) => {
            let [input] = plan.inputs()[..] else {
                return Ok(None);
            };
            match extend(input, missing)? {
                Some(input) => Ok(Some(plan.with_new_exprs(plan.expressions(), vec![input])?)),
                None => Ok(None),
            }
        }
        _ => Ok(None),
    }
}

type GridWrite = Arc<dyn Fn(&CopyTo) -> bool + Send + Sync>;

/// Applies [`add_grid_coordinates`] to the input of each grid `COPY TO`.
///
/// The host decides which writes are grid writes, for example by the file
/// type of the `COPY TO`. `INSERT` does not change, because the columns of a
/// table are fixed. Add the rule with `SessionStateBuilder::with_analyzer_rule`.
pub struct NdGridCoordinatesRule {
    is_grid_write: GridWrite,
}

impl NdGridCoordinatesRule {
    /// A rule for each `COPY TO` for which `is_grid_write` returns true.
    pub fn new(is_grid_write: impl Fn(&CopyTo) -> bool + Send + Sync + 'static) -> Self {
        Self {
            is_grid_write: Arc::new(is_grid_write),
        }
    }

    /// A rule for every `COPY TO`.
    pub fn all() -> Self {
        Self::new(|_| true)
    }
}

impl fmt::Debug for NdGridCoordinatesRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("NdGridCoordinatesRule")
    }
}

impl AnalyzerRule for NdGridCoordinatesRule {
    fn analyze(&self, plan: LogicalPlan, _config: &ConfigOptions) -> Result<LogicalPlan> {
        plan.transform_down(|node| match node {
            LogicalPlan::Copy(copy) if (self.is_grid_write)(&copy) => {
                let input = add_grid_coordinates(copy.input.as_ref().clone())?;
                Ok(Transformed::yes(LogicalPlan::Copy(CopyTo {
                    input: Arc::new(input),
                    ..copy
                })))
            }
            other => Ok(Transformed::no(other)),
        })
        .map(|transformed| transformed.data)
    }

    fn name(&self) -> &str {
        "nd_grid_coordinates"
    }
}

#[cfg(test)]
mod tests {
    use datafusion::prelude::SessionContext;

    use super::*;
    use crate::testing::grid_table;

    async fn unoptimized(sql: &str) -> LogicalPlan {
        let ctx = SessionContext::new();
        ctx.register_table("t", Arc::new(grid_table().unwrap()))
            .unwrap();
        ctx.sql(sql).await.unwrap().into_unoptimized_plan()
    }

    fn names(plan: &LogicalPlan) -> Vec<String> {
        plan.schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect()
    }

    async fn added(sql: &str) -> Vec<String> {
        names(&add_grid_coordinates(unoptimized(sql).await).unwrap())
    }

    #[tokio::test]
    async fn the_coordinates_of_the_output_axes_join_the_query() {
        let names = added("SELECT sst FROM t WHERE lat > 0").await;
        assert_eq!(names, ["sst", "time", "lat", "lon"]);
    }

    #[tokio::test]
    async fn only_the_axes_of_the_output_count() {
        assert_eq!(added("SELECT elev FROM t").await, ["elev", "lat", "lon"]);
    }

    #[tokio::test]
    async fn a_query_with_all_coordinates_does_not_change() {
        let all = names(&unoptimized("SELECT * FROM t").await);
        assert_eq!(added("SELECT * FROM t").await, all);
    }

    #[tokio::test]
    async fn the_columns_pass_through_a_sort_and_a_limit() {
        let names = added("SELECT time, sst FROM t ORDER BY time LIMIT 2").await;
        assert_eq!(names, ["time", "sst", "lat", "lon"]);
    }

    #[tokio::test]
    async fn a_computed_column_adds_no_axes() {
        assert_eq!(added("SELECT sst * 2 AS s FROM t").await, ["s"]);
    }

    #[tokio::test]
    async fn the_rule_changes_only_a_grid_copy() {
        let sql = "COPY (SELECT sst FROM t) TO 'out.csv' STORED AS CSV";
        let input_names = |plan: &LogicalPlan| match plan {
            LogicalPlan::Copy(copy) => names(&copy.input),
            other => panic!("not a copy: {other}"),
        };
        let config = ConfigOptions::default();
        let grid = NdGridCoordinatesRule::all()
            .analyze(unoptimized(sql).await, &config)
            .unwrap();
        assert_eq!(input_names(&grid), ["sst", "time", "lat", "lon"]);
        let flat = NdGridCoordinatesRule::new(|_| false)
            .analyze(unoptimized(sql).await, &config)
            .unwrap();
        assert_eq!(input_names(&flat), ["sst"]);
    }
}
