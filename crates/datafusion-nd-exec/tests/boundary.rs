//! The boundary rule moves nodes that operate on grids below the broadcast.

use std::sync::Arc;

use arrow::array::{Array, AsArray};
use datafusion::common::config::ConfigOptions;
use datafusion::error::Result;
use datafusion::execution::TaskContext;
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::expressions::{binary, col, lit};
use datafusion::physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::filter::{FilterExec, FilterExecBuilder};
use datafusion::physical_plan::projection::{ProjectionExec, ProjectionExpr};
use datafusion::physical_plan::{ExecutionPlan, collect, displayable};
use datafusion_nd_exec::testing::{Differential, grid_table, sorted_rows};
use datafusion_nd_exec::{NdBoundaryRule, NdNodeRegistry};

fn harness() -> Result<Differential> {
    let harness = Differential::new();
    harness.register("grid", grid_table()?)?;
    Ok(harness)
}

fn scan() -> Result<Arc<dyn ExecutionPlan>> {
    grid_table()?.nd_scan(None, NdNodeRegistry::shared_default())
}

fn optimize(plan: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
    NdBoundaryRule::new(NdNodeRegistry::shared_default()).optimize(plan, &ConfigOptions::default())
}

/// The node names of `plan`, top down.
fn node_names(plan: &Arc<dyn ExecutionPlan>) -> Vec<String> {
    displayable(plan.as_ref())
        .indent(true)
        .to_string()
        .lines()
        .map(|line| {
            line.trim_start()
                .split(':')
                .next()
                .unwrap_or("")
                .to_string()
        })
        .collect()
}

fn lat_above_zero(plan: &Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalExpr>> {
    let schema = plan.schema();
    binary(col("lat", &schema)?, Operator::Gt, lit(0.0f64), &schema)
}

fn random_expr(plan: &Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalExpr>> {
    Ok(Arc::new(ScalarFunctionExpr::try_new(
        datafusion::functions::math::random(),
        vec![],
        &plan.schema(),
        Arc::new(ConfigOptions::default()),
    )?))
}

#[tokio::test]
async fn an_elementwise_filter_runs_below_the_broadcast() -> Result<()> {
    let harness = harness()?;
    let sql = "SELECT * FROM grid WHERE lat > -1";
    harness
        .assert_plan_nodes(sql, &["NdBroadcastExec", "NdFilterExec", "NdSourceExec"])
        .await?;
    let plan = harness.nd_plan(sql).await?;
    assert!(
        !plan
            .lines()
            .any(|l| l.trim_start().starts_with("FilterExec:")),
        "no FilterExec may stay above the broadcast:\n{plan}"
    );
    Ok(())
}

#[tokio::test]
async fn a_filter_and_a_projection_both_run_below_the_broadcast() -> Result<()> {
    harness()?
        .assert_plan_nodes(
            "SELECT lat * 2 AS lat2 FROM grid WHERE lon <> 15",
            &[
                "NdBroadcastExec",
                "NdProjectionExec",
                "NdFilterExec",
                "NdSourceExec",
            ],
        )
        .await
}

#[tokio::test]
async fn a_volatile_conjunct_stays_above_the_broadcast() -> Result<()> {
    harness()?
        .assert_plan_nodes(
            "SELECT * FROM grid WHERE random() < 2 AND lat > 0",
            &[
                "FilterExec",
                "NdBroadcastExec",
                "NdFilterExec",
                "NdSourceExec",
            ],
        )
        .await
}

#[tokio::test]
async fn a_union_of_nd_scans_runs_below_the_broadcast() -> Result<()> {
    let harness = harness()?;
    let sql = "SELECT lat, sst FROM grid UNION ALL SELECT lat, sst FROM grid";
    harness
        .assert_plan_nodes(sql, &["NdBroadcastExec", "NdUnionExec", "NdSourceExec"])
        .await?;
    let plan = harness.nd_plan(sql).await?;
    assert!(
        !plan
            .lines()
            .any(|l| l.trim_start().starts_with("UnionExec")),
        "no UnionExec may stay above the broadcast:
{plan}"
    );
    Ok(())
}

#[tokio::test]
async fn explain_shows_the_nd_region() -> Result<()> {
    let harness = harness()?;
    let explained = harness
        .nd_context()
        .sql("EXPLAIN SELECT * FROM grid WHERE lat > -1")
        .await?
        .collect()
        .await?;
    let text: String = explained
        .iter()
        .flat_map(|batch| {
            let plans = batch.column(1).as_string::<i32>();
            (0..plans.len())
                .map(|i| plans.value(i).to_string())
                .collect::<Vec<_>>()
        })
        .collect();
    assert!(
        text.contains("NdBroadcastExec: region=[NdFilterExec, NdSourceExec]"),
        "{text}"
    );
    Ok(())
}

/// A filter that could sink stays when an unsinkable projection sits between
/// it and the boundary.
#[tokio::test]
async fn a_sinkable_node_above_an_unsinkable_node_stays() -> Result<()> {
    let scan = scan()?;
    let projection: Arc<dyn ExecutionPlan> = Arc::new(ProjectionExec::try_new(
        [
            ProjectionExpr {
                expr: random_expr(&scan)?,
                alias: "r".to_string(),
            },
            ProjectionExpr {
                expr: col("lat", &scan.schema())?,
                alias: "lat".to_string(),
            },
        ],
        scan,
    )?);
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(
        lat_above_zero(&projection)?,
        projection,
    )?);

    let optimized = optimize(filter)?;
    assert_eq!(
        node_names(&optimized),
        [
            "FilterExec",
            "ProjectionExec",
            "NdBroadcastExec",
            "NdSourceExec",
            "DataSourceExec"
        ]
    );
    Ok(())
}

/// An unsinkable projection above a sinkable filter: the filter sinks, the
/// projection stays, and the result does not change.
#[tokio::test]
async fn an_unsinkable_node_above_a_sinkable_node() -> Result<()> {
    let scan = scan()?;
    let filter: Arc<dyn ExecutionPlan> =
        Arc::new(FilterExec::try_new(lat_above_zero(&scan)?, scan)?);
    let random = random_expr(&filter)?;
    let lat = col("lat", &filter.schema())?;
    // `random() * 0` keeps the projection volatile but the result fixed.
    let fixed = binary(random, Operator::Multiply, lit(0.0f64), &filter.schema())?;
    let projection: Arc<dyn ExecutionPlan> = Arc::new(ProjectionExec::try_new(
        [
            ProjectionExpr {
                expr: fixed,
                alias: "zero".to_string(),
            },
            ProjectionExpr {
                expr: lat,
                alias: "lat".to_string(),
            },
        ],
        filter,
    )?);
    let expected = collect(projection.clone(), Arc::new(TaskContext::default())).await?;

    let optimized = optimize(projection)?;
    assert_eq!(
        node_names(&optimized),
        [
            "ProjectionExec",
            "NdBroadcastExec",
            "NdFilterExec",
            "NdSourceExec",
            "DataSourceExec"
        ]
    );
    let actual = collect(optimized, Arc::new(TaskContext::default())).await?;
    assert_eq!(sorted_rows(&actual)?, sorted_rows(&expected)?);
    Ok(())
}

/// A filter with a limit sinks with an `NdLimitExec` on top.
#[tokio::test]
async fn a_filter_with_a_limit_sinks_with_its_limit() -> Result<()> {
    let scan = scan()?;
    let filter: Arc<dyn ExecutionPlan> = Arc::new(
        FilterExecBuilder::new(lat_above_zero(&scan)?, scan)
            .with_fetch(Some(2))
            .build()?,
    );
    let expected = collect(filter.clone(), Arc::new(TaskContext::default())).await?;
    let optimized = optimize(filter)?;
    assert_eq!(
        node_names(&optimized),
        [
            "NdBroadcastExec",
            "NdLimitExec",
            "NdFilterExec",
            "NdSourceExec",
            "DataSourceExec"
        ]
    );
    let actual = collect(optimized, Arc::new(TaskContext::default())).await?;
    assert_eq!(
        actual.iter().map(|b| b.num_rows()).sum::<usize>(),
        expected.iter().map(|b| b.num_rows()).sum::<usize>()
    );
    Ok(())
}

#[tokio::test]
async fn a_limit_runs_below_the_broadcast() -> Result<()> {
    harness()?
        .assert_plan_nodes(
            "SELECT * FROM grid LIMIT 5",
            &["NdBroadcastExec", "NdLimitExec", "NdSourceExec"],
        )
        .await
}
