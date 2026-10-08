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
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties, collect, displayable};
use datafusion_nd_exec::testing::{Differential, grid_table, profile_table, sorted_rows};
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
    let sql = "SELECT lat, sst FROM grid WHERE lat > 0 UNION ALL SELECT lat, sst FROM grid";
    harness
        .assert_plan_nodes(sql, &["NdBroadcastExec", "NdUnionExec", "NdFilterExec"])
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

/// Whether the nd plan of `sql` sorts.
async fn sorts(harness: &Differential, sql: &str) -> Result<bool> {
    let plan = harness.nd_plan(sql).await?;
    Ok(plan.lines().any(|l| l.trim_start().starts_with("SortExec")))
}

#[tokio::test]
async fn an_order_on_the_outer_coordinates_needs_no_sort() -> Result<()> {
    let harness = harness()?;
    harness.register("profiles", profile_table()?)?;
    for sql in [
        "SELECT time, sst FROM grid ORDER BY time",
        "SELECT time, lat, lon, sst FROM grid ORDER BY time, lat, lon",
    ] {
        assert!(
            !sorts(&harness, sql).await?,
            "{sql}\n{}",
            harness.nd_plan(sql).await?
        );
    }
    for sql in [
        // `lat` is not the outer axis.
        "SELECT lat, sst FROM grid ORDER BY lat",
        "SELECT time, sst FROM grid ORDER BY time DESC",
        // A profile axis has no order.
        r#"SELECT "PRES" FROM profiles ORDER BY "PRES""#,
    ] {
        assert!(
            sorts(&harness, sql).await?,
            "{sql}\n{}",
            harness.nd_plan(sql).await?
        );
    }
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
        text.contains("NdBroadcastExec: region=[NdFilterExec, NdRepartitionExec, NdSourceExec]"),
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
async fn a_round_robin_repartition_runs_below_the_broadcast() -> Result<()> {
    let harness = harness()?;
    let sql = "SELECT * FROM grid WHERE lat > -1";
    harness
        .assert_plan_nodes(
            sql,
            &[
                "NdBroadcastExec",
                "NdFilterExec",
                "NdRepartitionExec",
                "NdSourceExec",
            ],
        )
        .await?;
    let plan = harness.nd_plan(sql).await?;
    assert!(
        !plan
            .lines()
            .any(|l| l.trim_start().starts_with("RepartitionExec")),
        "no RepartitionExec may stay above the broadcast:\n{plan}"
    );
    Ok(())
}

#[tokio::test]
async fn a_partition_merge_runs_below_the_broadcast() -> Result<()> {
    use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;

    for fetch in [None, Some(5)] {
        let scan = scan()?;
        let coalesce: Arc<dyn ExecutionPlan> =
            Arc::new(CoalescePartitionsExec::new(scan).with_fetch(fetch));
        let expected = collect(coalesce.clone(), Arc::new(TaskContext::default())).await?;
        let optimized = optimize(coalesce)?;
        let names = node_names(&optimized);
        assert_eq!(names[0], "NdBroadcastExec");
        assert!(
            names.contains(&"NdCoalescePartitionsExec".to_string()),
            "{names:?}"
        );
        assert_eq!(optimized.output_partitioning().partition_count(), 1);
        let actual = collect(optimized, Arc::new(TaskContext::default())).await?;
        let rows =
            |b: &[arrow::record_batch::RecordBatch]| b.iter().map(|b| b.num_rows()).sum::<usize>();
        assert_eq!(rows(&actual), rows(&expected));
        if fetch.is_none() {
            assert_eq!(sorted_rows(&actual)?, sorted_rows(&expected)?);
        }
    }
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

/// The declared grid axes of the nd child of the boundary at the root of `sql`.
async fn root_grid_axes(sql: &str) -> Result<Option<Vec<String>>> {
    use datafusion_nd_exec::exec::NdBroadcastExec;

    let harness = harness()?;
    let plan = harness
        .nd_context()
        .sql(sql)
        .await?
        .create_physical_plan()
        .await?;
    let boundary = plan
        .as_any()
        .downcast_ref::<NdBroadcastExec>()
        .expect("the plan ends in the nd region");
    let nd = NdNodeRegistry::shared_default()
        .as_nd_plan(boundary.input())
        .expect("an nd child");
    Ok(nd
        .grid_axes()
        .map(|axes| axes.axes().iter().map(|(name, _)| name.clone()).collect()))
}

#[tokio::test]
async fn the_grid_axes_pass_through_the_nd_region() -> Result<()> {
    let expected = Some(vec![
        "time".to_string(),
        "lat".to_string(),
        "lon".to_string(),
    ]);
    for sql in [
        "SELECT lat * 2 AS lat2, sst FROM grid WHERE lon <> 15",
        "SELECT * FROM grid UNION ALL SELECT * FROM grid",
        "SELECT * FROM grid LIMIT 3",
    ] {
        assert_eq!(root_grid_axes(sql).await?, expected, "{sql}");
    }
    Ok(())
}

#[tokio::test]
async fn a_union_with_other_axes_has_no_grid_axes() -> Result<()> {
    use datafusion_nd_exec::exec::NdUnionExec;
    use datafusion_nd_exec::testing::NdMemTable;

    let grid = grid_table()?.nd_scan(None, NdNodeRegistry::shared_default())?;
    // The same data, with no declared grid axes.
    let undeclared = NdMemTable::try_new(grid_table()?.partitions().to_vec())?;
    let other = undeclared.nd_scan(None, NdNodeRegistry::shared_default())?;
    let registry = NdNodeRegistry::shared_default();
    let same = NdUnionExec::try_new(
        vec![grid.children()[0].clone(), grid.children()[0].clone()],
        registry.clone(),
    )?;
    assert!(datafusion_nd_exec::exec::NdExecutionPlan::grid_axes(&same).is_some());
    let mixed = NdUnionExec::try_new(
        vec![grid.children()[0].clone(), other.children()[0].clone()],
        registry,
    )?;
    assert!(datafusion_nd_exec::exec::NdExecutionPlan::grid_axes(&mixed).is_none());
    Ok(())
}

#[tokio::test]
async fn the_sort_order_stops_at_an_unordered_axis() -> Result<()> {
    use datafusion_nd_exec::array::{AxisOrder, NdGridAxes};
    use datafusion_nd_exec::testing::NdMemTable;

    let axes = NdGridAxes::new([
        ("time", AxisOrder::Descending),
        ("lat", AxisOrder::Unordered),
        ("lon", AxisOrder::Ascending),
    ]);
    let table = NdMemTable::try_new(grid_table()?.partitions().to_vec())?
        .with_grid_axes(axes)
        .with_ordered_chunks();
    let scan = table.nd_scan(None, NdNodeRegistry::shared_default())?;
    let source = scan.children()[0].clone();
    let ordering = source.output_ordering().expect("an order on time");
    assert_eq!(ordering.len(), 1);
    assert_eq!(ordering[0].to_string(), "time@0 DESC");
    Ok(())
}
