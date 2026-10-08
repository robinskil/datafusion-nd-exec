//! The grid reaches the output through an nd sink.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::common::config::ConfigOptions;
use datafusion::datasource::sink::{DataSink, DataSinkExec};
use datafusion::error::Result;
use datafusion::execution::TaskContext;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, SendableRecordBatchStream, collect, displayable,
};
use datafusion_nd_exec::array::{AxisOrder, NdGridAxes, NdRecordBatch};
use datafusion_nd_exec::exec::NdBroadcastExec;
use datafusion_nd_exec::sink::{NdDataSink, NdDataSinkExec, NdSinkFactory};
use datafusion_nd_exec::testing::{
    Differential, MemoryGridSink, NdMemTable, grid_table, profile_table, sorted_rows,
};
use datafusion_nd_exec::{
    NdBoundaryRule, NdGridCoordinatesRule, NdNodeRegistry, NdSessionStateBuilderExt,
};

/// The count of the one row that a sink yields.
async fn run(plan: Arc<dyn ExecutionPlan>) -> Result<u64> {
    let batches = collect(plan, Arc::new(TaskContext::default())).await?;
    Ok(batches[0]
        .column(0)
        .as_primitive::<arrow::datatypes::UInt64Type>()
        .value(0))
}

/// The nd child of a scan boundary.
fn nd_scan(table: &datafusion_nd_exec::testing::NdMemTable) -> Result<Arc<dyn ExecutionPlan>> {
    let scan = table.nd_scan(None, NdNodeRegistry::shared_default())?;
    Ok(scan.children()[0].clone())
}

#[tokio::test]
async fn a_grid_round_trips_through_the_sink() -> Result<()> {
    let table = grid_table()?;
    let sink = Arc::new(MemoryGridSink::new(
        table
            .nd_scan(None, NdNodeRegistry::shared_default())?
            .schema(),
    ));
    let rows = run(Arc::new(NdDataSinkExec::try_new(
        nd_scan(&table)?,
        sink.clone(),
        NdNodeRegistry::shared_default(),
    )?))
    .await?;
    assert_eq!(rows, 24);

    let grid = sink.grid().unwrap();
    let names: Vec<(&str, usize)> = grid.target().iter().map(|d| (d.name(), d.size())).collect();
    assert_eq!(names, [("time", 4), ("lat", 3), ("lon", 2)]);

    // The dense grid holds the same rows as the scan.
    let expected = collect(
        table.nd_scan(None, NdNodeRegistry::shared_default())?,
        Arc::new(TaskContext::default()),
    )
    .await?;
    assert_eq!(
        sorted_rows(&[grid.materialize()?])?,
        sorted_rows(&expected)?
    );
    Ok(())
}

#[tokio::test]
async fn a_profile_table_cannot_go_to_a_grid_sink() -> Result<()> {
    let table = profile_table()?;
    let schema = table
        .nd_scan(None, NdNodeRegistry::shared_default())?
        .schema();
    // The profile axes have no coordinates, so the scan declares no grid axes.
    let error = NdDataSinkExec::try_new(
        nd_scan(&table)?,
        Arc::new(MemoryGridSink::new(schema)),
        NdNodeRegistry::shared_default(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("grid axes"), "{error}");

    // A sink without a grid streams the profiles.
    let counting = Arc::new(NdDataSinkExec::try_new(
        nd_scan(&table)?,
        Arc::new(CountingSink {
            schema: table
                .nd_scan(None, NdNodeRegistry::shared_default())?
                .schema(),
        }),
        NdNodeRegistry::shared_default(),
    )?);
    // 3 x 4 cells from the first file and 2 x 3 from the second.
    assert_eq!(run(counting).await?, 18);
    Ok(())
}

#[tokio::test]
async fn chunks_in_reverse_order_come_out_sorted() -> Result<()> {
    use arrow::datatypes::Int64Type;

    let table = grid_table()?;
    let schema = table
        .nd_scan(None, NdNodeRegistry::shared_default())?
        .schema();
    let reversed = NdMemTable::try_new(table.partitions().iter().rev().cloned().collect())?
        .with_grid_axes(grid_axes());

    let in_order = Arc::new(MemoryGridSink::new(schema.clone()));
    run(Arc::new(NdDataSinkExec::try_new(
        nd_scan(&table)?,
        in_order.clone(),
        NdNodeRegistry::shared_default(),
    )?))
    .await?;
    let backward = Arc::new(MemoryGridSink::new(schema));
    run(Arc::new(NdDataSinkExec::try_new(
        nd_scan(&reversed)?,
        backward.clone(),
        NdNodeRegistry::shared_default(),
    )?))
    .await?;

    let grid = backward.grid().unwrap();
    let time = grid.column(0).values().as_primitive::<Int64Type>();
    assert_eq!(time.values(), &[100, 101, 102, 103]);
    assert_eq!(
        grid.column(3).values(),
        in_order.grid().unwrap().column(3).values()
    );
    Ok(())
}

/// An nd sink that counts rows and needs no grid.
#[derive(Debug)]
struct CountingSink {
    schema: SchemaRef,
}

impl DisplayAs for CountingSink {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CountingSink")
    }
}

#[async_trait]
impl NdDataSink for CountingSink {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    async fn write_all(
        &self,
        mut data: datafusion_nd_exec::exec::SendableNdBatchStream,
        _context: &Arc<TaskContext>,
    ) -> Result<u64> {
        let mut rows = 0;
        while let Some(batch) = futures::StreamExt::next(&mut data).await {
            rows += batch?.num_rows() as u64;
        }
        Ok(rows)
    }
}

#[tokio::test]
async fn only_a_grid_sink_reads_through_the_regrid_step() -> Result<()> {
    let table = grid_table()?;
    let schema = table
        .nd_scan(None, NdNodeRegistry::shared_default())?
        .schema();
    let grid_sink = NdDataSinkExec::try_new(
        nd_scan(&table)?,
        Arc::new(MemoryGridSink::new(schema.clone())),
        NdNodeRegistry::shared_default(),
    )?;
    let lines = displayable(&grid_sink).indent(true).to_string();
    let lines: Vec<&str> = lines.lines().map(str::trim).collect();
    assert_eq!(lines[0], "NdDataSinkExec: sink=MemoryGridSink");
    assert_eq!(lines[1], "NdRegridExec");

    let counting = Arc::new(NdDataSinkExec::try_new(
        nd_scan(&table)?,
        Arc::new(CountingSink { schema }),
        NdNodeRegistry::shared_default(),
    )?);
    let rendered = displayable(counting.as_ref()).indent(true).to_string();
    assert!(!rendered.contains("NdRegridExec"), "{rendered}");
    assert_eq!(run(counting).await?, 24);
    Ok(())
}

/// A flat sink of a host.
#[derive(Debug)]
struct HostSink {
    schema: SchemaRef,
}

impl DisplayAs for HostSink {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HostSink")
    }
}

#[async_trait]
impl DataSink for HostSink {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    async fn write_all(
        &self,
        _data: SendableRecordBatchStream,
        _context: &Arc<TaskContext>,
    ) -> Result<u64> {
        unreachable!("the nd sink replaces the flat sink")
    }
}

/// The host maps its flat sink to the in-memory grid sink.
#[derive(Debug)]
struct HostFactory {
    sink: Arc<MemoryGridSink>,
}

impl NdSinkFactory for HostFactory {
    fn nd_sink(&self, sink: &dyn DataSink) -> Option<Arc<dyn NdDataSink>> {
        sink.as_any()
            .is::<HostSink>()
            .then(|| self.sink.clone() as Arc<dyn NdDataSink>)
    }
}

#[tokio::test]
async fn a_host_sink_becomes_an_nd_sink() -> Result<()> {
    let table = grid_table()?;
    let scan = table.nd_scan(None, NdNodeRegistry::shared_default())?;
    let schema = scan.schema();
    let grid_sink = Arc::new(MemoryGridSink::new(schema.clone()));
    let registry = Arc::new(
        NdNodeRegistry::new().with_sink_factory(Arc::new(HostFactory {
            sink: grid_sink.clone(),
        })),
    );

    // DataFusion puts a partition merge under a sink.
    let plan: Arc<dyn ExecutionPlan> = Arc::new(DataSinkExec::new(
        Arc::new(CoalescePartitionsExec::new(scan)),
        Arc::new(HostSink { schema }),
        None,
    ));
    let optimized = NdBoundaryRule::new(registry).optimize(plan, &ConfigOptions::default())?;
    let rendered = displayable(optimized.as_ref()).indent(true).to_string();
    assert!(
        rendered.starts_with("NdDataSinkExec: sink=MemoryGridSink")
            && rendered.contains("NdCoalescePartitionsExec")
            && !rendered.contains("NdBroadcastExec"),
        "{rendered}"
    );

    assert_eq!(run(optimized).await?, 24);
    assert_eq!(grid_sink.grid().unwrap().num_rows(), 24);
    Ok(())
}

#[tokio::test]
async fn without_a_factory_the_flat_sink_stays() -> Result<()> {
    let table = grid_table()?;
    let scan = table.nd_scan(None, NdNodeRegistry::shared_default())?;
    let schema = scan.schema();
    let plan: Arc<dyn ExecutionPlan> = Arc::new(DataSinkExec::new(
        Arc::new(CoalescePartitionsExec::new(scan)),
        Arc::new(HostSink { schema }),
        None,
    ));
    let optimized = NdBoundaryRule::new(NdNodeRegistry::shared_default())
        .optimize(plan, &ConfigOptions::default())?;
    assert_eq!(optimized.name(), "DataSinkExec");
    Ok(())
}

/// The grid axes of `grid_table`.
fn grid_axes() -> NdGridAxes {
    NdGridAxes::new([
        ("time", AxisOrder::Ascending),
        ("lat", AxisOrder::Ascending),
        ("lon", AxisOrder::Ascending),
    ])
}

/// Write the result of `sql` on `grid_table` to a `MemoryGridSink`.
async fn write_grid(sql: &str) -> Result<NdRecordBatch> {
    let harness = Differential::new();
    harness.register("t", grid_table()?)?;
    let ctx = harness.nd_context();
    let plan = ctx.sql(sql).await?.create_physical_plan().await?;
    let boundary = plan
        .as_any()
        .downcast_ref::<NdBroadcastExec>()
        .expect("the plan ends in the nd region");
    let sink = Arc::new(MemoryGridSink::new(boundary.schema()));
    let write = NdDataSinkExec::try_new(
        boundary.input().clone(),
        sink.clone(),
        NdNodeRegistry::shared_default(),
    )?;
    collect(Arc::new(write), ctx.task_ctx()).await?;
    Ok(sink.grid().expect("one grid"))
}

#[tokio::test]
async fn coordinates_come_from_the_column_names() -> Result<()> {
    use arrow::datatypes::Float64Type;

    let grid = write_grid("SELECT time, lat, lon, sst FROM t WHERE lat > 0").await?;
    let shape: Vec<(&str, usize)> = grid.target().iter().map(|d| (d.name(), d.size())).collect();
    assert_eq!(shape, [("time", 4), ("lat", 1), ("lon", 2)]);
    let lat = grid.column(1).values().as_primitive::<Float64Type>();
    assert_eq!(lat.values(), &[30.0]);
    Ok(())
}

#[tokio::test]
async fn a_grid_axis_without_its_coordinate_fails() {
    // The query does not select `lon`, so the chunks hold no `lon` coordinate.
    let error = write_grid("SELECT time, lat, sst FROM t WHERE lat > 0 AND sst > 5")
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("select the column 'lon'"),
        "{error}"
    );
}

/// The host maps the CSV sink of `COPY TO` to an in-memory grid sink.
#[derive(Debug)]
struct CsvAsGrid {
    sink: Arc<MemoryGridSink>,
}

impl NdSinkFactory for CsvAsGrid {
    fn nd_sink(&self, sink: &dyn DataSink) -> Option<Arc<dyn NdDataSink>> {
        use datafusion::datasource::file_format::csv::CsvSink;
        sink.as_any()
            .is::<CsvSink>()
            .then(|| self.sink.clone() as Arc<dyn NdDataSink>)
    }
}

#[tokio::test]
async fn a_grid_copy_gets_its_coordinates() -> Result<()> {
    use arrow::datatypes::Float64Type;
    use datafusion::catalog::TableProvider;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::SessionContext;

    let table = grid_table()?;
    // The sink gets the schema with the coordinates that the rule adds.
    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        table.schema().field_with_name("sst")?.clone(),
        table.schema().field_with_name("time")?.clone(),
        table.schema().field_with_name("lat")?.clone(),
        table.schema().field_with_name("lon")?.clone(),
    ]));
    let grid_sink = Arc::new(MemoryGridSink::new(schema));
    let registry = Arc::new(NdNodeRegistry::new().with_sink_factory(Arc::new(CsvAsGrid {
        sink: grid_sink.clone(),
    })));
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_nd_pipeline(registry)
        .with_analyzer_rule(Arc::new(NdGridCoordinatesRule::all()))
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_table("t", Arc::new(table))?;

    let out = std::env::temp_dir().join("nd-grid-copy.csv");
    let sql = format!(
        "COPY (SELECT sst FROM t WHERE lat > 0) TO '{}' STORED AS CSV",
        out.display().to_string().replace('\\', "/")
    );
    let rows = ctx.sql(&sql).await?.collect().await?;
    let count = rows[0]
        .column(0)
        .as_primitive::<arrow::datatypes::UInt64Type>();
    assert_eq!(count.value(0), 8);

    let grid = grid_sink.grid().expect("one grid");
    let shape: Vec<(&str, usize)> = grid.target().iter().map(|d| (d.name(), d.size())).collect();
    assert_eq!(shape, [("time", 4), ("lat", 1), ("lon", 2)]);
    let lat = grid.column(2).values().as_primitive::<Float64Type>();
    assert_eq!(lat.values(), &[30.0]);
    Ok(())
}
