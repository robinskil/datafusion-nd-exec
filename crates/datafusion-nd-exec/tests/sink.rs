//! The grid reaches the output through an nd sink.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::array::{Array, AsArray};
use arrow::datatypes::{Int32Type, SchemaRef};
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
use datafusion_nd_exec::sink::{NdDataSink, NdDataSinkExec, NdSinkFactory};
use datafusion_nd_exec::testing::{
    MemoryGridSink, NdMemTable, grid_table, profile_table, sorted_rows,
};
use datafusion_nd_exec::{NdBoundaryRule, NdNodeRegistry};

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
async fn profiles_append_and_pad_in_the_sink() -> Result<()> {
    let table = profile_table()?;
    let schema = table
        .nd_scan(None, NdNodeRegistry::shared_default())?
        .schema();
    let sink = Arc::new(MemoryGridSink::new(schema));
    let rows = run(Arc::new(NdDataSinkExec::try_new(
        nd_scan(&table)?,
        sink.clone(),
    )?))
    .await?;
    // 3 x 4 cells from the first file and 2 x 3 from the second.
    assert_eq!(rows, 18);

    let grid = sink.grid().unwrap();
    let names: Vec<(&str, usize)> = grid.target().iter().map(|d| (d.name(), d.size())).collect();
    assert_eq!(names, [("N_PROF", 5), ("N_LEVELS", 4)]);
    let platforms = grid.column(0).values().as_primitive::<Int32Type>();
    assert_eq!(platforms.values(), &[1901, 1901, 1902, 1903, 1904]);
    // The pad cells of the second file, level 3 of profiles 3 and 4, are null.
    let pres = grid.column(2).values();
    assert!(pres.is_null(3 * 4 + 3) && pres.is_null(4 * 4 + 3));
    assert!(pres.is_valid(3 * 4));
    Ok(())
}

#[tokio::test]
async fn chunks_in_reverse_order_come_out_sorted() -> Result<()> {
    use arrow::datatypes::Int64Type;

    let table = grid_table()?;
    let schema = table
        .nd_scan(None, NdNodeRegistry::shared_default())?
        .schema();
    let reversed = NdMemTable::try_new(table.partitions().iter().rev().cloned().collect())?;

    let in_order = Arc::new(MemoryGridSink::new(schema.clone()));
    run(Arc::new(NdDataSinkExec::try_new(
        nd_scan(&table)?,
        in_order.clone(),
    )?))
    .await?;
    let backward = Arc::new(MemoryGridSink::new(schema));
    run(Arc::new(NdDataSinkExec::try_new(
        nd_scan(&reversed)?,
        backward.clone(),
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
    )?;
    let lines = displayable(&grid_sink).indent(true).to_string();
    let lines: Vec<&str> = lines.lines().map(str::trim).collect();
    assert_eq!(lines[0], "NdDataSinkExec: sink=MemoryGridSink");
    assert_eq!(lines[1], "NdRegridExec");

    let counting = Arc::new(NdDataSinkExec::try_new(
        nd_scan(&table)?,
        Arc::new(CountingSink { schema }),
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
