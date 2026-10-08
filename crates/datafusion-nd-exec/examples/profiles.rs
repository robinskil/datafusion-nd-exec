//! Read three profile files with a varying outer axis, query them, and write
//! the result with an nd sink.
//!
//! Run it with `cargo run -p datafusion-nd-exec --example profiles`.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::array::{Float64Array, Int32Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::util::pretty::pretty_format_batches;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::Result;
use datafusion::execution::{SessionStateBuilder, TaskContext};
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, collect, displayable,
};
use datafusion::prelude::SessionContext;
use datafusion_nd_exec::array::encoding::{encode_nd_record_batch, logical_schema};
use datafusion_nd_exec::array::{Dimension, Dimensions, NdArrowArray, NdRecordBatch};
use datafusion_nd_exec::exec::{NdBroadcastExec, NdSourceExec, SendableNdBatchStream};
use datafusion_nd_exec::sink::{NdDataSink, NdDataSinkExec};
use datafusion_nd_exec::{NdNodeRegistry, NdSessionStateBuilderExt};
use futures::StreamExt;

/// One profile file: `n_prof` profiles of `n_levels` levels, with the levels
/// past each profile length as fill values (nulls).
fn profile_file(platform: i32, lengths: &[usize], n_levels: usize) -> Result<NdRecordBatch> {
    // Profile axes have no coordinate variable.
    let n_prof = Dimension::new("N_PROF", lengths.len());
    let levels = Dimension::new("N_LEVELS", n_levels);
    let grid = Dimensions::try_new(vec![n_prof.clone(), levels])?;

    let pres: Float64Array = lengths
        .iter()
        .flat_map(|&len| (0..n_levels).map(move |l| (l < len).then_some(10.0 * (l + 1) as f64)))
        .collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("PLATFORM_NUMBER", DataType::Int32, true),
        Field::new("PRES", DataType::Float64, true),
    ]));
    let columns = vec![
        // One value per profile: the column lives on `N_PROF` only.
        NdArrowArray::try_new(
            Arc::new(Int32Array::from(vec![platform; lengths.len()])),
            Dimensions::try_new(vec![n_prof])?,
        )?,
        // One value per cell of the grid.
        NdArrowArray::try_new(Arc::new(pres), grid.clone())?,
    ];
    Ok(NdRecordBatch::try_new(schema, columns, grid)?)
}

/// A minimal nd format: each file is one chunk, and each file is its own
/// partition. A real format reads chunks from files the same way.
#[derive(Debug)]
struct ProfileFiles {
    /// The logical schema of the table.
    schema: SchemaRef,
    /// The `nd.array`-encoded chunks, one partition per file.
    encoded: Vec<Vec<arrow::record_batch::RecordBatch>>,
    encoded_schema: SchemaRef,
}

impl ProfileFiles {
    fn try_new(files: Vec<NdRecordBatch>) -> Result<Self> {
        let encoded: Vec<_> = files
            .iter()
            .map(encode_nd_record_batch)
            .collect::<std::result::Result<_, _>>()?;
        let encoded_schema = encoded[0].schema();
        // Every chunk must carry the same encoded schema.
        let encoded = encoded
            .into_iter()
            .map(|batch| {
                Ok(vec![arrow::record_batch::RecordBatch::try_new(
                    encoded_schema.clone(),
                    batch.columns().to_vec(),
                )?])
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            schema: logical_schema(&encoded_schema)?,
            encoded,
            encoded_schema,
        })
    }
}

#[async_trait]
impl TableProvider for ProfileFiles {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // The scan shape of every nd format: encoded chunks, decoded by
        // NdSourceExec, then the boundary to flat rows.
        let memory = MemorySourceConfig::try_new_exec(
            &self.encoded,
            self.encoded_schema.clone(),
            projection.cloned(),
        )?;
        let source = Arc::new(NdSourceExec::try_new(memory)?);
        let registry = NdNodeRegistry::from_session_config(state.config());
        Ok(Arc::new(NdBroadcastExec::try_new(source, registry)?))
    }
}

/// A toy nd writer: it prints the grid of each chunk. With `grid`, it asks for
/// one output grid, as a netCDF or Zarr grid writer does.
#[derive(Debug)]
struct PrintingSink {
    schema: SchemaRef,
    grid: bool,
}

impl DisplayAs for PrintingSink {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PrintingSink")
    }
}

#[async_trait]
impl NdDataSink for PrintingSink {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn requires_grid(&self) -> bool {
        self.grid
    }

    async fn write_all(
        &self,
        mut data: SendableNdBatchStream,
        _context: &Arc<TaskContext>,
    ) -> Result<u64> {
        let mut rows = 0;
        while let Some(chunk) = data.next().await {
            let chunk = chunk?;
            println!("write chunk on {}", describe(chunk.target()));
            rows += chunk.num_rows() as u64;
        }
        Ok(rows)
    }
}

fn describe(dims: &Dimensions) -> String {
    let axes: Vec<String> = dims
        .iter()
        .map(|d| format!("{}={}", d.name(), d.size()))
        .collect();
    axes.join(", ")
}

#[tokio::main]
async fn main() -> Result<()> {
    // Three files: N_PROF is 3, 2 and 4, and N_LEVELS is 4, 3 and 5.
    let files = vec![
        profile_file(1901, &[4, 2, 3], 4)?,
        profile_file(1902, &[3, 1], 3)?,
        profile_file(1903, &[5, 5, 2, 4], 5)?,
    ];
    let table = Arc::new(ProfileFiles::try_new(files)?);

    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_nd_pipeline(Arc::new(NdNodeRegistry::new()))
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_table("profiles", table)?;

    let sql = r#"SELECT "PLATFORM_NUMBER", "PRES" FROM profiles WHERE "PRES" < 25"#;
    let plan = ctx.sql(sql).await?.create_physical_plan().await?;
    println!("plan:\n{}", displayable(plan.as_ref()).indent(true));
    let rows = collect(plan.clone(), ctx.task_ctx()).await?;
    println!("{}", pretty_format_batches(&rows)?);

    // Write the same query result with an nd sink: the sink takes the nd child
    // of the boundary at the root.
    let boundary = plan
        .as_any()
        .downcast_ref::<NdBroadcastExec>()
        .expect("the plan ends in the nd region");
    let registry = NdNodeRegistry::shared_default();
    let sink = |grid| {
        Arc::new(PrintingSink {
            schema: boundary.schema(),
            grid,
        })
    };

    // The profile axes have no coordinates, so the scan declares no grid axes
    // and a grid sink refuses the plan.
    let refused = NdDataSinkExec::try_new(boundary.input().clone(), sink(true), registry.clone());
    println!("grid sink: {}", refused.unwrap_err());

    // A sink without a grid streams each chunk on its own grid.
    let write = Arc::new(NdDataSinkExec::try_new(
        boundary.input().clone(),
        sink(false),
        registry,
    )?);
    println!("write plan:\n{}", displayable(write.as_ref()).indent(true));
    collect(write, ctx.task_ctx()).await?;
    Ok(())
}
