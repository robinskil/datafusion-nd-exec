//! End-to-end tests of the nd operators over an in-memory source.

use std::sync::Arc;

use arrow::array::{AsArray, Float64Array, Int32Array};
use arrow::compute::concat_batches;
use arrow::datatypes::{Float64Type, Int32Type, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::error::Result;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::ExecutionPlan;
use futures::TryStreamExt;

use nd_arrow_array::encoding::encode_nd_record_batch;
use nd_arrow_array::{Dimension, Dimensions, NdArrowArray, NdRecordBatch};

use crate::exec::{NdBroadcastExec, NdSourceExec};
use crate::registry::NdNodeRegistry;

fn dims(spec: &[(&str, usize)]) -> Dimensions {
    Dimensions::try_new(
        spec.iter()
            .map(|(name, size)| Dimension::new(*name, *size))
            .collect(),
    )
    .unwrap()
}

/// A (time=4, lat=3, lon=2) grid with coordinate variables and one data
/// variable, encoded into two nd-encoded `RecordBatch`es (split along
/// `time`) and served from an in-memory `DataSourceExec` — the shape a file
/// opener produces.
fn test_source() -> Arc<NdSourceExec> {
    let logical = Arc::new(Schema::new(vec![
        arrow::datatypes::Field::new("time", arrow::datatypes::DataType::Int32, true),
        arrow::datatypes::Field::new("lat", arrow::datatypes::DataType::Int32, true),
        arrow::datatypes::Field::new("lon", arrow::datatypes::DataType::Int32, true),
        arrow::datatypes::Field::new("sst", arrow::datatypes::DataType::Float64, true),
    ]));

    let lat = || {
        NdArrowArray::try_new(
            Arc::new(Int32Array::from(vec![-30, 0, 30])),
            dims(&[("lat", 3)]),
        )
        .unwrap()
    };
    let lon = || {
        NdArrowArray::try_new(Arc::new(Int32Array::from(vec![5, 15])), dims(&[("lon", 2)])).unwrap()
    };

    let mut encoded = Vec::new();
    for chunk in 0..2u32 {
        let t0 = chunk * 2;
        let time = NdArrowArray::try_new(
            Arc::new(Int32Array::from(vec![100 + t0 as i32, 101 + t0 as i32])),
            dims(&[("time", 2)]),
        )
        .unwrap();
        let sst_values: Vec<f64> = (0..12).map(|i| (t0 * 6 + i) as f64).collect();
        let sst = NdArrowArray::try_new(
            Arc::new(Float64Array::from(sst_values)),
            dims(&[("time", 2), ("lat", 3), ("lon", 2)]),
        )
        .unwrap();
        let nd = NdRecordBatch::try_new(
            logical.clone(),
            vec![time, lat(), lon(), sst],
            dims(&[("time", 2), ("lat", 3), ("lon", 2)]),
        )
        .unwrap();
        encoded.push(encode_nd_record_batch(&nd).unwrap());
    }

    let encoded_schema = encoded[0].schema();
    // One partition, each nd chunk its own single-row encoded batch.
    let source = MemorySourceConfig::try_new_exec(&[encoded], encoded_schema, None).unwrap();
    Arc::new(NdSourceExec::try_new(source).unwrap())
}

async fn run(plan: Arc<dyn ExecutionPlan>) -> Result<RecordBatch> {
    let schema = plan.schema();
    let batches: Vec<_> = plan
        .execute(0, Arc::new(TaskContext::default()))?
        .try_collect()
        .await?;
    Ok(concat_batches(&schema, &batches)?)
}

#[tokio::test]
async fn source_alone_materializes_full_grid() {
    // NdSourceExec's own `execute` materializes (no broadcast node needed).
    let batch = run(test_source()).await.unwrap();
    assert_eq!(batch.num_rows(), 24);
    assert_eq!(batch.column(0).as_primitive::<Int32Type>().value(0), 100);
    assert_eq!(batch.column(1).as_primitive::<Int32Type>().value(0), -30);
    assert_eq!(batch.column(2).as_primitive::<Int32Type>().value(0), 5);
}

#[tokio::test]
async fn source_then_broadcast_materializes_full_grid() {
    let plan = Arc::new(
        NdBroadcastExec::try_new(test_source(), NdNodeRegistry::shared_default()).unwrap(),
    );
    let batch = run(plan).await.unwrap();

    // 4 time x 3 lat x 2 lon = 24 rows, C-order (time outer, lon inner).
    assert_eq!(batch.num_rows(), 24);

    // time repeats over the 6 (lat,lon) cells of each step.
    assert_eq!(
        batch.column(0).as_primitive::<Int32Type>().values(),
        &[
            100, 100, 100, 100, 100, 100, 101, 101, 101, 101, 101, 101, 102, 102, 102, 102, 102,
            102, 103, 103, 103, 103, 103, 103
        ]
    );
    // lat repeats over lon, tiles over time.
    assert_eq!(
        &batch.column(1).as_primitive::<Int32Type>().values()[..6],
        &[-30, -30, 0, 0, 30, 30]
    );
    // lon tiles over everything.
    assert_eq!(
        &batch.column(2).as_primitive::<Int32Type>().values()[..6],
        &[5, 15, 5, 15, 5, 15]
    );
    // sst is full-rank: passes through unbroadcast, values 0..24.
    assert_eq!(
        batch.column(3).as_primitive::<Float64Type>().values(),
        &(0..24).map(|v| v as f64).collect::<Vec<_>>()[..]
    );
}

#[tokio::test]
async fn nodes_report_metrics() {
    let source = test_source();
    let broadcast = Arc::new(
        NdBroadcastExec::try_new(source.clone(), NdNodeRegistry::shared_default()).unwrap(),
    );

    // Drain the plan so the streams run to completion and finalize metrics.
    let out = run(broadcast.clone() as Arc<dyn ExecutionPlan>)
        .await
        .unwrap();
    assert_eq!(out.num_rows(), 24);

    // Broadcast reports the flattened output rows.
    let broadcast_metrics = broadcast.metrics().unwrap();
    assert_eq!(broadcast_metrics.output_rows(), Some(24));

    // Each chunk broadcasts time/lat/lon (3) and passes sst through (1);
    // two chunks → 6 implicit broadcasts, 2 pass-throughs.
    assert_eq!(
        broadcast_metrics
            .sum_by_name("implicit_broadcasts")
            .map(|v| v.as_usize()),
        Some(6)
    );
    assert_eq!(
        broadcast_metrics
            .sum_by_name("passthrough_columns")
            .map(|v| v.as_usize()),
        Some(2)
    );

    // Source reports the grid rows it decoded (12 + 12) and the number of
    // nd batches.
    let source_metrics = source.metrics().unwrap();
    assert_eq!(source_metrics.output_rows(), Some(24));
    assert_eq!(
        source_metrics
            .sum_by_name("nd_batches")
            .map(|v| v.as_usize()),
        Some(2)
    );
}

#[tokio::test]
async fn broadcast_requires_nd_input() {
    // A non-nd child is rejected at construction.
    let broadcast = Arc::new(
        NdBroadcastExec::try_new(test_source(), NdNodeRegistry::shared_default()).unwrap(),
    );
    // Wrapping a broadcast (which is not nd-aware) must fail.
    assert!(NdBroadcastExec::try_new(broadcast, NdNodeRegistry::shared_default()).is_err());
}

// ── projection ───────────────────────────────────────────────────────

use datafusion::logical_expr::Operator;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{binary, col, lit};
use datafusion::physical_plan::projection::{ProjectionExec, ProjectionExpr};

use crate::exec::NdProjectionExec;

/// Single-column expressions take the fast path: the projection does no
/// co-broadcast (footprint == the column's own dims), leaving the single
/// gather to the terminal broadcast. `lat * 2` (single column) and `sst`
/// (bare passthrough) both report zero implicit broadcasts, with unchanged
/// results.
#[tokio::test]
async fn single_column_projection_skips_broadcast() {
    let schema = test_source().schema();
    let exprs: Vec<(Arc<dyn PhysicalExpr>, String)> = vec![
        (
            binary(
                col("lat", &schema).unwrap(),
                Operator::Multiply,
                lit(2i32),
                &schema,
            )
            .unwrap(),
            "lat2".to_string(),
        ),
        (col("sst", &schema).unwrap(), "sst".to_string()),
    ];

    let reference = Arc::new(
        ProjectionExec::try_new(
            exprs
                .iter()
                .cloned()
                .map(|(expr, alias)| ProjectionExpr { expr, alias }),
            Arc::new(
                NdBroadcastExec::try_new(test_source(), NdNodeRegistry::shared_default()).unwrap(),
            ),
        )
        .unwrap(),
    );
    let expected = run(reference).await.unwrap();

    let projection = Arc::new(
        NdProjectionExec::try_new(test_source(), exprs, None, NdNodeRegistry::shared_default())
            .unwrap(),
    );
    let broadcast = Arc::new(
        NdBroadcastExec::try_new(projection.clone(), NdNodeRegistry::shared_default()).unwrap(),
    );
    let actual = run(broadcast).await.unwrap();

    assert_eq!(actual, expected);
    // The projection performed no gather; only the terminal broadcast did.
    assert_eq!(
        projection
            .metrics()
            .unwrap()
            .sum_by_name("implicit_broadcasts")
            .map(|v| v.as_usize()),
        Some(0)
    );
}

/// NdProjectionExec reports the work it did and saved: elements evaluated on
/// footprints, elements avoided versus the full grid, and implicit
/// co-broadcasts.
#[tokio::test]
async fn projection_reports_metrics() {
    let schema = test_source().schema();
    // lat{lat} + lon{lon} → footprint {lat, lon}; both inputs co-broadcast.
    let exprs: Vec<(Arc<dyn PhysicalExpr>, String)> = vec![(
        binary(
            col("lat", &schema).unwrap(),
            Operator::Plus,
            col("lon", &schema).unwrap(),
            &schema,
        )
        .unwrap(),
        "lat_plus_lon".to_string(),
    )];

    let projection = Arc::new(
        NdProjectionExec::try_new(test_source(), exprs, None, NdNodeRegistry::shared_default())
            .unwrap(),
    );
    let broadcast = Arc::new(
        NdBroadcastExec::try_new(projection.clone(), NdNodeRegistry::shared_default()).unwrap(),
    );
    let out = run(broadcast).await.unwrap();
    assert_eq!(out.num_rows(), 24);

    let m = projection.metrics().unwrap();
    let count = |name: &str| m.sum_by_name(name).map(|v| v.as_usize());
    // Per chunk: full grid = 2·3·2 = 12, footprint {lat=3, lon=2} = 6; two
    // chunks. Evaluated 6·2 = 12, saved (12−6)·2 = 12.
    assert_eq!(count("elements_evaluated"), Some(12));
    assert_eq!(count("elements_saved"), Some(12));
    // Both referenced columns gather onto the footprint: 2 per chunk · 2 = 4.
    assert_eq!(count("implicit_broadcasts"), Some(4));
}
