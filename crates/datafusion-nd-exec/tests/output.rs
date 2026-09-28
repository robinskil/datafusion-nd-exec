//! The nd output: encoded chunks that a client decodes to the same rows.

use arrow::array::{Array, RecordBatch};
use datafusion::error::Result;
use datafusion_nd_exec::array::encoding::{decode_nd_record_batch_row, nd_batch_count};
use datafusion_nd_exec::sink::nd_output_plan;
use datafusion_nd_exec::testing::{Differential, grid_table, profile_table, sorted_rows};

fn harness() -> Result<Differential> {
    let harness = Differential::new();
    harness.register("grid", grid_table()?)?;
    harness.register("profiles", profile_table()?)?;
    Ok(harness)
}

/// The nd output of `sql`, decoded and materialized.
async fn decoded(harness: &Differential, sql: &str) -> Result<Option<Vec<RecordBatch>>> {
    let state = harness.nd_context().state();
    let plan = harness
        .nd_context()
        .sql(sql)
        .await?
        .create_physical_plan()
        .await?;
    let Some(output) = nd_output_plan(&plan)? else {
        return Ok(None);
    };
    let encoded = datafusion::physical_plan::collect(output, state.task_ctx()).await?;
    let mut rows = Vec::new();
    for batch in &encoded {
        for row in 0..nd_batch_count(batch) {
            rows.push(decode_nd_record_batch_row(batch, row)?.materialize()?);
        }
    }
    Ok(Some(rows))
}

#[tokio::test]
async fn a_rectangle_decodes_to_the_flat_rows() -> Result<()> {
    let harness = harness()?;
    for sql in [
        "SELECT * FROM grid",
        "SELECT lat, sst FROM grid WHERE lat > 0",
        "SELECT lat * 2 AS lat2, time FROM grid WHERE time > 100",
    ] {
        let expected = harness.flat_context().sql(sql).await?.collect().await?;
        let actual = decoded(&harness, sql)
            .await?
            .expect("the plan ends in the nd region");
        assert_eq!(sorted_rows(&actual)?, sorted_rows(&expected)?, "{sql}");
    }
    Ok(())
}

#[tokio::test]
async fn a_cell_mask_decodes_with_nulls_at_the_dropped_cells() -> Result<()> {
    let harness = harness()?;
    let sql = r#"SELECT "PRES" FROM profiles WHERE "PRES" < 25"#;
    let expected = harness.flat_context().sql(sql).await?.collect().await?;
    let actual = decoded(&harness, sql)
        .await?
        .expect("the plan ends in the nd region");
    let kept = |batches: &[RecordBatch]| {
        batches
            .iter()
            .map(|b| b.num_rows() - b.column(0).null_count())
            .sum::<usize>()
    };
    assert_eq!(kept(&actual), kept(&expected));
    Ok(())
}

#[tokio::test]
async fn flat_operators_above_the_region_give_no_nd_output() -> Result<()> {
    let harness = harness()?;
    assert!(
        decoded(&harness, "SELECT count(*) FROM grid")
            .await?
            .is_none()
    );
    assert!(
        decoded(&harness, "SELECT lat, avg(sst) FROM grid GROUP BY lat")
            .await?
            .is_none()
    );
    Ok(())
}
