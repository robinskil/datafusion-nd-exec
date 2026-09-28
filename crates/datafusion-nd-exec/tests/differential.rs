//! Differential tests: each query must give the same rows on the flat path and
//! on the nd path.

use datafusion::error::Result;
use datafusion_nd_exec::testing::{Differential, grid_table, profile_table};

fn harness() -> Result<Differential> {
    let harness = Differential::new();
    harness.register("grid", grid_table()?)?;
    harness.register("profiles", profile_table()?)?;
    Ok(harness)
}

const GRID_QUERIES: &[&str] = &[
    "SELECT * FROM grid",
    "SELECT time, lat, sst FROM grid WHERE lat > -1",
    "SELECT * FROM grid WHERE time BETWEEN 101 AND 102 AND lon = 5",
    "SELECT * FROM grid WHERE time <> 101",
    "SELECT * FROM grid WHERE sst > 3",
    "SELECT * FROM grid WHERE sst IS NULL",
    "SELECT * FROM grid WHERE lat + lon > 10",
    "SELECT * FROM grid WHERE lat < 0 OR lon = 15",
    "SELECT * FROM grid WHERE lat > -30 AND sst < 20",
    "SELECT lat * 2 AS lat2, sst + elev AS s FROM grid WHERE lon <> 15",
    "SELECT source, elev FROM grid WHERE elev >= 0",
    "SELECT count(*) FROM grid",
    "SELECT count(*) FROM grid WHERE lat = 0",
    "SELECT lat, avg(sst), count(sst) FROM grid GROUP BY lat",
    "SELECT source, min(elev), max(sst) FROM grid GROUP BY source",
    "SELECT CASE WHEN sst > 5 THEN 'warm' ELSE 'cold' END AS c, count(*) FROM grid GROUP BY 1",
    "SELECT * FROM grid WHERE random() < 2 AND lat > 0",
    "SELECT time, lat, lon FROM grid ORDER BY time DESC, lat, lon LIMIT 5",
];

const PROFILE_QUERIES: &[&str] = &[
    "SELECT * FROM profiles",
    r#"SELECT * FROM profiles WHERE "PRES" IS NOT NULL"#,
    r#"SELECT * FROM profiles WHERE "PLATFORM_NUMBER" = 1901 AND "PRES" < 25"#,
    r#"SELECT * FROM profiles WHERE "LATITUDE" > 0"#,
    r#"SELECT "PLATFORM_NUMBER", max("PRES"), avg("TEMP") FROM profiles GROUP BY "PLATFORM_NUMBER""#,
    "SELECT count(*) FROM profiles",
    r#"SELECT count("TEMP") FROM profiles WHERE "TEMP" > 18"#,
];

#[tokio::test]
async fn grid_queries_match_the_flat_path() -> Result<()> {
    let harness = harness()?;
    for sql in GRID_QUERIES {
        harness.assert_same(sql).await?;
    }
    Ok(())
}

#[tokio::test]
async fn profile_queries_match_the_flat_path() -> Result<()> {
    let harness = harness()?;
    for sql in PROFILE_QUERIES {
        harness.assert_same(sql).await?;
    }
    Ok(())
}
