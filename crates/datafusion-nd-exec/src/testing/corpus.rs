//! Two test corpora: a grid with coordinate axes and a set of profiles.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::error::Result;
use nd_arrow_array::{AxisMeta, AxisOrder, Dimension, Dimensions, NdArrowArray, NdRecordBatch};

use super::table::NdMemTable;

fn coordinate(name: &str, size: usize) -> Dimension {
    Dimension::new(name, size).with_meta(Some(AxisMeta::coordinate(name, AxisOrder::Ascending)))
}

fn plain(name: &str, size: usize) -> Dimension {
    Dimension::new(name, size).with_meta(Some(AxisMeta::no_coordinate()))
}

fn nd(values: ArrayRef, dims: &[&Dimension]) -> Result<NdArrowArray> {
    let dims = Dimensions::try_new(dims.iter().map(|d| (*d).clone()).collect())?;
    Ok(NdArrowArray::try_new(values, dims)?)
}

/// The schema of [`grid_table`].
pub fn grid_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("time", DataType::Int64, true),
        Field::new("lat", DataType::Float64, true),
        Field::new("lon", DataType::Float64, true),
        Field::new("sst", DataType::Float64, true),
        Field::new("elev", DataType::Float64, true),
        Field::new("source", DataType::Utf8, true),
    ]))
}

/// A grid with axes `time=4, lat=3, lon=2` in two chunks of two time steps.
/// Each chunk is its own partition. The coordinates rise, so the table
/// declares ordered chunks.
///
/// | Column | Axes | Content |
/// |---|---|---|
/// | `time` | `time` | 100, 101, 102, 103 |
/// | `lat` | `lat` | -30, 0, 30 |
/// | `lon` | `lon` | 5, 15 |
/// | `sst` | `time, lat, lon` | `0.5 * cell`, null at every seventh cell |
/// | `elev` | `lat, lon` | -100, -50, 0, 50, 100, 150 |
/// | `source` | none | `"ship"` |
pub fn grid_table() -> Result<NdMemTable> {
    let lat = coordinate("lat", 3);
    let lon = coordinate("lon", 2);
    let mut partitions = Vec::new();
    for chunk in 0..2i64 {
        let time = coordinate("time", 2);
        let t0 = chunk * 2;
        let sst: Float64Array = (0..12)
            .map(|i| {
                let cell = t0 * 6 + i;
                (cell % 7 != 3).then_some(cell as f64 * 0.5)
            })
            .collect();
        let columns = vec![
            nd(
                Arc::new(Int64Array::from(vec![100 + t0, 101 + t0])),
                &[&time],
            )?,
            nd(
                Arc::new(Float64Array::from(vec![-30.0, 0.0, 30.0])),
                &[&lat],
            )?,
            nd(Arc::new(Float64Array::from(vec![5.0, 15.0])), &[&lon])?,
            nd(Arc::new(sst), &[&time, &lat, &lon])?,
            nd(
                Arc::new(Float64Array::from(vec![
                    -100.0, -50.0, 0.0, 50.0, 100.0, 150.0,
                ])),
                &[&lat, &lon],
            )?,
            nd(Arc::new(StringArray::from(vec!["ship"])), &[])?,
        ];
        let target = Dimensions::try_new(vec![time, lat.clone(), lon.clone()])?;
        partitions.push(vec![NdRecordBatch::try_new(
            grid_schema(),
            columns,
            target,
        )?]);
    }
    Ok(NdMemTable::try_new(partitions)?.with_ordered_chunks())
}

/// The schema of [`profile_table`].
pub fn profile_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("PLATFORM_NUMBER", DataType::Int32, true),
        Field::new("LATITUDE", DataType::Float64, true),
        Field::new("PRES", DataType::Float64, true),
        Field::new("TEMP", DataType::Float64, true),
    ]))
}

/// Profiles on axes `N_PROF, N_LEVELS` without coordinate variables, in two
/// files. Each file is its own partition and has its own axis sizes.
///
/// - File 1: `N_PROF=3, N_LEVELS=4`, profile lengths 4, 2, 3.
/// - File 2: `N_PROF=2, N_LEVELS=3`, profile lengths 3, 1.
///
/// The levels past the length of a profile hold fill values, which read as
/// nulls in `PRES` and `TEMP`.
pub fn profile_table() -> Result<NdMemTable> {
    let file = |platforms: Vec<i32>, lats: Vec<f64>, lengths: Vec<usize>, levels: usize| {
        let n_prof = plain("N_PROF", platforms.len());
        let n_levels = plain("N_LEVELS", levels);
        let mut pres = Vec::new();
        let mut temp = Vec::new();
        for (p, &len) in lengths.iter().enumerate() {
            for level in 0..levels {
                let valid = level < len;
                pres.push(valid.then_some(10.0 * (level + 1) as f64));
                temp.push(valid.then_some(20.0 - level as f64 - p as f64 * 0.5));
            }
        }
        let columns = vec![
            nd(Arc::new(Int32Array::from(platforms)), &[&n_prof])?,
            nd(Arc::new(Float64Array::from(lats)), &[&n_prof])?,
            nd(Arc::new(Float64Array::from(pres)), &[&n_prof, &n_levels])?,
            nd(Arc::new(Float64Array::from(temp)), &[&n_prof, &n_levels])?,
        ];
        let target = Dimensions::try_new(vec![n_prof, n_levels])?;
        Ok::<_, datafusion::error::DataFusionError>(NdRecordBatch::try_new(
            profile_schema(),
            columns,
            target,
        )?)
    };
    NdMemTable::try_new(vec![
        vec![file(
            vec![1901, 1901, 1902],
            vec![-10.0, -11.0, 42.5],
            vec![4, 2, 3],
            4,
        )?],
        vec![file(vec![1903, 1904], vec![60.0, 61.5], vec![3, 1], 3)?],
    ])
}
