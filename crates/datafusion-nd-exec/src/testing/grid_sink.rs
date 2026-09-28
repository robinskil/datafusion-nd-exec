//! An in-memory grid writer: the reference sink of the tests.

use std::any::Any;
use std::fmt;
use std::sync::{Arc, Mutex};

use arrow::array::{Array, ArrayRef, new_null_array};
use arrow::compute::interleave;
use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType};
use futures::StreamExt;
use nd_arrow_array::{Dimension, Dimensions, NdArrowArray, NdRecordBatch};

use crate::exec::SendableNdBatchStream;
use crate::sink::{AxisMode, NdDataSink, NdGridAccumulator, Placement};

/// An [`NdDataSink`] that places each chunk with the [`NdGridAccumulator`]
/// and builds the dense output grid in memory. A cell that no chunk writes is
/// null.
#[derive(Debug)]
pub struct MemoryGridSink {
    schema: SchemaRef,
    modes: Vec<(String, AxisMode)>,
    growth_axis: Option<String>,
    seeds: Vec<(String, ArrayRef)>,
    grid: Mutex<Option<NdRecordBatch>>,
}

impl MemoryGridSink {
    pub fn new(schema: SchemaRef) -> Self {
        Self {
            schema,
            modes: Vec::new(),
            growth_axis: None,
            seeds: Vec::new(),
            grid: Mutex::new(None),
        }
    }

    /// Set the accumulator mode of the axis `axis`.
    pub fn with_mode(mut self, axis: impl Into<String>, mode: AxisMode) -> Self {
        self.modes.push((axis.into(), mode));
        self
    }

    /// Set the growth axis of the accumulator.
    pub fn with_growth_axis(mut self, axis: impl Into<String>) -> Self {
        self.growth_axis = Some(axis.into());
        self
    }

    /// Seed the coordinate of the axis `axis`, see
    /// [`NdGridAccumulator::with_coordinate`].
    pub fn with_coordinate(mut self, axis: impl Into<String>, values: ArrayRef) -> Self {
        self.seeds.push((axis.into(), values));
        self
    }

    fn accumulator(&self) -> NdGridAccumulator {
        let mut acc = NdGridAccumulator::new();
        for (axis, mode) in &self.modes {
            acc = acc.with_mode(axis.clone(), *mode);
        }
        if let Some(axis) = &self.growth_axis {
            acc = acc.with_growth_axis(axis.clone());
        }
        for (axis, values) in &self.seeds {
            acc = acc.with_coordinate(axis.clone(), values.clone());
        }
        acc
    }

    /// The output grid of the last `write_all`.
    pub fn grid(&self) -> Option<NdRecordBatch> {
        self.grid.lock().expect("grid lock").clone()
    }
}

impl DisplayAs for MemoryGridSink {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MemoryGridSink")
    }
}

#[async_trait]
impl NdDataSink for MemoryGridSink {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    async fn write_all(
        &self,
        mut data: SendableNdBatchStream,
        _context: &Arc<TaskContext>,
    ) -> Result<u64> {
        let mut accumulator = self.accumulator();
        let mut placements = Vec::new();
        while let Some(batch) = data.next().await {
            placements.push(accumulator.place(&batch?)?);
        }
        let rows = placements.iter().map(|p| p.batch.num_rows() as u64).sum();
        let grid = build_grid(&self.schema, &accumulator, &placements)?;
        *self.grid.lock().expect("grid lock") = grid;
        Ok(rows)
    }
}

/// The dense output grid of all placements, or `None` without placements.
fn build_grid(
    schema: &SchemaRef,
    accumulator: &NdGridAccumulator,
    placements: &[Placement],
) -> Result<Option<NdRecordBatch>> {
    let Some(first) = placements.first() else {
        return Ok(None);
    };
    let extents = accumulator.extents();
    let meta_of = |name: &str| {
        first
            .batch
            .target()
            .iter()
            .find(|d| d.name() == name)
            .and_then(|d| d.meta().cloned())
    };
    let target = Dimensions::try_new(
        extents
            .iter()
            .map(|(name, size)| Dimension::new(name.as_str(), *size).with_meta(meta_of(name)))
            .collect(),
    )?;
    let columns = schema
        .fields()
        .iter()
        .enumerate()
        .map(|(index, field)| {
            // A seeded coordinate comes from its seed, so a value that no
            // chunk holds is not null.
            let seeded = extents.iter().find_map(|(axis, _)| {
                (accumulator.coordinate_column(axis) == Some(field.name().as_str()))
                    .then(|| accumulator.coordinate_seed(axis).map(|seed| (axis, seed)))
                    .flatten()
            });
            match seeded {
                Some((axis, seed)) => {
                    let dims = Dimensions::try_new(vec![
                        target
                            .get(target.position(axis).expect("output axis"))
                            .clone(),
                    ])?;
                    Ok(NdArrowArray::try_new(seed.clone(), dims)?)
                }
                None => build_column(index, &target, placements),
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(NdRecordBatch::try_new(
        schema.clone(),
        columns,
        target,
    )?))
}

/// One output column: each output cell takes the value of the last chunk that
/// writes it.
///
/// The column has the axes of the chunk where it has the most axes. A chunk
/// where it has fewer axes, for example a scalar null for a file that lacks
/// the column, broadcasts its values over its own block.
fn build_column(
    index: usize,
    target: &Dimensions,
    placements: &[Placement],
) -> Result<NdArrowArray> {
    let widest = placements
        .iter()
        .map(|p| p.batch.column(index))
        .max_by_key(|column| column.dims().rank())
        .expect("at least one placement");
    let names: Vec<&str> = widest.dims().iter().map(|d| d.name()).collect();
    let axis_of = |name: &str| {
        target.position(name).ok_or_else(|| {
            DataFusionError::Execution(format!("column axis '{name}' is not an output axis"))
        })
    };
    let dims = Dimensions::try_new(
        names
            .iter()
            .map(|name| Ok(target.get(axis_of(name)?).clone()))
            .collect::<Result<Vec<_>>>()?,
    )?;
    let strides = dims.c_strides();

    // The source of each output cell: (placement, index), or the null source.
    let null_source = placements.len();
    let mut sources = vec![(null_source, 0usize); dims.num_elements()];
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(placements.len() + 1);
    for (k, placement) in placements.iter().enumerate() {
        // The block of the chunk on the column axes, in output axis order.
        let block_axes: Vec<&crate::sink::AxisPlacement> = names
            .iter()
            .map(|name| {
                placement.axis(name).ok_or_else(|| {
                    DataFusionError::Execution(format!("the chunk has no placement for '{name}'"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let block = Dimensions::try_new(
            block_axes
                .iter()
                .map(|axis| Dimension::new(axis.name.as_str(), axis.len))
                .collect(),
        )?;
        arrays.push(placement.batch.column(index).materialize(&block)?);
        let block_strides = block.c_strides();
        for cell in 0..block.num_elements() {
            let out: usize = block_axes
                .iter()
                .enumerate()
                .map(|(axis, placed)| {
                    let coord = (cell / block_strides[axis]) % placed.len;
                    (placed.offset + coord) * strides[axis]
                })
                .sum();
            sources[out] = (k, cell);
        }
    }
    arrays.push(new_null_array(widest.values().data_type(), 1));
    let refs: Vec<&dyn Array> = arrays.iter().map(|a| a.as_ref()).collect();
    Ok(NdArrowArray::try_new(interleave(&refs, &sources)?, dims)?)
}

#[cfg(test)]
mod tests {
    use arrow::array::{AsArray, Float64Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Float64Type, Schema};
    use nd_arrow_array::{AxisMeta, AxisOrder};

    use super::*;

    /// A chunk on `time` with a `sst` column, or a scalar null `sst` for a file
    /// that lacks the column.
    fn chunk(times: Vec<i64>, sst: Option<Vec<f64>>) -> NdRecordBatch {
        let time = Dimension::new("time", times.len())
            .with_meta(Some(AxisMeta::coordinate("time", AxisOrder::Ascending)));
        let dims = Dimensions::try_new(vec![time]).unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("time", DataType::Int64, true),
            Field::new("sst", DataType::Float64, true),
        ]));
        let sst = match sst {
            Some(values) => {
                NdArrowArray::try_new(Arc::new(Float64Array::from(values)), dims.clone())
            }
            None => {
                NdArrowArray::try_new(new_null_array(&DataType::Float64, 1), Dimensions::scalar())
            }
        }
        .unwrap();
        let time = NdArrowArray::try_new(Arc::new(Int64Array::from(times)), dims.clone()).unwrap();
        NdRecordBatch::try_new(schema, vec![time, sst], dims).unwrap()
    }

    #[tokio::test]
    async fn a_file_without_the_column_writes_nulls() {
        let first = chunk(vec![100, 101], None);
        let schema = first.schema().clone();
        let sink = MemoryGridSink::new(schema);
        let data: SendableNdBatchStream = Box::pin(futures::stream::iter(vec![
            Ok(first),
            Ok(chunk(vec![102], Some(vec![1.5]))),
        ]));
        sink.write_all(data, &Arc::new(TaskContext::default()))
            .await
            .unwrap();

        let grid = sink.grid().unwrap();
        let sst = grid.column(1);
        assert_eq!(sst.dims().rank(), 1);
        let values = sst.values().as_primitive::<Float64Type>();
        assert!(values.is_null(0) && values.is_null(1));
        assert_eq!(values.value(2), 1.5);
    }
}
