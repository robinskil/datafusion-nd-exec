//! An in-memory grid writer: the reference sink of the tests.

use std::any::Any;
use std::fmt;
use std::sync::{Arc, Mutex};

use arrow::array::{Array, ArrayRef, UInt64Array, new_null_array};
use arrow::compute::interleave;
use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType};
use futures::StreamExt;
use nd_arrow_array::{Dimension, Dimensions, NdArrowArray, NdOutputGrid, NdRecordBatch};

use crate::exec::SendableNdBatchStream;
use crate::sink::NdDataSink;

/// An [`NdDataSink`] that builds each dense output grid in memory from the
/// placements of the regrid step. A cell that no batch writes is null.
#[derive(Debug)]
pub struct MemoryGridSink {
    schema: SchemaRef,
    grids: Mutex<Vec<NdRecordBatch>>,
}

impl MemoryGridSink {
    pub fn new(schema: SchemaRef) -> Self {
        Self {
            schema,
            grids: Mutex::new(Vec::new()),
        }
    }

    /// The first output grid of the last `write_all`.
    pub fn grid(&self) -> Option<NdRecordBatch> {
        self.grids().into_iter().next()
    }

    /// All output grids of the last `write_all`, in the order of their first
    /// batch.
    pub fn grids(&self) -> Vec<NdRecordBatch> {
        self.grids.lock().expect("grid lock").clone()
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

    fn requires_grid(&self) -> bool {
        true
    }

    async fn write_all(
        &self,
        mut data: SendableNdBatchStream,
        _context: &Arc<TaskContext>,
    ) -> Result<u64> {
        let mut groups: Vec<(Arc<NdOutputGrid>, Vec<NdRecordBatch>)> = Vec::new();
        let mut rows = 0;
        while let Some(batch) = data.next().await {
            let batch = batch?;
            let grid = batch
                .placement()
                .ok_or_else(|| {
                    DataFusionError::Execution("MemoryGridSink needs placed batches".to_string())
                })?
                .grid()
                .clone();
            rows += batch.num_rows() as u64;
            match groups.iter_mut().find(|(g, _)| Arc::ptr_eq(g, &grid)) {
                Some((_, batches)) => batches.push(batch),
                None => groups.push((grid, vec![batch])),
            }
        }
        let grids = groups
            .iter()
            .map(|(grid, batches)| build_grid(&self.schema, grid, batches))
            .collect::<Result<Vec<_>>>()?;
        *self.grids.lock().expect("grid lock") = grids;
        Ok(rows)
    }
}

/// The dense output grid of `batches`.
fn build_grid(
    schema: &SchemaRef,
    grid: &NdOutputGrid,
    batches: &[NdRecordBatch],
) -> Result<NdRecordBatch> {
    let target = grid.dims().clone();
    let columns = schema
        .fields()
        .iter()
        .enumerate()
        .map(|(index, field)| {
            // A coordinate column has the name of its axis, and its values come
            // from the grid, so each step has a value.
            match (target.position(field.name()), grid.coordinate(field.name())) {
                (Some(axis), Some(values)) => Ok(NdArrowArray::try_new(
                    values.clone(),
                    Dimensions::try_new(vec![target.get(axis).clone()])?,
                )?),
                _ => build_column(index, &target, batches),
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(NdRecordBatch::try_new(schema.clone(), columns, target)?)
}

/// One output column: each output cell takes the value of the batch that
/// writes it.
///
/// The column has the axes of the batch where it has the most axes. A batch
/// where it has fewer axes, for example a scalar null for a file that lacks
/// the column, broadcasts its values over its own block.
fn build_column(
    index: usize,
    target: &Dimensions,
    batches: &[NdRecordBatch],
) -> Result<NdArrowArray> {
    let widest = batches
        .iter()
        .map(|b| b.column(index))
        .max_by_key(|column| column.dims().rank())
        .expect("at least one batch");
    let names: Vec<&str> = widest.dims().iter().map(|d| d.name()).collect();
    let dims = Dimensions::try_new(
        names
            .iter()
            .map(|name| {
                let axis = target.position(name).ok_or_else(|| {
                    DataFusionError::Execution(format!("column axis '{name}' is not a grid axis"))
                })?;
                Ok(target.get(axis).clone())
            })
            .collect::<Result<Vec<_>>>()?,
    )?;
    let strides = dims.c_strides();

    // The source of each output cell: (batch, index), or the null source.
    let null_source = batches.len();
    let mut sources = vec![(null_source, 0usize); dims.num_elements()];
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(batches.len() + 1);
    for (k, batch) in batches.iter().enumerate() {
        let placement = batch.placement().expect("checked in write_all");
        let positions: Vec<&UInt64Array> = names
            .iter()
            .map(|name| {
                placement.axis_indices(name).ok_or_else(|| {
                    DataFusionError::Execution(format!("the placement has no axis '{name}'"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let block = Dimensions::try_new(
            names
                .iter()
                .zip(&positions)
                .map(|(name, p)| Dimension::new(*name, p.len()))
                .collect(),
        )?;
        arrays.push(batch.column(index).materialize(&block)?);
        let block_strides = block.c_strides();
        for cell in 0..block.num_elements() {
            let out: usize = positions
                .iter()
                .enumerate()
                .map(|(axis, p)| {
                    let coord = (cell / block_strides[axis]) % p.len();
                    p.value(coord) as usize * strides[axis]
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
    use nd_arrow_array::{AxisOrder, NdBatchRecord, NdGridAxes, NdGridBuilder};

    use super::*;

    /// A chunk on `time` with a `sst` column, or a scalar null `sst` for a file
    /// that lacks the column.
    fn chunk(times: Vec<i64>, sst: Option<Vec<f64>>) -> NdRecordBatch {
        let time = Dimension::new("time", times.len());
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
        let chunks = vec![
            chunk(vec![100, 101], None),
            chunk(vec![102], Some(vec![1.5])),
        ];
        let schema = chunks[0].schema().clone();
        let axes = NdGridAxes::new([("time", AxisOrder::Ascending)]);
        let mut builder = NdGridBuilder::new(Arc::new(axes));
        for (number, batch) in chunks.iter().enumerate() {
            builder
                .add(NdBatchRecord::of(0, number, batch).unwrap())
                .unwrap();
        }
        let placed: Vec<Result<NdRecordBatch>> = chunks
            .into_iter()
            .zip(builder.finish().unwrap())
            .map(|(batch, place)| Ok(batch.with_placement(place)?))
            .collect();
        let sink = MemoryGridSink::new(schema);
        sink.write_all(
            Box::pin(futures::stream::iter(placed)),
            &Arc::new(TaskContext::default()),
        )
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
