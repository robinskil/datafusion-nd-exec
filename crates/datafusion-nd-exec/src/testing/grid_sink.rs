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
    grid: Mutex<Option<NdRecordBatch>>,
}

impl MemoryGridSink {
    pub fn new(schema: SchemaRef) -> Self {
        Self {
            schema,
            modes: Vec::new(),
            grid: Mutex::new(None),
        }
    }

    /// Set the accumulator mode of the axis `axis`.
    pub fn with_mode(mut self, axis: impl Into<String>, mode: AxisMode) -> Self {
        self.modes.push((axis.into(), mode));
        self
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
        let mut accumulator = self
            .modes
            .iter()
            .fold(NdGridAccumulator::new(), |acc, (axis, mode)| {
                acc.with_mode(axis.clone(), *mode)
            });
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
    let columns = (0..schema.fields().len())
        .map(|index| build_column(index, &target, placements))
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(NdRecordBatch::try_new(
        schema.clone(),
        columns,
        target,
    )?))
}

/// One output column: each output cell takes the value of the last chunk that
/// writes it.
fn build_column(
    index: usize,
    target: &Dimensions,
    placements: &[Placement],
) -> Result<NdArrowArray> {
    let first = placements[0].batch.column(index);
    let names: Vec<&str> = first.dims().iter().map(|d| d.name()).collect();
    let dims = Dimensions::try_new(
        names
            .iter()
            .map(|name| {
                let axis = target.position(name).ok_or_else(|| {
                    DataFusionError::Execution(format!(
                        "column axis '{name}' is not an output axis"
                    ))
                })?;
                Ok(target.get(axis).clone())
            })
            .collect::<Result<Vec<_>>>()?,
    )?;
    let strides = dims.c_strides();

    // The source of each output cell: (placement, index), or the null source.
    let null_source = placements.len();
    let mut sources = vec![(null_source, 0usize); dims.num_elements()];
    for (k, placement) in placements.iter().enumerate() {
        let column = placement.batch.column(index);
        let chunk = column.dims();
        let chunk_strides = chunk.c_strides();
        for cell in 0..chunk.num_elements() {
            let mut out = 0;
            for (axis, name) in chunk.iter().map(|d| d.name()).enumerate() {
                let coord = (cell / chunk_strides[axis]) % chunk.get(axis).size();
                let offset = placement.axis(name).map_or(0, |a| a.offset);
                let position = names.iter().position(|n| *n == name).expect("same axes");
                out += (offset + coord) * strides[position];
            }
            sources[out] = (k, cell);
        }
    }
    let mut arrays: Vec<ArrayRef> = placements
        .iter()
        .map(|p| p.batch.column(index).values().clone())
        .collect();
    arrays.push(new_null_array(first.values().data_type(), 1));
    let refs: Vec<&dyn Array> = arrays.iter().map(|a| a.as_ref()).collect();
    Ok(NdArrowArray::try_new(interleave(&refs, &sources)?, dims)?)
}
