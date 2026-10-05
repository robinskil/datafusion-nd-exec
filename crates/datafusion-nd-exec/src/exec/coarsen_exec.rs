//! Block-reduce nd chunks along axes.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Float64Array, UInt64Array};
use arrow::compute::{cast, take};
use arrow::datatypes::{DataType, Field, Float64Type, Schema, SchemaRef};
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};
use futures::StreamExt;
use nd_arrow_array::selection::cartesian_sum;
use nd_arrow_array::{Dimension, Dimensions, NdArrowArray, NdRecordBatch, SelectionKind};

use super::{NdExecutionPlan, SendableNdBatchStream, execute_flat, one_child, require_nd_input};
use crate::registry::NdNodeRegistry;

/// How [`NdCoarsenExec`] reduces a block of cells to one value. Nulls do not
/// count, and a block of nulls gives null.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoarsenReduce {
    /// The mean, as `Float64`.
    Mean,
    /// The smallest value, in the column type.
    Min,
    /// The largest value, in the column type.
    Max,
    /// The value of the first cell of the block.
    First,
}

/// Reduce each block of `factor` cells along each named axis to one cell, for
/// example to make a map tile from a large grid.
///
/// - Blocks stay inside a chunk. A chunk size that is not a multiple of the
///   factor gives a short block at the chunk end, which reduces on its own.
/// - With [`CoarsenReduce::Mean`], every numeric column becomes `Float64`.
///   A non-numeric column takes the first value of each block with any
///   reduction.
/// - The chunks are compacted first, so the output selection is `Full`.
#[derive(Debug, Clone)]
pub struct NdCoarsenExec {
    input: Arc<dyn ExecutionPlan>,
    nd_input: Arc<dyn NdExecutionPlan>,
    registry: Arc<NdNodeRegistry>,
    factors: Vec<(String, usize)>,
    reduce: CoarsenReduce,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl NdCoarsenExec {
    /// The nd child is resolved through `registry`.
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        factors: Vec<(String, usize)>,
        reduce: CoarsenReduce,
        registry: Arc<NdNodeRegistry>,
    ) -> Result<Self> {
        let nd_input = require_nd_input("NdCoarsenExec", &input, &registry)?;
        if let Some((axis, _)) = factors.iter().find(|(_, factor)| *factor == 0) {
            return Err(DataFusionError::Plan(format!(
                "NdCoarsenExec needs a factor of 1 or more for axis '{axis}'"
            )));
        }
        let input_schema = input.schema();
        let schema = Arc::new(Schema::new_with_metadata(
            input_schema
                .fields()
                .iter()
                .map(|field| {
                    Field::new(field.name(), output_type(field.data_type(), reduce), true)
                        .with_metadata(field.metadata().clone())
                })
                .collect::<Vec<_>>(),
            input_schema.metadata().clone(),
        ));
        let properties = Arc::new(
            input
                .properties()
                .as_ref()
                .clone()
                .with_eq_properties(EquivalenceProperties::new(schema.clone())),
        );
        Ok(Self {
            input,
            nd_input,
            registry,
            factors,
            reduce,
            schema,
            properties,
        })
    }

    fn factor(&self, axis: &str) -> usize {
        self.factors
            .iter()
            .find(|(name, _)| name == axis)
            .map_or(1, |(_, factor)| *factor)
    }

    fn coarsen(&self, batch: &NdRecordBatch) -> Result<NdRecordBatch> {
        let batch = batch.compact()?;
        let old = batch.target();
        let target = Dimensions::try_new(
            old.iter()
                .map(|dim| {
                    Dimension::new(dim.name(), dim.size().div_ceil(self.factor(dim.name())))
                        .with_meta(dim.meta().cloned())
                })
                .collect(),
        )?;
        let columns = batch
            .columns()
            .iter()
            .zip(self.schema.fields())
            .map(|(column, field)| self.coarsen_column(column, old, field.data_type()))
            .collect::<Result<Vec<_>>>()?;
        Ok(NdRecordBatch::try_new(
            self.schema.clone(),
            columns,
            target,
        )?)
    }

    fn coarsen_column(
        &self,
        column: &NdArrowArray,
        target: &Dimensions,
        output: &DataType,
    ) -> Result<NdArrowArray> {
        let dims = column.dims();
        // An axis of size 1 broadcasts, so only full axes coarsen.
        let factors: Vec<usize> = dims
            .iter()
            .map(|dim| match target.position(dim.name()) {
                Some(axis) if target.get(axis).size() == dim.size() => self.factor(dim.name()),
                _ => 1,
            })
            .collect();
        if factors.iter().all(|&f| f == 1) {
            let values = cast(column.values(), output)?;
            return Ok(NdArrowArray::try_new(values, dims.clone())?);
        }
        let new_dims = Dimensions::try_new(
            dims.iter()
                .zip(&factors)
                .map(|(dim, &f)| {
                    Dimension::new(dim.name(), dim.size().div_ceil(f))
                        .with_meta(dim.meta().cloned())
                })
                .collect(),
        )?;
        let blocks = blocks(dims, &new_dims, &factors);
        let values = column.values();
        let numeric = values.data_type().is_numeric();
        let reduced = match self.reduce {
            CoarsenReduce::First => first(values, &blocks)?,
            _ if !numeric => first(values, &blocks)?,
            reduce => {
                let floats = cast(values, &DataType::Float64)?;
                let floats = floats.as_primitive::<Float64Type>();
                let out: Float64Array = blocks
                    .iter()
                    .map(|block| fold(floats, block, reduce))
                    .collect();
                cast(&(Arc::new(out) as ArrayRef), output)?
            }
        };
        Ok(NdArrowArray::try_new(reduced, new_dims)?)
    }
}

/// The type of a column after a reduction.
fn output_type(input: &DataType, reduce: CoarsenReduce) -> DataType {
    if reduce == CoarsenReduce::Mean && input.is_numeric() {
        DataType::Float64
    } else {
        input.clone()
    }
}

/// The source cells of each output cell, in row-major output order.
fn blocks(dims: &Dimensions, new_dims: &Dimensions, factors: &[usize]) -> Vec<Vec<u64>> {
    let strides = dims.c_strides();
    (0..new_dims.num_elements())
        .map(|out| {
            let mut rem = out;
            let mut ranges = vec![Vec::new(); dims.rank()];
            for axis in (0..dims.rank()).rev() {
                let size = new_dims.get(axis).size();
                let coord = rem % size;
                rem /= size;
                let start = coord * factors[axis];
                let end = (start + factors[axis]).min(dims.get(axis).size());
                ranges[axis] = (start..end).map(|c| (c * strides[axis]) as u64).collect();
            }
            cartesian_sum(&ranges)
        })
        .collect()
}

fn first(values: &ArrayRef, blocks: &[Vec<u64>]) -> Result<ArrayRef> {
    let indices: UInt64Array = blocks.iter().map(|block| block[0]).collect();
    Ok(take(values.as_ref(), &indices, None)?)
}

fn fold(values: &Float64Array, block: &[u64], reduce: CoarsenReduce) -> Option<f64> {
    let valid = block
        .iter()
        .map(|&i| i as usize)
        .filter(|&i| values.is_valid(i))
        .map(|i| values.value(i));
    match reduce {
        CoarsenReduce::Mean => {
            let (sum, count) = valid.fold((0.0, 0usize), |(s, c), v| (s + v, c + 1));
            (count > 0).then(|| sum / count as f64)
        }
        CoarsenReduce::Min => valid.reduce(f64::min),
        CoarsenReduce::Max => valid.reduce(f64::max),
        CoarsenReduce::First => unreachable!("First takes the first cell"),
    }
}

impl DisplayAs for NdCoarsenExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let factors: Vec<String> = self
            .factors
            .iter()
            .map(|(axis, factor)| format!("{axis}={factor}"))
            .collect();
        write!(
            f,
            "NdCoarsenExec: factors=[{}], reduce={:?}",
            factors.join(", "),
            self.reduce
        )
    }
}

impl ExecutionPlan for NdCoarsenExec {
    fn name(&self) -> &str {
        "NdCoarsenExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let input = one_child("NdCoarsenExec", children)?;
        Ok(Arc::new(Self::try_new(
            input,
            self.factors.clone(),
            self.reduce,
            self.registry.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        execute_flat(self, &self.registry, partition, context)
    }

    /// The nd child must stay a direct child.
    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }
}

impl NdExecutionPlan for NdCoarsenExec {
    fn execute_nd(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableNdBatchStream> {
        let this = self.clone();
        let stream = self
            .nd_input
            .execute_nd(partition, context)?
            .map(move |item| this.coarsen(&item?));
        Ok(Box::pin(stream))
    }

    fn max_output_selection(&self) -> SelectionKind {
        SelectionKind::Full
    }
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::Int64Type;
    use futures::TryStreamExt;

    use super::*;
    use crate::testing::grid_table;

    async fn coarsened(factors: Vec<(&str, usize)>, reduce: CoarsenReduce) -> Vec<NdRecordBatch> {
        let scan = grid_table()
            .unwrap()
            .nd_scan(None, NdNodeRegistry::shared_default())
            .unwrap();
        let factors = factors
            .into_iter()
            .map(|(axis, f)| (axis.to_string(), f))
            .collect();
        NdCoarsenExec::try_new(
            scan.children()[0].clone(),
            factors,
            reduce,
            NdNodeRegistry::shared_default(),
        )
        .unwrap()
        .execute_nd(0, Arc::new(TaskContext::default()))
        .unwrap()
        .try_collect()
        .await
        .unwrap()
    }

    fn column<'a>(batch: &'a NdRecordBatch, name: &str) -> &'a NdArrowArray {
        batch.column(batch.schema().index_of(name).unwrap())
    }

    #[tokio::test]
    async fn a_mean_reduces_each_block() {
        let batches = coarsened(vec![("lat", 2)], CoarsenReduce::Mean).await;
        let batch = &batches[0];
        let sizes: Vec<usize> = batch.target().iter().map(|d| d.size()).collect();
        // lat 3 in blocks of 2 gives 2 cells: a full block and a short one.
        assert_eq!(sizes, [2, 2, 2]);
        let lat = column(batch, "lat").values().as_primitive::<Float64Type>();
        assert_eq!(lat.values(), &[-15.0, 30.0]);
        // elev{lat, lon} = [-100, -50, 0, 50, 100, 150].
        let elev = column(batch, "elev").values().as_primitive::<Float64Type>();
        assert_eq!(elev.values(), &[-50.0, 0.0, 100.0, 150.0]);
        // `time` does not touch `lat`, but a mean makes every numeric column Float64.
        assert_eq!(
            column(batch, "time").values().data_type(),
            &DataType::Float64
        );
        // A string column takes the first value.
        assert_eq!(column(batch, "source").values().len(), 1);
    }

    #[tokio::test]
    async fn min_and_max_keep_the_type() {
        let min = coarsened(vec![("time", 2)], CoarsenReduce::Min).await;
        let time = column(&min[0], "time").values().as_primitive::<Int64Type>();
        assert_eq!(time.values(), &[100]);
        let max = coarsened(vec![("time", 2)], CoarsenReduce::Max).await;
        let time = column(&max[0], "time").values().as_primitive::<Int64Type>();
        assert_eq!(time.values(), &[101]);
    }

    #[tokio::test]
    async fn nulls_do_not_count() {
        // sst is null at every seventh cell: cell 3 of chunk 0 is null.
        let batches = coarsened(
            vec![("time", 2), ("lat", 3), ("lon", 2)],
            CoarsenReduce::Mean,
        )
        .await;
        let sst = column(&batches[0], "sst")
            .values()
            .as_primitive::<Float64Type>();
        let valid: Vec<f64> = (0..12)
            .filter(|i| i % 7 != 3)
            .map(|i| i as f64 * 0.5)
            .collect();
        let mean = valid.iter().sum::<f64>() / valid.len() as f64;
        assert_eq!(sst.values(), &[mean]);
    }

    #[tokio::test]
    async fn a_factor_of_one_keeps_the_grid() {
        let batches = coarsened(vec![("lat", 1)], CoarsenReduce::First).await;
        assert_eq!(batches[0].num_rows(), 12);
    }

    #[test]
    fn a_factor_of_zero_is_rejected() {
        let scan = grid_table()
            .unwrap()
            .nd_scan(None, NdNodeRegistry::shared_default())
            .unwrap();
        let result = NdCoarsenExec::try_new(
            scan.children()[0].clone(),
            vec![("lat".to_string(), 0)],
            CoarsenReduce::Mean,
            NdNodeRegistry::shared_default(),
        );
        assert!(result.is_err());
    }
}
