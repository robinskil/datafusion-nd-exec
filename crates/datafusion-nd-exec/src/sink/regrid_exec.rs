//! Collect all nd batches, regrid them, and restream each batch with its place.

use std::any::Any;
use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use arrow::array::RecordBatchOptions;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::execution::disk_manager::RefCountedTempFile;
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{
    ExecutionPlanMetricsSet, MetricBuilder, MetricsSet, SpillMetrics,
};
use datafusion::physical_plan::spill::SpillManager;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PlanProperties, SendableRecordBatchStream,
};
use futures::{StreamExt, TryStreamExt};
use nd_arrow_array::encoding::{
    decode_nd_record_batch_row, encode_nd_record_batch, encoded_schema, nd_batch_count,
};
use nd_arrow_array::grid::axis_origins;
use nd_arrow_array::{
    Dimensions, NdArrowArray, NdBatchRecord, NdGridBuilder, NdOutputGrid, NdPlacement,
    NdRecordBatch, SelectionKind,
};

use crate::exec::{NdBroadcastExec, NdExecutionPlan, SendableNdBatchStream, require_nd_input};
use crate::registry::NdNodeRegistry;

/// Collects all nd batches of its input, builds the output grids, and sends
/// each batch again with its [`NdPlacement`]. A grid sink reads through it.
///
/// The batches stay in memory while the memory pool allows it, and spill to
/// IPC files when it does not. The node has one output partition.
#[derive(Debug, Clone)]
pub struct NdRegridExec {
    input: Arc<dyn ExecutionPlan>,
    nd_input: Arc<dyn NdExecutionPlan>,
    registry: Arc<NdNodeRegistry>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl NdRegridExec {
    pub fn try_new(input: Arc<dyn ExecutionPlan>) -> Result<Self> {
        Self::try_new_with_registry(input, NdNodeRegistry::shared_default())
    }

    /// Like [`try_new`](Self::try_new), but resolves the nd child through
    /// `registry`.
    pub fn try_new_with_registry(
        input: Arc<dyn ExecutionPlan>,
        registry: Arc<NdNodeRegistry>,
    ) -> Result<Self> {
        let nd_input = require_nd_input("NdRegridExec", &input, &registry)?;
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(input.schema()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Ok(Self {
            input,
            nd_input,
            registry,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }
}

/// The batches of one execution, in memory or in spill files.
struct Collector {
    schema: SchemaRef,
    spill_schema: SchemaRef,
    builder: NdGridBuilder,
    /// The grid of each batch, by batch number.
    targets: Vec<Dimensions>,
    /// The batches in memory with their numbers, in collection order.
    held: Vec<(usize, NdRecordBatch)>,
    held_bytes: usize,
    /// Each spill file with the numbers of its batches, in collection order.
    spills: Vec<(RefCountedTempFile, Vec<usize>)>,
    /// The next batch number of each input partition.
    next: Vec<usize>,
    spill: SpillManager,
    reservation: MemoryReservation,
}

impl Collector {
    fn new(
        schema: SchemaRef,
        partitions: usize,
        metrics: &ExecutionPlanMetricsSet,
        context: &TaskContext,
    ) -> Self {
        let spill_schema = Arc::new(encoded_schema(&schema));
        let spill = SpillManager::new(
            context.runtime_env(),
            SpillMetrics::new(metrics, 0),
            spill_schema.clone(),
        );
        let reservation = MemoryConsumer::new("NdRegridExec")
            .with_can_spill(true)
            .register(context.memory_pool());
        Self {
            schema,
            spill_schema,
            builder: NdGridBuilder::new(),
            targets: Vec::new(),
            held: Vec::new(),
            held_bytes: 0,
            spills: Vec::new(),
            next: vec![0; partitions],
            spill,
            reservation,
        }
    }

    fn push(&mut self, partition: usize, batch: NdRecordBatch) -> Result<()> {
        let number = self.next[partition];
        self.next[partition] += 1;
        let origins = axis_origins(batch.selection(), batch.target().rank());
        let batch = batch.compact()?;
        if batch.target().num_elements() == 0 {
            return Ok(());
        }
        let record = NdBatchRecord::of(partition, number, &batch).with_origins(origins)?;
        // The records stay in memory until the regrid, whatever the pool says.
        self.reservation.grow(record.memory_size());
        let id = self.builder.add(record);
        self.targets.push(batch.target().clone());

        let size: usize = batch
            .columns()
            .iter()
            .map(|c| c.values().get_array_memory_size())
            .sum();
        self.held.push((id, batch));
        match self.reservation.try_grow(size) {
            Ok(()) => self.held_bytes += size,
            Err(_) => self.spill_held()?,
        }
        Ok(())
    }

    /// Write all held batches to one spill file and free their memory.
    fn spill_held(&mut self) -> Result<()> {
        let batches = self
            .held
            .iter()
            .map(|(_, batch)| self.encode(batch))
            .collect::<Result<Vec<_>>>()?;
        let file = self
            .spill
            .spill_record_batch_and_finish(&batches, "NdRegridExec")
            .map_err(|e| e.context("NdRegridExec cannot spill its batches"))?;
        let ids = self.held.drain(..).map(|(id, _)| id).collect();
        if let Some(file) = file {
            self.spills.push((file, ids));
        }
        self.reservation.shrink(self.held_bytes);
        self.held_bytes = 0;
        Ok(())
    }

    /// The encoded batch on the spill schema. The record keeps the axis metadata.
    fn encode(&self, batch: &NdRecordBatch) -> Result<RecordBatch> {
        let encoded = encode_nd_record_batch(batch)?;
        let options = RecordBatchOptions::new().with_row_count(Some(1));
        Ok(RecordBatch::try_new_with_options(
            self.spill_schema.clone(),
            encoded.columns().to_vec(),
            &options,
        )?)
    }

    /// Build the grids, then stream the spilled batches and the held batches
    /// with their places.
    fn finish(self, metrics: &ExecutionPlanMetricsSet) -> Result<SendableNdBatchStream> {
        let Collector {
            schema,
            builder,
            targets,
            held,
            spills,
            spill,
            reservation,
            ..
        } = self;
        let placements = builder.finish()?;

        // Count each grid once: its batches share one `Arc`.
        let mut seen: HashSet<*const NdOutputGrid> = HashSet::new();
        let cells: usize = placements
            .iter()
            .filter(|p| seen.insert(Arc::as_ptr(p.grid())))
            .map(|p| p.grid().num_cells())
            .sum();
        MetricBuilder::new(metrics)
            .global_counter("grids")
            .add(seen.len());
        MetricBuilder::new(metrics)
            .global_counter("output_cells")
            .add(cells);

        let place = Arc::new(Place {
            schema,
            targets,
            placements,
        });
        let mut parts: Vec<SendableNdBatchStream> = Vec::new();
        for (file, ids) in spills {
            let encoded = spill.read_spill_as_stream(file, None)?;
            let place = place.clone();
            let mut ids = ids.into_iter();
            let decoded = encoded
                .map(move |encoded| -> Result<Vec<Result<NdRecordBatch>>> {
                    let encoded = encoded?;
                    (0..nd_batch_count(&encoded))
                        .map(|row| {
                            let id = ids.next().ok_or_else(|| {
                                DataFusionError::Internal(
                                    "a spill file holds more batches than records".to_string(),
                                )
                            })?;
                            Ok(place.restore(id, decode_nd_record_batch_row(&encoded, row)?))
                        })
                        .collect()
                })
                .map_ok(futures::stream::iter)
                .try_flatten();
            parts.push(Box::pin(decoded));
        }
        let place_held = place.clone();
        parts.push(Box::pin(futures::stream::iter(
            held.into_iter()
                .map(move |(id, batch)| place_held.attach(id, batch)),
        )));
        let stream = futures::stream::iter(parts).flatten().inspect(move |_| {
            // The stream owns the reservation until the last batch.
            let _ = &reservation;
        });
        Ok(Box::pin(stream))
    }
}

/// The places of all batches, and the grid of each batch for the restore.
struct Place {
    schema: SchemaRef,
    targets: Vec<Dimensions>,
    placements: Vec<Arc<NdPlacement>>,
}

impl Place {
    fn attach(&self, id: usize, batch: NdRecordBatch) -> Result<NdRecordBatch> {
        Ok(batch.with_placement(self.placements[id].clone())?)
    }

    /// Rebuild a batch from a spill file on its own grid, with its axis metadata.
    fn restore(&self, id: usize, decoded: NdRecordBatch) -> Result<NdRecordBatch> {
        let target = &self.targets[id];
        let meta = |name: &str| {
            target
                .position(name)
                .and_then(|axis| target.get(axis).meta().cloned())
        };
        let columns = decoded
            .columns()
            .iter()
            .map(|c| NdArrowArray::try_new(c.values().clone(), c.dims().with_axis_meta(meta)))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let batch = NdRecordBatch::try_new(self.schema.clone(), columns, target.clone())?;
        self.attach(id, batch)
    }
}

impl DisplayAs for NdRegridExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NdRegridExec")
    }
}

impl ExecutionPlan for NdRegridExec {
    fn name(&self) -> &str {
        "NdRegridExec"
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
        let [input] = <[_; 1]>::try_from(children).map_err(|_| {
            DataFusionError::Internal("NdRegridExec expects exactly one child".to_string())
        })?;
        Ok(Arc::new(Self::try_new_with_registry(
            input,
            self.registry.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        NdBroadcastExec::try_new_with_registry(Arc::new(self.clone()), self.registry.clone())?
            .execute(partition, context)
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    /// The nd child must stay a direct child.
    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }
}

impl NdExecutionPlan for NdRegridExec {
    fn execute_nd(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableNdBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "NdRegridExec has one partition, got partition {partition}"
            )));
        }
        let partitions = self.input.output_partitioning().partition_count();
        let inputs = (0..partitions)
            .map(|p| {
                let stream = self.nd_input.execute_nd(p, context.clone())?;
                Ok(stream.map_ok(move |batch| (p, batch)).boxed())
            })
            .collect::<Result<Vec<_>>>()?;
        let mut collector = Collector::new(self.schema(), partitions, &self.metrics, &context);
        let metrics = self.metrics.clone();
        let stream = futures::stream::once(async move {
            let mut merged = futures::stream::select_all(inputs);
            while let Some(item) = merged.next().await {
                let (partition, batch) = item?;
                collector.push(partition, batch)?;
            }
            collector.finish(&metrics)
        })
        .try_flatten();
        Ok(Box::pin(stream))
    }

    fn max_output_selection(&self) -> SelectionKind {
        SelectionKind::Full
    }
}

#[cfg(test)]
mod tests {
    use datafusion::execution::memory_pool::FairSpillPool;
    use datafusion::execution::runtime_env::RuntimeEnvBuilder;
    use futures::TryStreamExt;

    use super::*;
    use crate::testing::{grid_table, sorted_rows};

    fn source() -> Arc<dyn ExecutionPlan> {
        let scan = grid_table()
            .unwrap()
            .nd_scan(None, NdNodeRegistry::shared_default())
            .unwrap();
        scan.children()[0].clone()
    }

    async fn regrid(context: Arc<TaskContext>) -> (NdRegridExec, Vec<NdRecordBatch>) {
        let exec = NdRegridExec::try_new(source()).unwrap();
        let batches = exec
            .execute_nd(0, context)
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        (exec, batches)
    }

    fn time_positions(batches: &[NdRecordBatch]) -> Vec<Vec<u64>> {
        let mut positions: Vec<Vec<u64>> = batches
            .iter()
            .map(|b| {
                b.placement()
                    .unwrap()
                    .axis_indices("time")
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        positions.sort();
        positions
    }

    fn rows(batches: &[NdRecordBatch]) -> Vec<String> {
        let flat: Vec<_> = batches.iter().map(|b| b.materialize().unwrap()).collect();
        sorted_rows(&flat).unwrap()
    }

    #[tokio::test]
    async fn every_batch_gets_its_place() {
        let (exec, batches) = regrid(Arc::new(TaskContext::default())).await;
        assert_eq!(batches.len(), 2);
        let grid = batches[0].placement().unwrap().grid().clone();
        assert!(Arc::ptr_eq(&grid, batches[1].placement().unwrap().grid()));
        let shape: Vec<(&str, usize)> = grid.dims().iter().map(|d| (d.name(), d.size())).collect();
        assert_eq!(shape, [("time", 4), ("lat", 3), ("lon", 2)]);
        assert_eq!(time_positions(&batches), [vec![0, 1], vec![2, 3]]);
        let metrics = exec.metrics().unwrap();
        assert_eq!(metrics.sum_by_name("grids").unwrap().as_usize(), 1);
        assert_eq!(metrics.sum_by_name("output_cells").unwrap().as_usize(), 24);
    }

    #[tokio::test]
    async fn a_small_pool_spills_and_keeps_the_output() {
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::new(FairSpillPool::new(1)))
            .build_arc()
            .unwrap();
        let context = Arc::new(TaskContext::default().with_runtime(runtime));
        let (exec, spilled) = regrid(context).await;
        let (_, held) = regrid(Arc::new(TaskContext::default())).await;

        assert!(exec.metrics().unwrap().spill_count().unwrap() > 0);
        assert_eq!(rows(&spilled), rows(&held));
        assert_eq!(time_positions(&spilled), time_positions(&held));
        // The axis metadata comes back from the record of each batch.
        assert!(spilled.iter().all(|b| b.target() == held[0].target()));
        let axes = |b: &NdRecordBatch| {
            b.columns()
                .iter()
                .map(|c| c.dims().clone())
                .collect::<Vec<_>>()
        };
        assert!(spilled.iter().all(|b| axes(b) == axes(&held[0])));
    }

    #[tokio::test]
    async fn the_node_has_one_partition() {
        let exec = NdRegridExec::try_new(source()).unwrap();
        assert_eq!(exec.properties().partitioning.partition_count(), 1);
        assert!(
            exec.execute_nd(1, Arc::new(TaskContext::default()))
                .is_err()
        );
    }

    #[tokio::test]
    async fn a_scan_without_an_inner_coordinate_fails() {
        // The projection keeps time, lat and sst, but not lon.
        let scan = grid_table()
            .unwrap()
            .nd_scan(Some(&vec![0, 1, 3]), NdNodeRegistry::shared_default())
            .unwrap();
        let exec = NdRegridExec::try_new(scan.children()[0].clone()).unwrap();
        let stream = exec
            .execute_nd(0, Arc::new(TaskContext::default()))
            .unwrap();
        let error = stream
            .try_collect::<Vec<_>>()
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("select the column 'lon'"), "{error}");
    }
}
