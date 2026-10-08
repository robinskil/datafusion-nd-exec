//! Round-robin repartition of whole nd batches.

use std::any::Any;
use std::fmt;
use std::sync::{Arc, Mutex};

use datafusion::common::runtime::SpawnedTask;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PlanProperties, SendableRecordBatchStream,
};
use futures::StreamExt;
use futures::channel::mpsc::{UnboundedReceiver, UnboundedSender, unbounded};
use nd_arrow_array::{NdGridAxes, NdRecordBatch, SelectionKind};

use super::{NdExecutionPlan, SendableNdBatchStream, execute_flat, one_child, require_nd_input};
use crate::registry::NdNodeRegistry;

type Item = Result<NdRecordBatch>;

/// The channels and tasks of one execution. The first `execute_nd` builds it.
struct Channels {
    /// One receiver per output partition. An execution takes its receiver.
    receivers: Vec<Option<UnboundedReceiver<Item>>>,
    /// One task per input partition. The output streams keep them alive, and
    /// dropping the last stream aborts them.
    tasks: Arc<Vec<SpawnedTask<()>>>,
}

/// Sends whole nd batches round robin to `partitions` output partitions. Each
/// batch keeps its grid and selection. The broadcast above then runs on all
/// output partitions in parallel.
///
/// The channels are unbounded: a consumer that reads the output partitions in
/// sequence cannot block an input task.
#[derive(Debug, Clone)]
pub struct NdRepartitionExec {
    input: Arc<dyn ExecutionPlan>,
    nd_input: Arc<dyn NdExecutionPlan>,
    registry: Arc<NdNodeRegistry>,
    partitions: usize,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
    channels: Arc<Mutex<Option<Channels>>>,
}

impl fmt::Debug for Channels {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Channels")
            .field("receivers", &self.receivers.len())
            .field("tasks", &self.tasks.len())
            .finish()
    }
}

impl NdRepartitionExec {
    /// The nd child is resolved through `registry`.
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        partitions: usize,
        registry: Arc<NdNodeRegistry>,
    ) -> Result<Self> {
        let nd_input = require_nd_input("NdRepartitionExec", &input, &registry)?;
        if partitions == 0 {
            return Err(DataFusionError::Plan(
                "NdRepartitionExec needs at least one output partition".to_string(),
            ));
        }
        let properties = Arc::new(
            input
                .properties()
                .as_ref()
                .clone()
                .with_eq_properties(EquivalenceProperties::new(input.schema()))
                .with_partitioning(Partitioning::RoundRobinBatch(partitions)),
        );
        Ok(Self {
            input,
            nd_input,
            registry,
            partitions,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
            channels: Arc::new(Mutex::new(None)),
        })
    }

    /// Spawn one task per input partition. Each task sends its batches round
    /// robin to the output channels, and sends an error to every output.
    fn start(&self, context: Arc<TaskContext>) -> Result<Channels> {
        let (senders, receivers): (Vec<UnboundedSender<Item>>, Vec<_>) =
            (0..self.partitions).map(|_| unbounded()).unzip();
        let inputs = self.input.output_partitioning().partition_count();
        let tasks = (0..inputs)
            .map(|input_partition| {
                let mut stream = self.nd_input.execute_nd(input_partition, context.clone())?;
                let senders = senders.clone();
                Ok(SpawnedTask::spawn(async move {
                    // Start each input at its own output, so small inputs spread.
                    let mut next = input_partition % senders.len();
                    while let Some(item) = stream.next().await {
                        match item {
                            Ok(batch) => {
                                // A closed channel means its consumer is gone.
                                let _ = senders[next].unbounded_send(Ok(batch));
                                next = (next + 1) % senders.len();
                            }
                            Err(error) => {
                                let shared = Arc::new(error);
                                for sender in &senders {
                                    let _ = sender.unbounded_send(Err(DataFusionError::Shared(
                                        shared.clone(),
                                    )));
                                }
                                return;
                            }
                        }
                    }
                }))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Channels {
            receivers: receivers.into_iter().map(Some).collect(),
            tasks: Arc::new(tasks),
        })
    }
}

impl DisplayAs for NdRepartitionExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NdRepartitionExec: partitioning=RoundRobinBatch({}), input_partitions={}",
            self.partitions,
            self.input.output_partitioning().partition_count()
        )
    }
}

impl ExecutionPlan for NdRepartitionExec {
    fn name(&self) -> &str {
        "NdRepartitionExec"
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
        let input = one_child("NdRepartitionExec", children)?;
        Ok(Arc::new(Self::try_new(
            input,
            self.partitions,
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

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    /// The nd child must stay a direct child.
    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }
}

impl NdExecutionPlan for NdRepartitionExec {
    fn grid_axes(&self) -> Option<Arc<NdGridAxes>> {
        self.nd_input.grid_axes()
    }

    fn execute_nd(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableNdBatchStream> {
        let mut channels = self.channels.lock().map_err(|_| {
            DataFusionError::Internal("NdRepartitionExec state lock is poisoned".to_string())
        })?;
        // A new execution starts when every output of the last one is taken.
        let spent = channels
            .as_ref()
            .is_none_or(|c| c.receivers.iter().all(Option::is_none));
        if spent {
            *channels = Some(self.start(context)?);
        }
        let channels = channels.as_mut().expect("set above");
        let receiver = channels
            .receivers
            .get_mut(partition)
            .and_then(Option::take)
            .ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "NdRepartitionExec partition {partition} is missing or already executed"
                ))
            })?;
        let tasks = channels.tasks.clone();
        let baseline = BaselineMetrics::new(&self.metrics, partition);
        let stream = receiver.inspect(move |item| {
            // The stream owns the tasks, so the tasks run while it is read.
            let _ = &tasks;
            if let Ok(batch) = item {
                baseline.record_output(batch.num_rows());
            }
        });
        Ok(Box::pin(stream))
    }

    fn max_output_selection(&self) -> SelectionKind {
        self.nd_input.max_output_selection()
    }
}

#[cfg(test)]
mod tests {
    use datafusion::execution::TaskContext;
    use futures::TryStreamExt;

    use super::*;
    use crate::testing::grid_table;

    #[tokio::test]
    async fn every_batch_reaches_one_output_partition() {
        let scan = grid_table()
            .unwrap()
            .nd_scan(None, NdNodeRegistry::shared_default())
            .unwrap();
        let source = scan.children()[0].clone();
        let repartition =
            NdRepartitionExec::try_new(source, 3, NdNodeRegistry::shared_default()).unwrap();
        let context = Arc::new(TaskContext::default());

        // Read the outputs in sequence: the unbounded channels must not block.
        let mut rows = Vec::new();
        for partition in 0..3 {
            let batches: Vec<_> = repartition
                .execute_nd(partition, context.clone())
                .unwrap()
                .try_collect()
                .await
                .unwrap();
            rows.push(batches.iter().map(|b| b.num_rows()).sum::<usize>());
        }
        // Two chunks of 12 cells go to outputs 0 and 1.
        assert_eq!(rows, vec![12, 12, 0]);
    }

    #[tokio::test]
    async fn an_output_partition_runs_once() {
        let scan = grid_table()
            .unwrap()
            .nd_scan(None, NdNodeRegistry::shared_default())
            .unwrap();
        let source = scan.children()[0].clone();
        let repartition =
            NdRepartitionExec::try_new(source, 2, NdNodeRegistry::shared_default()).unwrap();
        let context = Arc::new(TaskContext::default());
        assert!(repartition.execute_nd(0, context.clone()).is_ok());
        assert!(repartition.execute_nd(0, context).is_err());
    }

    #[tokio::test]
    async fn a_plan_runs_again_after_every_output_is_taken() {
        let scan = grid_table()
            .unwrap()
            .nd_scan(None, NdNodeRegistry::shared_default())
            .unwrap();
        let repartition = NdRepartitionExec::try_new(
            scan.children()[0].clone(),
            2,
            NdNodeRegistry::shared_default(),
        )
        .unwrap();
        let context = Arc::new(TaskContext::default());
        for _ in 0..2 {
            let mut rows = 0;
            for partition in 0..2 {
                let batches: Vec<_> = repartition
                    .execute_nd(partition, context.clone())
                    .unwrap()
                    .try_collect()
                    .await
                    .unwrap();
                rows += batches.iter().map(|b| b.num_rows()).sum::<usize>();
            }
            assert_eq!(rows, 24);
        }
    }
}
