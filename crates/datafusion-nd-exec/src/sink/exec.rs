//! The sink terminal of the nd pipeline.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::array::{RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::datasource::sink::DataSink;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};

use crate::exec::{NdExecutionPlan, SendableNdBatchStream, merge_partitions, require_nd_input};
use crate::registry::NdNodeRegistry;

/// A writer that takes nd batches, so the grid reaches the output. A grid
/// format implements it with the [`NdGridAccumulator`] and its own writer.
///
/// [`NdGridAccumulator`]: crate::sink::NdGridAccumulator
#[async_trait]
pub trait NdDataSink: DisplayAs + fmt::Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;

    /// The schema of the nd batches.
    fn schema(&self) -> &SchemaRef;

    /// Write all batches of `data` and return the number of rows written.
    async fn write_all(
        &self,
        data: SendableNdBatchStream,
        context: &Arc<TaskContext>,
    ) -> Result<u64>;
}

/// Maps a flat [`DataSink`] of a host to an [`NdDataSink`]. The registry holds
/// the factories. The boundary rule then replaces a `DataSinkExec` over the
/// nd region with an [`NdDataSinkExec`].
pub trait NdSinkFactory: fmt::Debug + Send + Sync {
    /// The nd sink for `sink`, or `None` when this factory does not know it.
    fn nd_sink(&self, sink: &dyn DataSink) -> Option<Arc<dyn NdDataSink>>;
}

/// The schema of the row count that a sink yields.
fn count_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "count",
        DataType::UInt64,
        false,
    )]))
}

/// Write all partitions of an nd input with an [`NdDataSink`]. The node yields
/// one row with the row count, like `DataSinkExec`.
#[derive(Debug, Clone)]
pub struct NdDataSinkExec {
    input: Arc<dyn ExecutionPlan>,
    nd_input: Arc<dyn NdExecutionPlan>,
    registry: Arc<NdNodeRegistry>,
    sink: Arc<dyn NdDataSink>,
    properties: Arc<PlanProperties>,
}

impl NdDataSinkExec {
    pub fn try_new(input: Arc<dyn ExecutionPlan>, sink: Arc<dyn NdDataSink>) -> Result<Self> {
        Self::try_new_with_registry(input, sink, NdNodeRegistry::shared_default())
    }

    /// Like [`try_new`](Self::try_new), but resolves the nd child through
    /// `registry`.
    pub fn try_new_with_registry(
        input: Arc<dyn ExecutionPlan>,
        sink: Arc<dyn NdDataSink>,
        registry: Arc<NdNodeRegistry>,
    ) -> Result<Self> {
        let nd_input = require_nd_input("NdDataSinkExec", &input, &registry)?;
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(count_schema()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Ok(Self {
            input,
            nd_input,
            registry,
            sink,
            properties,
        })
    }

    pub fn sink(&self) -> &Arc<dyn NdDataSink> {
        &self.sink
    }
}

impl DisplayAs for NdDataSinkExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NdDataSinkExec: sink=")?;
        self.sink.fmt_as(t, f)
    }
}

impl ExecutionPlan for NdDataSinkExec {
    fn name(&self) -> &str {
        "NdDataSinkExec"
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
            DataFusionError::Internal("NdDataSinkExec expects exactly one child".to_string())
        })?;
        Ok(Arc::new(Self::try_new_with_registry(
            input,
            self.sink.clone(),
            self.registry.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "NdDataSinkExec has one partition, got partition {partition}"
            )));
        }
        let data = merge_partitions(&self.input, &self.nd_input, context.clone())?;
        let sink = self.sink.clone();
        let count = futures::stream::once(async move {
            let rows = sink.write_all(data, &context).await?;
            Ok(RecordBatch::try_new(
                count_schema(),
                vec![Arc::new(UInt64Array::from(vec![rows]))],
            )?)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            count_schema(),
            count,
        )))
    }

    /// The nd child must stay a direct child.
    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }
}
