//! nd output: encoded chunks instead of flat rows.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::array::{RecordBatch, RecordBatchOptions};
use arrow::datatypes::{Schema, SchemaRef};
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};
use futures::StreamExt;
use nd_arrow_array::NdRecordBatch;
use nd_arrow_array::encoding::{encode_nd_record_batch, encoded_schema};

use crate::exec::{NdBroadcastExec, NdExecutionPlan, require_nd_input};
use crate::registry::NdNodeRegistry;

/// A terminal that yields one `nd.array`-encoded row per nd chunk. Each chunk
/// is compacted first, so the rows carry dense grids. A host sends these rows
/// over Arrow IPC or Flight, and a client decodes each column into its values,
/// axis names and axis sizes.
#[derive(Debug, Clone)]
pub struct NdEncodeExec {
    input: Arc<dyn ExecutionPlan>,
    nd_input: Arc<dyn NdExecutionPlan>,
    registry: Arc<NdNodeRegistry>,
    properties: Arc<PlanProperties>,
}

impl NdEncodeExec {
    pub fn try_new(input: Arc<dyn ExecutionPlan>) -> Result<Self> {
        Self::try_new_with_registry(input, NdNodeRegistry::shared_default())
    }

    /// Like [`try_new`](Self::try_new), but resolves the nd child through
    /// `registry`.
    pub fn try_new_with_registry(
        input: Arc<dyn ExecutionPlan>,
        registry: Arc<NdNodeRegistry>,
    ) -> Result<Self> {
        let nd_input = require_nd_input("NdEncodeExec", &input, &registry)?;
        let schema = Arc::new(encoded_schema(&input.schema()));
        let properties = Arc::new(
            input
                .properties()
                .as_ref()
                .clone()
                .with_eq_properties(EquivalenceProperties::new(schema)),
        );
        Ok(Self {
            input,
            nd_input,
            registry,
            properties,
        })
    }
}

/// One encoded row for `batch`, with the columns of `schema`. A batch with no
/// columns becomes a row carrier with one row per cell, as a `count(*)` scan
/// expects.
fn encode(batch: &NdRecordBatch, schema: &SchemaRef) -> Result<RecordBatch> {
    if batch.columns().is_empty() {
        return Ok(RecordBatch::try_new_with_options(
            Arc::new(Schema::empty()),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
        )?);
    }
    let encoded = encode_nd_record_batch(&batch.compact()?)?;
    Ok(RecordBatch::try_new(
        schema.clone(),
        encoded.columns().to_vec(),
    )?)
}

impl DisplayAs for NdEncodeExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NdEncodeExec")
    }
}

impl ExecutionPlan for NdEncodeExec {
    fn name(&self) -> &str {
        "NdEncodeExec"
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
            DataFusionError::Internal("NdEncodeExec expects exactly one child".to_string())
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
        let schema = self.schema();
        let out = schema.clone();
        let stream = self
            .nd_input
            .execute_nd(partition, context)?
            // A chunk with no rows carries no data.
            .filter(|item| futures::future::ready(!matches!(item, Ok(b) if b.num_rows() == 0)))
            .map(move |item| encode(&item?, &out));
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }

    /// The nd child must stay a direct child.
    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }
}

/// The nd output plan of an optimized plan: when the root is a boundary,
/// an [`NdEncodeExec`] over its nd child replaces it. `None` when flat
/// operators run above the nd region, for example an aggregate.
pub fn nd_output_plan(plan: &Arc<dyn ExecutionPlan>) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let Some(boundary) = plan.as_any().downcast_ref::<NdBroadcastExec>() else {
        return Ok(None);
    };
    Ok(Some(Arc::new(NdEncodeExec::try_new_with_registry(
        boundary.input().clone(),
        boundary.registry().clone(),
    )?)))
}
