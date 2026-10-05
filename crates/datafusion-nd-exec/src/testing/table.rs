//! An in-memory table of nd batches.

use std::any::Any;
use std::sync::Arc;

use arrow::datatypes::{Schema, SchemaRef};
use arrow::record_batch::{RecordBatch, RecordBatchOptions};
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::ExecutionPlan;
use nd_arrow_array::NdRecordBatch;
use nd_arrow_array::encoding::{encode_nd_record_batch, logical_schema};

use crate::exec::{NdBroadcastExec, NdSourceExec};
use crate::registry::NdNodeRegistry;

/// A [`TableProvider`] over in-memory nd batches.
///
/// A scan returns `NdBroadcastExec(NdSourceExec(DataSourceExec))`, the plan
/// shape that an nd file format produces. The scan resolves nd nodes through
/// the registry of the session.
#[derive(Debug, Clone)]
pub struct NdMemTable {
    /// The logical schema. Every field is nullable.
    schema: SchemaRef,
    encoded_schema: SchemaRef,
    partitions: Vec<Vec<NdRecordBatch>>,
    encoded: Vec<Vec<RecordBatch>>,
    ordered_chunks: bool,
}

impl NdMemTable {
    /// A table with one list of nd batches per partition. All batches must
    /// have the same column names and types. The first batch sets the axis
    /// metadata of the encoded schema.
    pub fn try_new(partitions: Vec<Vec<NdRecordBatch>>) -> Result<Self> {
        let first =
            partitions.iter().flatten().next().ok_or_else(|| {
                DataFusionError::Plan("NdMemTable needs at least one batch".into())
            })?;
        let encoded_schema = encode_nd_record_batch(first)?.schema();

        let encoded = partitions
            .iter()
            .map(|batches| {
                batches
                    .iter()
                    .map(|batch| {
                        let encoded = encode_nd_record_batch(batch)?;
                        Ok(RecordBatch::try_new(
                            encoded_schema.clone(),
                            encoded.columns().to_vec(),
                        )?)
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            schema: logical_schema(&encoded_schema)?,
            encoded_schema,
            partitions,
            encoded,
            ordered_chunks: false,
        })
    }

    /// Declare that each partition holds its chunks in the order of the outer
    /// axis, see [`NdSourceExec::with_ordered_chunks`].
    pub fn with_ordered_chunks(mut self) -> Self {
        self.ordered_chunks = true;
        self
    }

    pub fn partitions(&self) -> &[Vec<NdRecordBatch>] {
        &self.partitions
    }
}

#[async_trait]
impl TableProvider for NdMemTable {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.nd_scan(
            projection,
            NdNodeRegistry::from_session_config(state.config()),
        )
    }
}

impl NdMemTable {
    /// The scan plan `NdBroadcastExec(NdSourceExec(DataSourceExec))` of the
    /// columns in `projection`, with nd nodes resolved through `registry`.
    pub fn nd_scan(
        &self,
        projection: Option<&Vec<usize>>,
        registry: Arc<NdNodeRegistry>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let memory = match projection {
            // A zero-column scan (`count(*)`) carries the cell count of each
            // batch as the row count.
            Some(columns) if columns.is_empty() => {
                let empty = Arc::new(Schema::empty());
                let partitions = self
                    .partitions
                    .iter()
                    .map(|batches| {
                        batches
                            .iter()
                            .map(|batch| {
                                Ok(RecordBatch::try_new_with_options(
                                    empty.clone(),
                                    vec![],
                                    &RecordBatchOptions::new()
                                        .with_row_count(Some(batch.num_rows())),
                                )?)
                            })
                            .collect::<Result<Vec<_>>>()
                    })
                    .collect::<Result<Vec<_>>>()?;
                MemorySourceConfig::try_new_exec(&partitions, empty, None)?
            }
            _ => MemorySourceConfig::try_new_exec(
                &self.encoded,
                self.encoded_schema.clone(),
                projection.cloned(),
            )?,
        };
        let source = NdSourceExec::try_new(memory)?;
        let source = Arc::new(if self.ordered_chunks {
            source.with_ordered_chunks()?
        } else {
            source
        });
        Ok(Arc::new(NdBroadcastExec::try_new(source, registry)?))
    }
}
