//! Run one query on the flat path and on the nd path, and compare.

use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use arrow::util::display::{ArrayFormatter, FormatOptions};
use datafusion::error::Result;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_plan::displayable;
use datafusion::prelude::SessionContext;

use super::table::NdMemTable;
use crate::registry::NdNodeRegistry;
use crate::session::NdSessionStateBuilderExt;

/// Two sessions over the same tables. The flat session scans materialized
/// rows with no nd nodes. The nd session scans nd batches with the nd
/// pipeline enabled.
pub struct Differential {
    flat: SessionContext,
    nd: SessionContext,
}

impl Default for Differential {
    fn default() -> Self {
        Self::new()
    }
}

impl Differential {
    /// Sessions with the default registry.
    pub fn new() -> Self {
        Self::with_registry(NdNodeRegistry::shared_default())
    }

    /// Sessions whose nd side uses `registry`.
    pub fn with_registry(registry: Arc<NdNodeRegistry>) -> Self {
        let nd_state = SessionStateBuilder::new()
            .with_default_features()
            .with_nd_pipeline(registry)
            .build();
        Self {
            flat: SessionContext::new(),
            nd: SessionContext::new_with_state(nd_state),
        }
    }

    /// Register `table` as `name` in both sessions.
    pub fn register(&self, name: &str, table: NdMemTable) -> Result<()> {
        self.flat
            .register_table(name, Arc::new(table.to_flat_table()?))?;
        self.nd.register_table(name, Arc::new(table))?;
        Ok(())
    }

    pub fn flat_context(&self) -> &SessionContext {
        &self.flat
    }

    pub fn nd_context(&self) -> &SessionContext {
        &self.nd
    }

    /// Run `sql` on both paths and assert the same rows in any order. Return
    /// the nd result.
    pub async fn assert_same(&self, sql: &str) -> Result<Vec<RecordBatch>> {
        let expected = self.flat.sql(sql).await?.collect().await?;
        let actual = self.nd.sql(sql).await?.collect().await?;
        assert_eq!(
            sorted_rows(&actual)?,
            sorted_rows(&expected)?,
            "the nd path and the flat path disagree for: {sql}\nnd plan:\n{}",
            self.nd_plan(sql).await?
        );
        Ok(actual)
    }

    /// The indented physical plan of `sql` on the nd path.
    pub async fn nd_plan(&self, sql: &str) -> Result<String> {
        let plan = self.nd.sql(sql).await?.create_physical_plan().await?;
        Ok(displayable(plan.as_ref()).indent(true).to_string())
    }

    /// Assert that the nd plan of `sql` holds `nodes`, top down in that order.
    pub async fn assert_plan_nodes(&self, sql: &str, nodes: &[&str]) -> Result<()> {
        let plan = self.nd_plan(sql).await?;
        let names: Vec<&str> = plan
            .lines()
            .map(|line| line.trim_start().split(':').next().unwrap_or(""))
            .collect();
        let mut from = 0;
        for node in nodes {
            match names[from..].iter().position(|name| name == node) {
                Some(at) => from += at + 1,
                None => {
                    panic!("expected {node} in order {nodes:?} in the nd plan of: {sql}\n{plan}")
                }
            }
        }
        Ok(())
    }
}

/// Each row of `batches` formatted as one string, sorted.
pub fn sorted_rows(batches: &[RecordBatch]) -> Result<Vec<String>> {
    let options = FormatOptions::default().with_null("NULL");
    let mut rows = Vec::new();
    for batch in batches {
        let formatters = batch
            .columns()
            .iter()
            .map(|column| ArrayFormatter::try_new(column.as_ref(), &options))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for row in 0..batch.num_rows() {
            let cells: Vec<String> = formatters
                .iter()
                .map(|f| f.value(row).to_string())
                .collect();
            rows.push(cells.join(" | "));
        }
    }
    rows.sort();
    Ok(rows)
}
