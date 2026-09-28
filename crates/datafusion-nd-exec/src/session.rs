//! Session setup for the nd pipeline.

use std::sync::Arc;

use datafusion::execution::SessionStateBuilder;

use crate::boundary::NdBoundaryRule;
use crate::registry::NdNodeRegistry;

/// Enable the nd pipeline on a [`SessionStateBuilder`].
pub trait NdSessionStateBuilderExt {
    /// Store `registry` in the session config and append the
    /// [`NdBoundaryRule`] after the default physical rules.
    ///
    /// Call this after `with_config`, which replaces the session config.
    fn with_nd_pipeline(self, registry: Arc<NdNodeRegistry>) -> Self;
}

impl NdSessionStateBuilderExt for SessionStateBuilder {
    fn with_nd_pipeline(mut self, registry: Arc<NdNodeRegistry>) -> Self {
        let config = self
            .config()
            .take()
            .unwrap_or_default()
            .with_extension(registry.clone());
        // The builder appends these to the default rules. The defaults must
        // stay: without `EnforceDistribution`, a final aggregate does not merge
        // its partitions.
        self.with_config(config)
            .with_physical_optimizer_rule(Arc::new(NdBoundaryRule::new(registry)))
    }
}
