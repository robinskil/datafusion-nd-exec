//! The registry of nd plan nodes.
//!
//! DataFusion passes plan children as `Arc<dyn ExecutionPlan>`. An nd operator
//! needs the [`NdExecutionPlan`] side of its child, so it asks the registry. A
//! probe recognizes the node types of one crate. Each crate registers its
//! probes and sinkers at start time, so a crate can add nd nodes without a
//! change to this crate.

use std::fmt;
use std::sync::{Arc, OnceLock};

use datafusion::error::Result;
use datafusion::execution::config::SessionConfig;
use datafusion::physical_plan::ExecutionPlan;

use crate::exec::{NdExecutionPlan, NdFilterExec, NdProjectionExec, NdSourceExec, NdUnionExec};
use crate::sinkers::{FilterSinker, ProjectionSinker, UnionSinker};

/// Recognizes the nd node types of one crate. Returns `None` for any other
/// node.
pub type NdProbe =
    Arc<dyn Fn(&Arc<dyn ExecutionPlan>) -> Option<Arc<dyn NdExecutionPlan>> + Send + Sync>;

/// A probe that recognizes the node type `T`.
pub fn probe_for<T>() -> NdProbe
where
    T: NdExecutionPlan + Clone + 'static,
{
    Arc::new(|plan: &Arc<dyn ExecutionPlan>| {
        plan.as_any()
            .downcast_ref::<T>()
            .map(|node| Arc::new(node.clone()) as Arc<dyn NdExecutionPlan>)
    })
}

/// The result of an [`NdSinker`]: an nd node that replaces a flat node above
/// the boundary.
pub struct Sunk {
    /// The nd node that goes below the boundary, over the old nd children.
    pub nd: Arc<dyn NdExecutionPlan>,
    /// A flat node that stays above the boundary, or `None`. The boundary
    /// rule replaces its one child with the new boundary.
    pub residual: Option<Arc<dyn ExecutionPlan>>,
}

/// The sink check of one flat node kind: can the node move below the nd
/// boundary?
pub trait NdSinker: Send + Sync + fmt::Debug {
    /// Return the nd replacement of `parent` over `children`, or `None` when
    /// `parent` is not a node kind of this sinker or cannot operate on grids.
    /// `children` holds the nd child of the boundary under each child of
    /// `parent`, in order. Build the replacement with `registry`.
    fn try_sink(
        &self,
        parent: &Arc<dyn ExecutionPlan>,
        children: &[Arc<dyn ExecutionPlan>],
        registry: &Arc<NdNodeRegistry>,
    ) -> Result<Option<Sunk>>;
}

/// The probes and sinkers of all nd node types in a session.
#[derive(Clone, Default)]
pub struct NdNodeRegistry {
    probes: Vec<NdProbe>,
    sinkers: Vec<Arc<dyn NdSinker>>,
}

impl NdNodeRegistry {
    /// A registry with the built-in nd nodes and sinkers of this crate.
    pub fn new() -> Self {
        Self::empty()
            .with_probe(probe_for::<NdSourceExec>())
            .with_probe(probe_for::<NdProjectionExec>())
            .with_probe(probe_for::<NdFilterExec>())
            .with_probe(probe_for::<NdUnionExec>())
            .with_sinker(Arc::new(FilterSinker))
            .with_sinker(Arc::new(ProjectionSinker))
            .with_sinker(Arc::new(UnionSinker))
    }

    /// A registry with no probes and no sinkers.
    pub fn empty() -> Self {
        Self::default()
    }

    /// The shared registry with the built-in nd nodes.
    pub fn shared_default() -> Arc<Self> {
        static DEFAULT: OnceLock<Arc<NdNodeRegistry>> = OnceLock::new();
        DEFAULT.get_or_init(|| Arc::new(Self::new())).clone()
    }

    /// The registry of a session, or [`NdNodeRegistry::shared_default`] when
    /// the session has none.
    pub fn from_session_config(config: &SessionConfig) -> Arc<Self> {
        config
            .get_extension::<Self>()
            .unwrap_or_else(Self::shared_default)
    }

    pub fn with_probe(mut self, probe: NdProbe) -> Self {
        self.register_probe(probe);
        self
    }

    pub fn register_probe(&mut self, probe: NdProbe) {
        self.probes.push(probe);
    }

    pub fn with_sinker(mut self, sinker: Arc<dyn NdSinker>) -> Self {
        self.register_sinker(sinker);
        self
    }

    pub fn register_sinker(&mut self, sinker: Arc<dyn NdSinker>) {
        self.sinkers.push(sinker);
    }

    pub fn sinkers(&self) -> &[Arc<dyn NdSinker>] {
        &self.sinkers
    }

    /// Recover the nd side of a plan node, or `None` when no probe knows it.
    pub fn as_nd_plan(&self, plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn NdExecutionPlan>> {
        self.probes.iter().find_map(|probe| probe(plan))
    }
}

impl fmt::Debug for NdNodeRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NdNodeRegistry")
            .field("probes", &self.probes.len())
            .field("sinkers", &self.sinkers)
            .finish()
    }
}
