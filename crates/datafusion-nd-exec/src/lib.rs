//! Grid-native DataFusion execution for n-dimensional data.
//!
//! The nd operators keep columns un-broadcast through the physical plan. A
//! terminal [`exec::NdBroadcastExec`] materializes them into flat
//! `RecordBatch`es only where a flat operator needs them.

pub mod exec;
pub mod optimizer;
pub mod registry;
pub mod session;
#[cfg(any(test, feature = "test-utils"))]
pub mod testing;

/// The nd array types of the input layer.
pub use nd_arrow_array as array;

pub use optimizer::{NdFilterPushdown, NdProjectionPushdown, is_pushable_expr};
pub use registry::{NdNodeRegistry, NdProbe, NdSinker, Sunk, probe_for};
pub use session::NdSessionStateBuilderExt;

#[cfg(test)]
mod tests;
