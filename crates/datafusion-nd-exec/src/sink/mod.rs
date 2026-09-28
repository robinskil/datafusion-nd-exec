//! Output terminals that keep the grid.
//!
//! A writer of a grid format implements [`NdDataSink`]. It places each nd
//! chunk into one output grid with the [`NdGridAccumulator`], then writes the
//! chunk at its place. A host maps its flat `DataSink` to the nd sink with an
//! [`NdSinkFactory`] in the registry.

mod accumulator;
mod exec;

pub use accumulator::{AxisMode, AxisPlacement, NdGridAccumulator, Placement};
pub use exec::{NdDataSink, NdDataSinkExec, NdSinkFactory};
