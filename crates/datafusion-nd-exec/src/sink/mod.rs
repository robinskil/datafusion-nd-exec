//! Output terminals that keep the grid.
//!
//! A writer of a grid format implements [`NdDataSink`] and sets
//! `requires_grid`. [`NdDataSinkExec`] then reads the input through an
//! [`NdRegridExec`], which gives each batch its place in one output grid. A
//! host maps its flat `DataSink` to the nd sink with an [`NdSinkFactory`] in
//! the registry.

mod encode;
mod exec;
mod regrid_exec;

pub use encode::{NdEncodeExec, nd_output_plan};
pub use exec::{NdDataSink, NdDataSinkExec, NdSinkFactory};
pub use regrid_exec::NdRegridExec;
