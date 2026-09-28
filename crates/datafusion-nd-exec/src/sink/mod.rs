//! Output terminals that keep the grid.
//!
//! A writer of a grid format places each nd chunk into one output grid with
//! the [`NdGridAccumulator`], then writes the chunk at its place.

mod accumulator;

pub use accumulator::{AxisMode, AxisPlacement, NdGridAccumulator, Placement};
