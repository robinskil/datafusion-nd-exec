//! A differential test harness for nd plan nodes.
//!
//! Each query runs on the flat path and on the nd path. The results must be
//! the same, and the nd plan must hold the expected nd nodes. Two corpora
//! cover the two data shapes: a grid with coordinate axes and a set of
//! profiles without coordinates. Enable the `test-utils` feature to use the
//! harness from another crate.

mod corpus;
mod differential;
mod grid_sink;
mod table;

pub use corpus::{grid_schema, grid_table, profile_schema, profile_table};
pub use differential::{Differential, sorted_rows};
pub use grid_sink::MemoryGridSink;
pub use table::NdMemTable;
