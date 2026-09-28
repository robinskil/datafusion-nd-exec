//! N-dimensional Arrow arrays and record batches with named axes.
//!
//! An [`NdArrowArray`] is a flat Arrow array plus named C-order [`Dimensions`].
//! An [`NdRecordBatch`] holds columns over a shared target grid. Each column
//! can use a subset of the target axes. [`BroadcastMap`] maps a column onto
//! the grid by index arithmetic. The [`encoding`] module carries nd data
//! through plain Arrow operators.

pub mod array;
pub mod batch;
pub mod broadcast;
pub mod dimensions;
pub mod encoding;
pub mod error;

pub use array::NdArrowArray;
pub use batch::NdRecordBatch;
pub use broadcast::BroadcastMap;
pub use dimensions::{Dimension, Dimensions};
pub use error::{ArrowError, Result};
