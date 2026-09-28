//! Error and result types of the crate.

pub use arrow_schema::ArrowError;

/// Result type of the crate.
pub type Result<T> = std::result::Result<T, ArrowError>;

/// Return early with an [`ArrowError::InvalidArgumentError`].
macro_rules! nd_err {
    ($($arg:tt)*) => {
        Err($crate::error::ArrowError::InvalidArgumentError(format!($($arg)*)))
    };
}

pub(crate) use nd_err;
