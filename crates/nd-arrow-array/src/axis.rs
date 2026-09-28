//! Coordinate and order metadata of an axis.
//!
//! A grid axis such as `time` usually has a coordinate variable of the same
//! name, and its values rise or fall. A profile axis such as `N_PROF` has no
//! coordinate and no order. Readers detect the metadata at scan time. Plans
//! use it to report sort order and to align grids.

use std::sync::Arc;

use arrow::array::{Array, BooleanArray};
use arrow::compute::kernels::cmp::{gt_eq, lt_eq};
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// The monotone direction of an axis coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AxisOrder {
    /// The coordinate never falls along the axis.
    Ascending,
    /// The coordinate never rises along the axis.
    Descending,
    /// The axis has no coordinate, or the coordinate is not monotone.
    #[default]
    Unordered,
}

impl AxisOrder {
    /// Detect the order of a 1-D coordinate array.
    ///
    /// A coordinate with a null value is [`AxisOrder::Unordered`]. A coordinate
    /// with fewer than two values, or with all values equal, is
    /// [`AxisOrder::Ascending`].
    pub fn detect(values: &dyn Array) -> Result<Self> {
        if values.null_count() > 0 {
            return Ok(Self::Unordered);
        }
        if values.len() < 2 {
            return Ok(Self::Ascending);
        }
        let head = values.slice(0, values.len() - 1);
        let tail = values.slice(1, values.len() - 1);
        if all_true(&lt_eq(&head, &tail)?) {
            return Ok(Self::Ascending);
        }
        if all_true(&gt_eq(&head, &tail)?) {
            return Ok(Self::Descending);
        }
        Ok(Self::Unordered)
    }
}

fn all_true(mask: &BooleanArray) -> bool {
    mask.null_count() == 0 && mask.true_count() == mask.len()
}

/// Optional metadata of one axis.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct AxisMeta {
    coordinate: Option<Arc<str>>,
    order: AxisOrder,
}

impl AxisMeta {
    pub fn new(coordinate: Option<Arc<str>>, order: AxisOrder) -> Self {
        Self { coordinate, order }
    }

    /// Metadata of an axis whose coordinate column is `coordinate`.
    pub fn coordinate(coordinate: impl Into<Arc<str>>, order: AxisOrder) -> Self {
        Self::new(Some(coordinate.into()), order)
    }

    /// Metadata of an axis with no coordinate, such as a profile axis.
    pub fn no_coordinate() -> Self {
        Self::default()
    }

    /// The name of the coordinate column, if the axis has one.
    pub fn coordinate_column(&self) -> Option<&str> {
        self.coordinate.as_deref()
    }

    pub fn order(&self) -> AxisOrder {
        self.order
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{Float64Array, Int32Array, StringArray};

    use super::*;

    #[test]
    fn rise_is_ascending() {
        let values = Float64Array::from(vec![1.0, 2.0, 2.0, 5.0]);
        assert_eq!(AxisOrder::detect(&values).unwrap(), AxisOrder::Ascending);
    }

    #[test]
    fn fall_is_descending() {
        let values = Int32Array::from(vec![90, 45, 0, -45]);
        assert_eq!(AxisOrder::detect(&values).unwrap(), AxisOrder::Descending);
    }

    #[test]
    fn a_non_monotone_coordinate_is_unordered() {
        let values = Int32Array::from(vec![1, 3, 2]);
        assert_eq!(AxisOrder::detect(&values).unwrap(), AxisOrder::Unordered);
    }

    #[test]
    fn a_null_coordinate_is_unordered() {
        let values = Int32Array::from(vec![Some(1), None, Some(3)]);
        assert_eq!(AxisOrder::detect(&values).unwrap(), AxisOrder::Unordered);
    }

    #[test]
    fn a_short_coordinate_is_ascending() {
        assert_eq!(
            AxisOrder::detect(&Int32Array::from(vec![7])).unwrap(),
            AxisOrder::Ascending
        );
        assert_eq!(
            AxisOrder::detect(&Int32Array::from(Vec::<i32>::new())).unwrap(),
            AxisOrder::Ascending
        );
    }

    #[test]
    fn strings_have_an_order() {
        let values = StringArray::from(vec!["c", "b", "a"]);
        assert_eq!(AxisOrder::detect(&values).unwrap(), AxisOrder::Descending);
    }

    #[test]
    fn a_profile_axis_has_no_order() {
        let meta = AxisMeta::no_coordinate();
        assert_eq!(meta.coordinate_column(), None);
        assert_eq!(meta.order(), AxisOrder::Unordered);
    }
}
