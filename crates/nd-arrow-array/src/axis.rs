//! The order of an axis, and the grid axes that a format declares.
//!
//! A grid axis such as `time` has a coordinate column of the same name, and
//! its values rise or fall. A format declares the grid axes of its scan with
//! [`NdGridAxes`]. Plans use them to report a sort order and to write grids.

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

/// The grid axes of a scan, outer first, each with the order of its
/// coordinate. A format declares them on its `NdSourceExec`. Each axis has a
/// coordinate column with the name of the axis.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NdGridAxes {
    axes: Vec<(String, AxisOrder)>,
}

impl NdGridAxes {
    pub fn new<S: Into<String>>(axes: impl IntoIterator<Item = (S, AxisOrder)>) -> Self {
        Self {
            axes: axes
                .into_iter()
                .map(|(name, order)| (name.into(), order))
                .collect(),
        }
    }

    /// The axes, outer first.
    pub fn axes(&self) -> &[(String, AxisOrder)] {
        &self.axes
    }

    /// The order of the axis `axis`, or `None` when it is not a grid axis.
    pub fn order(&self, axis: &str) -> Option<AxisOrder> {
        self.axes
            .iter()
            .find(|(name, _)| name == axis)
            .map(|(_, order)| *order)
    }

    /// The position of the axis `axis` in the declaration.
    pub fn position(&self, axis: &str) -> Option<usize> {
        self.axes.iter().position(|(name, _)| name == axis)
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
    fn grid_axes_give_the_order_and_position_of_each_axis() {
        let axes = NdGridAxes::new([
            ("time", AxisOrder::Ascending),
            ("lat", AxisOrder::Descending),
        ]);
        assert_eq!(axes.order("lat"), Some(AxisOrder::Descending));
        assert_eq!(axes.order("lon"), None);
        assert_eq!(axes.position("lat"), Some(1));
        assert_eq!(
            axes,
            NdGridAxes::new([
                ("time", AxisOrder::Ascending),
                ("lat", AxisOrder::Descending)
            ])
        );
    }
}
