//! Output grids of a regrid step, and the place of a batch in one.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, UInt64Array};

use crate::dimensions::Dimensions;
use crate::error::{Result, nd_err};

/// One output grid: its axes and the coordinate values of each axis.
#[derive(Debug, Clone)]
pub struct NdOutputGrid {
    dims: Dimensions,
    coordinates: Vec<Option<ArrayRef>>,
}

impl NdOutputGrid {
    /// A grid over `dims`. `coordinates` holds one entry per axis: the values
    /// of the axis, or `None` for an axis without a coordinate.
    pub fn try_new(dims: Dimensions, coordinates: Vec<Option<ArrayRef>>) -> Result<Self> {
        if coordinates.len() != dims.rank() {
            return nd_err!(
                "an output grid of rank {} has {} coordinate entries",
                dims.rank(),
                coordinates.len()
            );
        }
        for (dim, values) in dims.iter().zip(&coordinates) {
            if let Some(values) = values
                && values.len() != dim.size()
            {
                return nd_err!(
                    "the coordinate of axis '{}' has {} values, but the axis has size {}",
                    dim.name(),
                    values.len(),
                    dim.size()
                );
            }
        }
        Ok(Self { dims, coordinates })
    }

    pub fn dims(&self) -> &Dimensions {
        &self.dims
    }

    /// One entry per axis: the coordinate values, or `None`.
    pub fn coordinates(&self) -> &[Option<ArrayRef>] {
        &self.coordinates
    }

    /// The coordinate values of the axis `axis`, or `None`.
    pub fn coordinate(&self, axis: &str) -> Option<&ArrayRef> {
        self.coordinates[self.dims.position(axis)?].as_ref()
    }

    /// The number of cells of the grid.
    pub fn num_cells(&self) -> usize {
        self.dims.num_elements()
    }
}

/// The place of one batch in its output grid.
#[derive(Debug, Clone)]
pub struct NdPlacement {
    grid: Arc<NdOutputGrid>,
    indices: Vec<UInt64Array>,
}

impl NdPlacement {
    /// `indices` holds one entry per axis of the grid: the grid position of
    /// each index of the batch on that axis.
    pub fn try_new(grid: Arc<NdOutputGrid>, indices: Vec<UInt64Array>) -> Result<Self> {
        let dims = grid.dims();
        if indices.len() != dims.rank() {
            return nd_err!(
                "a placement in a grid of rank {} has {} index arrays",
                dims.rank(),
                indices.len()
            );
        }
        for (dim, positions) in dims.iter().zip(&indices) {
            if positions.null_count() > 0 {
                return nd_err!("the positions on axis '{}' hold nulls", dim.name());
            }
            if let Some(&max) = positions.values().iter().max()
                && max as usize >= dim.size()
            {
                return nd_err!(
                    "position {max} on axis '{}' is outside the axis size {}",
                    dim.name(),
                    dim.size()
                );
            }
        }
        Ok(Self { grid, indices })
    }

    pub fn grid(&self) -> &Arc<NdOutputGrid> {
        &self.grid
    }

    /// One entry per axis of the grid.
    pub fn indices(&self) -> &[UInt64Array] {
        &self.indices
    }

    /// The grid positions of the batch on the axis `axis`.
    pub fn axis_indices(&self, axis: &str) -> Option<&UInt64Array> {
        Some(&self.indices[self.grid.dims().position(axis)?])
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::Int64Array;
    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{Dimension, NdArrowArray, NdRecordBatch, Selection};

    fn time_batch(values: Vec<i64>) -> NdRecordBatch {
        let dims = Dimensions::try_new(vec![Dimension::new("time", values.len())]).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("time", DataType::Int64, true)]));
        let column =
            NdArrowArray::try_new(Arc::new(Int64Array::from(values)), dims.clone()).unwrap();
        NdRecordBatch::try_new(schema, vec![column], dims).unwrap()
    }

    fn time_grid(size: usize) -> Arc<NdOutputGrid> {
        let dims = Dimensions::try_new(vec![Dimension::new("time", size)]).unwrap();
        let values: ArrayRef = Arc::new(Int64Array::from_iter_values(0..size as i64));
        Arc::new(NdOutputGrid::try_new(dims, vec![Some(values)]).unwrap())
    }

    fn placement(size: usize, positions: Vec<u64>) -> Arc<NdPlacement> {
        Arc::new(NdPlacement::try_new(time_grid(size), vec![UInt64Array::from(positions)]).unwrap())
    }

    #[test]
    fn a_coordinate_must_match_its_axis_size() {
        let dims = Dimensions::try_new(vec![Dimension::new("time", 3)]).unwrap();
        let values: ArrayRef = Arc::new(Int64Array::from(vec![1, 2]));
        assert!(NdOutputGrid::try_new(dims.clone(), vec![Some(values)]).is_err());
        assert!(NdOutputGrid::try_new(dims, vec![]).is_err());
    }

    #[test]
    fn a_position_must_be_inside_the_grid() {
        assert!(NdPlacement::try_new(time_grid(3), vec![UInt64Array::from(vec![1, 3])]).is_err());
        assert!(NdPlacement::try_new(time_grid(3), vec![]).is_err());
    }

    #[test]
    fn a_batch_carries_its_placement() {
        let place = placement(4, vec![1, 3]);
        let batch = time_batch(vec![10, 30])
            .with_placement(place.clone())
            .unwrap();
        assert!(Arc::ptr_eq(batch.placement().unwrap(), &place));
        let copy = batch.clone();
        assert_eq!(
            copy.placement()
                .unwrap()
                .axis_indices("time")
                .unwrap()
                .values(),
            &[1, 3]
        );
        assert_eq!(
            copy.placement()
                .unwrap()
                .grid()
                .coordinate("time")
                .unwrap()
                .len(),
            4
        );
    }

    #[test]
    fn a_new_selection_drops_the_placement() {
        let batch = time_batch(vec![10, 30])
            .with_placement(placement(4, vec![1, 3]))
            .unwrap()
            .with_selection(Selection::Full)
            .unwrap();
        assert!(batch.placement().is_none());
    }

    #[test]
    fn a_placement_must_fit_the_batch() {
        assert!(
            time_batch(vec![10, 30])
                .with_placement(placement(4, vec![0, 1, 2]))
                .is_err()
        );
        let depth = Dimensions::try_new(vec![Dimension::new("depth", 4)]).unwrap();
        let other = Arc::new(NdOutputGrid::try_new(depth, vec![None]).unwrap());
        let other =
            Arc::new(NdPlacement::try_new(other, vec![UInt64Array::from(vec![0, 1])]).unwrap());
        assert!(time_batch(vec![10, 30]).with_placement(other).is_err());
    }

    #[test]
    fn a_placed_batch_needs_a_full_selection() {
        let batch = time_batch(vec![10, 30])
            .with_selection(Selection::CellMask(UInt64Array::from(vec![0])))
            .unwrap();
        assert!(batch.with_placement(placement(4, vec![1, 3])).is_err());
    }
}
