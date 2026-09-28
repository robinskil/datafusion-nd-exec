//! Virtual broadcast views over flat C-order arrays.
//!
//! A [`BroadcastMap`] captures a name-aligned (xarray-style) broadcast of a
//! source array onto a target dimension grid as pure index arithmetic: per
//! target axis, a source stride. A stride of `0` means the source does not
//! advance along that axis (the axis is missing from the source, or has size
//! 1 in it). No data moves until [`BroadcastMap::gather_indices`] feeds an
//! Arrow `take`.

use crate::error::Result;
use crate::error::nd_err;
use arrow::array::UInt64Array;

use super::dimensions::Dimensions;
use super::selection::{Selection, cartesian_sum};

/// Broadcast of a source dimension set onto a target dimension set,
/// represented as one source stride per target axis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BroadcastMap {
    target_shape: Vec<usize>,
    /// Source stride per target axis; `0` = source does not advance.
    strides: Vec<usize>,
    source_len: usize,
}

impl BroadcastMap {
    /// Build the broadcast of `source` onto `target`.
    ///
    /// Rules (matching `beacon-nd-array`'s `permuted_axes`-based broadcast):
    /// - every source dimension must appear in `target` (matched by name;
    ///   reordering is allowed and expressed through the strides);
    /// - a source axis must have the same size as the target axis, or size 1
    ///   (which expands).
    pub fn try_new(source: &Dimensions, target: &Dimensions) -> Result<Self> {
        let source_strides = source.c_strides();
        let mut strides = vec![0usize; target.rank()];

        for (source_axis, dim) in source.iter().enumerate() {
            let Some(target_axis) = target.position(dim.name()) else {
                return nd_err!(
                    "cannot broadcast {source} to {target}: dimension '{}' is missing from the target",
                    dim.name()
                );
            };

            let target_size = target.get(target_axis).size();
            if dim.size() == target_size {
                strides[target_axis] = source_strides[source_axis];
            } else if dim.size() == 1 {
                strides[target_axis] = 0;
            } else {
                return nd_err!(
                    "cannot broadcast {source} to {target}: dimension '{}' has size {} but the target expects {}",
                    dim.name(),
                    dim.size(),
                    target_size
                );
            }
        }

        Ok(Self {
            target_shape: target.shape(),
            strides,
            source_len: source.num_elements(),
        })
    }

    fn num_target_elements(&self) -> usize {
        self.target_shape.iter().product()
    }

    /// True when the broadcast is a no-op: every target row maps to the source
    /// row of the same index.
    pub fn is_identity(&self) -> bool {
        if self.source_len != self.num_target_elements() {
            return false;
        }
        let mut acc = 1usize;
        for axis in (0..self.target_shape.len()).rev() {
            let size = self.target_shape[axis];
            // A stride mismatch on a size-1 axis is irrelevant: its coordinate
            // is always 0.
            if size > 1 && self.strides[axis] != acc {
                return false;
            }
            acc = acc.saturating_mul(size);
        }
        true
    }

    /// Source take-indices for a *subset* of target cells, given their
    /// row-major target linear indices. For each `t`, decompose it into per-axis
    /// target coordinates and dot them with the source strides — the same
    /// mapping [`gather_indices`](Self::gather_indices) applies to every cell,
    /// evaluated only for the retained ones. This is how a broadcast composes
    /// with an accumulated selection into a single gather: the un-broadcast
    /// cross-product of filtered-out cells is never built.
    pub fn gather_indices_at(&self, targets: &UInt64Array) -> UInt64Array {
        let rank = self.target_shape.len();
        let out: Vec<u64> = targets
            .values()
            .iter()
            .map(|&t| {
                let mut rem = t as usize;
                let mut source = 0usize;
                // Decode C-order coordinates from the innermost axis outward.
                for axis in (0..rank).rev() {
                    let size = self.target_shape[axis];
                    if size == 0 {
                        continue;
                    }
                    let coord = rem % size;
                    rem /= size;
                    source += coord * self.strides[axis];
                }
                source as u64
            })
            .collect();
        UInt64Array::from(out)
    }

    /// Take indices for the broadcast view, in row-major target order.
    pub fn gather_indices(&self) -> UInt64Array {
        UInt64Array::from(cartesian_sum(&self.axis_offsets(None)))
    }

    /// Source take-indices for the target cells that `selection` keeps, in
    /// row-major target order. `selection` must be valid for the target grid.
    ///
    /// An [`Selection::AxisIndices`] selection fuses into the stride walk: the
    /// gather visits only the kept coordinates of each axis. A
    /// [`Selection::Ragged`] selection walks the outer cells and the kept
    /// prefix of the innermost axis. A [`Selection::CellMask`] selection
    /// decodes each kept cell.
    pub fn gather_indices_for(&self, selection: &Selection) -> UInt64Array {
        match selection {
            Selection::Full => self.gather_indices(),
            Selection::AxisIndices(axes) => {
                debug_assert_eq!(axes.len(), self.target_shape.len());
                UInt64Array::from(cartesian_sum(&self.axis_offsets(Some(axes))))
            }
            Selection::Ragged { lengths } => {
                let rank = self.target_shape.len();
                debug_assert!(rank > 0);
                let mut outer = self.axis_offsets(None);
                outer.pop();
                let inner_stride = self.strides[rank - 1] as u64;
                let mut out = Vec::with_capacity(lengths.values().iter().sum::<u64>() as usize);
                for (base, &len) in cartesian_sum(&outer).into_iter().zip(lengths.values()) {
                    out.extend((0..len).map(|c| base + c * inner_stride));
                }
                UInt64Array::from(out)
            }
            Selection::CellMask(cells) => self.gather_indices_at(cells),
        }
    }

    /// Per target axis, the source offset of each kept coordinate.
    fn axis_offsets(&self, axes: Option<&[Option<UInt64Array>]>) -> Vec<Vec<u64>> {
        (0..self.target_shape.len())
            .map(|axis| {
                let stride = self.strides[axis] as u64;
                match axes.and_then(|axes| axes[axis].as_ref()) {
                    Some(indices) => indices.values().iter().map(|&c| c * stride).collect(),
                    None => (0..self.target_shape[axis] as u64)
                        .map(|c| c * stride)
                        .collect(),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimensions::Dimension;

    fn dims(spec: &[(&str, usize)]) -> Dimensions {
        Dimensions::try_new(
            spec.iter()
                .map(|(name, size)| Dimension::new(*name, *size))
                .collect(),
        )
        .unwrap()
    }

    fn indices(map: &BroadcastMap) -> Vec<u64> {
        map.gather_indices().values().to_vec()
    }

    #[test]
    fn outer_dim_broadcast_repeats() {
        // time[2] onto (time, lon=3): each time value repeats 3x contiguously.
        let map = BroadcastMap::try_new(&dims(&[("time", 2)]), &dims(&[("time", 2), ("lon", 3)]))
            .unwrap();
        assert_eq!(indices(&map), vec![0, 0, 0, 1, 1, 1]);
    }

    #[test]
    fn inner_dim_broadcast_tiles() {
        // lon[3] onto (time=2, lon): the lon pattern tiles.
        let map =
            BroadcastMap::try_new(&dims(&[("lon", 3)]), &dims(&[("time", 2), ("lon", 3)])).unwrap();
        assert_eq!(indices(&map), vec![0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn middle_dim_broadcast() {
        // lat[2] onto (time=2, lat=2, lon=2): runs of 2, tiled twice.
        let map = BroadcastMap::try_new(
            &dims(&[("lat", 2)]),
            &dims(&[("time", 2), ("lat", 2), ("lon", 2)]),
        )
        .unwrap();
        assert_eq!(indices(&map), vec![0, 0, 1, 1, 0, 0, 1, 1]);
    }

    #[test]
    fn scalar_broadcast() {
        let map = BroadcastMap::try_new(&Dimensions::scalar(), &dims(&[("time", 2), ("lon", 2)]))
            .unwrap();
        assert_eq!(indices(&map), vec![0, 0, 0, 0]);
    }

    #[test]
    fn identity_detected() {
        let target = dims(&[("time", 2), ("lon", 3)]);
        let map = BroadcastMap::try_new(&target, &target).unwrap();
        assert!(map.is_identity());

        let map =
            BroadcastMap::try_new(&dims(&[("lon", 3)]), &dims(&[("time", 2), ("lon", 3)])).unwrap();
        assert!(!map.is_identity());
    }

    #[test]
    fn size_one_expansion() {
        // time[1] onto time[4]: stride 0.
        let map = BroadcastMap::try_new(&dims(&[("time", 1)]), &dims(&[("time", 4)])).unwrap();
        assert_eq!(indices(&map), vec![0, 0, 0, 0]);
    }

    #[test]
    fn transposed_dims_gather_correctly() {
        // Source stored (lon, time); target order (time, lon). Matching
        // beacon-nd-array's permuted_axes broadcast, the view transposes.
        let map = BroadcastMap::try_new(
            &dims(&[("lon", 3), ("time", 2)]),
            &dims(&[("time", 2), ("lon", 3)]),
        )
        .unwrap();
        // source flat layout: (l0,t0),(l0,t1),(l1,t0),(l1,t1),(l2,t0),(l2,t1)
        assert_eq!(indices(&map), vec![0, 2, 4, 1, 3, 5]);
        assert!(!map.is_identity());
    }

    #[test]
    fn gather_indices_at_selects_subset() {
        // lat[3] onto (time=2, lat=3): full gather tiles the lat pattern.
        let map =
            BroadcastMap::try_new(&dims(&[("lat", 3)]), &dims(&[("time", 2), ("lat", 3)])).unwrap();
        let full = indices(&map);
        assert_eq!(full, vec![0, 1, 2, 0, 1, 2]);

        // Selecting a subset of target cells returns exactly those source offsets.
        let targets = UInt64Array::from(vec![0u64, 2, 4, 5]);
        let picked = map.gather_indices_at(&targets).values().to_vec();
        assert_eq!(picked, vec![full[0], full[2], full[4], full[5]]);
        assert_eq!(picked, vec![0, 2, 1, 2]);
    }

    /// The fused gather must equal the full gather at the kept cells.
    fn assert_fused_matches_full(map: &BroadcastMap, target: &Dimensions, selection: &Selection) {
        let full = indices(map);
        let expected: Vec<u64> = selection
            .cell_indices(target)
            .values()
            .iter()
            .map(|&c| full[c as usize])
            .collect();
        let fused = map.gather_indices_for(selection).values().to_vec();
        assert_eq!(fused, expected, "{selection:?}");
    }

    #[test]
    fn gather_for_each_selection_state_matches_the_full_gather() {
        let target = dims(&[("time", 2), ("lat", 3), ("lon", 2)]);
        let sources = [
            dims(&[("lat", 3)]),
            dims(&[("time", 2), ("lat", 3), ("lon", 2)]),
            dims(&[("lon", 2), ("time", 2)]),
            Dimensions::scalar(),
        ];
        let selections = [
            Selection::Full,
            Selection::along_axis(&target, 1, UInt64Array::from(vec![0u64, 2])).unwrap(),
            Selection::AxisIndices(vec![
                Some(UInt64Array::from(vec![1u64])),
                None,
                Some(UInt64Array::from(vec![0u64])),
            ]),
            Selection::Ragged {
                lengths: UInt64Array::from(vec![2u64, 0, 1, 2, 1, 0]),
            },
            Selection::CellMask(UInt64Array::from(vec![0u64, 5, 7, 11])),
        ];
        for source in &sources {
            let map = BroadcastMap::try_new(source, &target).unwrap();
            for selection in &selections {
                selection.validate(&target).unwrap();
                assert_fused_matches_full(&map, &target, selection);
            }
        }
    }

    #[test]
    fn size_mismatch_rejected() {
        let result = BroadcastMap::try_new(&dims(&[("time", 2)]), &dims(&[("time", 4)]));
        assert!(result.is_err());
    }

    #[test]
    fn missing_target_dimension_rejected() {
        // A source axis the target does not carry cannot be broadcast away —
        // dropping it would silently collapse values.
        let result =
            BroadcastMap::try_new(&dims(&[("depth", 2)]), &dims(&[("time", 2), ("lon", 3)]));
        assert!(result.is_err());
    }

    #[test]
    fn size_one_axis_does_not_break_identity() {
        // A degenerate axis carries no information, so a stride mismatch on it
        // must not disqualify the zero-copy identity path.
        let source = dims(&[("time", 1), ("lon", 3)]);
        let map = BroadcastMap::try_new(&source, &source).unwrap();
        assert!(map.is_identity());
        assert_eq!(indices(&map), vec![0, 1, 2]);
    }

    #[test]
    fn empty_axis_yields_no_indices() {
        let map = BroadcastMap::try_new(&dims(&[("time", 0)]), &dims(&[("time", 0), ("lon", 3)]))
            .unwrap();
        assert_eq!(indices(&map), Vec::<u64>::new());
    }
}
