//! The cells of a target grid that an nd batch keeps.
//!
//! A [`Selection`] has four states, from coarse to fine:
//!
//! | State | Keeps | Shape of the result |
//! |---|---|---|
//! | [`Selection::Full`] | all cells | the grid |
//! | [`Selection::AxisIndices`] | an index set for each axis | a rectangle |
//! | [`Selection::Ragged`] | a valid prefix of the innermost axis for each outer cell | ragged rows |
//! | [`Selection::CellMask`] | an explicit cell set | any |
//!
//! An operator returns the coarsest state that holds its result. A consumer
//! declares the finest [`SelectionKind`] that it accepts. The retained cells
//! always materialize in row-major order of the target grid.

use arrow::array::{Array, BooleanArray, UInt64Array};

use crate::dimensions::Dimensions;
use crate::error::{Result, nd_err};

/// The state of a [`Selection`], ordered from coarse to fine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SelectionKind {
    Full,
    AxisIndices,
    Ragged,
    CellMask,
}

/// The retained cells of a target grid.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Selection {
    /// Keep every cell.
    #[default]
    Full,
    /// Keep the cross product of one index set per target axis. `None` keeps
    /// the full axis. Each index set is strictly ascending.
    AxisIndices(Vec<Option<UInt64Array>>),
    /// Keep the first `lengths[o]` cells of the innermost axis for each outer
    /// cell `o`. The outer cells are the row-major product of all other axes.
    Ragged { lengths: UInt64Array },
    /// Keep an explicit set of cells, as strictly ascending row-major indices.
    CellMask(UInt64Array),
}

impl Selection {
    pub fn kind(&self) -> SelectionKind {
        match self {
            Selection::Full => SelectionKind::Full,
            Selection::AxisIndices(_) => SelectionKind::AxisIndices,
            Selection::Ragged { .. } => SelectionKind::Ragged,
            Selection::CellMask(_) => SelectionKind::CellMask,
        }
    }

    /// True when the retained cells form a rectangle of the grid.
    pub fn is_rectangle(&self) -> bool {
        self.kind() <= SelectionKind::AxisIndices
    }

    /// A selection that keeps `indices` on target axis `axis` and all other
    /// axes in full. `indices` must be strictly ascending.
    pub fn along_axis(target: &Dimensions, axis: usize, indices: UInt64Array) -> Result<Self> {
        let mut axes = vec![None; target.rank()];
        if axis >= axes.len() {
            return nd_err!("nd selection axis {axis} is out of range for {target}");
        }
        axes[axis] = Some(indices);
        let selection = Selection::AxisIndices(axes);
        selection.validate(target)?;
        Ok(selection)
    }

    /// A selection of the cells where `mask` is true. `mask` has one value per
    /// target cell in row-major order. A null mask value drops its cell.
    pub fn from_cell_mask(target: &Dimensions, mask: &BooleanArray) -> Result<Self> {
        if mask.len() != target.num_elements() {
            return nd_err!(
                "nd cell mask has {} values but the grid {target} has {} cells",
                mask.len(),
                target.num_elements()
            );
        }
        if mask.null_count() == 0 && mask.true_count() == mask.len() {
            return Ok(Selection::Full);
        }
        Ok(Selection::CellMask(true_indices(mask)))
    }

    /// Check the selection against `target`.
    pub fn validate(&self, target: &Dimensions) -> Result<()> {
        match self {
            Selection::Full => Ok(()),
            Selection::AxisIndices(axes) => {
                if axes.len() != target.rank() {
                    return nd_err!(
                        "nd axis selection has {} axes but the grid {target} has {}",
                        axes.len(),
                        target.rank()
                    );
                }
                for (axis, indices) in axes.iter().enumerate() {
                    if let Some(indices) = indices {
                        check_ascending(indices, target.get(axis).size() as u64, "axis")?;
                    }
                }
                Ok(())
            }
            Selection::Ragged { lengths } => {
                let Some((outer, inner)) = split_innermost(target) else {
                    return nd_err!("a ragged nd selection needs a grid of rank 1 or more");
                };
                if lengths.len() != outer {
                    return nd_err!(
                        "ragged nd selection has {} lengths but the grid {target} has {outer} outer cells",
                        lengths.len()
                    );
                }
                if lengths.null_count() > 0 {
                    return nd_err!("ragged nd selection must not contain null lengths");
                }
                if let Some(&max) = lengths.values().iter().max()
                    && max > inner as u64
                {
                    return nd_err!(
                        "ragged nd selection length {max} exceeds the innermost axis size {inner}"
                    );
                }
                Ok(())
            }
            Selection::CellMask(cells) => {
                check_ascending(cells, target.num_elements() as u64, "cell")
            }
        }
    }

    /// The number of retained cells.
    pub fn num_rows(&self, target: &Dimensions) -> usize {
        match self {
            Selection::Full => target.num_elements(),
            Selection::AxisIndices(axes) => axes
                .iter()
                .enumerate()
                .map(|(axis, indices)| match indices {
                    Some(indices) => indices.len(),
                    None => target.get(axis).size(),
                })
                .product(),
            Selection::Ragged { lengths } => lengths.values().iter().sum::<u64>() as usize,
            Selection::CellMask(cells) => cells.len(),
        }
    }

    /// The retained cells as strictly ascending row-major indices.
    pub fn cell_indices(&self, target: &Dimensions) -> UInt64Array {
        match self {
            Selection::Full => UInt64Array::from_iter_values(0..target.num_elements() as u64),
            Selection::AxisIndices(axes) => {
                let strides = target.c_strides();
                let offsets: Vec<Vec<u64>> = axes
                    .iter()
                    .enumerate()
                    .map(|(axis, indices)| {
                        let stride = strides[axis] as u64;
                        match indices {
                            Some(indices) => indices.values().iter().map(|i| i * stride).collect(),
                            None => (0..target.get(axis).size() as u64)
                                .map(|i| i * stride)
                                .collect(),
                        }
                    })
                    .collect();
                UInt64Array::from(cartesian_sum(&offsets))
            }
            Selection::Ragged { lengths } => {
                let inner = split_innermost(target).map_or(0, |(_, inner)| inner) as u64;
                let mut out = Vec::with_capacity(self.num_rows(target));
                for (outer, &len) in lengths.values().iter().enumerate() {
                    let base = outer as u64 * inner;
                    out.extend(base..base + len);
                }
                UInt64Array::from(out)
            }
            Selection::CellMask(cells) => cells.clone(),
        }
    }

    /// The cells that both selections keep, in the coarsest state that holds
    /// them.
    pub fn intersect(&self, other: &Selection, target: &Dimensions) -> Result<Selection> {
        self.validate(target)?;
        other.validate(target)?;
        let result = match (self, other) {
            (Selection::Full, s) | (s, Selection::Full) => s.clone(),
            (Selection::AxisIndices(a), Selection::AxisIndices(b)) => {
                let axes: Vec<Option<UInt64Array>> = a
                    .iter()
                    .zip(b)
                    .map(|(a, b)| match (a, b) {
                        (None, x) | (x, None) => x.clone(),
                        (Some(a), Some(b)) => Some(intersect_sorted(a, b)),
                    })
                    .collect();
                if axes.iter().all(Option::is_none) {
                    Selection::Full
                } else {
                    Selection::AxisIndices(axes)
                }
            }
            (Selection::Ragged { lengths: a }, Selection::Ragged { lengths: b }) => {
                let lengths: UInt64Array = a
                    .values()
                    .iter()
                    .zip(b.values())
                    .map(|(a, b)| *a.min(b))
                    .collect();
                Selection::Ragged { lengths }
            }
            (a, b) => Selection::CellMask(intersect_sorted(
                &a.cell_indices(target),
                &b.cell_indices(target),
            )),
        };
        Ok(result)
    }

    /// The coarsest state that keeps the same cells. The selection must be
    /// valid for `target`.
    pub fn coarsen(&self, target: &Dimensions) -> Selection {
        match self {
            Selection::Full => Selection::Full,
            Selection::AxisIndices(axes) => axis_selection(
                axes.iter()
                    .enumerate()
                    .map(|(axis, indices)| {
                        indices
                            .as_ref()
                            .filter(|i| i.len() != target.get(axis).size())
                            .cloned()
                    })
                    .collect(),
            ),
            Selection::Ragged { lengths } => {
                let rank = target.rank();
                let inner = target.get(rank - 1).size() as u64;
                match lengths.values().first() {
                    // Equal lengths keep a rectangle: a prefix of the inner axis.
                    Some(&len) if lengths.values().iter().all(|&l| l == len) => {
                        let mut axes = vec![None; rank];
                        if len != inner {
                            axes[rank - 1] = Some(UInt64Array::from_iter_values(0..len));
                        }
                        axis_selection(axes)
                    }
                    _ => self.clone(),
                }
            }
            Selection::CellMask(cells) => coarsen_cells(cells, target),
        }
    }

    /// The retained cells `[offset, offset + len)` in row-major order, in the
    /// coarsest state that holds them. A cut on outer-axis boundaries keeps a
    /// rectangle.
    pub fn slice(&self, target: &Dimensions, offset: usize, len: usize) -> Selection {
        let rows = self.num_rows(target);
        let start = offset.min(rows);
        let end = offset.saturating_add(len).min(rows);
        if start == 0 && end == rows {
            return self.clone();
        }
        let cells = match self {
            Selection::Full => UInt64Array::from_iter_values(start as u64..end as u64),
            other => other.cell_indices(target).slice(start, end - start),
        };
        coarsen_cells(&cells, target)
    }
}

/// `Full` when no axis is constrained, else `AxisIndices`.
fn axis_selection(axes: Vec<Option<UInt64Array>>) -> Selection {
    if axes.iter().all(Option::is_none) {
        Selection::Full
    } else {
        Selection::AxisIndices(axes)
    }
}

/// The coarsest state for a strictly ascending cell set.
fn coarsen_cells(cells: &UInt64Array, target: &Dimensions) -> Selection {
    let total = target.num_elements();
    if cells.len() == total {
        return Selection::Full;
    }
    let rank = target.rank();
    if rank == 0 {
        return Selection::CellMask(cells.clone());
    }
    if let Some(rectangle) = as_rectangle(cells, target) {
        return rectangle;
    }
    if let Some(lengths) = as_ragged(cells, target) {
        return Selection::Ragged { lengths };
    }
    Selection::CellMask(cells.clone())
}

/// The cells as axis index sets, when they are the cross product of those
/// sets.
fn as_rectangle(cells: &UInt64Array, target: &Dimensions) -> Option<Selection> {
    let shape = target.shape();
    let strides = target.c_strides();
    let mut seen: Vec<Vec<bool>> = shape.iter().map(|&size| vec![false; size]).collect();
    for &cell in cells.values() {
        for (axis, (&stride, &size)) in strides.iter().zip(&shape).enumerate() {
            seen[axis][(cell as usize / stride) % size] = true;
        }
    }
    let sets: Vec<Vec<u64>> = seen
        .iter()
        .map(|axis| {
            axis.iter()
                .enumerate()
                .filter(|(_, kept)| **kept)
                .map(|(i, _)| i as u64)
                .collect()
        })
        .collect();
    // The cells are unique and inside the product, so equal counts mean equal sets.
    if sets.iter().map(Vec::len).product::<usize>() != cells.len() {
        return None;
    }
    let axes = sets
        .into_iter()
        .zip(&shape)
        .map(|(set, &size)| (set.len() != size).then(|| UInt64Array::from(set)))
        .collect::<Vec<_>>();
    if cells.is_empty() {
        // An empty set: constrain the outer axis to no index.
        let mut axes = vec![None; shape.len()];
        axes[0] = Some(UInt64Array::from(Vec::<u64>::new()));
        return Some(Selection::AxisIndices(axes));
    }
    Some(axis_selection(axes))
}

/// The prefix length of the innermost axis for each outer cell, when the
/// cells are such prefixes.
fn as_ragged(cells: &UInt64Array, target: &Dimensions) -> Option<UInt64Array> {
    let (outer, inner) = split_innermost(target)?;
    let inner = inner as u64;
    let mut lengths = vec![0u64; outer];
    for &cell in cells.values() {
        let (o, pos) = ((cell / inner) as usize, cell % inner);
        if pos != lengths[o] {
            return None;
        }
        lengths[o] += 1;
    }
    Some(UInt64Array::from(lengths))
}

/// Split `target` into the number of outer cells and the innermost axis size.
fn split_innermost(target: &Dimensions) -> Option<(usize, usize)> {
    let rank = target.rank();
    if rank == 0 {
        return None;
    }
    let inner = target.get(rank - 1).size();
    let outer = (0..rank - 1).map(|axis| target.get(axis).size()).product();
    Some((outer, inner))
}

fn check_ascending(indices: &UInt64Array, bound: u64, what: &str) -> Result<()> {
    if indices.null_count() > 0 {
        return nd_err!("nd {what} selection must not contain null indices");
    }
    let values = indices.values();
    if let Some(&last) = values.last()
        && last >= bound
    {
        return nd_err!("nd {what} selection index {last} is out of bounds for size {bound}");
    }
    if values.windows(2).any(|w| w[0] >= w[1]) {
        return nd_err!("nd {what} selection indices must be strictly ascending");
    }
    Ok(())
}

fn true_indices(mask: &BooleanArray) -> UInt64Array {
    let values = mask.values();
    match mask.nulls() {
        None => values.set_indices().map(|i| i as u64).collect(),
        Some(nulls) => values
            .set_indices()
            .filter(|&i| nulls.is_valid(i))
            .map(|i| i as u64)
            .collect(),
    }
}

/// Intersection of two strictly ascending index sets.
fn intersect_sorted(a: &UInt64Array, b: &UInt64Array) -> UInt64Array {
    let (a, b) = (a.values(), b.values());
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::with_capacity(a.len().min(b.len()));
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    UInt64Array::from(out)
}

/// Row-major cartesian sum of per-axis offsets.
pub(crate) fn cartesian_sum(axis_offsets: &[Vec<u64>]) -> Vec<u64> {
    let total: usize = axis_offsets.iter().map(Vec::len).product();
    let mut out = Vec::with_capacity(total);
    if total > 0 {
        fill_cartesian(axis_offsets, 0, 0, &mut out);
    }
    out
}

// The innermost axis is a tight loop; outer axes recurse (rank is small).
fn fill_cartesian(axis_offsets: &[Vec<u64>], axis: usize, base: u64, out: &mut Vec<u64>) {
    if axis_offsets.is_empty() {
        out.push(base);
        return;
    }
    if axis == axis_offsets.len() - 1 {
        out.extend(axis_offsets[axis].iter().map(|&off| base + off));
        return;
    }
    for &off in &axis_offsets[axis] {
        fill_cartesian(axis_offsets, axis + 1, base + off, out);
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

    fn grid() -> Dimensions {
        dims(&[("time", 2), ("lat", 3)])
    }

    fn u64s(values: &[u64]) -> UInt64Array {
        UInt64Array::from(values.to_vec())
    }

    fn cells(selection: &Selection, target: &Dimensions) -> Vec<u64> {
        selection.cell_indices(target).values().to_vec()
    }

    fn axis_selection(axes: Vec<Option<&[u64]>>) -> Selection {
        Selection::AxisIndices(axes.into_iter().map(|a| a.map(u64s)).collect())
    }

    #[test]
    fn full_keeps_every_cell() {
        let s = Selection::Full;
        assert_eq!(s.kind(), SelectionKind::Full);
        assert!(s.is_rectangle());
        assert_eq!(s.num_rows(&grid()), 6);
        assert_eq!(cells(&s, &grid()), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn axis_indices_keep_a_rectangle() {
        // lat in {0, 2} over both time steps.
        let s = Selection::along_axis(&grid(), 1, u64s(&[0, 2])).unwrap();
        assert_eq!(s.kind(), SelectionKind::AxisIndices);
        assert!(s.is_rectangle());
        assert_eq!(s.num_rows(&grid()), 4);
        assert_eq!(cells(&s, &grid()), vec![0, 2, 3, 5]);
    }

    #[test]
    fn ragged_keeps_a_prefix_per_profile() {
        let profiles = dims(&[("N_PROF", 3), ("N_LEVELS", 4)]);
        let s = Selection::Ragged {
            lengths: u64s(&[2, 0, 4]),
        };
        s.validate(&profiles).unwrap();
        assert_eq!(s.kind(), SelectionKind::Ragged);
        assert!(!s.is_rectangle());
        assert_eq!(s.num_rows(&profiles), 6);
        assert_eq!(cells(&s, &profiles), vec![0, 1, 8, 9, 10, 11]);
    }

    #[test]
    fn cell_mask_keeps_an_explicit_set() {
        let mask = BooleanArray::from(vec![
            Some(true),
            Some(false),
            None,
            Some(true),
            Some(false),
            Some(true),
        ]);
        let s = Selection::from_cell_mask(&grid(), &mask).unwrap();
        assert_eq!(s.kind(), SelectionKind::CellMask);
        assert!(!s.is_rectangle());
        assert_eq!(cells(&s, &grid()), vec![0, 3, 5]);
    }

    #[test]
    fn an_all_true_mask_is_full() {
        let mask = BooleanArray::from(vec![true; 6]);
        assert_eq!(
            Selection::from_cell_mask(&grid(), &mask).unwrap(),
            Selection::Full
        );
    }

    #[test]
    fn kinds_order_coarse_to_fine() {
        assert!(SelectionKind::Full < SelectionKind::AxisIndices);
        assert!(SelectionKind::AxisIndices < SelectionKind::Ragged);
        assert!(SelectionKind::Ragged < SelectionKind::CellMask);
    }

    #[test]
    fn full_intersect_keeps_the_other_state() {
        let a = Selection::along_axis(&grid(), 0, u64s(&[1])).unwrap();
        assert_eq!(Selection::Full.intersect(&a, &grid()).unwrap(), a);
        assert_eq!(a.intersect(&Selection::Full, &grid()).unwrap(), a);
    }

    #[test]
    fn axis_indices_intersect_per_axis() {
        let a = axis_selection(vec![Some(&[0, 1]), Some(&[0, 1])]);
        let b = axis_selection(vec![None, Some(&[1, 2])]);
        let s = a.intersect(&b, &grid()).unwrap();
        assert_eq!(s, axis_selection(vec![Some(&[0, 1]), Some(&[1])]));
        assert_eq!(cells(&s, &grid()), vec![1, 4]);
    }

    #[test]
    fn axis_indices_with_no_constraint_become_full() {
        let a = axis_selection(vec![None, None]);
        assert_eq!(a.intersect(&Selection::Full, &grid()).unwrap(), a);
        assert_eq!(a.intersect(&a, &grid()).unwrap(), Selection::Full);
    }

    #[test]
    fn ragged_intersect_takes_the_shorter_prefix() {
        let profiles = dims(&[("N_PROF", 2), ("N_LEVELS", 4)]);
        let a = Selection::Ragged {
            lengths: u64s(&[3, 1]),
        };
        let b = Selection::Ragged {
            lengths: u64s(&[2, 4]),
        };
        assert_eq!(
            a.intersect(&b, &profiles).unwrap(),
            Selection::Ragged {
                lengths: u64s(&[2, 1])
            }
        );
    }

    #[test]
    fn mixed_states_intersect_to_a_cell_mask() {
        let axis = Selection::along_axis(&grid(), 1, u64s(&[1, 2])).unwrap();
        let mask = Selection::CellMask(u64s(&[0, 2, 4]));
        let s = axis.intersect(&mask, &grid()).unwrap();
        assert_eq!(s, Selection::CellMask(u64s(&[2, 4])));

        let profiles = dims(&[("time", 2), ("lat", 3)]);
        let ragged = Selection::Ragged {
            lengths: u64s(&[3, 1]),
        };
        let s = ragged.intersect(&axis, &profiles).unwrap();
        assert_eq!(s, Selection::CellMask(u64s(&[1, 2])));
    }

    #[test]
    fn an_empty_axis_selection_keeps_no_rows() {
        let s = Selection::along_axis(&grid(), 0, u64s(&[])).unwrap();
        assert_eq!(s.num_rows(&grid()), 0);
        assert!(cells(&s, &grid()).is_empty());
    }

    fn mask(cells: &[u64]) -> Selection {
        Selection::CellMask(u64s(cells))
    }

    #[test]
    fn a_mask_of_every_cell_coarsens_to_full() {
        assert_eq!(mask(&[0, 1, 2, 3, 4, 5]).coarsen(&grid()), Selection::Full);
    }

    #[test]
    fn a_rectangular_mask_coarsens_to_axis_indices() {
        // time 1, lat {0, 2}.
        assert_eq!(
            mask(&[3, 5]).coarsen(&grid()),
            axis_selection(vec![Some(&[1]), Some(&[0, 2])])
        );
        // lat {1} over all time steps.
        assert_eq!(
            mask(&[1, 4]).coarsen(&grid()),
            axis_selection(vec![None, Some(&[1])])
        );
    }

    #[test]
    fn a_prefix_mask_coarsens_to_ragged() {
        let profiles = dims(&[("N_PROF", 3), ("N_LEVELS", 4)]);
        assert_eq!(
            mask(&[0, 1, 8, 9, 10]).coarsen(&profiles),
            Selection::Ragged {
                lengths: u64s(&[2, 0, 3])
            }
        );
    }

    #[test]
    fn other_masks_stay_masks() {
        let s = mask(&[0, 4]);
        assert_eq!(s.coarsen(&grid()), s);
    }

    #[test]
    fn an_empty_mask_coarsens_to_an_empty_rectangle() {
        let s = mask(&[]).coarsen(&grid());
        assert!(s.is_rectangle());
        assert_eq!(s.num_rows(&grid()), 0);
    }

    #[test]
    fn axis_indices_with_every_index_coarsen_to_full() {
        let s = axis_selection(vec![Some(&[0, 1]), None]);
        assert_eq!(s.coarsen(&grid()), Selection::Full);
        let s = axis_selection(vec![Some(&[0, 1]), Some(&[2])]);
        assert_eq!(s.coarsen(&grid()), axis_selection(vec![None, Some(&[2])]));
    }

    #[test]
    fn equal_ragged_lengths_coarsen_to_a_rectangle() {
        let profiles = dims(&[("N_PROF", 2), ("N_LEVELS", 4)]);
        let s = Selection::Ragged {
            lengths: u64s(&[2, 2]),
        };
        assert_eq!(
            s.coarsen(&profiles),
            axis_selection(vec![None, Some(&[0, 1])])
        );
        let s = Selection::Ragged {
            lengths: u64s(&[4, 4]),
        };
        assert_eq!(s.coarsen(&profiles), Selection::Full);
    }

    #[test]
    fn a_slice_on_outer_boundaries_keeps_a_rectangle() {
        // The second time step: cells 3, 4, 5.
        assert_eq!(
            Selection::Full.slice(&grid(), 3, 3),
            axis_selection(vec![Some(&[1]), None])
        );
    }

    #[test]
    fn a_slice_inside_an_outer_cell_is_ragged_or_a_mask() {
        // Cells 0..4: time 0 in full, time 1 at lat 0. A prefix per time step.
        assert_eq!(
            Selection::Full.slice(&grid(), 0, 4),
            Selection::Ragged {
                lengths: u64s(&[3, 1])
            }
        );
        // Cells 1, 2: a rectangle.
        assert_eq!(
            Selection::Full.slice(&grid(), 1, 2),
            axis_selection(vec![Some(&[0]), Some(&[1, 2])])
        );
        // Cells 2, 3: no coarser state.
        assert_eq!(Selection::Full.slice(&grid(), 2, 2), mask(&[2, 3]));
    }

    #[test]
    fn a_slice_follows_the_retained_cells() {
        // Keep lat {0, 2}: cells 0, 2, 3, 5. Skip one, take two: 2, 3.
        let s = axis_selection(vec![None, Some(&[0, 2])]);
        assert_eq!(s.slice(&grid(), 1, 2), mask(&[2, 3]));
        // Past the end: no rows.
        assert_eq!(s.slice(&grid(), 10, 2).num_rows(&grid()), 0);
        // All rows: the selection does not change.
        assert_eq!(s.slice(&grid(), 0, 100), s);
    }

    #[test]
    fn invalid_selections_are_rejected() {
        // Out of bounds, not ascending, wrong rank.
        assert!(Selection::along_axis(&grid(), 1, u64s(&[3])).is_err());
        assert!(Selection::along_axis(&grid(), 1, u64s(&[2, 1])).is_err());
        assert!(Selection::along_axis(&grid(), 2, u64s(&[0])).is_err());
        assert!(axis_selection(vec![None]).validate(&grid()).is_err());
        assert!(Selection::CellMask(u64s(&[6])).validate(&grid()).is_err());
        assert!(
            Selection::CellMask(u64s(&[1, 1]))
                .validate(&grid())
                .is_err()
        );
        // Wrong length count, prefix longer than the axis.
        let ragged = Selection::Ragged {
            lengths: u64s(&[1]),
        };
        assert!(ragged.validate(&grid()).is_err());
        let ragged = Selection::Ragged {
            lengths: u64s(&[1, 4]),
        };
        assert!(ragged.validate(&grid()).is_err());
        let ragged = Selection::Ragged {
            lengths: u64s(&[1]),
        };
        assert!(ragged.validate(&Dimensions::scalar()).is_err());
    }
}
