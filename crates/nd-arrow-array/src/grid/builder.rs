//! Build output grids from the records of the collected batches.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, UInt64Array};
use arrow::compute::{concat, take};
use arrow::row::{RowConverter, Rows, SortField};

use super::{NdOutputGrid, NdPlacement};
use crate::axis::{AxisMeta, AxisOrder};
use crate::batch::NdRecordBatch;
use crate::dimensions::{Dimension, Dimensions};
use crate::error::{ArrowError, Result, nd_err};
use crate::selection::Selection;

/// What the grid builder keeps of one batch: its origin, its grid and the
/// coordinate values of each axis.
#[derive(Debug, Clone)]
pub struct NdBatchRecord {
    /// The input partition of the batch.
    pub partition: usize,
    /// The number of the batch in its partition.
    pub batch: usize,
    /// The grid of the batch, with its axis metadata.
    pub dims: Dimensions,
    /// One entry per axis of `dims`: the coordinate values, or `None`.
    pub coordinates: Vec<Option<ArrayRef>>,
    /// One entry per axis of `dims`: the original index of each index, or
    /// `None` when the indices are `0..size`.
    pub origins: Vec<Option<UInt64Array>>,
}

impl NdBatchRecord {
    /// The record of `batch`. An axis has coordinate values when a column of
    /// the batch lives on that axis alone, with one value per index. The axis
    /// metadata names that column. Without metadata, it is the column with the
    /// name of the axis. `AxisMeta::no_coordinate()` turns the rule off.
    pub fn of(partition: usize, number: usize, batch: &NdRecordBatch) -> Self {
        let coordinates = batch
            .target()
            .iter()
            .map(|dim| coordinate_values(batch, dim))
            .collect();
        Self {
            partition,
            batch: number,
            dims: batch.target().clone(),
            coordinates,
            origins: vec![None; batch.target().rank()],
        }
    }

    /// Set the original index of each index per axis, see [`axis_origins`].
    pub fn with_origins(mut self, origins: Vec<Option<UInt64Array>>) -> Result<Self> {
        if origins.len() != self.dims.rank() {
            return nd_err!(
                "{} has {} axes but {} origin entries",
                self.label(),
                self.dims.rank(),
                origins.len()
            );
        }
        for (dim, origin) in self.dims.iter().zip(&origins) {
            if let Some(origin) = origin
                && origin.len() != dim.size()
            {
                return nd_err!(
                    "axis '{}' of {} has size {} but {} origins",
                    dim.name(),
                    self.label(),
                    dim.size(),
                    origin.len()
                );
            }
        }
        self.origins = origins;
        Ok(self)
    }

    /// The memory of the coordinate values, in bytes.
    pub fn memory_size(&self) -> usize {
        self.coordinates
            .iter()
            .flatten()
            .map(|values| values.get_array_memory_size())
            .sum()
    }

    fn label(&self) -> String {
        format!("partition {} batch {}", self.partition, self.batch)
    }
}

/// The original index of each index per axis, after
/// [`NdRecordBatch::compact`] of a batch with `selection` on `rank` axes. Only
/// an `AxisIndices` selection cuts the grid, so the other states give `None`.
pub fn axis_origins(selection: &Selection, rank: usize) -> Vec<Option<UInt64Array>> {
    match selection {
        Selection::AxisIndices(axes) => axes.clone(),
        _ => vec![None; rank],
    }
}

fn coordinate_values(batch: &NdRecordBatch, dim: &Dimension) -> Option<ArrayRef> {
    // Without metadata, the column with the name of the axis is its coordinate.
    let name = match dim.meta() {
        Some(meta) => meta.coordinate_column()?,
        None => dim.name(),
    };
    let column = batch.column(batch.schema().index_of(name).ok()?);
    let dims = column.dims();
    let on_axis = dims.rank() == 1 && dims.get(0).name() == dim.name();
    (on_axis && column.values().len() == dim.size()).then(|| column.values().clone())
}

/// Collects the records of all batches, then builds the output grids and the
/// place of each batch.
#[derive(Debug, Default)]
pub struct NdGridBuilder {
    records: Vec<NdBatchRecord>,
}

/// One axis of an output grid, with the positions of each group member.
struct OutputAxis {
    size: usize,
    meta: Option<AxisMeta>,
    values: Option<ArrayRef>,
    positions: Vec<UInt64Array>,
}

impl NdGridBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add the record of one batch. Returns its number: the index of its
    /// placement in [`finish`](Self::finish).
    pub fn add(&mut self, record: NdBatchRecord) -> usize {
        self.records.push(record);
        self.records.len() - 1
    }

    /// The placement of each batch, by batch number. Fails on an overlap, a
    /// null coordinate, a repeated coordinate value, or an inner axis whose
    /// coordinate column the batches do not hold.
    pub fn finish(self) -> Result<Vec<Arc<NdPlacement>>> {
        let mut placements: Vec<Option<Arc<NdPlacement>>> = vec![None; self.records.len()];
        for members in self.groups() {
            let (grid, indices) = self.build_group(&members)?;
            self.check_overlaps(&members, &indices)?;
            for (&member, indices) in members.iter().zip(indices) {
                placements[member] = Some(Arc::new(NdPlacement::try_new(grid.clone(), indices)?));
            }
        }
        Ok(placements
            .into_iter()
            .map(|p| p.expect("every record is in a group"))
            .collect())
    }

    /// The record numbers of each group, in first-seen order.
    fn groups(&self) -> Vec<Vec<usize>> {
        let mut keys: Vec<Vec<&str>> = Vec::new();
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for (number, record) in self.records.iter().enumerate() {
            let key: Vec<&str> = record.dims.iter().map(|d| d.name()).collect();
            match keys.iter().position(|k| *k == key) {
                Some(group) => groups[group].push(number),
                None => {
                    keys.push(key);
                    groups.push(vec![number]);
                }
            }
        }
        groups
    }

    /// The grid of one group, and the positions of each member per axis.
    fn build_group(&self, members: &[usize]) -> Result<(Arc<NdOutputGrid>, Vec<Vec<UInt64Array>>)> {
        let first = &self.records[members[0]];
        let rank = first.dims.rank();
        let mut dims = Vec::with_capacity(rank);
        let mut coordinates = Vec::with_capacity(rank);
        let mut indices: Vec<Vec<UInt64Array>> = vec![Vec::with_capacity(rank); members.len()];
        for axis in 0..rank {
            let dim = first.dims.get(axis);
            let with_values = members
                .iter()
                .filter(|&&m| self.records[m].coordinates[axis].is_some())
                .count();
            let output = if with_values == members.len() {
                self.coordinate_axis(members, axis)?
            } else if with_values == 0 {
                // A pad puts a cut chunk at index 0, so an inner axis needs its coordinate.
                let column = members.iter().find_map(|&m| {
                    let meta = self.records[m].dims.get(axis).meta()?;
                    meta.coordinate_column()
                });
                if let Some(column) = column
                    && axis > 0
                {
                    return nd_err!(
                        "inner axis '{}' has the coordinate column '{column}', but the batches do not hold it: select the column '{column}' to write a grid",
                        dim.name()
                    );
                }
                self.plain_axis(members, axis)
            } else {
                return nd_err!(
                    "axis '{}' has a coordinate in some batches but not in others",
                    dim.name()
                );
            };
            dims.push(Dimension::new(dim.name(), output.size).with_meta(output.meta));
            coordinates.push(output.values);
            for (member, positions) in indices.iter_mut().zip(output.positions) {
                member.push(positions);
            }
        }
        let grid = NdOutputGrid::try_new(Dimensions::try_new(dims)?, coordinates)?;
        Ok((Arc::new(grid), indices))
    }

    fn coordinate_axis(&self, members: &[usize], axis: usize) -> Result<OutputAxis> {
        let dim = self.records[members[0]].dims.get(axis);
        let arrays = self.coordinate_arrays(members, axis)?;
        let converter = RowConverter::new(vec![SortField::new(arrays[0].data_type().clone())])?;
        let refs: Vec<&dyn Array> = arrays.iter().map(|a| a.as_ref()).collect();
        let all = concat(&refs)?;
        let rows = converter.convert_columns(std::slice::from_ref(&all))?;
        let lengths: Vec<usize> = arrays.iter().map(|a| a.len()).collect();
        let order = axis_order(&rows, &lengths);

        // The union of all values, sorted in `order`, without duplicates.
        let mut union: Vec<usize> = (0..rows.num_rows()).collect();
        union.sort_by(|&a, &b| match order {
            AxisOrder::Descending => rows.row(b).cmp(&rows.row(a)),
            _ => rows.row(a).cmp(&rows.row(b)),
        });
        union.dedup_by(|a, b| rows.row(*a) == rows.row(*b));
        let keep = UInt64Array::from_iter_values(union.iter().map(|&i| i as u64));
        let values = take(all.as_ref(), &keep, None)?;
        let position: HashMap<&[u8], u64> = union
            .iter()
            .enumerate()
            .map(|(out, &i)| (rows.row(i).data(), out as u64))
            .collect();

        let mut start = 0;
        let mut positions = Vec::with_capacity(arrays.len());
        for (&member, len) in members.iter().zip(lengths) {
            let member_positions: Vec<u64> = (start..start + len)
                .map(|i| position[rows.row(i).data()])
                .collect();
            if member_positions.iter().collect::<HashSet<_>>().len() != len {
                return nd_err!(
                    "the coordinate of axis '{}' repeats a value in {}",
                    dim.name(),
                    self.records[member].label()
                );
            }
            positions.push(UInt64Array::from(member_positions));
            start += len;
        }

        let column = dim
            .meta()
            .and_then(|m| m.coordinate_column())
            .unwrap_or(dim.name());
        Ok(OutputAxis {
            size: values.len(),
            meta: Some(AxisMeta::coordinate(column, order)),
            values: Some(values),
            positions,
        })
    }

    /// The coordinate values of each member on `axis`. All must have one data
    /// type and no nulls.
    fn coordinate_arrays(&self, members: &[usize], axis: usize) -> Result<Vec<&ArrayRef>> {
        let name = self.records[members[0]].dims.get(axis).name();
        let arrays: Vec<&ArrayRef> = members
            .iter()
            .map(|&m| {
                self.records[m].coordinates[axis]
                    .as_ref()
                    .expect("checked by the caller")
            })
            .collect();
        let data_type = arrays[0].data_type();
        for (&member, values) in members.iter().zip(&arrays) {
            let label = self.records[member].label();
            if values.data_type() != data_type {
                return nd_err!(
                    "the coordinate of axis '{name}' is {} in {label}, but {data_type} in {}",
                    values.data_type(),
                    self.records[members[0]].label()
                );
            }
            if values.null_count() > 0 {
                return nd_err!("the coordinate of axis '{name}' holds nulls in {label}");
            }
        }
        Ok(arrays)
    }

    fn plain_axis(&self, members: &[usize], axis: usize) -> OutputAxis {
        let sizes: Vec<usize> = members
            .iter()
            .map(|&m| self.records[m].dims.get(axis).size())
            .collect();
        let positions: Vec<UInt64Array> = if axis == 0 {
            // The outer axis appends in the order of partition, then batch number.
            let mut order: Vec<usize> = (0..members.len()).collect();
            order.sort_by_key(|&k| {
                let record = &self.records[members[k]];
                (record.partition, record.batch)
            });
            let mut offsets = vec![0; members.len()];
            let mut next = 0;
            for k in order {
                offsets[k] = next;
                next += sizes[k];
            }
            offsets
                .iter()
                .zip(&sizes)
                .map(|(&offset, &len)| {
                    UInt64Array::from_iter_values((offset..offset + len).map(|i| i as u64))
                })
                .collect()
        } else {
            // An inner axis keeps the original index, so a cut does not move cells.
            members
                .iter()
                .zip(&sizes)
                .map(|(&m, &len)| match &self.records[m].origins[axis] {
                    Some(origin) => origin.clone(),
                    None => UInt64Array::from_iter_values(0..len as u64),
                })
                .collect()
        };
        let size = positions
            .iter()
            .filter_map(|p| p.values().iter().max())
            .map(|&max| max as usize + 1)
            .max()
            .unwrap_or(0);
        OutputAxis {
            size,
            meta: self.records[members[0]].dims.get(axis).meta().cloned(),
            values: None,
            positions,
        }
    }

    /// Fail when two members of a group write the same cell.
    fn check_overlaps(&self, members: &[usize], indices: &[Vec<UInt64Array>]) -> Result<()> {
        let bounds = |positions: &UInt64Array| {
            let values = positions.values();
            Some((*values.iter().min()?, *values.iter().max()?))
        };
        // A member with an empty axis has no cells.
        let boxes: Vec<Option<Vec<(u64, u64)>>> = indices
            .iter()
            .map(|axes| axes.iter().map(bounds).collect())
            .collect();
        let mut order: Vec<usize> = (0..members.len()).filter(|&k| boxes[k].is_some()).collect();
        let lowest = |k: usize| boxes[k].as_ref().and_then(|b| b.first()).map_or(0, |b| b.0);
        order.sort_by_key(|&k| lowest(k));
        for (i, &a) in order.iter().enumerate() {
            let box_a = boxes[a].as_ref().expect("filtered");
            for &b in &order[i + 1..] {
                let box_b = boxes[b].as_ref().expect("filtered");
                // The members are sorted on axis 0, so no later member overlaps.
                if let (Some(first_a), Some(first_b)) = (box_a.first(), box_b.first())
                    && first_b.0 > first_a.1
                {
                    break;
                }
                if indices[a]
                    .iter()
                    .zip(&indices[b])
                    .all(|(pa, pb)| intersects(pa, pb))
                {
                    return Err(self.overlap_error(members[a], members[b], box_a, box_b));
                }
            }
        }
        Ok(())
    }

    fn overlap_error(
        &self,
        a: usize,
        b: usize,
        box_a: &[(u64, u64)],
        box_b: &[(u64, u64)],
    ) -> ArrowError {
        let ranges: Vec<String> = self.records[a]
            .dims
            .iter()
            .zip(box_a.iter().zip(box_b))
            .map(|(dim, (ra, rb))| {
                format!("{} {}..={}", dim.name(), ra.0.max(rb.0), ra.1.min(rb.1))
            })
            .collect();
        ArrowError::InvalidArgumentError(format!(
            "{} and {} write the same cells of the output grid: {}",
            self.records[a].label(),
            self.records[b].label(),
            ranges.join(", ")
        ))
    }
}

/// Descending when every member with two or more values falls strictly, and
/// at least one such member exists. Else ascending. `lengths` gives the rows
/// of each member in `rows`, in order.
fn axis_order(rows: &Rows, lengths: &[usize]) -> AxisOrder {
    let mut long = 0;
    let mut falling = 0;
    let mut start = 0;
    for &len in lengths {
        if len >= 2 {
            long += 1;
            if (start + 1..start + len).all(|i| rows.row(i) < rows.row(i - 1)) {
                falling += 1;
            }
        }
        start += len;
    }
    if long > 0 && falling == long {
        AxisOrder::Descending
    } else {
        AxisOrder::Ascending
    }
}

/// True when the two position sets share a position.
fn intersects(a: &UInt64Array, b: &UInt64Array) -> bool {
    let set: HashSet<u64> = a.values().iter().copied().collect();
    b.values().iter().any(|p| set.contains(p))
}

#[cfg(test)]
mod tests {
    use arrow::array::{AsArray, Int64Array};
    use arrow::datatypes::{DataType, Field, Int64Type, Schema};

    use super::*;
    use crate::{NdArrowArray, Selection};

    /// One axis of a test record: coordinate values, or a plain size.
    enum Axis {
        C(&'static str, Vec<i64>),
        P(&'static str, usize),
        /// An axis with a coordinate column that the query does not select.
        U(&'static str, usize),
    }

    fn record(partition: usize, batch: usize, axes: Vec<Axis>) -> NdBatchRecord {
        let mut dims = Vec::new();
        let mut coordinates = Vec::new();
        for axis in axes {
            match axis {
                Axis::C(name, values) => {
                    let meta = AxisMeta::coordinate(name, AxisOrder::Unordered);
                    dims.push(Dimension::new(name, values.len()).with_meta(Some(meta)));
                    coordinates.push(Some(Arc::new(Int64Array::from(values)) as ArrayRef));
                }
                Axis::P(name, size) => {
                    dims.push(Dimension::new(name, size));
                    coordinates.push(None);
                }
                Axis::U(name, size) => {
                    let meta = AxisMeta::coordinate(name, AxisOrder::Ascending);
                    dims.push(Dimension::new(name, size).with_meta(Some(meta)));
                    coordinates.push(None);
                }
            }
        }
        let origins = vec![None; dims.len()];
        NdBatchRecord {
            partition,
            batch,
            dims: Dimensions::try_new(dims).unwrap(),
            coordinates,
            origins,
        }
    }

    fn finish(records: Vec<NdBatchRecord>) -> Result<Vec<Arc<NdPlacement>>> {
        let mut builder = NdGridBuilder::new();
        for record in records {
            builder.add(record);
        }
        builder.finish()
    }

    fn positions(placement: &NdPlacement, axis: &str) -> Vec<u64> {
        placement.axis_indices(axis).unwrap().values().to_vec()
    }

    fn coordinate(placement: &NdPlacement, axis: &str) -> Vec<i64> {
        let values = placement.grid().coordinate(axis).unwrap();
        values.as_primitive::<Int64Type>().values().to_vec()
    }

    fn shape(placement: &NdPlacement) -> Vec<(String, usize)> {
        let dims = placement.grid().dims();
        dims.iter()
            .map(|d| (d.name().to_string(), d.size()))
            .collect()
    }

    #[test]
    fn coordinates_join_in_ascending_order() {
        let placements = finish(vec![
            record(0, 0, vec![Axis::C("time", vec![102, 103])]),
            record(1, 0, vec![Axis::C("time", vec![100, 101])]),
        ])
        .unwrap();
        assert!(Arc::ptr_eq(placements[0].grid(), placements[1].grid()));
        assert_eq!(coordinate(&placements[0], "time"), [100, 101, 102, 103]);
        assert_eq!(positions(&placements[0], "time"), [2, 3]);
        assert_eq!(positions(&placements[1], "time"), [0, 1]);
        let meta = placements[0].grid().dims().get(0).meta().unwrap();
        assert_eq!(meta.order(), AxisOrder::Ascending);
        assert_eq!(meta.coordinate_column(), Some("time"));
    }

    #[test]
    fn strictly_descending_batches_give_a_descending_axis() {
        let placements = finish(vec![
            record(0, 0, vec![Axis::C("lat", vec![30, 20])]),
            record(0, 1, vec![Axis::C("lat", vec![10])]),
            record(0, 2, vec![Axis::C("lat", vec![0, -10])]),
        ])
        .unwrap();
        assert_eq!(coordinate(&placements[0], "lat"), [30, 20, 10, 0, -10]);
        assert_eq!(positions(&placements[2], "lat"), [3, 4]);
        let meta = placements[0].grid().dims().get(0).meta().unwrap();
        assert_eq!(meta.order(), AxisOrder::Descending);
    }

    #[test]
    fn mixed_or_single_values_give_an_ascending_axis() {
        let mixed = finish(vec![
            record(0, 0, vec![Axis::C("lat", vec![3, 2])]),
            record(0, 1, vec![Axis::C("lat", vec![0, 1])]),
        ])
        .unwrap();
        assert_eq!(coordinate(&mixed[0], "lat"), [0, 1, 2, 3]);
        let single = finish(vec![
            record(0, 0, vec![Axis::C("lat", vec![5])]),
            record(0, 1, vec![Axis::C("lat", vec![4])]),
        ])
        .unwrap();
        assert_eq!(coordinate(&single[0], "lat"), [4, 5]);
    }

    #[test]
    fn different_grids_fill_a_sparse_grid() {
        let placements = finish(vec![
            record(
                0,
                0,
                vec![Axis::C("time", vec![100]), Axis::C("lat", vec![0])],
            ),
            record(
                0,
                1,
                vec![Axis::C("time", vec![101]), Axis::C("lat", vec![10])],
            ),
        ])
        .unwrap();
        let grid = placements[0].grid();
        assert_eq!(grid.num_cells(), 4);
        assert_eq!(positions(&placements[1], "time"), [1]);
        assert_eq!(positions(&placements[1], "lat"), [1]);
    }

    #[test]
    fn the_outer_plain_axis_appends_and_inner_axes_pad() {
        let placements = finish(vec![
            record(1, 0, vec![Axis::P("N_PROF", 2), Axis::P("N_LEVELS", 4)]),
            record(0, 1, vec![Axis::P("N_PROF", 3), Axis::P("N_LEVELS", 2)]),
            record(0, 0, vec![Axis::P("N_PROF", 1), Axis::P("N_LEVELS", 5)]),
        ])
        .unwrap();
        let expected = vec![("N_PROF".to_string(), 6), ("N_LEVELS".to_string(), 5)];
        assert_eq!(shape(&placements[0]), expected);
        assert_eq!(positions(&placements[2], "N_PROF"), [0]);
        assert_eq!(positions(&placements[1], "N_PROF"), [1, 2, 3]);
        assert_eq!(positions(&placements[0], "N_PROF"), [4, 5]);
        assert_eq!(positions(&placements[1], "N_LEVELS"), [0, 1]);
        assert!(placements[0].grid().coordinate("N_PROF").is_none());
    }

    #[test]
    fn each_axis_set_gets_its_own_grid() {
        let placements = finish(vec![
            record(0, 0, vec![Axis::C("time", vec![1])]),
            record(0, 1, vec![Axis::P("N_PROF", 2)]),
            record(0, 2, vec![Axis::C("time", vec![2])]),
        ])
        .unwrap();
        assert!(Arc::ptr_eq(placements[0].grid(), placements[2].grid()));
        assert!(!Arc::ptr_eq(placements[0].grid(), placements[1].grid()));
    }

    #[test]
    fn an_overlap_is_an_error() {
        let error = finish(vec![
            record(0, 0, vec![Axis::C("time", vec![100, 101])]),
            record(1, 0, vec![Axis::C("time", vec![101, 102])]),
        ])
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("partition 0 batch 0 and partition 1 batch 0"),
            "{error}"
        );
        assert!(error.contains("time 1..=1"), "{error}");
    }

    #[test]
    fn tiles_on_another_axis_do_not_overlap() {
        let tile = |batch, lat| {
            record(
                0,
                batch,
                vec![Axis::C("time", vec![100, 101]), Axis::C("lat", vec![lat])],
            )
        };
        assert!(finish(vec![tile(0, 0), tile(1, 10)]).is_ok());
    }

    #[test]
    fn a_repeated_value_in_one_batch_is_an_error() {
        let error = finish(vec![record(0, 0, vec![Axis::C("time", vec![1, 1])])]);
        assert!(error.is_err());
    }

    #[test]
    fn a_null_coordinate_is_an_error() {
        let mut bad = record(0, 0, vec![Axis::C("time", vec![1, 2])]);
        bad.coordinates[0] = Some(Arc::new(Int64Array::from(vec![Some(1), None])));
        assert!(finish(vec![bad]).unwrap_err().to_string().contains("nulls"));
    }

    #[test]
    fn a_coordinate_in_some_batches_only_is_an_error() {
        let error = finish(vec![
            record(0, 0, vec![Axis::C("time", vec![1])]),
            record(0, 1, vec![Axis::P("time", 1)]),
        ]);
        assert!(error.unwrap_err().to_string().contains("some batches"));
    }

    #[test]
    fn a_record_takes_the_coordinate_column_of_its_batch() {
        let meta = AxisMeta::coordinate("time", AxisOrder::Ascending);
        let dims =
            Dimensions::try_new(vec![Dimension::new("time", 2).with_meta(Some(meta))]).unwrap();
        let values = || NdArrowArray::try_new(Arc::new(Int64Array::from(vec![5, 6])), dims.clone());
        let field = |name| Field::new(name, DataType::Int64, true);

        let with = NdRecordBatch::try_new(
            Arc::new(Schema::new(vec![field("time"), field("sst")])),
            vec![values().unwrap(), values().unwrap()],
            dims.clone(),
        )
        .unwrap();
        assert!(NdBatchRecord::of(0, 0, &with).coordinates[0].is_some());

        // The query does not select `time`, so the axis has no coordinate.
        let without = NdRecordBatch::try_new(
            Arc::new(Schema::new(vec![field("sst")])),
            vec![values().unwrap()],
            dims,
        )
        .unwrap();
        let record = NdBatchRecord::of(0, 1, &without);
        assert!(record.coordinates[0].is_none());
        let placements =
            finish(vec![record.clone(), NdBatchRecord { batch: 2, ..record }]).unwrap();
        assert_eq!(positions(&placements[1], "time"), [2, 3]);
    }

    #[test]
    fn an_inner_axis_without_its_coordinate_column_is_an_error() {
        let error = finish(vec![
            record(0, 0, vec![Axis::C("time", vec![1]), Axis::U("lon", 1)]),
            record(0, 1, vec![Axis::C("time", vec![2]), Axis::U("lon", 2)]),
        ])
        .unwrap_err()
        .to_string();
        assert!(error.contains("select the column 'lon'"), "{error}");
    }

    #[test]
    fn an_outer_axis_without_its_coordinate_column_appends() {
        let placements = finish(vec![
            record(0, 0, vec![Axis::U("time", 2), Axis::C("lat", vec![0])]),
            record(1, 0, vec![Axis::U("time", 1), Axis::C("lat", vec![0])]),
        ])
        .unwrap();
        assert_eq!(positions(&placements[1], "time"), [2]);
    }

    /// A chunk on `time` with one column per name. Column `i` holds `10 * i + 5, 10 * i + 6`.
    fn on_time(meta: Option<AxisMeta>, names: &[&str]) -> NdRecordBatch {
        let dims = Dimensions::try_new(vec![Dimension::new("time", 2).with_meta(meta)]).unwrap();
        let fields: Vec<Field> = names
            .iter()
            .map(|n| Field::new(*n, DataType::Int64, true))
            .collect();
        let columns = (0..names.len() as i64)
            .map(|i| {
                let values = Int64Array::from(vec![10 * i + 5, 10 * i + 6]);
                NdArrowArray::try_new(Arc::new(values), dims.clone()).unwrap()
            })
            .collect();
        NdRecordBatch::try_new(Arc::new(Schema::new(fields)), columns, dims).unwrap()
    }

    fn first_value(record: &NdBatchRecord) -> Option<i64> {
        let values = record.coordinates[0].as_ref()?;
        Some(values.as_primitive::<Int64Type>().value(0))
    }

    #[test]
    fn a_column_with_the_axis_name_is_its_coordinate() {
        let record = NdBatchRecord::of(0, 0, &on_time(None, &["sst", "time"]));
        assert_eq!(first_value(&record), Some(15));
        assert_eq!(
            first_value(&NdBatchRecord::of(0, 0, &on_time(None, &["sst"]))),
            None
        );
    }

    #[test]
    fn axis_metadata_overrides_the_convention() {
        let off = on_time(Some(AxisMeta::no_coordinate()), &["time"]);
        assert_eq!(first_value(&NdBatchRecord::of(0, 0, &off)), None);
        let other = AxisMeta::coordinate("t", AxisOrder::Unordered);
        let named = on_time(Some(other), &["time", "t"]);
        assert_eq!(first_value(&NdBatchRecord::of(0, 0, &named)), Some(15));
    }

    #[test]
    fn a_column_on_two_axes_is_not_a_coordinate() {
        let time = Dimension::new("time", 2);
        let lat = Dimension::new("lat", 1);
        let dims = Dimensions::try_new(vec![time, lat]).unwrap();
        let values = Int64Array::from(vec![1, 2]);
        let column = NdArrowArray::try_new(Arc::new(values), dims.clone()).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("time", DataType::Int64, true)]));
        let batch = NdRecordBatch::try_new(schema, vec![column], dims).unwrap();
        assert!(NdBatchRecord::of(0, 0, &batch).coordinates[0].is_none());
    }

    #[test]
    fn a_discovered_coordinate_names_its_column_in_the_grid() {
        let record = NdBatchRecord::of(0, 0, &on_time(None, &["time"]));
        let placements = finish(vec![record]).unwrap();
        let meta = placements[0].grid().dims().get(0).meta().unwrap();
        assert_eq!(meta.coordinate_column(), Some("time"));
        assert_eq!(meta.order(), AxisOrder::Ascending);
    }

    #[test]
    fn origins_come_from_an_axis_indices_selection() {
        let kept = vec![None, Some(UInt64Array::from(vec![2, 3]))];
        assert_eq!(axis_origins(&Selection::AxisIndices(kept.clone()), 2), kept);
        assert_eq!(axis_origins(&Selection::Full, 2), vec![None, None]);
        let mask = Selection::CellMask(UInt64Array::from(vec![0]));
        assert_eq!(axis_origins(&mask, 2), vec![None, None]);
    }

    #[test]
    fn an_inner_axis_without_a_coordinate_keeps_the_original_index() {
        let first = record(0, 0, vec![Axis::C("time", vec![1]), Axis::P("lev", 2)])
            .with_origins(vec![None, Some(UInt64Array::from(vec![2, 3]))])
            .unwrap();
        let second = record(0, 1, vec![Axis::C("time", vec![2]), Axis::P("lev", 1)])
            .with_origins(vec![None, Some(UInt64Array::from(vec![0]))])
            .unwrap();
        let placements = finish(vec![first, second]).unwrap();
        assert_eq!(positions(&placements[0], "lev"), [2, 3]);
        assert_eq!(positions(&placements[1], "lev"), [0]);
        let expected = vec![("time".to_string(), 2), ("lev".to_string(), 4)];
        assert_eq!(shape(&placements[0]), expected);
    }

    #[test]
    fn the_outer_axis_appends_whatever_the_origins() {
        let first = record(0, 0, vec![Axis::P("N_PROF", 2)])
            .with_origins(vec![Some(UInt64Array::from(vec![5, 7]))])
            .unwrap();
        let second = record(0, 1, vec![Axis::P("N_PROF", 1)]);
        let placements = finish(vec![first, second]).unwrap();
        assert_eq!(positions(&placements[0], "N_PROF"), [0, 1]);
        assert_eq!(positions(&placements[1], "N_PROF"), [2]);
    }

    #[test]
    fn origins_must_fit_the_axes() {
        let record = record(0, 0, vec![Axis::P("lev", 2)]);
        assert!(record.clone().with_origins(vec![]).is_err());
        let short = vec![Some(UInt64Array::from(vec![1]))];
        assert!(record.with_origins(short).is_err());
    }
}
