//! Build output grids from the records of the collected batches.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, UInt64Array};
use arrow::compute::{concat, take};
use arrow::row::{RowConverter, SortField};

use super::{NdOutputGrid, NdPlacement};
use crate::axis::{AxisMeta, AxisOrder};
use crate::batch::NdRecordBatch;
use crate::dimensions::{Dimension, Dimensions};
use crate::error::{Result, nd_err};

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
}

impl NdBatchRecord {
    /// The record of `batch`. An axis has coordinate values when its metadata
    /// names a column of the batch that lives on that axis alone.
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
        }
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

fn coordinate_values(batch: &NdRecordBatch, dim: &Dimension) -> Option<ArrayRef> {
    let name = dim.meta()?.coordinate_column()?;
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

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
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
        let name = dim.name();
        let arrays: Vec<&ArrayRef> = members
            .iter()
            .map(|&m| {
                self.records[m].coordinates[axis]
                    .as_ref()
                    .expect("checked by the caller")
            })
            .collect();
        let data_type = arrays[0].data_type().clone();
        for (&member, values) in members.iter().zip(&arrays) {
            let label = self.records[member].label();
            if values.data_type() != &data_type {
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

        let converter = RowConverter::new(vec![SortField::new(data_type)])?;
        let refs: Vec<&dyn Array> = arrays.iter().map(|a| a.as_ref()).collect();
        let all = concat(&refs)?;
        let rows = converter.convert_columns(std::slice::from_ref(&all))?;

        // Descending only when every batch with two or more values falls strictly.
        let mut long = 0;
        let mut falling = 0;
        let mut start = 0;
        for values in &arrays {
            let end = start + values.len();
            if values.len() >= 2 {
                long += 1;
                if (start + 1..end).all(|i| rows.row(i) < rows.row(i - 1)) {
                    falling += 1;
                }
            }
            start = end;
        }
        let descending = long > 0 && falling == long;

        let mut order: Vec<usize> = (0..rows.num_rows()).collect();
        if descending {
            order.sort_by(|&a, &b| rows.row(b).cmp(&rows.row(a)));
        } else {
            order.sort_by(|&a, &b| rows.row(a).cmp(&rows.row(b)));
        }
        order.dedup_by(|a, b| rows.row(*a) == rows.row(*b));
        let keep = UInt64Array::from_iter_values(order.iter().map(|&i| i as u64));
        let values = take(all.as_ref(), &keep, None)?;
        let position: HashMap<&[u8], u64> = order
            .iter()
            .enumerate()
            .map(|(out, &i)| (rows.row(i).data(), out as u64))
            .collect();

        let mut start = 0;
        let mut positions = Vec::with_capacity(arrays.len());
        for (&member, values) in members.iter().zip(&arrays) {
            let end = start + values.len();
            let member_positions: Vec<u64> =
                (start..end).map(|i| position[rows.row(i).data()]).collect();
            let unique: HashSet<u64> = member_positions.iter().copied().collect();
            if unique.len() != member_positions.len() {
                return nd_err!(
                    "the coordinate of axis '{name}' repeats a value in {}",
                    self.records[member].label()
                );
            }
            positions.push(UInt64Array::from(member_positions));
            start = end;
        }

        let column = dim
            .meta()
            .and_then(|m| m.coordinate_column())
            .unwrap_or(name)
            .to_string();
        let order = if descending {
            AxisOrder::Descending
        } else {
            AxisOrder::Ascending
        };
        Ok(OutputAxis {
            size: values.len(),
            meta: Some(AxisMeta::coordinate(column, order)),
            values: Some(values),
            positions,
        })
    }

    fn plain_axis(&self, members: &[usize], axis: usize) -> OutputAxis {
        let sizes: Vec<usize> = members
            .iter()
            .map(|&m| self.records[m].dims.get(axis).size())
            .collect();
        let mut offsets = vec![0; members.len()];
        let size = if axis == 0 {
            // The outer axis appends in the order of partition, then batch number.
            let mut order: Vec<usize> = (0..members.len()).collect();
            order.sort_by_key(|&k| {
                let record = &self.records[members[k]];
                (record.partition, record.batch)
            });
            let mut next = 0;
            for k in order {
                offsets[k] = next;
                next += sizes[k];
            }
            next
        } else {
            sizes.iter().copied().max().unwrap_or(0)
        };
        let positions = offsets
            .iter()
            .zip(&sizes)
            .map(|(&offset, &len)| {
                UInt64Array::from_iter_values((offset..offset + len).map(|i| i as u64))
            })
            .collect();
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
                if let (Some(first_a), Some(first_b)) = (box_a.first(), box_b.first())
                    && first_b.0 > first_a.1
                {
                    break;
                }
                let overlap = indices[a].iter().zip(&indices[b]).all(|(pa, pb)| {
                    let set: HashSet<u64> = pa.values().iter().copied().collect();
                    pb.values().iter().any(|p| set.contains(p))
                });
                if overlap {
                    let dims = &self.records[members[a]].dims;
                    let ranges: Vec<String> = dims
                        .iter()
                        .zip(box_a.iter().zip(box_b))
                        .map(|(dim, (ra, rb))| {
                            format!("{} {}..={}", dim.name(), ra.0.max(rb.0), ra.1.min(rb.1))
                        })
                        .collect();
                    return nd_err!(
                        "{} and {} write the same cells of the output grid: {}",
                        self.records[members[a]].label(),
                        self.records[members[b]].label(),
                        ranges.join(", ")
                    );
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{AsArray, Int64Array};
    use arrow::datatypes::{DataType, Field, Int64Type, Schema};

    use super::*;
    use crate::NdArrowArray;

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
        NdBatchRecord {
            partition,
            batch,
            dims: Dimensions::try_new(dims).unwrap(),
            coordinates,
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
}
