//! Build one output grid from the records of the collected batches.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, UInt64Array};
use arrow::compute::{concat, take};
use arrow::row::{RowConverter, SortField};

use super::{NdOutputGrid, NdPlacement};
use crate::axis::{AxisOrder, NdGridAxes};
use crate::batch::NdRecordBatch;
use crate::dimensions::{Dimension, Dimensions};
use crate::error::{ArrowError, Result, nd_err};

/// What the grid builder keeps of one batch: its origin, its grid and the
/// coordinate values of each axis.
#[derive(Debug, Clone)]
pub struct NdBatchRecord {
    /// The input partition of the batch.
    pub partition: usize,
    /// The number of the batch in its partition.
    pub batch: usize,
    /// The grid of the batch.
    pub dims: Dimensions,
    /// The coordinate values of each axis of `dims`.
    pub coordinates: Vec<ArrayRef>,
}

impl NdBatchRecord {
    /// The record of `batch`. Each axis needs its coordinate column: the
    /// column with the name of the axis, on that axis alone, with one value per
    /// index.
    pub fn of(partition: usize, number: usize, batch: &NdRecordBatch) -> Result<Self> {
        let label = format!("partition {partition} batch {number}");
        let coordinates = batch
            .target()
            .iter()
            .map(|dim| {
                coordinate_values(batch, dim).ok_or_else(|| {
                    ArrowError::InvalidArgumentError(format!(
                        "axis '{0}' of {label} has no coordinate column: select the column '{0}' to write a grid",
                        dim.name()
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            partition,
            batch: number,
            dims: batch.target().clone(),
            coordinates,
        })
    }

    /// The memory of the coordinate values, in bytes.
    pub fn memory_size(&self) -> usize {
        self.coordinates
            .iter()
            .map(|values| values.get_array_memory_size())
            .sum()
    }

    fn label(&self) -> String {
        format!("partition {} batch {}", self.partition, self.batch)
    }

    fn axis_names(&self) -> Vec<&str> {
        self.dims.iter().map(|d| d.name()).collect()
    }
}

fn coordinate_values(batch: &NdRecordBatch, dim: &Dimension) -> Option<ArrayRef> {
    let column = batch.column(batch.schema().index_of(dim.name()).ok()?);
    let dims = column.dims();
    let on_axis = dims.rank() == 1 && dims.get(0).name() == dim.name();
    (on_axis && column.values().len() == dim.size()).then(|| column.values().clone())
}

/// Collects the records of all batches, then builds the output grid and the
/// place of each batch.
///
/// Each batch must lie on declared grid axes, in the declared order, and all
/// batches must have the same axes. [`add`](Self::add) checks this at once,
/// so a bad batch fails before the rest of the input is read.
#[derive(Debug)]
pub struct NdGridBuilder {
    axes: Arc<NdGridAxes>,
    records: Vec<NdBatchRecord>,
}

impl NdGridBuilder {
    /// A builder for a scan with the grid axes `axes`.
    pub fn new(axes: Arc<NdGridAxes>) -> Self {
        Self {
            axes,
            records: Vec::new(),
        }
    }

    /// Add the record of one batch. Returns its number: the index of its
    /// placement in [`finish`](Self::finish).
    pub fn add(&mut self, record: NdBatchRecord) -> Result<usize> {
        self.check(&record)?;
        self.records.push(record);
        Ok(self.records.len() - 1)
    }

    fn check(&self, record: &NdBatchRecord) -> Result<()> {
        let declared: Vec<&str> = self
            .axes
            .axes()
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        let mut last = None;
        for dim in record.dims.iter() {
            let Some(position) = self.axes.position(dim.name()) else {
                return nd_err!(
                    "{} has the axis '{}', which is not a grid axis of the scan [{}]",
                    record.label(),
                    dim.name(),
                    declared.join(", ")
                );
            };
            if last.is_some_and(|last| position <= last) {
                return nd_err!(
                    "{} has the axes [{}], not in the order of the grid axes [{}]",
                    record.label(),
                    record.axis_names().join(", "),
                    declared.join(", ")
                );
            }
            last = Some(position);
        }
        let Some(first) = self.records.first() else {
            return check_values(record, None);
        };
        if first.axis_names() != record.axis_names() {
            return nd_err!(
                "{} has the axes [{}], but {} has [{}]",
                record.label(),
                record.axis_names().join(", "),
                first.label(),
                first.axis_names().join(", ")
            );
        }
        check_values(record, Some(first))
    }

    /// The placement of each batch, by batch number. Fails on an overlap or a
    /// repeated coordinate value.
    pub fn finish(self) -> Result<Vec<Arc<NdPlacement>>> {
        let Some(first) = self.records.first() else {
            return Ok(Vec::new());
        };
        let rank = first.dims.rank();
        let mut dims = Vec::with_capacity(rank);
        let mut coordinates = Vec::with_capacity(rank);
        let mut indices: Vec<Vec<UInt64Array>> = vec![Vec::with_capacity(rank); self.records.len()];
        for axis in 0..rank {
            let name = first.dims.get(axis).name();
            let order = self.axes.order(name).unwrap_or(AxisOrder::Ascending);
            let (values, positions) = self.join_axis(axis, order)?;
            dims.push(Dimension::new(name, values.len()));
            coordinates.push(values);
            for (record, positions) in indices.iter_mut().zip(positions) {
                record.push(positions);
            }
        }
        self.check_overlaps(&indices)?;
        let grid = Arc::new(NdOutputGrid::try_new(
            Dimensions::try_new(dims)?,
            coordinates,
        )?);
        indices
            .into_iter()
            .map(|indices| Ok(Arc::new(NdPlacement::try_new(grid.clone(), indices)?)))
            .collect()
    }

    /// The union of the values of all records on `axis`, sorted in `order`
    /// without duplicates, and the position of each value of each record.
    fn join_axis(&self, axis: usize, order: AxisOrder) -> Result<(ArrayRef, Vec<UInt64Array>)> {
        let arrays: Vec<&ArrayRef> = self.records.iter().map(|r| &r.coordinates[axis]).collect();
        let converter = RowConverter::new(vec![SortField::new(arrays[0].data_type().clone())])?;
        let refs: Vec<&dyn Array> = arrays.iter().map(|a| a.as_ref()).collect();
        let all = concat(&refs)?;
        let rows = converter.convert_columns(std::slice::from_ref(&all))?;

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
        for (record, values) in self.records.iter().zip(&arrays) {
            let len = values.len();
            let record_positions: Vec<u64> = (start..start + len)
                .map(|i| position[rows.row(i).data()])
                .collect();
            if record_positions.iter().collect::<HashSet<_>>().len() != len {
                return nd_err!(
                    "the coordinate of axis '{}' repeats a value in {}",
                    record.dims.get(axis).name(),
                    record.label()
                );
            }
            positions.push(UInt64Array::from(record_positions));
            start += len;
        }
        Ok((values, positions))
    }

    /// Fail when two records write the same cell.
    fn check_overlaps(&self, indices: &[Vec<UInt64Array>]) -> Result<()> {
        let bounds = |positions: &UInt64Array| {
            let values = positions.values();
            Some((*values.iter().min()?, *values.iter().max()?))
        };
        // A record with an empty axis has no cells.
        let boxes: Vec<Option<Vec<(u64, u64)>>> = indices
            .iter()
            .map(|axes| axes.iter().map(bounds).collect())
            .collect();
        let mut order: Vec<usize> = (0..indices.len()).filter(|&k| boxes[k].is_some()).collect();
        let lowest = |k: usize| boxes[k].as_ref().and_then(|b| b.first()).map_or(0, |b| b.0);
        order.sort_by_key(|&k| lowest(k));
        for (i, &a) in order.iter().enumerate() {
            let box_a = boxes[a].as_ref().expect("filtered");
            for &b in &order[i + 1..] {
                let box_b = boxes[b].as_ref().expect("filtered");
                // The records are sorted on axis 0, so no later record overlaps.
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
                    return Err(self.overlap_error(a, b, box_a, box_b));
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

/// The coordinates of `record` have no nulls, and the data types of `first`.
fn check_values(record: &NdBatchRecord, first: Option<&NdBatchRecord>) -> Result<()> {
    for (axis, values) in record.coordinates.iter().enumerate() {
        let name = record.dims.get(axis).name();
        if values.null_count() > 0 {
            return nd_err!(
                "the coordinate of axis '{name}' holds nulls in {}",
                record.label()
            );
        }
        if let Some(first) = first
            && values.data_type() != first.coordinates[axis].data_type()
        {
            return nd_err!(
                "the coordinate of axis '{name}' is {} in {}, but {} in {}",
                values.data_type(),
                record.label(),
                first.coordinates[axis].data_type(),
                first.label()
            );
        }
    }
    Ok(())
}

/// True when the two position sets share a position.
fn intersects(a: &UInt64Array, b: &UInt64Array) -> bool {
    let set: HashSet<u64> = a.values().iter().copied().collect();
    b.values().iter().any(|p| set.contains(p))
}

#[cfg(test)]
mod tests {
    use arrow::array::{AsArray, Float64Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Int64Type, Schema};

    use super::*;
    use crate::NdArrowArray;

    fn axes(spec: &[(&str, AxisOrder)]) -> Arc<NdGridAxes> {
        Arc::new(NdGridAxes::new(
            spec.iter().map(|(name, order)| (*name, *order)),
        ))
    }

    fn time_lat() -> Arc<NdGridAxes> {
        axes(&[
            ("time", AxisOrder::Ascending),
            ("lat", AxisOrder::Ascending),
        ])
    }

    fn record(partition: usize, batch: usize, spec: Vec<(&str, Vec<i64>)>) -> NdBatchRecord {
        let dims = spec
            .iter()
            .map(|(name, values)| Dimension::new(*name, values.len()))
            .collect();
        let coordinates = spec
            .into_iter()
            .map(|(_, values)| Arc::new(Int64Array::from(values)) as ArrayRef)
            .collect();
        NdBatchRecord {
            partition,
            batch,
            dims: Dimensions::try_new(dims).unwrap(),
            coordinates,
        }
    }

    fn finish(axes: Arc<NdGridAxes>, records: Vec<NdBatchRecord>) -> Result<Vec<Arc<NdPlacement>>> {
        let mut builder = NdGridBuilder::new(axes);
        for record in records {
            builder.add(record)?;
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

    #[test]
    fn coordinates_join_in_ascending_order() {
        let placements = finish(
            time_lat(),
            vec![
                record(0, 0, vec![("time", vec![102, 103])]),
                record(1, 0, vec![("time", vec![100, 101])]),
            ],
        )
        .unwrap();
        assert!(Arc::ptr_eq(placements[0].grid(), placements[1].grid()));
        assert_eq!(coordinate(&placements[0], "time"), [100, 101, 102, 103]);
        assert_eq!(positions(&placements[0], "time"), [2, 3]);
        assert_eq!(positions(&placements[1], "time"), [0, 1]);
    }

    #[test]
    fn a_descending_axis_comes_out_descending() {
        let lat = axes(&[("lat", AxisOrder::Descending)]);
        let placements = finish(
            lat,
            vec![
                record(0, 0, vec![("lat", vec![0, 10])]),
                record(0, 1, vec![("lat", vec![30, 20])]),
            ],
        )
        .unwrap();
        assert_eq!(coordinate(&placements[0], "lat"), [30, 20, 10, 0]);
        assert_eq!(positions(&placements[0], "lat"), [3, 2]);
    }

    #[test]
    fn different_grids_fill_a_sparse_grid() {
        let placements = finish(
            time_lat(),
            vec![
                record(0, 0, vec![("time", vec![100]), ("lat", vec![0])]),
                record(0, 1, vec![("time", vec![101]), ("lat", vec![10])]),
            ],
        )
        .unwrap();
        assert_eq!(placements[0].grid().num_cells(), 4);
        assert_eq!(positions(&placements[1], "time"), [1]);
        assert_eq!(positions(&placements[1], "lat"), [1]);
    }

    #[test]
    fn an_overlap_is_an_error() {
        let error = finish(
            time_lat(),
            vec![
                record(0, 0, vec![("time", vec![100, 101])]),
                record(1, 0, vec![("time", vec![101, 102])]),
            ],
        )
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
        let tile =
            |batch, lat| record(0, batch, vec![("time", vec![100, 101]), ("lat", vec![lat])]);
        assert!(finish(time_lat(), vec![tile(0, 0), tile(1, 10)]).is_ok());
    }

    #[test]
    fn a_repeated_value_in_one_batch_is_an_error() {
        let error = finish(time_lat(), vec![record(0, 0, vec![("time", vec![1, 1])])]);
        assert!(error.unwrap_err().to_string().contains("repeats"));
    }

    #[test]
    fn a_batch_must_lie_on_declared_axes_in_order() {
        let mut builder = NdGridBuilder::new(time_lat());
        let undeclared = builder.add(record(0, 0, vec![("depth", vec![5])]));
        assert!(
            undeclared
                .unwrap_err()
                .to_string()
                .contains("not a grid axis")
        );
        let reversed = builder.add(record(0, 1, vec![("lat", vec![0]), ("time", vec![1])]));
        assert!(
            reversed
                .unwrap_err()
                .to_string()
                .contains("not in the order")
        );
    }

    #[test]
    fn all_batches_must_have_the_same_axes() {
        let mut builder = NdGridBuilder::new(time_lat());
        builder
            .add(record(0, 0, vec![("time", vec![1]), ("lat", vec![0])]))
            .unwrap();
        let error = builder
            .add(record(0, 1, vec![("time", vec![2])]))
            .unwrap_err();
        assert!(
            error.to_string().contains("but partition 0 batch 0 has"),
            "{error}"
        );
    }

    #[test]
    fn a_null_or_mixed_type_coordinate_is_an_error() {
        let mut builder = NdGridBuilder::new(time_lat());
        let mut nulls = record(0, 0, vec![("time", vec![1, 2])]);
        nulls.coordinates[0] = Arc::new(Int64Array::from(vec![Some(1), None]));
        assert!(
            builder
                .add(nulls)
                .unwrap_err()
                .to_string()
                .contains("nulls")
        );

        builder.add(record(0, 1, vec![("time", vec![1])])).unwrap();
        let mut floats = record(0, 2, vec![("time", vec![2])]);
        floats.coordinates[0] = Arc::new(Float64Array::from(vec![2.0]));
        assert!(
            builder
                .add(floats)
                .unwrap_err()
                .to_string()
                .contains("Float64")
        );
    }

    #[test]
    fn no_batches_give_no_placements() {
        assert!(finish(time_lat(), vec![]).unwrap().is_empty());
    }

    /// A batch on `time` with a column per name.
    fn on_time(names: &[&str]) -> NdRecordBatch {
        let dims = Dimensions::try_new(vec![Dimension::new("time", 2)]).unwrap();
        let fields: Vec<Field> = names
            .iter()
            .map(|n| Field::new(*n, DataType::Int64, true))
            .collect();
        let columns = names
            .iter()
            .map(|_| {
                let values = Int64Array::from(vec![5, 6]);
                NdArrowArray::try_new(Arc::new(values), dims.clone()).unwrap()
            })
            .collect();
        NdRecordBatch::try_new(Arc::new(Schema::new(fields)), columns, dims).unwrap()
    }

    #[test]
    fn a_record_takes_the_column_with_the_axis_name() {
        let record = NdBatchRecord::of(0, 0, &on_time(&["sst", "time"])).unwrap();
        assert_eq!(record.coordinates[0].len(), 2);
        let error = NdBatchRecord::of(0, 1, &on_time(&["sst"])).unwrap_err();
        assert!(
            error.to_string().contains("select the column 'time'"),
            "{error}"
        );
    }

    #[test]
    fn a_column_on_two_axes_is_not_a_coordinate() {
        let dims =
            Dimensions::try_new(vec![Dimension::new("time", 2), Dimension::new("lat", 1)]).unwrap();
        let values = Int64Array::from(vec![1, 2]);
        let column = NdArrowArray::try_new(Arc::new(values), dims.clone()).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("time", DataType::Int64, true)]));
        let batch = NdRecordBatch::try_new(schema, vec![column], dims).unwrap();
        assert!(NdBatchRecord::of(0, 0, &batch).is_err());
    }
}
