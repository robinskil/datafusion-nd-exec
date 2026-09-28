//! Place nd chunks into one output grid.

use std::collections::HashMap;

use arrow::array::{Array, ArrayRef};
use arrow::row::{RowConverter, SortField};
use arrow::util::display::array_value_to_string;
use datafusion::error::{DataFusionError, Result};
use nd_arrow_array::{Dimension, NdRecordBatch};

/// How the accumulator places chunks along one axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AxisMode {
    /// Match the coordinate values of the chunk against the output coordinate.
    /// An exact match of a contiguous slice gives its offset. Values with no
    /// overlap are appended. An overlap without an exact match is an error.
    Coordinate,
    /// Put each chunk after the previous chunks, for example `N_PROF`.
    Append,
    /// Align each chunk at index 0 and grow the axis to the largest chunk, for
    /// example `N_LEVELS`. A writer fills the cells past a short chunk.
    Pad,
    /// All chunks have the same size on the axis.
    Fixed,
}

/// Where a chunk goes on one output axis.
#[derive(Debug, Clone)]
pub struct AxisPlacement {
    pub name: String,
    /// The first output index of the chunk.
    pub offset: usize,
    /// The size of the chunk on the axis.
    pub len: usize,
    /// The size of the output axis after this chunk.
    pub extent: usize,
    /// The coordinate values that the chunk appends to the output axis, at
    /// `offset`. `None` when the chunk adds no coordinate values.
    pub new_coordinates: Option<ArrayRef>,
}

/// Where a chunk goes in the output grid.
#[derive(Debug, Clone)]
pub struct Placement {
    /// One entry per output axis, in output order.
    pub axes: Vec<AxisPlacement>,
    /// The chunk, compacted: its selection is `Full`.
    pub batch: NdRecordBatch,
}

impl Placement {
    /// The placement of the output axis `name`.
    pub fn axis(&self, name: &str) -> Option<&AxisPlacement> {
        self.axes.iter().find(|axis| axis.name == name)
    }
}

/// The state of one output axis.
struct AxisState {
    name: String,
    mode: AxisMode,
    extent: usize,
    coordinate: Option<CoordinateState>,
}

/// The coordinate values of one output axis, as Arrow rows.
struct CoordinateState {
    column: String,
    converter: RowConverter,
    /// The output index of each coordinate value.
    index: HashMap<Vec<u8>, usize>,
    /// The first and last output values, for error messages.
    first: Option<String>,
    last: Option<String>,
}

/// Places each nd chunk into one output grid. It holds the axis state only,
/// not the data, so a writer can write each chunk as it arrives.
///
/// The output axes are the axes of the first chunk, in its order. Every chunk
/// must have the same set of axes. The mode of an axis is the mode that
/// [`with_mode`](Self::with_mode) sets. Else it is [`AxisMode::Coordinate`]
/// when the chunk has the coordinate column of the axis, and
/// [`AxisMode::Fixed`] when it does not.
#[derive(Default)]
pub struct NdGridAccumulator {
    modes: HashMap<String, AxisMode>,
    axes: Vec<AxisState>,
}

impl std::fmt::Debug for NdGridAccumulator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NdGridAccumulator")
            .field("modes", &self.modes)
            .field("extents", &self.extents())
            .finish()
    }
}

impl NdGridAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the mode of the axis `axis`.
    pub fn with_mode(mut self, axis: impl Into<String>, mode: AxisMode) -> Self {
        self.modes.insert(axis.into(), mode);
        self
    }

    /// The output axes and their sizes, in output order.
    pub fn extents(&self) -> Vec<(String, usize)> {
        self.axes
            .iter()
            .map(|axis| (axis.name.clone(), axis.extent))
            .collect()
    }

    /// Compact `batch` and place it into the output grid.
    pub fn place(&mut self, batch: &NdRecordBatch) -> Result<Placement> {
        let batch = batch.compact()?;
        let target = batch.target().clone();
        if self.axes.is_empty() {
            for dim in target.iter() {
                let state = self.new_axis(&batch, dim)?;
                self.axes.push(state);
            }
        }
        let names: Vec<&str> = target.iter().map(|d| d.name()).collect();
        if names.len() != self.axes.len()
            || !self.axes.iter().all(|a| names.contains(&a.name.as_str()))
        {
            let output: Vec<&str> = self.axes.iter().map(|a| a.name.as_str()).collect();
            return Err(DataFusionError::Execution(format!(
                "a chunk with axes {names:?} does not fit the output axes {output:?}"
            )));
        }

        let axes = self
            .axes
            .iter_mut()
            .map(|state| {
                let dim = target.get(target.position(&state.name).expect("checked above"));
                place_axis(state, dim, &batch)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Placement { axes, batch })
    }

    fn new_axis(&self, batch: &NdRecordBatch, dim: &Dimension) -> Result<AxisState> {
        let column = coordinate_column(batch, dim);
        let mode = match self.modes.get(dim.name()) {
            Some(mode) => *mode,
            None if column.is_some() => AxisMode::Coordinate,
            None => AxisMode::Fixed,
        };
        let coordinate = match mode {
            AxisMode::Coordinate => {
                let column = column.ok_or_else(|| {
                    DataFusionError::Plan(format!(
                        "axis '{}' has mode Coordinate but the chunk has no coordinate column for it",
                        dim.name()
                    ))
                })?;
                let values = coordinate_values(batch, dim, &column)?;
                Some(CoordinateState {
                    converter: RowConverter::new(vec![SortField::new(values.data_type().clone())])?,
                    column,
                    index: HashMap::new(),
                    first: None,
                    last: None,
                })
            }
            _ => None,
        };
        Ok(AxisState {
            name: dim.name().to_string(),
            mode,
            extent: 0,
            coordinate,
        })
    }
}

/// The name of the coordinate column of `dim` in `batch`: the column that the
/// axis metadata names, on `dim` alone.
fn coordinate_column(batch: &NdRecordBatch, dim: &Dimension) -> Option<String> {
    let name = dim.meta()?.coordinate_column()?;
    let index = batch.schema().index_of(name).ok()?;
    let dims = batch.column(index).dims();
    (dims.rank() == 1 && dims.get(0).name() == dim.name()).then(|| name.to_string())
}

fn coordinate_values(batch: &NdRecordBatch, dim: &Dimension, column: &str) -> Result<ArrayRef> {
    let index = batch.schema().index_of(column).map_err(|_| {
        DataFusionError::Execution(format!(
            "a chunk has no coordinate column '{column}' for axis '{}'",
            dim.name()
        ))
    })?;
    Ok(batch.column(index).values().clone())
}

fn place_axis(
    state: &mut AxisState,
    dim: &Dimension,
    batch: &NdRecordBatch,
) -> Result<AxisPlacement> {
    let len = dim.size();
    let (offset, new_coordinates) = match state.mode {
        AxisMode::Append => (state.extent, None),
        AxisMode::Pad => (0, None),
        AxisMode::Fixed => {
            if state.extent != 0 && state.extent != len {
                return Err(DataFusionError::Execution(format!(
                    "axis '{}' has size {} in one chunk and {len} in another; set mode Append or Pad for it",
                    state.name, state.extent
                )));
            }
            (0, None)
        }
        AxisMode::Coordinate => {
            let coordinate = state.coordinate.as_mut().expect("set for mode Coordinate");
            let values = coordinate_values(batch, dim, &coordinate.column)?;
            place_coordinate(&state.name, state.extent, coordinate, &values)?
        }
    };
    state.extent = state.extent.max(offset + len);
    Ok(AxisPlacement {
        name: state.name.clone(),
        offset,
        len,
        extent: state.extent,
        new_coordinates,
    })
}

/// The offset of a chunk with coordinate `values`, and the values it appends.
fn place_coordinate(
    axis: &str,
    extent: usize,
    state: &mut CoordinateState,
    values: &ArrayRef,
) -> Result<(usize, Option<ArrayRef>)> {
    let rows = state
        .converter
        .convert_columns(std::slice::from_ref(values))?;
    let keys: Vec<Vec<u8>> = (0..rows.num_rows())
        .map(|i| rows.row(i).as_ref().to_vec())
        .collect();
    let found: Vec<Option<usize>> = keys
        .iter()
        .map(|key| state.index.get(key).copied())
        .collect();
    let range = |values: &ArrayRef| -> Result<String> {
        if values.is_empty() {
            return Ok("[]".to_string());
        }
        Ok(format!(
            "[{}, {}]",
            array_value_to_string(values, 0)?,
            array_value_to_string(values, values.len() - 1)?
        ))
    };
    let conflict = || -> Result<DataFusionError> {
        Ok(DataFusionError::Execution(format!(
            "axis '{axis}': chunk coordinates {} overlap the output coordinates [{}, {}] without a match",
            range(values)?,
            state.first.as_deref().unwrap_or(""),
            state.last.as_deref().unwrap_or("")
        )))
    };

    if found.iter().all(Option::is_none) {
        // No overlap: append the values at the end.
        for (i, key) in keys.into_iter().enumerate() {
            if state.index.insert(key, extent + i).is_some() {
                return Err(DataFusionError::Execution(format!(
                    "axis '{axis}': chunk coordinates {} hold a value twice",
                    range(values)?
                )));
            }
        }
        if !values.is_empty() {
            if state.first.is_none() {
                state.first = Some(array_value_to_string(values, 0)?);
            }
            state.last = Some(array_value_to_string(values, values.len() - 1)?);
        }
        return Ok((extent, Some(values.clone())));
    }
    // An exact match: every value sits at the next index of the first one.
    let Some(start) = found[0] else {
        return Err(conflict()?);
    };
    if found
        .iter()
        .enumerate()
        .all(|(i, index)| *index == Some(start + i))
    {
        return Ok((start, None));
    }
    Err(conflict()?)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::Int64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use nd_arrow_array::{AxisMeta, AxisOrder, Dimensions, NdArrowArray};

    use super::*;
    use crate::testing::{grid_table, profile_table};

    fn chunks(table: crate::testing::NdMemTable) -> Vec<NdRecordBatch> {
        table.partitions().iter().flatten().cloned().collect()
    }

    /// A batch with one `time` coordinate column.
    fn times(values: Vec<i64>) -> NdRecordBatch {
        let time = Dimension::new("time", values.len())
            .with_meta(Some(AxisMeta::coordinate("time", AxisOrder::Ascending)));
        let dims = Dimensions::try_new(vec![time]).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("time", DataType::Int64, true)]));
        let column =
            NdArrowArray::try_new(Arc::new(Int64Array::from(values)), dims.clone()).unwrap();
        NdRecordBatch::try_new(schema, vec![column], dims).unwrap()
    }

    #[test]
    fn grid_chunks_place_by_coordinate() {
        let mut acc = NdGridAccumulator::new();
        let chunks = chunks(grid_table().unwrap());
        let first = acc.place(&chunks[0]).unwrap();
        let second = acc.place(&chunks[1]).unwrap();
        let time = second.axis("time").unwrap();
        assert_eq!((time.offset, time.len, time.extent), (2, 2, 4));
        assert!(time.new_coordinates.is_some());
        // `lat` matches at offset 0 and adds no values.
        let lat = second.axis("lat").unwrap();
        assert_eq!((lat.offset, lat.extent), (0, 3));
        assert!(lat.new_coordinates.is_none());
        assert!(first.axis("lat").unwrap().new_coordinates.is_some());
        assert_eq!(
            acc.extents(),
            vec![
                ("time".to_string(), 4),
                ("lat".to_string(), 3),
                ("lon".to_string(), 2)
            ]
        );
    }

    #[test]
    fn a_repeated_chunk_matches_its_slice() {
        let mut acc = NdGridAccumulator::new();
        acc.place(&times(vec![100, 101, 102])).unwrap();
        let again = acc.place(&times(vec![101, 102])).unwrap();
        let time = again.axis("time").unwrap();
        assert_eq!((time.offset, time.extent), (1, 3));
        assert!(time.new_coordinates.is_none());
    }

    #[test]
    fn an_overlap_without_a_match_is_a_conflict() {
        let mut acc = NdGridAccumulator::new();
        acc.place(&times(vec![100, 101])).unwrap();
        let error = acc.place(&times(vec![101, 105])).unwrap_err().to_string();
        assert!(
            error.contains("[101, 105]") && error.contains("[100, 101]"),
            "{error}"
        );
        // Values out of order also conflict.
        let error = acc.place(&times(vec![101, 100])).unwrap_err();
        assert!(error.to_string().contains("without a match"));
    }

    #[test]
    fn profiles_append_and_pad() {
        let mut acc = NdGridAccumulator::new()
            .with_mode("N_PROF", AxisMode::Append)
            .with_mode("N_LEVELS", AxisMode::Pad);
        let chunks = chunks(profile_table().unwrap());
        acc.place(&chunks[0]).unwrap();
        let second = acc.place(&chunks[1]).unwrap();
        let prof = second.axis("N_PROF").unwrap();
        let levels = second.axis("N_LEVELS").unwrap();
        assert_eq!((prof.offset, prof.len, prof.extent), (3, 2, 5));
        assert_eq!((levels.offset, levels.len, levels.extent), (0, 3, 4));
    }

    #[test]
    fn a_fixed_axis_rejects_another_size() {
        let mut acc = NdGridAccumulator::new().with_mode("N_PROF", AxisMode::Append);
        let chunks = chunks(profile_table().unwrap());
        acc.place(&chunks[0]).unwrap();
        let error = acc.place(&chunks[1]).unwrap_err().to_string();
        assert!(
            error.contains("N_LEVELS") && error.contains("Pad"),
            "{error}"
        );
    }

    #[test]
    fn chunks_must_have_the_same_axes() {
        let mut acc = NdGridAccumulator::new();
        acc.place(&times(vec![100])).unwrap();
        let chunks = chunks(profile_table().unwrap());
        assert!(acc.place(&chunks[0]).is_err());
    }

    #[test]
    fn a_coordinate_mode_needs_a_coordinate_column() {
        let mut acc = NdGridAccumulator::new().with_mode("N_PROF", AxisMode::Coordinate);
        let chunks = chunks(profile_table().unwrap());
        assert!(acc.place(&chunks[0]).is_err());
    }
}
