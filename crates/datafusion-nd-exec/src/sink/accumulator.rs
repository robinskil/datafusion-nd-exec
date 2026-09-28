//! Place nd chunks into one output grid.

use std::collections::HashMap;

use arrow::array::{Array, ArrayRef, UInt64Array};
use arrow::compute::{sort, take};
use arrow::row::{RowConverter, SortField};
use arrow::util::display::array_value_to_string;
use datafusion::error::{DataFusionError, Result};
use nd_arrow_array::{AxisOrder, Dimension, NdRecordBatch};

/// How the accumulator places chunks along one axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AxisMode {
    /// Match the coordinate values of the chunk against the output coordinate.
    /// An exact match of a contiguous slice gives its offset. Values with no
    /// overlap are appended, unless the axis is seeded. An overlap without an
    /// exact match is an error.
    Coordinate,
    /// Put each chunk after the previous chunks, for example `N_PROF`. Only the
    /// growth axis can append.
    Append,
    /// Align each chunk at index 0 and grow the axis to the largest chunk, for
    /// example `N_LEVELS`. A writer fills the cells past a short chunk. Only an
    /// axis without a coordinate can pad.
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

/// The coordinate values of one output axis, as Arrow rows. A row compares
/// in the sort order of its value.
struct CoordinateState {
    column: String,
    converter: RowConverter,
    /// The output index of each coordinate value.
    index: HashMap<Vec<u8>, usize>,
    /// The seed values, when the host gave the whole axis in advance.
    seed: Option<ArrayRef>,
    /// The row of the last value, for the order report.
    last_key: Option<Vec<u8>>,
    ascending: bool,
    descending: bool,
    /// The first and last output values, for error messages.
    first: Option<String>,
    last: Option<String>,
}

impl CoordinateState {
    fn order(&self) -> AxisOrder {
        match (self.ascending, self.descending) {
            (true, _) => AxisOrder::Ascending,
            (false, true) => AxisOrder::Descending,
            (false, false) => AxisOrder::Unordered,
        }
    }
}

/// Places each nd chunk into one output grid. It holds the axis state only,
/// not the data, so a writer can write each chunk as it arrives.
///
/// The output axes are the axes of the first chunk, in its order. Every chunk
/// must have the same set of axes. One axis is the growth axis: by default
/// the outer axis, else the axis of [`with_growth_axis`](Self::with_growth_axis).
///
/// | Axis | Allowed modes | Default |
/// |---|---|---|
/// | growth axis | `Coordinate`, `Append` | `Coordinate` with a coordinate column, else `Append` |
/// | other axis | `Coordinate`, `Pad`, `Fixed` | `Coordinate` with a coordinate column, else `Fixed` |
///
/// `Pad` needs an axis without a coordinate column. A seeded axis, see
/// [`with_coordinate`](Self::with_coordinate), is `Coordinate` and strict.
#[derive(Default)]
pub struct NdGridAccumulator {
    modes: HashMap<String, AxisMode>,
    growth_axis: Option<String>,
    seeds: HashMap<String, ArrayRef>,
    axes: Vec<AxisState>,
}

impl std::fmt::Debug for NdGridAccumulator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NdGridAccumulator")
            .field("modes", &self.modes)
            .field("growth_axis", &self.growth_axis)
            .field("seeds", &self.seeds.keys())
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

    /// Set the growth axis. By default it is the outer axis of the first chunk.
    pub fn with_growth_axis(mut self, axis: impl Into<String>) -> Self {
        self.growth_axis = Some(axis.into());
        self
    }

    /// Give the whole coordinate of the axis `axis` in advance, for example
    /// from file metadata.
    ///
    /// The values are sorted ascending, unless they are strictly descending,
    /// and duplicates are removed. Each chunk then gets its offset by value,
    /// so the arrival order does not matter, and the extent of the axis is
    /// known before the first chunk. The axis is strict: a chunk value that is
    /// not in the seed is an error.
    pub fn with_coordinate(mut self, axis: impl Into<String>, values: ArrayRef) -> Self {
        self.seeds.insert(axis.into(), values);
        self
    }

    /// The output axes and their sizes, in output order.
    pub fn extents(&self) -> Vec<(String, usize)> {
        self.axes
            .iter()
            .map(|axis| (axis.name.clone(), axis.extent))
            .collect()
    }

    /// The order of the output coordinate of the axis `axis`, or `None` when
    /// the axis has no coordinate. An unseeded axis takes its values in
    /// arrival order, so it can be `Unordered`.
    pub fn coordinate_order(&self, axis: &str) -> Option<AxisOrder> {
        self.state(axis)?
            .coordinate
            .as_ref()
            .map(CoordinateState::order)
    }

    /// The seed of the axis `axis`, sorted, when the host gave one.
    pub fn coordinate_seed(&self, axis: &str) -> Option<&ArrayRef> {
        self.state(axis)?.coordinate.as_ref()?.seed.as_ref()
    }

    /// The name of the coordinate column of the axis `axis`.
    pub fn coordinate_column(&self, axis: &str) -> Option<&str> {
        Some(self.state(axis)?.coordinate.as_ref()?.column.as_str())
    }

    fn state(&self, axis: &str) -> Option<&AxisState> {
        self.axes.iter().find(|state| state.name == axis)
    }

    /// Compact `batch` and place it into the output grid.
    pub fn place(&mut self, batch: &NdRecordBatch) -> Result<Placement> {
        let batch = batch.compact()?;
        let target = batch.target().clone();
        if self.axes.is_empty() {
            self.init_axes(&batch)?;
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

    /// Set up the output axes from the first chunk.
    fn init_axes(&mut self, batch: &NdRecordBatch) -> Result<()> {
        let target = batch.target();
        let growth = match &self.growth_axis {
            Some(axis) => target.position(axis).ok_or_else(|| {
                DataFusionError::Plan(format!(
                    "the growth axis '{axis}' is not an axis of the chunk {target}"
                ))
            })?,
            None => 0,
        };
        for name in self.seeds.keys() {
            if target.position(name).is_none() {
                return Err(DataFusionError::Plan(format!(
                    "the seeded axis '{name}' is not an axis of the chunk {target}"
                )));
            }
        }
        let axes = target
            .iter()
            .enumerate()
            .map(|(axis, dim)| self.new_axis(batch, dim, axis == growth))
            .collect::<Result<Vec<_>>>()?;
        self.axes = axes;
        Ok(())
    }

    fn new_axis(&self, batch: &NdRecordBatch, dim: &Dimension, growth: bool) -> Result<AxisState> {
        let name = dim.name();
        let column = coordinate_column(batch, dim);
        let seed = self.seeds.get(name);
        let plan_error = |message: String| Err(DataFusionError::Plan(message));
        let mode = match (self.modes.get(name).copied(), seed) {
            (Some(mode), Some(_)) if mode != AxisMode::Coordinate => {
                return plan_error(format!("the seeded axis '{name}' must use Coordinate"));
            }
            (_, Some(_)) => AxisMode::Coordinate,
            (Some(mode), None) => mode,
            (None, None) if column.is_some() => AxisMode::Coordinate,
            (None, None) if growth => AxisMode::Append,
            (None, None) => AxisMode::Fixed,
        };
        match mode {
            AxisMode::Pad | AxisMode::Fixed if growth => {
                return plan_error(format!(
                    "the growth axis '{name}' must use Coordinate or Append, not {mode:?}"
                ));
            }
            AxisMode::Append if !growth => {
                return plan_error(format!(
                    "only the growth axis can use Append, not axis '{name}'"
                ));
            }
            AxisMode::Pad if column.is_some() => {
                return plan_error(format!(
                    "axis '{name}' has a coordinate, so it cannot use Pad"
                ));
            }
            _ => {}
        }
        let coordinate = match mode {
            AxisMode::Coordinate => {
                let column = column.ok_or_else(|| {
                    DataFusionError::Plan(format!(
                        "axis '{name}' has mode Coordinate but the chunk has no coordinate column for it"
                    ))
                })?;
                let values = coordinate_values(batch, dim, &column)?;
                Some(new_coordinate(name, column, values.data_type(), seed)?)
            }
            _ => None,
        };
        let extent = coordinate
            .as_ref()
            .and_then(|c| c.seed.as_ref())
            .map_or(0, |seed| seed.len());
        Ok(AxisState {
            name: name.to_string(),
            mode,
            extent,
            coordinate,
        })
    }
}

/// The coordinate state of one axis, with its seed placed.
fn new_coordinate(
    axis: &str,
    column: String,
    data_type: &arrow::datatypes::DataType,
    seed: Option<&ArrayRef>,
) -> Result<CoordinateState> {
    let converter = RowConverter::new(vec![SortField::new(data_type.clone())])?;
    let mut state = CoordinateState {
        column,
        converter,
        index: HashMap::new(),
        seed: None,
        last_key: None,
        ascending: true,
        descending: true,
        first: None,
        last: None,
    };
    let Some(seed) = seed else {
        return Ok(state);
    };
    if seed.null_count() > 0 {
        return Err(DataFusionError::Plan(format!(
            "the seed of axis '{axis}' must not hold nulls"
        )));
    }
    let seed = arrow::compute::cast(seed, data_type)?;
    let rows = state
        .converter
        .convert_columns(std::slice::from_ref(&seed))?;
    let descending = (1..rows.num_rows()).all(|i| rows.row(i) < rows.row(i - 1));
    let sorted = if descending {
        seed
    } else {
        sort(seed.as_ref(), None)?
    };
    // Remove duplicates: keep a value when it differs from the one before.
    let rows = state
        .converter
        .convert_columns(std::slice::from_ref(&sorted))?;
    let keep: UInt64Array = (0..rows.num_rows())
        .filter(|&i| i == 0 || rows.row(i) != rows.row(i - 1))
        .map(|i| i as u64)
        .collect();
    let sorted = take(sorted.as_ref(), &keep, None)?;
    let rows = state
        .converter
        .convert_columns(std::slice::from_ref(&sorted))?;
    for i in 0..rows.num_rows() {
        state.index.insert(rows.row(i).as_ref().to_vec(), i);
    }
    state.ascending = !descending;
    state.descending = descending;
    if !sorted.is_empty() {
        state.first = Some(array_value_to_string(&sorted, 0)?);
        state.last = Some(array_value_to_string(&sorted, sorted.len() - 1)?);
    }
    state.seed = Some(sorted);
    Ok(state)
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
                    "axis '{}' has size {} in one chunk and {len} in another; set mode Pad for it",
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
    let error = |problem: &str| -> Result<DataFusionError> {
        Ok(DataFusionError::Execution(format!(
            "axis '{axis}': chunk coordinates {} {problem} the output coordinates [{}, {}]",
            range(values)?,
            state.first.as_deref().unwrap_or(""),
            state.last.as_deref().unwrap_or("")
        )))
    };

    if state.seed.is_none() && found.iter().all(Option::is_none) {
        // No overlap: append the values at the end.
        for (i, key) in keys.into_iter().enumerate() {
            if let Some(last) = &state.last_key {
                match key.cmp(last) {
                    std::cmp::Ordering::Greater => state.descending = false,
                    std::cmp::Ordering::Less => state.ascending = false,
                    std::cmp::Ordering::Equal => {}
                }
            }
            if state.index.insert(key.clone(), extent + i).is_some() {
                return Err(DataFusionError::Execution(format!(
                    "axis '{axis}': chunk coordinates {} hold a value twice",
                    range(values)?
                )));
            }
            state.last_key = Some(key);
        }
        if !values.is_empty() {
            if state.first.is_none() {
                state.first = Some(array_value_to_string(values, 0)?);
            }
            state.last = Some(array_value_to_string(values, values.len() - 1)?);
        }
        return Ok((extent, Some(values.clone())));
    }
    if state.seed.is_some() && found.iter().any(Option::is_none) {
        return Err(error("fall outside the seed of")?);
    }
    // An exact match: every value sits at the next index of the first one.
    match found.first() {
        None => Ok((0, None)),
        Some(Some(start))
            if found
                .iter()
                .enumerate()
                .all(|(i, index)| *index == Some(start + i)) =>
        {
            Ok((*start, None))
        }
        Some(Some(start))
            if state.seed.is_some()
                && found
                    .iter()
                    .enumerate()
                    .all(|(i, index)| *index == start.checked_sub(i)) =>
        {
            Err(error("run against the order of the seed of")?)
        }
        _ => Err(error("overlap without a match")?),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use nd_arrow_array::{AxisMeta, Dimensions, NdArrowArray};

    use super::*;
    use crate::testing::{grid_table, profile_table};

    fn chunks(table: crate::testing::NdMemTable) -> Vec<NdRecordBatch> {
        table.partitions().iter().flatten().cloned().collect()
    }

    fn coordinate_dim(name: &str, size: usize) -> Dimension {
        Dimension::new(name, size).with_meta(Some(AxisMeta::coordinate(name, AxisOrder::Ascending)))
    }

    /// A batch with one `time` coordinate column.
    fn times(values: Vec<i64>) -> NdRecordBatch {
        let dims = Dimensions::try_new(vec![coordinate_dim("time", values.len())]).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("time", DataType::Int64, true)]));
        let column =
            NdArrowArray::try_new(Arc::new(Int64Array::from(values)), dims.clone()).unwrap();
        NdRecordBatch::try_new(schema, vec![column], dims).unwrap()
    }

    /// A batch on `time, lat` with both coordinate columns.
    fn time_lat(times: Vec<i64>, lats: Vec<f64>) -> NdRecordBatch {
        let time = coordinate_dim("time", times.len());
        let lat = coordinate_dim("lat", lats.len());
        let schema = Arc::new(Schema::new(vec![
            Field::new("time", DataType::Int64, true),
            Field::new("lat", DataType::Float64, true),
        ]));
        let columns = vec![
            NdArrowArray::try_new(
                Arc::new(Int64Array::from(times)),
                Dimensions::try_new(vec![time.clone()]).unwrap(),
            )
            .unwrap(),
            NdArrowArray::try_new(
                Arc::new(Float64Array::from(lats)),
                Dimensions::try_new(vec![lat.clone()]).unwrap(),
            )
            .unwrap(),
        ];
        NdRecordBatch::try_new(
            schema,
            columns,
            Dimensions::try_new(vec![time, lat]).unwrap(),
        )
        .unwrap()
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
        assert_eq!(acc.coordinate_order("time"), Some(AxisOrder::Ascending));
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
    fn an_arrival_out_of_order_is_reported() {
        let mut acc = NdGridAccumulator::new();
        let later = acc.place(&times(vec![102, 103])).unwrap();
        let earlier = acc.place(&times(vec![100, 101])).unwrap();
        // The offsets follow the arrival order, so the axis is not sorted.
        assert_eq!(later.axis("time").unwrap().offset, 0);
        assert_eq!(earlier.axis("time").unwrap().offset, 2);
        assert_eq!(acc.coordinate_order("time"), Some(AxisOrder::Unordered));
    }

    #[test]
    fn a_seed_places_by_value_in_any_arrival_order() {
        let seed = Arc::new(Int64Array::from(vec![103, 101, 102, 100, 101]));
        let mut acc = NdGridAccumulator::new().with_coordinate("time", seed);
        let later = acc.place(&times(vec![102, 103])).unwrap();
        let earlier = acc.place(&times(vec![100, 101])).unwrap();
        let time = later.axis("time").unwrap();
        assert_eq!((time.offset, time.extent), (2, 4));
        assert!(time.new_coordinates.is_none());
        assert_eq!(earlier.axis("time").unwrap().offset, 0);
        assert_eq!(acc.coordinate_order("time"), Some(AxisOrder::Ascending));
        let seed = acc.coordinate_seed("time").unwrap();
        assert_eq!(seed.len(), 4);
    }

    #[test]
    fn a_seeded_axis_is_strict() {
        let seed = Arc::new(Int64Array::from(vec![100, 101]));
        let mut acc = NdGridAccumulator::new().with_coordinate("time", seed);
        let error = acc.place(&times(vec![101, 102])).unwrap_err().to_string();
        assert!(error.contains("outside the seed"), "{error}");
    }

    #[test]
    fn a_descending_seed_keeps_its_order() {
        let seed = Arc::new(Int64Array::from(vec![103, 102, 101, 100]));
        let mut acc = NdGridAccumulator::new().with_coordinate("time", seed);
        let placed = acc.place(&times(vec![101, 100])).unwrap();
        assert_eq!(placed.axis("time").unwrap().offset, 2);
        assert_eq!(acc.coordinate_order("time"), Some(AxisOrder::Descending));
    }

    #[test]
    fn a_chunk_against_the_seed_order_is_named() {
        let seed = Arc::new(Int64Array::from(vec![103, 102, 101, 100]));
        let mut acc = NdGridAccumulator::new().with_coordinate("time", seed);
        let error = acc.place(&times(vec![100, 101])).unwrap_err().to_string();
        assert!(error.contains("against the order"), "{error}");
    }

    #[test]
    fn a_split_inner_axis_appends_its_values() {
        let mut acc = NdGridAccumulator::new();
        acc.place(&time_lat(vec![100], vec![-30.0, 0.0])).unwrap();
        let placed = acc.place(&time_lat(vec![100], vec![30.0])).unwrap();
        let lat = placed.axis("lat").unwrap();
        assert_eq!((lat.offset, lat.extent), (2, 3));
        assert_eq!(placed.axis("time").unwrap().offset, 0);
    }

    #[test]
    fn a_seeded_inner_axis_rejects_another_grid() {
        let lat = Arc::new(Float64Array::from(vec![-30.0, 0.0, 30.0]));
        let mut acc = NdGridAccumulator::new().with_coordinate("lat", lat);
        acc.place(&time_lat(vec![100], vec![-30.0, 0.0, 30.0]))
            .unwrap();
        // The next chunk comes from a grid with another resolution.
        assert!(acc.place(&time_lat(vec![101], vec![-15.0, 15.0])).is_err());
    }

    #[test]
    fn profiles_append_and_pad() {
        let mut acc = NdGridAccumulator::new().with_mode("N_LEVELS", AxisMode::Pad);
        let chunks = chunks(profile_table().unwrap());
        acc.place(&chunks[0]).unwrap();
        let second = acc.place(&chunks[1]).unwrap();
        // `N_PROF` is the outer axis, so it grows by Append.
        let prof = second.axis("N_PROF").unwrap();
        let levels = second.axis("N_LEVELS").unwrap();
        assert_eq!((prof.offset, prof.len, prof.extent), (3, 2, 5));
        assert_eq!((levels.offset, levels.len, levels.extent), (0, 3, 4));
        assert_eq!(acc.coordinate_order("N_PROF"), None);
    }

    #[test]
    fn a_fixed_axis_rejects_another_size() {
        let mut acc = NdGridAccumulator::new();
        let chunks = chunks(profile_table().unwrap());
        acc.place(&chunks[0]).unwrap();
        let error = acc.place(&chunks[1]).unwrap_err().to_string();
        assert!(
            error.contains("N_LEVELS") && error.contains("Pad"),
            "{error}"
        );
    }

    #[test]
    fn the_modes_follow_the_axis_role() {
        let chunk = &chunks(profile_table().unwrap())[0];
        let place = |acc: NdGridAccumulator| {
            let mut acc = acc;
            acc.place(chunk).map(|_| ())
        };
        // Only the growth axis appends.
        assert!(place(NdGridAccumulator::new().with_mode("N_LEVELS", AxisMode::Append)).is_err());
        // The growth axis cannot pad or stay fixed.
        assert!(place(NdGridAccumulator::new().with_mode("N_PROF", AxisMode::Pad)).is_err());
        // Another growth axis is possible.
        assert!(
            place(
                NdGridAccumulator::new()
                    .with_growth_axis("N_LEVELS")
                    .with_mode("N_PROF", AxisMode::Fixed)
            )
            .is_ok()
        );
        assert!(place(NdGridAccumulator::new().with_growth_axis("depth")).is_err());
        // An axis with a coordinate cannot pad.
        let grid = &chunks(grid_table().unwrap())[0];
        let mut acc = NdGridAccumulator::new().with_mode("lat", AxisMode::Pad);
        assert!(acc.place(grid).is_err());
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
