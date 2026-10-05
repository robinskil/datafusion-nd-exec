//! Axis ranges for readers.
//!
//! A reader gets the query filters as pruning hints. Before it reads, it can
//! turn the conjuncts on one coordinate column into the index ranges of that
//! axis, and skip each chunk that holds no kept index. The filter above still
//! applies every conjunct, so the ranges only have to keep every row that the
//! filter keeps.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanArray, RecordBatch};
use arrow::datatypes::{Schema, SchemaRef};
use datafusion::error::{DataFusionError, Result};
use datafusion::physical_expr::utils::{collect_columns, reassign_expr_columns};
use datafusion::physical_expr::{PhysicalExpr, split_conjunction};

use crate::pushable::is_pushable_expr;

/// The 1-D coordinate values of one axis.
#[derive(Debug, Clone)]
pub struct AxisCoordinate {
    /// The name of the axis.
    pub axis: String,
    /// The name of the coordinate column in the table schema.
    pub column: String,
    /// One value per index of the axis.
    pub values: ArrayRef,
}

/// The kept index ranges of each constrained axis. An axis that is absent is
/// kept in full.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AxisRanges {
    axes: BTreeMap<String, Vec<Range<usize>>>,
}

impl AxisRanges {
    /// True when no axis is constrained.
    pub fn is_empty(&self) -> bool {
        self.axes.is_empty()
    }

    /// The kept ranges of `axis`, ascending and disjoint, or `None` when the
    /// axis is kept in full.
    pub fn get(&self, axis: &str) -> Option<&[Range<usize>]> {
        self.axes.get(axis).map(Vec::as_slice)
    }

    /// The constrained axes and their kept ranges.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &[Range<usize>])> {
        self.axes
            .iter()
            .map(|(axis, ranges)| (axis.as_str(), ranges.as_slice()))
    }

    /// True when the chunk at `origin` with `shape` on the axes `axis_names`
    /// holds a kept index on every constrained axis. A constrained axis that
    /// the chunk does not have does not count.
    pub fn overlaps(&self, origin: &[usize], shape: &[usize], axis_names: &[&str]) -> bool {
        axis_names
            .iter()
            .zip(origin.iter().zip(shape))
            .all(|(axis, (&start, &len))| match self.axes.get(*axis) {
                None => true,
                Some(ranges) => ranges
                    .iter()
                    .any(|range| range.start < start + len && start < range.end),
            })
    }
}

/// The kept index ranges per axis for the filters `predicates` over
/// `schema`.
///
/// Each top-level conjunct that references only the coordinate column of one
/// axis in `coordinates` is evaluated on the coordinate values. A null value
/// drops its index, as in a `WHERE`. Other conjuncts do not constrain an axis.
pub fn axis_ranges(
    predicates: &[Arc<dyn PhysicalExpr>],
    schema: &SchemaRef,
    coordinates: &[AxisCoordinate],
) -> Result<AxisRanges> {
    let mut kept: BTreeMap<&str, Vec<bool>> = BTreeMap::new();
    for conjunct in predicates.iter().flat_map(|p| split_conjunction(p)) {
        if !is_pushable_expr(conjunct) {
            continue;
        }
        let columns = collect_columns(conjunct);
        let [column] = columns.iter().collect::<Vec<_>>()[..] else {
            continue;
        };
        let Some(coordinate) = coordinates.iter().find(|c| c.column == column.name()) else {
            continue;
        };
        let field = schema.field_with_name(column.name())?;
        let compact = Arc::new(Schema::new(vec![field.clone()]));
        let expr = reassign_expr_columns(conjunct.clone(), &compact)?;
        let batch = RecordBatch::try_new(compact, vec![coordinate.values.clone()])?;
        let mask = expr.evaluate(&batch)?.into_array(batch.num_rows())?;
        let mask = mask
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| {
                DataFusionError::Plan(format!("the conjunct {conjunct} is not a boolean"))
            })?;
        let axis = kept
            .entry(coordinate.axis.as_str())
            .or_insert_with(|| vec![true; mask.len()]);
        for (i, keep) in axis.iter_mut().enumerate() {
            *keep &= mask.is_valid(i) && mask.value(i);
        }
    }
    Ok(AxisRanges {
        axes: kept
            .into_iter()
            .map(|(axis, keep)| (axis.to_string(), runs(&keep)))
            .collect(),
    })
}

/// The runs of true values as index ranges.
fn runs(keep: &[bool]) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = None;
    for (i, &k) in keep.iter().enumerate() {
        match (k, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                ranges.push(s..i);
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        ranges.push(s..keep.len());
    }
    ranges
}

#[cfg(test)]
mod tests {
    use arrow::array::{Float64Array, Int64Array};
    use arrow::datatypes::{DataType, Field};
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::expressions::{binary, col, lit};

    use super::*;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("time", DataType::Int64, true),
            Field::new("lat", DataType::Float64, true),
            Field::new("sst", DataType::Float64, true),
        ]))
    }

    fn coordinates() -> Vec<AxisCoordinate> {
        vec![
            AxisCoordinate {
                axis: "time".to_string(),
                column: "time".to_string(),
                values: Arc::new(Int64Array::from(vec![100, 101, 102, 103])),
            },
            AxisCoordinate {
                axis: "lat".to_string(),
                column: "lat".to_string(),
                values: Arc::new(Float64Array::from(vec![Some(-30.0), None, Some(30.0)])),
            },
        ]
    }

    fn pred(
        name: &str,
        op: Operator,
        value: impl datafusion::logical_expr::Literal,
    ) -> Arc<dyn PhysicalExpr> {
        let s = schema();
        binary(col(name, &s).unwrap(), op, lit(value), &s).unwrap()
    }

    #[test]
    fn one_axis_conjuncts_give_index_ranges() {
        let time = binary(
            pred("time", Operator::GtEq, 101i64),
            Operator::And,
            pred("time", Operator::NotEq, 102i64),
            &schema(),
        )
        .unwrap();
        let ranges = axis_ranges(
            &[time, pred("lat", Operator::Gt, -40.0f64)],
            &schema(),
            &coordinates(),
        )
        .unwrap();
        assert_eq!(ranges.get("time"), Some(&[1..2, 3..4][..]));
        // A null coordinate value drops its index.
        assert_eq!(ranges.get("lat"), Some(&[0..1, 2..3][..]));
    }

    #[test]
    fn other_conjuncts_do_not_constrain_an_axis() {
        let cross = binary(
            pred("time", Operator::Gt, 0i64),
            Operator::Or,
            pred("sst", Operator::Gt, 1.0f64),
            &schema(),
        )
        .unwrap();
        let ranges = axis_ranges(
            &[pred("sst", Operator::Gt, 1.0f64), cross],
            &schema(),
            &coordinates(),
        )
        .unwrap();
        assert!(ranges.is_empty());
    }

    #[test]
    fn a_chunk_overlaps_when_it_holds_a_kept_index() {
        let ranges = axis_ranges(
            &[pred("time", Operator::Eq, 101i64)],
            &schema(),
            &coordinates(),
        )
        .unwrap();
        let axes = ["time", "lat"];
        assert!(ranges.overlaps(&[0, 0], &[2, 3], &axes));
        assert!(!ranges.overlaps(&[2, 0], &[2, 3], &axes));
        // A chunk without the constrained axis overlaps.
        assert!(ranges.overlaps(&[0], &[3], &["lat"]));
    }
}
