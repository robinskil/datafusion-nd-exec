# Design: discover the coordinates of the axes

## Context

The regrid step needs to know which column holds the values of an axis. Today
a reader declares this with `AxisMeta::coordinate(column, order)` on each
axis. A reader that does not declare it gets no coordinate axes, so its grids
append and pad.

netCDF, CF, xarray and Zarr share one convention: a 1-D variable with the name
of its axis is the coordinate of that axis, such as `time(time)`. The repo can
apply that convention itself, so a reader only returns nd record batches with
named axes.

A second problem: an inner axis without a coordinate pads, and each chunk
starts at index 0. A chunk that a filter cuts on that axis then moves. For
example, `N_LEVELS [2, 3]` goes to 0 and 1.

Scope: the regrid step, and the coordinate columns of a grid `COPY TO`. The
sort order of `with_ordered_chunks()` stays declared, because the order of the
values is unknown at plan time.

## 1. Discovery by convention

The coordinate column of an axis `a` in a chunk:

| Axis metadata | Coordinate column |
|---|---|
| none | the column named `a`, by convention |
| `AxisMeta::coordinate(c, …)` | the column `c` |
| `AxisMeta::no_coordinate()` | none |

The column counts only when it lies on `a` alone and has one value per index.
Else the axis has no coordinate in that chunk.

The regrid step detects the order of the values, as today. The order in the
metadata is not used.

The error from the last fix stays for explicit metadata only: an inner axis
whose metadata names a column that the chunk does not hold fails. Without
metadata, the regrid step cannot know that the table has such a column, so
section 2 applies.

The output grid gets `AxisMeta::coordinate(column, order)` on each coordinate
axis, so a sink finds the coordinate column of each axis in the grid.

## 2. The original index on an inner axis without a coordinate

`NdRecordBatch::compact` cuts an `AxisIndices` selection to a smaller grid. The
record of a chunk keeps, per axis, the original index of each index of the
compacted chunk:

```rust
pub struct NdBatchRecord {
    // ...
    /// One entry per axis: the original index of each index, or `None` when
    /// the indices are `0..size`.
    pub origins: Vec<Option<UInt64Array>>,
}
```

- `axis_origins(selection, rank)` gives them: the index sets of an `AxisIndices` selection, and `None` for each other state, because `compact` keeps the full grid there.
- `NdBatchRecord::with_origins(origins)` sets them. `NdBatchRecord::of` sets `None` on each axis.

Rules for an axis without a coordinate:

- **Inner axis:** the position of each index is its original index. The output size is the largest position plus 1, over all chunks of the group. So a cut does not move cells.
- **Outer axis:** it appends the kept indices one after another, as today. The origins are not used.

## 3. Plan-time discovery and the grid coordinates rule

A grid sink needs the coordinate columns in the query when the grids of the
chunks differ. A user who writes `SELECT sst` does not list them. A logical
rule adds them.

**The axes in the table schema.** `logical_schema` copies the nd metadata of
each encoded field (its axis names and axis entries) into the field metadata
of the logical field, under the key `nd.array.metadata`. The table schema, the
`NdSourceExec` schema and the decoded batches all come from `logical_schema`,
so the three stay equal. A format that records its axes with
`nd_encoded_field_with_dims` needs no extra code. `nd_logical_metadata(field)`
reads the key.

**Discovery on a schema.** `schema_coordinates(schema)` gives the coordinate
column of each axis that a field names, with the rule of section 1: an axis
entry names a column or turns the rule off; without an entry, the field with
the name of the axis, on that axis alone, is the coordinate.

**Add the missing columns.** `add_grid_coordinates(plan)` works on a logical
plan before the optimizer:

1. It finds the coordinate columns of each `TableScan` source in the plan.
2. It finds the axes of the output columns from their field metadata. A computed column has no metadata, so it adds no axes.
3. It adds each coordinate column of those axes that the output does not hold. The column goes at the end of the top projection, under its own name. The rule passes through a `Sort`, a `Limit`, a `Filter` and a `SubqueryAlias` to that projection.
4. When it finds no projection, or the input of the projection does not hold the column, it changes nothing. The regrid step then falls back to section 2.

**The rule.** `NdGridCoordinatesRule` is an `AnalyzerRule`. It applies
`add_grid_coordinates` to the input of each `COPY TO` for which a predicate of
the host returns true, for example "the format is Zarr". `INSERT` does not
change, because the columns of the table are fixed. A host adds the rule with
`SessionStateBuilder::with_analyzer_rule`.

A plan-time check in `NdDataSinkExec` is not necessary: without metadata, a
missing coordinate falls back to the original index, and explicit metadata
still fails in the regrid step.

## Tests

- Discovery: a column with the name of its axis is the coordinate without metadata; `no_coordinate()` turns it off; `coordinate("t", …)` names another column; a column with the name of the axis but on two axes is not a coordinate.
- Origins: `axis_origins` for each selection state; an inner axis with origins `[2, 3]` gets the positions `[2, 3]`; the output size covers the largest position; the outer axis ignores origins.
- End to end, on a copy of `grid_table` without axis metadata:
  - `SELECT time, lat, lon, sst ... WHERE lat > 0` gives the grid `time=4, lat=1, lon=2`, with coordinates by discovery;
  - `SELECT time, lat, sst ... WHERE lat > 0 AND sst > 5` has no `lon` column, and the one cell of the first chunk lands at its own `lon` index.

- Plan-time discovery: `logical_schema` carries the metadata; `schema_coordinates` follows the convention and the axis entries.
- `add_grid_coordinates`: `SELECT sst ... WHERE lat > 0` gets `time`, `lat` and `lon`; `SELECT elev` gets `lat` and `lon` only; `SELECT *` does not change; the columns pass through `ORDER BY` and `LIMIT`.
- End to end: `COPY (SELECT sst FROM t WHERE lat > 0) TO ...` with the rule writes the grid `time=4, lat=1, lon=2` with its coordinates.

## Steps

1. Discovery in `NdBatchRecord::of`, with `AxisMeta` as an override.
2. Origins in `NdBatchRecord`, `axis_origins`, the inner-axis rule, and the regrid step.
3. The metadata in `logical_schema`, `nd_logical_metadata` and `schema_coordinates`.
4. `add_grid_coordinates` and `NdGridCoordinatesRule`.
5. README text.

Verification per step: `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test --workspace --all-features`.
