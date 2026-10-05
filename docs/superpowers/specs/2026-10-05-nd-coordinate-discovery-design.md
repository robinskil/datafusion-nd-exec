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

Scope: the regrid step only. The sort order of `with_ordered_chunks()` stays
declared, because the order of the values is unknown at plan time. Plan-time
discovery and the automatic addition of coordinate columns get their own
design.

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

## Tests

- Discovery: a column with the name of its axis is the coordinate without metadata; `no_coordinate()` turns it off; `coordinate("t", …)` names another column; a column with the name of the axis but on two axes is not a coordinate.
- Origins: `axis_origins` for each selection state; an inner axis with origins `[2, 3]` gets the positions `[2, 3]`; the output size covers the largest position; the outer axis ignores origins.
- End to end, on a copy of `grid_table` without axis metadata:
  - `SELECT time, lat, lon, sst ... WHERE lat > 0` gives the grid `time=4, lat=1, lon=2`, with coordinates by discovery;
  - `SELECT time, lat, sst ... WHERE lat > 0 AND sst > 5` has no `lon` column, and the one cell of the first chunk lands at its own `lon` index.

## Steps

1. Discovery in `NdBatchRecord::of`, with `AxisMeta` as an override.
2. Origins in `NdBatchRecord`, `axis_origins`, the inner-axis rule, and the regrid step.
3. README and spec text.

Verification per step: `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test --workspace --all-features`.
