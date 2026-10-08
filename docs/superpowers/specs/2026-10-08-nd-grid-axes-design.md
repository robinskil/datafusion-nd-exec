# Design: declared grid axes

## Context

A grid sink writes the result of a query as one grid, for example one netCDF
or Zarr file. Today the regrid step accepts any batch:

- An axis with a coordinate column joins the coordinate values of all batches.
- An axis without one appends (outer axis) or keeps its original index (inner axis).
- Batches with other axis sets go to other output grids.
- Axis metadata (`AxisMeta`) overrides the coordinate convention and gives the sort order of `with_ordered_chunks()`.

The rules for axes without a coordinate are complex, and they give an output
grid only by position, not by value. This design keeps the grid sink for real
grids only, and removes the axis metadata.

A flat format, such as Parquet, plans no nd scan, so it does not change.

## 1. Remove the axis metadata

- `AxisMeta`, `Dimension::meta`, `Dimension::with_meta`, `Dimension::order`, `Dimensions::with_axis_meta` and `AxisEntry` go away. A `Dimension` is a name and a size.
- `NdArrayMetadata` keeps `version` and `dims`. The `axes` entries go away.
- A coordinate is found only by convention: a column with the name of its axis, on that axis alone, with one value per index.
- `AxisOrder` (`Ascending`, `Descending`, `Unordered`) and `AxisOrder::detect` stay.

## 2. The format declares the grid axes

```rust
pub struct NdGridAxes { /* (axis name, AxisOrder), outer first */ }

impl NdGridAxes {
    pub fn new<S: Into<String>>(axes: impl IntoIterator<Item = (S, AxisOrder)>) -> Self;
    pub fn axes(&self) -> &[(String, AxisOrder)];
    pub fn order(&self, axis: &str) -> Option<AxisOrder>;
}
```

`NdGridAxes` lives in `nd-arrow-array`, module `axis`. A format declares it
on its scan:

```rust
let source = NdSourceExec::try_new(scan)?.with_grid_axes(axes);
```

The order of each axis comes from the format, for example from the header or
the coordinate values of the file.

## 3. The declaration goes up through the nd region

`NdExecutionPlan::grid_axes(&self) -> Option<Arc<NdGridAxes>>`, with default
`None`:

- `NdSourceExec` returns its declared axes.
- `NdFilterExec`, `NdProjectionExec`, `NdLimitExec`, `NdRepartitionExec`, `NdCoalescePartitionsExec`, `NdCoarsenExec` and `NdRegridExec` return the axes of their child.
- `NdUnionExec` returns them when all inputs declare equal axes, else `None`.

## 4. The sort order

`NdSourceExec::with_ordered_chunks()` reports the order of the outer declared
axes. The order stops at the first axis that is `Unordered`, or whose
coordinate column is not in the schema. Without declared axes, it reports no
order.

## 5. The grid sink

**Plan time.** `NdDataSinkExec::try_new` fails with a plan error when the sink
requires a grid and the input declares no grid axes. A sink without
`requires_grid` does not change.

**Run time.** `NdGridBuilder::new(axes)` takes the declared axes.
`NdGridBuilder::add` checks each batch at once, so a bad batch fails before
the rest of the input is read or spilled:

- Each axis of the batch is a declared axis, in the declared order.
- Each axis has its coordinate column. The error asks for the column.
- All batches have the same axes, so there is one output grid.
- A coordinate has no nulls and the data type of the first batch.

The output grid of each axis is the union of the values of all batches,
without duplicates. The sort is descending when the declared order is
`Descending`, else ascending. A value that repeats in one batch, and two
batches that write the same cell, fail as today.

**What goes away:** the append axis, the pad axis, the original indices
(`origins`, `axis_origins`), the groups by axis set, and the error for an
inner axis without its coordinate column.

The output grid holds no axis metadata. A sink finds the coordinate of an
axis with `NdOutputGrid::coordinate(axis)`, and the coordinate column has the
name of the axis.

## 6. Plan-time discovery

`schema_coordinates` and `NdGridCoordinatesRule` use the convention only.
They do not change otherwise.

## Tests

- `NdGridAxes`: the order of an axis; equality.
- Builder: a batch with an undeclared axis, with the axes in another order, without a coordinate column, or with other axes than the first batch fails at `add`; the declared order gives the sort.
- Plan: a grid sink over a scan without declared axes fails at plan time; a streaming sink does not.
- Propagation: the declared axes pass through a filter, a projection, a repartition, a coalesce and a union.
- Sort order: `with_ordered_chunks` reports the declared order, and stops at an `Unordered` axis.
- End to end: `grid_table` declares `time, lat, lon` and writes its grid; the profile table cannot go to a grid sink.

## Steps

1. Remove the axis metadata in `nd-arrow-array`, and add `NdGridAxes`.
2. Change `NdGridBuilder` to the declared axes.
3. `grid_axes` on the nodes, `with_grid_axes`, the sort order, and the plan-time check.
4. The regrid step, the test sink, the corpus, the tests and the example.
5. README.

Steps 1 to 4 change the API together, so they go in as one commit after the
whole workspace builds, with fmt, clippy and the tests. Step 5 is its own
commit.
