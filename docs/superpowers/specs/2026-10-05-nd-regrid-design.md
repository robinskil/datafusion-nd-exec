# Design: the regrid step for grid sinks

## Context

A grid sink writes the result of a query as one nd grid, for example a Zarr
store or a netCDF file. To write a chunk, the sink must know the full output
grid and the place of the chunk in it.

The `NdGridAccumulator` places each chunk on arrival. It needs seeds, axis
modes and a known arrival order to get a sorted grid. A planned statistics
design gives the seeds at plan time, but it needs a provider per format, a
merge and a grid hint on each nd node.

This design replaces both with a simpler model. The sink waits until it has
all the data. The data then gives the grid:

1. **Collect** all nd batches, and spill them to disk when the memory is full.
2. **Regrid**: build the output grid and the place of each batch from the collected axis values.
3. **Restream** the batches to the sink, each with its place in the grid.

So no plan-time statistics, seeds or axis modes are necessary.

Scope: this repo only. Beacon integration comes later.

## 1. Grid rules

**Before collection:** each batch goes through `NdRecordBatch::compact()`, so
it is a dense block with a `Full` selection. A cell that a filter removes is
null.

**Groups:** batches with the same axis names in the same order make one group.
Each group gets one output grid. The same axes in another order make another
group.

**Axis with a coordinate:** the axis metadata names a coordinate column, and
that column is in the batch.

- The output values are the union of the values of all batches, without duplicates.
- The sort is ascending. It is descending only when every batch is strictly descending.
- The sort and the duplicate check use the Arrow row format, so they work for each data type.
- A null coordinate value causes a failure.
- When the query does not select the coordinate column, the axis counts as an axis without a coordinate.

**Axis without a coordinate:**

- The outermost axis of the group appends: each batch gets the next range.
- Each inner axis pads to the largest size of the group. Each batch starts at offset 0.
- The append order is the input partition number, then the batch number in that partition. This order does not change between runs on the same input.

**Index maps:** each batch gets one `UInt64Array` per axis with its positions
in the output grid. On a coordinate axis, the positions do not have to be
contiguous. So batches with different grids fill a sparse output grid, and
each cell that no batch writes is null.

**Overlap:** two batches of one group fail when their positions intersect on
every axis of the grid. The error names the two batches and the cell ranges.
Columns on fewer axes do not count: two spatial tiles can write the same
`time` values.

**Coordinates in the output:** the output grid holds the coordinate array of
each axis. A sink writes the coordinates from the grid, not from the batches.

## 2. Types (`nd-arrow-array`, module `grid`)

```rust
/// One output grid.
pub struct NdOutputGrid {
    pub dims: Dimensions,
    /// One entry per axis of `dims`: the coordinate values, or `None`.
    pub coordinates: Vec<Option<ArrayRef>>,
}

/// The place of one batch in its output grid.
pub struct NdPlacement {
    pub grid: Arc<NdOutputGrid>,
    /// One entry per axis of the grid: the position of each batch index.
    pub indices: Vec<UInt64Array>,
}
```

- `NdRecordBatch` gets `placement: Option<Arc<NdPlacement>>`, with `with_placement` and `placement()`.
- A new batch has no placement. `with_selection` drops the placement, because a new selection breaks the dense block.
- All batches of one grid share one `Arc<NdOutputGrid>`.

`NdGridBuilder` applies the rules of section 1:

- `add(&mut self, record: NdBatchRecord) -> usize`: add the record of one batch, and get its number.
- `NdBatchRecord`: the partition, the batch number in the partition, the axis names and sizes, and the coordinate values per axis.
- `finish(self) -> Result<Vec<Arc<NdPlacement>>>`: the placement of each batch, by batch number. It fails on an overlap or a null coordinate.

The builder uses Arrow only, so a host can use it without DataFusion.

## 3. `NdRegridExec` (`datafusion-nd-exec`, `sink/regrid_exec.rs`)

An nd node with one output partition. It sits directly below the sink:

```
NdDataSinkExec: sink=ZarrSink
  NdRegridExec: grids=1
    NdFilterExec / NdSourceExec / ...
```

`execute_nd(0)` runs three phases.

**Collect:**

- Read all input partitions at the same time. Call `compact()` on each batch.
- Give the record of each batch to the `NdGridBuilder`. The records stay in memory and count against the reservation.
- Encode each batch with `encode_nd_record_batch` and hold it in a `MemoryReservation`.
- When the reservation cannot grow, spill all held batches to one IPC file through DataFusion's `SpillManager`, then free the memory.

**Regrid:**

- Call `NdGridBuilder::finish`. The overlap check runs here, before the node reads data back.

**Restream:**

- Read the spill files in order, then the batches still in memory.
- Decode each batch and set its placement.
- The restream order is the collection order. Each batch carries its exact place, so the sink does not need a sorted order.

**Metrics:** DataFusion's spill metrics (spill count, bytes and rows), plus the
number of grids and output cells.

**Errors:** an overlap, a null coordinate, or a spill without a `DiskManager`
fails the query with a clear message.

## 4. Sink side

- `NdDataSink::requires_grid(&self) -> bool`, with default `false`.
- `NdDataSinkExec::try_new` adds `NdRegridExec` above its child when the sink requires a grid. The DataSink sinker does not change.
- `write_all(data, context)` does not change. A grid sink reads `batch.placement()` on each batch. The first batch of each grid gives the grid, so the sink can write its metadata before the data.
- A sink that does not require a grid gets no `NdRegridExec` and streams at once.
- An empty result gives an empty stream, and the sink writes nothing.

Only `NdRegridExec` sets a placement. A node that makes new batches drops it,
so `NdRegridExec` must be the direct child of the sink.

## 5. Removal

- `sink/accumulator.rs`: `NdGridAccumulator`, `AxisMode`, `AxisPlacement`, `Placement`, the seeds and `coordinate_order`.
- `2026-09-28-nd-statistics-design.md`: this design replaces it.
- `MemoryGridSink` loses `with_mode`, `with_growth_axis` and `with_coordinate`.

The older phase specs stay as they are.

## 6. Test support

- `MemoryGridSink` sets `requires_grid` to true.
- It builds the dense grid from the placements, and takes the coordinates from `NdOutputGrid`.
- `examples/profiles.rs` prints the grid and the place of each batch.

## Tests

- `NdGridBuilder`:
  - the union of coordinates, ascending and descending;
  - the append order on the outer axis, and the pad on inner axes;
  - groups by axis set;
  - a gap gives null cells;
  - an overlap error and a null coordinate error;
  - an axis whose coordinate column the query does not select appends.
- Placement: `with_selection` drops it.
- Spill: a small `FairSpillPool` forces spills. The output is the same as without spills, and the spill metrics are above 0.
- Plan: `EXPLAIN` shows `NdRegridExec` below `NdDataSinkExec` only when the sink requires a grid.
- End to end:
  - grid chunks that arrive in reverse order give a sorted output grid, with no seeds;
  - profile files with different `N_LEVELS` give a padded grid.

## Steps

1. Write this design and remove the statistics design.
2. Add `NdOutputGrid`, `NdPlacement` and the placement field.
3. Add `NdGridBuilder` with its tests.
4. Add `NdRegridExec`: collect, spill and restream.
5. Add `requires_grid` and the change in `NdDataSinkExec`. Change `MemoryGridSink` and the sink tests, and remove the accumulator.
6. Change the example and the README.

Each step is one commit with its tests. Verification per step: `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test --workspace --all-features`.

## Later

- Output blocks that match the storage chunks of the sink, so each Zarr chunk is written once.
- An option for "the last batch wins" on an overlap.
- One empty batch with the grid for an empty result, if a host must write an empty grid.
