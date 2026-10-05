# datafusion-nd-exec

Grid-native DataFusion execution for n-dimensional data, such as netCDF,
Zarr, HDF5 and GeoTIFF files. The work follows
[maris-development/beacon#442](https://github.com/maris-development/beacon/issues/442).

## The problem

A table engine works on flat rows. To make a grid flat, the engine repeats
each coordinate value once for each cell of the other axes:

```text
Grid form (as stored):              Flat form (after broadcast):

  time: [t0 t1]        2 values       time lat lon  sst
  lat:  [y0 y1 y2]     3 values       t0   y0  x0   s0
  lon:  [x0 x1 x2 x3]  4 values       t0   y0  x1   s1
  sst:  2x3x4         24 values       ...
  --------------------------          24 rows x 4 columns
  total:              33 values       total: 96 values
```

This repo keeps the data in grid form through the plan. Filters, projections,
unions, limits and repartitions run on the grid. The data becomes flat rows
only where an operator needs flat rows, or not at all when the output is a
grid too.

## Crates

| Crate | Content | Depends on |
|---|---|---|
| `nd-arrow-array` | The nd data types: `NdArrowArray`, `NdRecordBatch`, `Dimensions` with axis metadata, the `Selection` lattice, `BroadcastMap`, the `nd.array` Arrow extension type with its encoding, and the output grids: `NdOutputGrid`, `NdPlacement` and `NdGridBuilder`. | `arrow` |
| `datafusion-nd-exec` | The nd plan nodes, the node registry, the `NdBoundaryRule`, the output terminals, the regrid step, axis ranges for readers, and a test harness (feature `test-utils`). It re-exports `nd-arrow-array` as `datafusion_nd_exec::array`. | `nd-arrow-array`, `datafusion` |

Both crates target DataFusion 53 and Arrow 58.

## How it works

### The path of a query

```text
files -> DataSourceExec -> NdSourceExec -> nd nodes ------------> NdBroadcastExec -> flat nodes -> rows
         one encoded row   nd batches      filter, projection,   the boundary       aggregate,
         per chunk                         limit, union, ...                        sort, join
                                              |
                                              +--> NdEncodeExec ----------------------> encoded chunks
                                              |
                                              +--> NdDataSinkExec -> NdDataSink         streams the chunks
                                              |
                                              +--> NdDataSinkExec -> NdRegridExec ----> a grid sink
                                                                     (one output grid)
```

1. A format reads its files in chunks. Each chunk goes through DataFusion as one row of Arrow data.
2. `NdSourceExec` decodes each row into an nd batch: a grid with its columns, without repeated values.
3. The nd nodes run on the grid. They change the selection of a chunk, not its data.
4. The data leaves the nd region in one of three ways:
   - `NdBroadcastExec` makes flat rows, where a flat operator needs them.
   - `NdEncodeExec` sends the chunks to a client as they are.
   - `NdDataSinkExec` writes the chunks with an `NdDataSink`. A grid sink gets all chunks with their place in one output grid.

### The nd batch

An `NdRecordBatch` is one chunk of a grid:

- **A target grid:** named axes in C order, for example `time=2, lat=3, lon=4`.
- **Columns on their own axes:** each column is an `NdArrowArray`, a flat Arrow array plus its own axes. A coordinate `lat` lives on `lat` only, a data variable `sst` on `time, lat, lon`, and an attribute on no axis. Nothing is repeated.
- **A selection:** the cells of the grid that the chunk keeps.

The selection has four states, from coarse to fine:

| State | Keeps | Example |
|---|---|---|
| `Full` | every cell | a scan |
| `AxisIndices` | an index set per axis, so a rectangle | `WHERE lat > 0` |
| `Ragged` | a prefix of the inner axis for each outer cell | `WHERE PRES < 100` on profiles |
| `CellMask` | an explicit cell set | `WHERE sst > 5` |

An operator returns the coarsest state that holds its result. When the data
becomes flat, one gather per column applies the broadcast and the selection
together, so the cells that a filter drops are never built.

### The nd region and the boundary

A format plans its scan in this shape:

```text
NdBroadcastExec          <- the boundary: flat rows above, nd batches below
  NdSourceExec           <- decodes the encoded chunks into nd batches
    DataSourceExec       <- the file scan: one encoded row per chunk
```

The scan passes chunks through DataFusion as ordinary Arrow data: each column
is a `Struct { values, dim_sizes, dim_names }` with the `nd.array` extension
type. `NdSourceExec` decodes each row into one nd batch.

The part of the plan below `NdBroadcastExec` is the **nd region**. The
`NdBoundaryRule` grows it. It runs after the DataFusion physical rules and
looks at each node above a boundary. When the node can run on grids, the rule
puts its nd form below the boundary and looks at the next node up:

```text
Before:                                After:
ProjectionExec [lat * 2]               NdBroadcastExec: region=[NdProjectionExec, NdFilterExec, NdRepartitionExec, NdSourceExec]
  FilterExec [lon <> 15]                 NdProjectionExec [lat * 2]
    RepartitionExec [round robin]          NdFilterExec [lon <> 15]
      NdBroadcastExec                        NdRepartitionExec [round robin]
        NdSourceExec                           NdSourceExec
          DataSourceExec                         DataSourceExec
```

`EXPLAIN` shows the region on the boundary line.

### The plan nodes

| Node | Replaces | What it does |
|---|---|---|
| `NdSourceExec` | the scan of a format | Decodes encoded chunks. With `with_ordered_chunks()`, it reports the sort order of the outer coordinate axes, so `ORDER BY time` needs no sort. |
| `NdFilterExec` | `FilterExec` | Evaluates each condition on the axes of its columns only. A condition on one axis gives a rectangle and runs first. The other conditions read their mask only at the cells that are still kept. A volatile condition stays above in a flat `FilterExec`. |
| `NdProjectionExec` | `ProjectionExec` | Evaluates each element-wise expression on its own axes: `lat * 2` runs on the 3 values of `lat`, not on 24 cells. |
| `NdLimitExec` | `LocalLimitExec`, `GlobalLimitExec` | Cuts the kept cells in stream order. A cut on outer-axis borders stays a rectangle. |
| `NdUnionExec` | `UnionExec` | Lists the partitions of all inputs. Each chunk keeps its own grid. |
| `NdRepartitionExec` | round-robin `RepartitionExec` | Sends whole chunks round robin to more partitions, so the broadcast runs in parallel. |
| `NdCoalescePartitionsExec` | `CoalescePartitionsExec` | Merges all partitions into one nd stream. |
| `NdCoarsenExec` | none, a host plans it | Reduces blocks of cells along axes, with mean, min, max or first, for example for map tiles. |
| `NdAxisReorderExec`, `NdEmptyExec` | none | Building blocks for sinks. |
| `NdBroadcastExec` | the boundary | Makes flat rows. |
| `NdDataSinkExec` | `DataSinkExec` | A terminal: writes the grid with an `NdDataSink`. |
| `NdRegridExec` | none, the sink adds it | Collects all chunks with spill, builds the output grids, and gives each chunk its place. |
| `NdEncodeExec` | the boundary at the root | A terminal: yields the encoded chunks for Arrow IPC or Flight. |

### The row count

The grid of a scan comes from the axes of the **selected** columns. A filter
column counts as selected. For a table with `sst` on `time, lat, lon` and
`elev` on `lat, lon`:

| Query | Grid per chunk |
|---|---|
| `SELECT sst FROM t` | `time, lat, lon` |
| `SELECT elev FROM t` | `lat, lon` |
| `SELECT elev FROM t WHERE time > 100` | `time, lat, lon` |
| `SELECT count(*) FROM t` | every cell of the full grid of the chunk |

A column that a file does not have decodes as a null on no axis, so it is null
in every row of that file.

### Follow one query

The test table `grid_table` has two chunks. Each chunk has the grid
`time=2, lat=3, lon=2`, with `sst` on all three axes. The query:

```sql
SELECT time, lat, lon, sst FROM t WHERE lat > 0
```

1. **Plan.** DataFusion plans a `FilterExec` over a round-robin `RepartitionExec` over the scan. The `NdBoundaryRule` moves both below the boundary. The scan reads only the selected columns, so the plan needs no projection node:

   ```text
   NdBroadcastExec: region=[NdFilterExec, NdRepartitionExec, NdSourceExec]
     NdFilterExec: predicate=[lat@1 > 0]
       NdRepartitionExec: partitioning=RoundRobinBatch(24), input_partitions=2
         NdSourceExec
           DataSourceExec: partitions=2, partition_sizes=[1, 1]
   ```

2. **Scan.** Each partition yields one encoded row. `NdSourceExec` decodes it into an nd batch with the grid `time=2, lat=3, lon=2` and a `Full` selection. The columns hold 2, 3, 2 and 12 values.
3. **Repartition.** `NdRepartitionExec` sends each chunk whole to one output partition.
4. **Filter.** `lat > 0` reads the `lat` column only, so it compares 3 values, not 12 cells. The result is an `AxisIndices` selection with `lat = [2]`: a rectangle of 2 x 1 x 2 cells.
5. **Boundary.** `NdBroadcastExec` gathers each column at the 4 kept cells of each chunk. The query gives 8 rows, and no other cell is built.

To write the same result as one grid, a grid sink replaces the boundary:

```text
NdDataSinkExec: sink=MemoryGridSink
  NdRegridExec
    NdFilterExec: predicate=[lat@1 > 0]
      ...
```

6. **Compact.** `NdRegridExec` compacts each chunk. The rectangle becomes a dense block of `time=2, lat=1, lon=2`.
7. **Regrid.** The union of the coordinates gives `time = 100, 101, 102, 103`, `lat = 30` and `lon = 5, 15`, so the output grid is `time=4, lat=1, lon=2`. The first chunk goes at `time 0..=1`, the second at `time 2..=3`.
8. **Write.** The sink gets both chunks with their placements and writes 8 cells. Two `sst` cells are null, because the test table has a null at every seventh cell.

## Use

### Enable the nd pipeline

```rust
use std::sync::Arc;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use datafusion_nd_exec::{NdNodeRegistry, NdSessionStateBuilderExt};

let state = SessionStateBuilder::new()
    .with_default_features()
    .with_nd_pipeline(Arc::new(NdNodeRegistry::new()))
    .build();
let ctx = SessionContext::new_with_state(state);
```

`with_nd_pipeline` stores the registry in the session config and adds the
`NdBoundaryRule` after the default physical rules. Call it after
`with_config`, because `with_config` replaces the session config.

### Plug in a format

A format reads chunks, encodes them, and plans the scan with the nd shape. The
complete, runnable version is
[crates/datafusion-nd-exec/examples/profiles.rs](crates/datafusion-nd-exec/examples/profiles.rs).

1. **Build one nd batch per chunk.** Give each axis its metadata: a
   coordinate column and its order for a grid axis, no coordinate for a
   profile axis. `AxisOrder::detect` finds the order of a coordinate array.

   ```rust
   let time = Dimension::new("time", 2)
       .with_meta(Some(AxisMeta::coordinate("time", AxisOrder::Ascending)));
   let n_prof = Dimension::new("N_PROF", 3).with_meta(Some(AxisMeta::no_coordinate()));
   ```

2. **Encode each chunk** with `encode_nd_record_batch`. Each encoded row
   carries one chunk. All chunks of a scan must carry the same encoded schema.
   For a sort order, build the encoded schema with `nd_encoded_field_with_dims`,
   so the fields record the axes of each column.

3. **Plan the scan:**

   ```rust
   let source = Arc::new(NdSourceExec::try_new(file_scan)?);
   let registry = NdNodeRegistry::from_session_config(state.config());
   let plan = NdBroadcastExec::try_new_with_registry(source, registry)?;
   ```

   Call `NdSourceExec::with_ordered_chunks()` only when each partition yields
   its chunks split along the outer axis alone, in the order of that axis.

4. **Prune chunks.** The file source gets the query filters as pruning hints.
   `axis_ranges(filters, schema, coordinates)` turns the conditions on one
   coordinate column into index ranges per axis. `AxisRanges::overlaps`
   then tells if a chunk holds a kept index, so the reader skips the others.

### Add your own nodes

The `NdNodeRegistry` connects crates that do not know each other:

| Hook | Use |
|---|---|
| `with_probe(probe_for::<MyNdExec>())` | Make a node of your crate nd-aware, so other nd nodes can read its nd batches. |
| `with_sinker(Arc::new(MySinker))` | Tell the boundary rule how a flat node becomes an nd node (`Sunk::Below`) or a terminal (`Sunk::Terminal`). |
| `with_sink_factory(Arc::new(MyFactory))` | Map a flat `DataSink` of your host to an `NdDataSink`. |

## Read many files with different grids

Every chunk carries its own grid, so the files of one table can differ:

| Difference between files | Result |
|---|---|
| Another size on an axis, for example `N_PROF` 3, 2 and 4 | Fine. Each chunk runs on its own grid. |
| Other coordinate values, for example other days or other regions | Fine. |
| Another set of axes, for example one file without `lon` | Fine. A grid sink gets one output grid per axis set. |
| A column that a file does not have | The column is null for the rows of that file. |

Filters, projections, unions, limits, repartitions and coarsening all run per
chunk. Flat output is the rows of each chunk in sequence, like a `UNION ALL`
of each file made flat on its own grid. The nd output (`NdEncodeExec`) keeps
the grid of each chunk.

A sort order needs the same grid in every file: a format must not call
`with_ordered_chunks()` for a table whose files have another axis order.

## Write grids

### The terminals

- **`NdDataSinkExec`** writes the nd batches with an `NdDataSink`:

  ```rust
  #[async_trait]
  pub trait NdDataSink: DisplayAs + Debug + Send + Sync {
      fn as_any(&self) -> &dyn Any;
      fn schema(&self) -> &SchemaRef;
      fn requires_grid(&self) -> bool { false }
      async fn write_all(&self, data: SendableNdBatchStream, context: &Arc<TaskContext>) -> Result<u64>;
  }
  ```

  When a host plans `COPY TO` or `INSERT` as a `DataSinkExec`, an
  `NdSinkFactory` in the registry maps the flat sink to an nd sink. The
  boundary rule then replaces the `DataSinkExec` and the boundary with an
  `NdDataSinkExec`, so no flat row is built.

- **`NdEncodeExec`** yields one `nd.array`-encoded row per chunk. A client
  decodes each column to its values, axis names and axis sizes.
  `nd_output_plan(&plan)` swaps the boundary at the root for this node. It
  returns `None` when flat operators, for example an aggregate, run above the
  nd region.

`NdEncodeExec` and `NdRegridExec` compact each chunk with
`NdRecordBatch::compact`:

- A rectangle selection becomes a smaller dense grid.
- A ragged or cell-mask selection keeps its grid, and each full-grid column gets null at the dropped cells.

A sink without `requires_grid` gets each chunk as it is, with its selection.
It calls `compact()` or `materialize()` itself.

### The regrid step

A grid writer, such as a Zarr or netCDF writer, needs the full output grid
and the place of each chunk in it. A sink states this need with
`requires_grid() == true`. `NdDataSinkExec` then reads its input through an
`NdRegridExec`:

```text
NdDataSinkExec: sink=PrintingGridSink
  NdRegridExec
    NdFilterExec: predicate=[PRES@1 < 25]
      ...
```

`NdRegridExec` has one output partition and runs in three phases:

1. **Collect.** It reads all input partitions and compacts each chunk. It keeps a small record of each chunk: its axes, sizes and coordinate values. The chunks stay in a `MemoryReservation`. When the memory pool is full, the node spills the chunks to IPC files through the `SpillManager` of DataFusion.
2. **Regrid.** After the last chunk, it builds the output grids and the place of each chunk from the records alone. It also checks for overlaps here.
3. **Restream.** It reads the chunks back in collection order and gives each chunk its placement.

A sink that does not require a grid gets no `NdRegridExec` and streams at once.

### Read a placement

Each chunk of a grid sink has `placement()`:

```rust
while let Some(chunk) = data.next().await {
    let chunk = chunk?;
    let placement = chunk.placement().expect("a grid sink gets placed chunks");
    let grid = placement.grid();          // shared by all chunks of one grid
    // On the first chunk of a grid: write `grid.dims()` and `grid.coordinates()`.
    for (dim, positions) in grid.dims().iter().zip(placement.indices()) {
        // `positions.value(i)` is the grid index of index `i` of the chunk on `dim`.
    }
    // Write the chunk columns at those positions.
}
```

- `NdOutputGrid` holds the axes of the grid and the coordinate values of each axis. A writer takes the coordinates from the grid, not from the chunks.
- `NdPlacement` holds one `UInt64Array` per grid axis. On a coordinate axis, the positions do not have to be contiguous.
- The first chunk of each grid gives the grid, so a writer can write its metadata before the data.
- An empty result gives no chunks, and the sink writes nothing.

`NdGridBuilder` in `nd-arrow-array` holds the rules below. It uses Arrow only,
so a host can use it without DataFusion.

### Write a grid sink

1. **Implement `NdDataSink`** and return `true` from `requires_grid`.
2. **Open each output grid once.** Chunks of one grid share one `Arc<NdOutputGrid>`, so compare with `Arc::ptr_eq`. For a new grid, create the output arrays with the shape `grid.dims().shape()`, and write the coordinate arrays from `grid.coordinates()`.
3. **Write each column on its own axes.** A column lives on a subset of the grid axes. Take the positions of those axes from `placement.axis_indices(name)`.
   - When the positions rise by 1 on every axis, the chunk is one block. Write it at the first position of each axis, as one hyperslab or one region.
   - Otherwise, write it in parts. Each part is a run of positions that rise by 1.
   - A column on no axis, for example a null for a file that lacks the column, has no cells to write. The output keeps its fill value there.
4. **Connect the sink.** For `COPY TO` or `INSERT`, map the flat `DataSink` of your host to your sink with an `NdSinkFactory`, and add the factory to the registry:

   ```rust
   let registry = Arc::new(NdNodeRegistry::new().with_sink_factory(Arc::new(MyFactory)));
   let state = SessionStateBuilder::new()
       .with_default_features()
       .with_nd_pipeline(registry)
       .build();
   ```

   Or plan the write yourself: `NdDataSinkExec::try_new(nd_child_of_the_boundary, sink)`.
5. **Set the memory.** `NdRegridExec` holds the chunks until the input ends, and spills when the memory pool is full. Give the session a memory limit and a disk manager, for example `RuntimeEnvBuilder::new().with_memory_limit(4 << 30, 1.0)`. The default disk manager writes the spill files to the temp directory of the OS.

`testing::MemoryGridSink` follows these steps in memory, and
[examples/profiles.rs](crates/datafusion-nd-exec/examples/profiles.rs) has a
sink that prints each step.

### The grid rules

**Groups:** chunks with the same axis names in the same order make one group.
Each group gets one output grid.

**Axis with a coordinate:** the axis metadata names a coordinate column, and
the query selects that column.

- The output values are the union of the values of all chunks, without duplicates.
- The sort is ascending. It is descending only when every chunk with two or more values is strictly descending.
- A null value, a value that repeats in one chunk, or a coordinate in some chunks only causes a failure.

**Axis without a coordinate:**

- The outer axis appends: each chunk gets the next range. The order is the input partition, then the chunk number in that partition, so it does not change between runs. An outer axis whose coordinate column the query does not select also appends.
- Each inner axis pads to the largest size, with each chunk at index 0.

**Select the coordinate column of each inner axis.** An inner axis whose
metadata names a coordinate column, but whose chunks do not hold it, causes a
failure. A pad puts a chunk that a filter cuts on that axis at index 0, which
is not its true place, so the error asks for the column instead.

**Sparse grids:** each cell that no chunk writes is null. So chunks with
other grids, for example two regions, fill one union grid.

**Overlap:** two chunks fail when they write the same cells. The error names
both chunks and the cell ranges. Columns on fewer axes do not count, so two
spatial tiles can write the same `time` values.

### Examples

**Many Zarr stores into one store.** A collection holds many Zarr stores on
`time, lat, lon`. All stores have the same `lat` and `lon`, but each store has
other days. The query:

```sql
COPY (SELECT time, lat, lon, sst FROM stores WHERE time >= '2020-01-01') TO 'all.zarr'
```

1. The Zarr reader yields chunks with axis metadata: `time`, `lat` and `lon` each name their coordinate column. It can skip the stores and chunks before 2020 with `axis_ranges`.
2. The `NdSinkFactory` of the host maps the `COPY TO` sink to a Zarr grid sink. The plan is `NdDataSinkExec` over `NdRegridExec` over the nd region.
3. `NdRegridExec` collects all chunks and spills them when the memory is full. Only the records of the chunks stay in memory: their axis sizes and coordinate values.
4. The regrid joins the days of all stores into one sorted `time` axis. `lat` and `lon` are equal in all stores, so the union keeps them as they are. Two stores with the same day fail with an overlap error.
5. The sink creates `all.zarr` with the shape `[days, lat, lon]` and writes `time`, `lat` and `lon` from the grid. Then it writes each chunk at its `time` positions.

The arrival order of the stores does not matter, and no metadata is necessary
before the query runs.

**Daily grid files** `time, lat, lon`, one day per file, in any order: the
output `time` axis holds all days in ascending order. `lat` and `lon` are the
same in each file, so they stay the same.

**Profile files** `N_PROF, N_LEVELS`, where both sizes change per file. This
is the output of the example:

```text
output grid: N_PROF=9, N_LEVELS=3
write chunk at N_PROF 0..=2, N_LEVELS 0..=1
write chunk at N_PROF 3..=4, N_LEVELS 0..=2
write chunk at N_PROF 5..=8, N_LEVELS 0..=1
```

`N_PROF` appends, and `N_LEVELS` pads to the largest chunk. The query keeps
only the first levels, so most compacted chunks have fewer levels than their
files. The second file has a profile with one level only, so its filter is not
a rectangle, and its chunk keeps all 3 levels with nulls.

**One file read in split chunks**, for example `lat 0..90` then `lat 90..180`:
the union of the `lat` values gives the full axis, and together the chunks
fill the grid.

**Two resolutions**, for example `lat -15, 15` against `lat -30, 0, 30`: the
union gives `lat -30, -15, 0, 15, 30`, and the output is a mostly empty grid.
Resample first to prevent this.

### Compared with xarray

| xarray | Here |
|---|---|
| `concat(dim=...)`, `combine_nested(concat_dim=...)` | The outer axis without a coordinate. |
| `combine_by_coords` (sorts by coordinate, then concatenates) | An axis with a coordinate, in any arrival order. |
| `join="outer"` on the other dimensions | The union of the coordinate values, with null cells. |
| `compat="no_conflicts"` | An overlap fails, with no check of the values. |
| Another length on a dimension without an index | The axis pads, which gives the CF incomplete ragged layout. |

`testing::MemoryGridSink` is a complete in-memory writer on top of the
placements. It is the reference for the tests and a model for a real writer.
A real writer must take the axes of each column from the chunk where the
column has the most axes, because a file without the column gives it on no
axis.

## Test

```bash
cargo test --workspace --all-features
```

The harness in `datafusion_nd_exec::testing` (feature `test-utils`) runs each
query twice: once with the nd rules, and once without them, where each scan
becomes flat rows at once. Both results must be the same. It has two corpora:
a grid with coordinate axes, and profile files with other sizes per file.

Run the example:

```bash
cargo run -p datafusion-nd-exec --example profiles
```

## Limits

- A grid sink writes nothing until its input ends. Spilled chunks go to disk once and come back once. The input must be bounded.
- `NdRegridExec` sends whole chunks, not blocks that match the storage chunks of the output. A Zarr sink must read, change and write a storage chunk again when a chunk covers only part of it.
- A grid sink needs the coordinate column of each inner axis in the query. The plan does not add a missing coordinate column by itself.
- The spatial box of `st_within` and `st_intersects` does not narrow axes yet.
- `NdRepartitionExec` uses unbounded channels, so memory is not limited when a consumer is slow.
- A filter adds a round-robin repartition, which loses the order, so `WHERE ... ORDER BY time` still sorts. The flat path does the same.
- `NdCoarsenExec` computes `Min` and `Max` in `Float64`, so an integer above 2^53 can lose precision. Its blocks do not cross chunks.

The design documents are in [docs/superpowers/specs](docs/superpowers/specs).

## License

AGPL-3.0. See [LICENSE](LICENSE).
