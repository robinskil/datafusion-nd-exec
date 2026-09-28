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
| `nd-arrow-array` | The nd data types: `NdArrowArray`, `NdRecordBatch`, `Dimensions` with axis metadata, the `Selection` lattice, `BroadcastMap`, and the `nd.array` Arrow extension type with its encoding. | `arrow` |
| `datafusion-nd-exec` | The nd plan nodes, the node registry, the `NdBoundaryRule`, the output terminals, the grid accumulator, axis ranges for readers, and a test harness (feature `test-utils`). It re-exports `nd-arrow-array` as `datafusion_nd_exec::array`. | `nd-arrow-array`, `datafusion` |

Both crates target DataFusion 53 and Arrow 58.

## How it works

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
| Another set of axes, for example one file without `lon` | Fine for queries. A grid sink fails, see below. |
| A column that a file does not have | The column is null for the rows of that file. |

Filters, projections, unions, limits, repartitions and coarsening all run per
chunk. Flat output is the rows of each chunk in sequence, like a `UNION ALL`
of each file made flat on its own grid. The nd output (`NdEncodeExec`) keeps
the grid of each chunk.

Two things need the same grid in every file:

- **Sort order:** a format must not call `with_ordered_chunks()` for a table whose files have another axis order.
- **One output grid:** the grid sink places all chunks into one grid, with the rules below.

## Write grids

### The terminals

- **`NdDataSinkExec`** writes the nd batches with an `NdDataSink`:

  ```rust
  #[async_trait]
  pub trait NdDataSink: DisplayAs + Debug + Send + Sync {
      fn as_any(&self) -> &dyn Any;
      fn schema(&self) -> &SchemaRef;
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

Both terminals first compact each chunk with `NdRecordBatch::compact`:

- A rectangle selection becomes a smaller dense grid.
- A ragged or cell-mask selection keeps its grid, and each full-grid column gets null at the dropped cells.

### The grid accumulator

A grid writer does not keep the data. It places each chunk in one output grid
with the `NdGridAccumulator`, writes the chunk at that place, and drops it.
The accumulator keeps only the axis state.

```rust
let mut accumulator = NdGridAccumulator::new()
    .with_mode("N_LEVELS", AxisMode::Pad);

while let Some(chunk) = data.next().await {
    let placement = accumulator.place(&chunk?)?;
    for axis in &placement.axes {
        // Grow the output axis to `axis.extent`.
        // Write `axis.new_coordinates`, if any, at `axis.offset`.
    }
    // Write `placement.batch`, the compacted chunk, at the offsets.
}
```

`place` returns one `AxisPlacement` per output axis:

| Field | Meaning |
|---|---|
| `offset` | the first output index of the chunk on the axis |
| `len` | the size of the chunk on the axis |
| `extent` | the size of the output axis after this chunk |
| `new_coordinates` | the coordinate values that the chunk adds at `offset`, or `None` |

### The axis rules

The first chunk sets the output axes and their order. Every chunk must have
the same set of axes, or `place` fails.

One axis is the **growth axis**: the outer axis, or the axis of
`with_growth_axis`. It is the axis along which files follow each other, such
as `time` for daily files or `N_PROF` for profile files. The other axes are
**inner axes**.

| Mode | Axis | Default | Rule |
|---|---|---|---|
| `Coordinate` | growth or inner, with a coordinate column | yes | **Match:** values that are an exact contiguous slice of the output go at that slice. **Append:** values that do not overlap go at the end. **Conflict:** a partial overlap fails, and the error shows both value ranges. |
| `Append` | growth only | on the growth axis without a coordinate | The chunk goes after all previous chunks. |
| `Pad` | inner only, without a coordinate | no | Aligned at 0. The axis grows to the largest chunk, and the writer fills the rest. |
| `Fixed` | inner only | on an inner axis without a coordinate | All chunks have the same size. |

### Seeds: sort up front

Without a seed, appended coordinate values follow the arrival order of the
chunks, and that order is not fixed when partitions merge.
`coordinate_order(axis)` reports whether an axis came out ascending,
descending or unordered.

When the host knows a coordinate before it reads data, for example from file
metadata, it gives the whole axis first:

```rust
let accumulator = NdGridAccumulator::new().with_coordinate("time", all_times);
```

- **Sorting:** the accumulator sorts the seed ascending, unless the seed is strictly descending, and removes duplicates.
- **Order-free placement:** each chunk goes at the offset of its values, whatever the arrival order. The output axis is sorted.
- **Known extent:** the writer knows the extent before the first chunk, so it can create fixed dimensions.
- **Strict:** a seeded axis rejects a value outside the seed. On an inner axis, this rejects a file with another grid.

A writer can take the seed back with `coordinate_seed(axis)`, and the name of
the coordinate column with `coordinate_column(axis)`.

### Examples

**Daily grid files** `time, lat, lon`, one day per file:

```rust
NdGridAccumulator::new()                               // time grows by coordinate,
    .with_coordinate("time", all_days)                 // sorted whatever the order;
    .with_coordinate("lat", lat).with_coordinate("lon", lon) // another grid fails.
```

**Profile files** `N_PROF, N_LEVELS`, where both sizes change per file. This
is the output of the example:

```text
write chunk at N_PROF 0..3 of 3, N_LEVELS 0..2 of 2
write chunk at N_PROF 3..5 of 5, N_LEVELS 0..3 of 3
write chunk at N_PROF 5..9 of 9, N_LEVELS 0..2 of 3
output grid: [("N_PROF", 9), ("N_LEVELS", 3)]
```

`N_PROF` grows by `Append`, and `N_LEVELS` pads to the largest file. The
query kept only the first levels, so the compacted chunks have fewer levels
than the files.

**One file read in split chunks**, for example `lat 0..90` then `lat 90..180`:
the second chunk appends to the unseeded `lat` axis, and together the chunks
fill the grid.

**Two resolutions**, for example `lat -15, 15` against `lat -30, 0, 30`:
unseeded, the values do not overlap, so they append, and the output is a
mostly empty union grid. Seed `lat` to make this fail, or resample first.

### Compared with xarray

The accumulator behaves like xarray's combine functions, but it streams: it
sees one chunk at a time and never goes back.

| xarray | Here |
|---|---|
| `concat(dim=...)`, `combine_nested(concat_dim=...)` | The growth axis with `Append`. |
| `combine_by_coords` (sorts by coordinate, then concatenates) | The growth axis with `Coordinate` and a seed. Without a seed, the order is the arrival order. |
| `join="exact"` on the other dimensions | A seeded inner axis. |
| `join="outer"` | An unseeded inner axis, for values that do not overlap. A partial overlap fails. |
| Another length on a dimension without an index | `Fixed` fails in the same way. `Pad` gives the CF incomplete ragged layout. |
| A repeated slice | An exact match goes to the same place, and the last chunk wins. |

`testing::MemoryGridSink` is a complete in-memory writer on top of the
accumulator. It is the reference for the tests and a model for a real writer.
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

- The grid sink needs the same set of axes in every chunk. There is no grouping of chunks per grid yet.
- The spatial box of `st_within` and `st_intersects` does not narrow axes yet.
- `NdRepartitionExec` uses unbounded channels, so memory is not limited when a consumer is slow.
- A filter adds a round-robin repartition, which loses the order, so `WHERE ... ORDER BY time` still sorts. The flat path does the same.
- `NdCoarsenExec` computes `Min` and `Max` in `Float64`, so an integer above 2^53 can lose precision. Its blocks do not cross chunks.

The design documents are in [docs/superpowers/specs](docs/superpowers/specs).

## License

AGPL-3.0. See [LICENSE](LICENSE).
