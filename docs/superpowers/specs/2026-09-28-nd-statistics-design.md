# Design: nd statistics and the grid hint

## Context

A grid sink places chunks with the `NdGridAccumulator`. It works best with
seeds: the full coordinate of each axis before the first chunk. Then the
output axis is sorted whatever the chunk order, the output shape is known up
front, and an inner axis rejects a dataset with another grid.

Today a host collects the seeds by hand and passes them to its sink. This
design adds a generic way:

1. A format describes the grid of each dataset from metadata only: the **nd statistics**.
2. This repo merges the statistics of all datasets of a table at plan time.
3. The merged result rides on the scan as a **grid hint** to the sink, which seeds its accumulator.

The same statistics also give pruning, the dataset order, a grid check, the
output extents and row counts.

A dataset is one file or one store. Scope: generic parts only. A format
implements the provider; this repo defines the types, the merge, the hint and
their use. Persistence of statistics stays with the host.

## 1. Statistics types (`nd-arrow-array`, module `statistics`)

The values are Arrow arrays, so the types have no DataFusion dependency.

```rust
pub struct NdDatasetStatistics {
    pub axes: Vec<NdAxisStatistics>,
    pub variables: Vec<NdVariableStatistics>,
}

pub struct NdAxisStatistics {
    pub name: String,
    pub size: usize,
    /// `None` for an axis without a coordinate, such as `N_PROF`.
    pub coordinate: Option<NdCoordinateStatistics>,
}

pub struct NdCoordinateStatistics {
    pub column: String,
    pub order: AxisOrder,
    pub min: ArrayRef,          // one value
    pub max: ArrayRef,          // one value
    pub values: CoordinateValues,
    pub fingerprint: u64,
}

pub enum CoordinateValues {
    /// All values: a small or irregular axis.
    Explicit(ArrayRef),
    /// `start + i * step` for `i` in `0..count`, in the physical unit of the
    /// type: `i64` for integer and timestamp types, `f64` for float types.
    Regular { start: ArrayRef, step: RegularStep, count: usize },
    /// Only min, max and the fingerprint are known.
    Summary,
}

pub enum RegularStep { Int(i64), Float(f64) }

pub struct NdVariableStatistics {
    pub name: String,
    /// The axes of the variable, in stored order.
    pub dims: Vec<String>,
    /// The storage chunk shape, when the format has one.
    pub chunk_shape: Option<Vec<usize>>,
}
```

Helpers:

- `NdCoordinateStatistics::from_values(column, &values)`: compute the order with `AxisOrder::detect`, the min and max, and the fingerprint. The values are `Regular` when the steps are equal, and `Explicit` else. A float axis counts as regular within a relative tolerance of `1e-9` of the step.
- `CoordinateValues::materialize(&data_type) -> Option<ArrayRef>`: all values, or `None` for `Summary`.
- `fingerprint(&values) -> u64`: FNV-1a over the Arrow row format of the values. The algorithm is fixed, so cached statistics stay valid between releases.

Calendar steps, such as months, are `Explicit`.

## 2. The provider trait (`datafusion-nd-exec`, module `statistics`)

```rust
#[async_trait]
pub trait NdStatisticsProvider: Send + Sync {
    /// The grid of the dataset `object` in `store`, from metadata only.
    async fn dataset_statistics(
        &self,
        state: &dyn Session,
        store: &Arc<dyn ObjectStore>,
        object: &ObjectMeta,
    ) -> Result<NdDatasetStatistics>;
}

/// One dataset of a table: its object and its statistics.
pub struct NdDataset {
    pub object: ObjectMeta,
    pub statistics: NdDatasetStatistics,
}
```

The signature follows DataFusion's `FileFormat::infer_stats`, and
`ObjectStore` and `ObjectMeta` come from the `object_store` crate that
DataFusion re-exports. So:

- the provider reads the metadata through the store of the table, with the session config;
- the merge returns the dataset order as `ObjectMeta`s, so a format builds its `PartitionedFile`s from them at once;
- a host can key a cache on the location with `e_tag` or `last_modified`.

The statistics types stay in `nd-arrow-array` without a location, because that
crate depends on Arrow only. `NdDataset` pairs them with their object.

A format implements the provider. For Zarr: the array metadata
(`dimension_names`, `shape`, `chunks`) and the 1-D coordinate arrays. For
netCDF: the header and the coordinate variables. A host caches the results.

A Zarr store is a prefix, not one object. Its `ObjectMeta` is the root
metadata object (`zarr.json`, or `.zmetadata` for consolidated v2 metadata).
A rewrite of the coordinate chunks alone does not change that object, so a
cache keyed on its `e_tag` misses that change. A host that rewrites
coordinates must drop the cache entry.

## 3. Merge (`NdTableStatistics`)

`NdTableStatistics::merge(datasets: Vec<NdDataset>, growth_axis: Option<&str>)`
combines the datasets of a table:

- **Reference grid:** the axis set of the first dataset. The growth axis is the named axis, else the first axis of the widest variable.
- **Seeds:** per axis, the union of the values of all datasets, sorted ascending unless every dataset is strictly descending, with duplicates removed. An axis with a `Summary` dataset gets no seed.
- **Extents:** the seed length per axis, or `None` without a seed.
- **Dataset order:** the `ObjectMeta`s sorted by the minimum of the growth-axis coordinate. Without a coordinate on the growth axis, the input order.
- **Issues:** a list, not errors, so the host decides:
  - `AxisSetDiffers { location, axes }`: a dataset with another axis set. `location` is the object path.
  - `InnerGridDiffers { location, axis }`: another fingerprint on an inner axis.
  - `GrowthOverlap { first, second }`: two datasets whose growth-axis ranges overlap, unless their values are equal.
  - `NoSeed { axis }`: a coordinate axis with a `Summary` dataset.
- `ordered_chunks_safe()`: true when no two datasets overlap on the growth axis, and every variable chunk spans each inner axis in full. A format may then order its file groups by the dataset order and call `NdSourceExec::with_ordered_chunks()`.

`narrow(&filters, &schema)` cuts the merged result to a query:

- The one-axis conjuncts on a coordinate column cut the seed of that axis, with `axis_ranges`.
- A dataset whose growth-axis range holds no kept value is marked as pruned.

## 4. DataFusion statistics

`column_statistics(&dataset.statistics, &schema) -> datafusion::common::Statistics`:

- The min and max of each coordinate column, exact.
- The row count: the cell count of the full grid, as `Inexact`, because the row count of a scan depends on the selected columns.

A format attaches this to each `PartitionedFile`. DataFusion's file pruning
then skips datasets on coordinate filters with no extra code.

## 5. The grid hint

```rust
pub struct NdGridHint {
    pub growth_axis: String,
    pub axes: Vec<NdHintAxis>,
}

pub struct NdHintAxis {
    pub name: String,
    pub seed: Option<ArrayRef>,
    pub mode: Option<AxisMode>,
}
```

- `NdTableStatistics::grid_hint()` builds it. The mode is `Some` only when the statistics decide it: an inner axis without a coordinate whose size differs between datasets gets `Pad`.
- `NdSourceExec::with_grid_hint(hint)` stores it.
- `NdExecutionPlan` gets `fn grid_hint(&self) -> Option<Arc<NdGridHint>>`, with default `None`:
  - `NdSourceExec` returns its hint.
  - `NdFilterExec` returns the hint of its child, narrowed by its one-axis conjuncts.
  - `NdProjectionExec`, `NdLimitExec`, `NdRepartitionExec`, `NdCoalescePartitionsExec` and `NdAxisReorderExec` return the hint of their child.
  - `NdUnionExec` merges the hints of its children. It returns `None` when they differ in axis set or growth axis.
  - `NdCoarsenExec` returns `None`, because it changes the grid.
  - A node of another crate returns `None` unless it forwards the hint.
- `NdDataSink::write_all` gets the hint: `write_all(&self, data, hint: Option<&NdGridHint>, context)`. `NdDataSinkExec` takes it from its nd child.
- `NdGridAccumulator::with_grid_hint(&hint)` seeds each axis that has a seed, sets the growth axis, and sets each mode that the hint decides. An explicit `with_mode` or `with_coordinate` call wins over the hint.

A `LIMIT` does not cut the hint, so a limited write can keep axis steps that
no chunk fills. Those cells are null.

## 6. Test support

- `NdMemTable` implements `NdStatisticsProvider`: it derives the statistics of each partition from its chunks. Each partition gets an `ObjectMeta` with a made-up location, and the store argument is not used.
- `NdMemTable::with_grid_hint()` merges them and puts the hint on its scan.
- `MemoryGridSink` seeds its accumulator from the hint it gets.

## Tests

- `from_values`: regular integer, float and timestamp axes; an irregular axis; a descending axis.
- `fingerprint`: equal for equal values, different for different values, fixed for a known input.
- `merge`: the same grid with other times; overlapping times; another `lat`; another axis set; a `Summary` axis; the dataset order; `ordered_chunks_safe`.
- `narrow`: a time range cuts the seed and prunes datasets.
- `column_statistics`: DataFusion prunes a dataset on a coordinate filter.
- Hint flow: the hint passes a filter, a projection, a repartition and a merge, and a filter narrows it.
- End to end: grid chunks that arrive in reversed order write a sorted output grid through `MemoryGridSink`, with no manual seeds.

## Steps

1. Statistics types and helpers in `nd-arrow-array`.
2. `NdStatisticsProvider`, `NdTableStatistics::merge` and the issues.
3. `narrow` and `column_statistics`.
4. `NdGridHint`, `NdExecutionPlan::grid_hint`, and the hint on the nodes.
5. The hint in `NdDataSink`, `NdDataSinkExec` and the accumulator.
6. Test support and the end-to-end test.

Each step is one commit with its tests. Verification per step: `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test --workspace --all-features`.
