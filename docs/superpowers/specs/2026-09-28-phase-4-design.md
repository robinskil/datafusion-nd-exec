# Design: phase 4, output terminals (beacon #452 to #456)

## Context

Phases 1 to 3 are on `main`. Every query still ends in `NdBroadcastExec`, so a
result always leaves the plan as flat rows. Phase 4 lets the grid flow to the
output.

Scope: only the generic parts. This repo has no file-format code. The netCDF,
Zarr and raster writers, the output format, the WMS wiring and the TypeScript
decoder come later, in the integration of a host such as beacon.

The old branch `features/nd-pipeline-sink` does not exist any more, so the
accumulator is new code.

## 1. Terminal nodes in the boundary rule

- `Sunk` becomes an enum:
  - `Sunk::Below { nd, residual }`: as today, the nd node goes below a new boundary.
  - `Sunk::Terminal(plan)`: the node replaces the boundary. It reads nd batches and yields flat output, such as a row count.
- `NdCoalescePartitionsExec`: merge all input partitions into one nd stream. Sinker: a `CoalescePartitionsExec` over a boundary. A `fetch` on it becomes an `NdLimitExec` on top. `DataSinkExec` needs one input partition, so DataFusion puts this node above the boundary.

## 2. Compact batches (`nd-arrow-array`)

`NdRecordBatch::compact(&self) -> Result<NdRecordBatch>` gives a batch with a `Full` selection and the same rows:

- `AxisIndices`: take the kept indices along each axis. The grid gets smaller, and each column keeps only its kept cells.
- `Ragged` and `CellMask`: keep the grid. A column that spans the whole grid gets null at each cell that is not kept. A column on fewer axes stays as it is, because it holds coordinates or attributes.

A writer then only sees dense chunks.

## 3. Grid accumulator (#452, #453)

`NdGridAccumulator` places each chunk into one output grid. It holds the axis state only, not the data.

Axis modes:

| Mode | Rule |
|---|---|
| `Coordinate` | Match the chunk coordinate values against the output coordinate. An exact match of a contiguous slice gives its offset. Values with no overlap are appended. An overlap without an exact match is a conflict error with both value ranges. |
| `Append` | Put the chunk after the previous chunks, for example `N_PROF` 3 then 2 gives 5. |
| `Pad` | Align at index 0 and grow the axis to the largest size, for example `N_LEVELS`. |
| `Fixed` | All chunks must have the same size. |

- Default mode: `Coordinate` when the batch has the coordinate column of the axis (from `AxisMeta`), else `Fixed`. A caller sets a mode per axis name.
- `place(&mut self, batch) -> Result<Placement>`: compact the batch, then return per output axis the offset, the chunk size, the new extent, and the coordinate values that the chunk adds.
- The output axis order is the order of the first chunk. An axis that a later chunk adds goes at the end.
- Values are compared with the Arrow row format, so any orderable type works.

## 4. Sink terminal (#452, #456)

- `trait NdDataSink`: `schema()`, and `async fn write_all(&self, data: SendableNdBatchStream, context) -> Result<u64>`. A format implements it with the accumulator and its own writer.
- `NdDataSinkExec { input, sink }`: one output partition with the row count, like `DataSinkExec`. It reads all input partitions as one nd stream.
- `trait NdSinkFactory`: `fn nd_sink(&self, sink: &dyn DataSink) -> Option<Arc<dyn NdDataSink>>`. The registry holds the factories (`with_sink_factory`).
- Sinker: a `DataSinkExec` over a boundary, when a factory gives an nd sink, becomes a `Sunk::Terminal(NdDataSinkExec)`.
- `testing::MemoryGridSink`: an `NdDataSink` that builds the dense output grid in memory with the accumulator. It is the reference writer for the tests.

## 5. nd output (#454)

- `NdEncodeExec`: a terminal that yields one `nd.array`-encoded row per compacted nd batch. A host sends these over Arrow IPC or Flight.
- `nd_output_plan(plan) -> Result<Option<Arc<dyn ExecutionPlan>>>`: when the root of an optimized plan is a boundary, replace it with an `NdEncodeExec` over its nd child. Else `None`: flat operators run above the nd region.

## 6. Coarsen (#455)

- `NdCoarsenExec { factors: [(axis, factor)], reduce }`, with `reduce` one of `Mean`, `Min`, `Max`, `First`.
- Per chunk: compact, then reduce each block of `factor` cells along each named axis. A trailing short block reduces on its own. Nulls do not count. An all-null block is null.
- `Mean` gives `Float64` for every numeric column. `Min`, `Max` and `First` keep the type. A non-numeric column takes the first value.
- Blocks do not cross chunks. A chunk size that is not a multiple of the factor gives a short block at the chunk end.
- No sinker. A host plans it for tiles.

## Tests

- Unit tests for `compact`, each accumulator mode, the conflict error, `NdCoalescePartitionsExec`, `NdEncodeExec` and `NdCoarsenExec`.
- Round trip: grid table through `MemoryGridSink` gives one dense grid that equals the flat scan. Profiles with `Append` on `N_PROF` and `Pad` on `N_LEVELS` give extents 5 x 4 with the pad cells null.
- Plan: `DataSinkExec(CoalescePartitionsExec(boundary))` with a test factory becomes `NdDataSinkExec(NdCoalescePartitionsExec(...))`.
- nd output: decode the `NdEncodeExec` rows and compare with the flat result.

## Steps

1. `Sunk` enum and `NdCoalescePartitionsExec`.
2. `NdRecordBatch::compact`.
3. `NdGridAccumulator`.
4. `NdDataSink`, `NdDataSinkExec`, the factory and `MemoryGridSink`.
5. `NdEncodeExec` and `nd_output_plan`.
6. `NdCoarsenExec`.

Each step is one commit with its tests. Verification per step: `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test --workspace --all-features`.
