# datafusion-nd-exec

Grid-native DataFusion execution for n-dimensional data. The work follows
[maris-development/beacon#442](https://github.com/maris-development/beacon/issues/442).

## Crates

| Crate | Content | Depends on |
|---|---|---|
| `nd-arrow-array` | `NdArrowArray`, `NdRecordBatch`, `Dimensions` with axis metadata, the `Selection` lattice, `BroadcastMap`, and the `nd.array` Arrow extension type | `arrow` |
| `datafusion-nd-exec` | The nd plan nodes, the node registry, the `NdBoundaryRule`, axis ranges for readers, and a differential test harness (feature `test-utils`) | `nd-arrow-array`, `datafusion` |

## Use

Enable the nd pipeline on a session:

```rust
use std::sync::Arc;
use datafusion::execution::SessionStateBuilder;
use datafusion_nd_exec::{NdNodeRegistry, NdSessionStateBuilderExt};

let registry = Arc::new(NdNodeRegistry::new());
let state = SessionStateBuilder::new()
    .with_default_features()
    .with_nd_pipeline(registry)
    .build();
```

A format that produces nd data plans its scan as
`NdBroadcastExec(NdSourceExec(scan))`, with the scan columns encoded as
`nd.array`. A crate adds its own nd nodes with `NdNodeRegistry::with_probe`
and `NdNodeRegistry::with_sinker`.

## Plan nodes

| Node | Sinks from |
|---|---|
| `NdSourceExec` | the scan of a format |
| `NdFilterExec` | `FilterExec` |
| `NdProjectionExec` | `ProjectionExec` |
| `NdUnionExec` | `UnionExec` |
| `NdLimitExec` | `LocalLimitExec`, `GlobalLimitExec` |
| `NdRepartitionExec` | round-robin `RepartitionExec` |
| `NdCoalescePartitionsExec` | `CoalescePartitionsExec` |
| `NdEmptyExec`, `NdAxisReorderExec`, `NdCoarsenExec` | none: building blocks that a host plans |
| `NdBroadcastExec` | the boundary to flat rows |

## Output terminals

A terminal ends the nd region instead of `NdBroadcastExec`, so the grid
reaches the output.

| Terminal | Use |
|---|---|
| `NdDataSinkExec` | Write the grid with an `NdDataSink`. It replaces a `DataSinkExec` when an `NdSinkFactory` of the registry maps its sink. |
| `NdEncodeExec` | Yield one `nd.array`-encoded row per chunk, for Arrow IPC or Flight. `nd_output_plan` puts it at the root. |

A grid writer places each chunk with the `NdGridAccumulator`. One axis grows
(`Coordinate` or `Append`); the other axes match (`Coordinate` or `Fixed`) or
pad (`Pad`). A host that knows a coordinate up front seeds it with
`with_coordinate`, so the output axis is sorted whatever the chunk order.

## Test

```bash
cargo test --workspace --all-features
```

## License

AGPL-3.0. See [LICENSE](LICENSE).
