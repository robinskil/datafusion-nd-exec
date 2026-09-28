# datafusion-nd-exec

Grid-native DataFusion execution for n-dimensional data. The work follows
[maris-development/beacon#442](https://github.com/maris-development/beacon/issues/442).

## Crates

| Crate | Content | Depends on |
|---|---|---|
| `nd-arrow-array` | `NdArrowArray`, `NdRecordBatch`, `Dimensions` with axis metadata, the `Selection` lattice, `BroadcastMap`, and the `nd.array` Arrow extension type | `arrow` |
| `datafusion-nd-exec` | The nd plan nodes, the node registry, the `NdBoundaryRule`, and a differential test harness (feature `test-utils`) | `nd-arrow-array`, `datafusion` |

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

## Test

```bash
cargo test --workspace --all-features
```

## License

AGPL-3.0. See [LICENSE](LICENSE).
