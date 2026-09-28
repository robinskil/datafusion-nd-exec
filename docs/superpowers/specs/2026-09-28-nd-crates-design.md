# Design: nd crates for grid-native execution (beacon #442, phases 1 and 2)

## Context

Beacon issue maris-development/beacon#442 removes the broadcast of n-dimensional data to flat rows.
The nd pipeline lives in `beacon-datafusion-ext::nd` today (about 4,700 lines).
That module imports only `std`, `arrow`, `datafusion` and `futures`, so we can extract it.
This repo (`robinskil/datafusion-nd-exec`) becomes the home of the nd pipeline.
Beacon then consumes it as a dependency and plugs in its readers and sinks.

Decisions from the user:
- Scope: build the two crates, port the current nd code, then do phase 1 (#443, #444, #445, #446) and phase 2 (#447). Later phases get their own plans.
- Encoding: use a registered Arrow `ExtensionType`. Do not keep the `beacon.nd` field-metadata convention.
- Beacon-specific work (#449 readers, #452/#456 sinks, #454 TS, #455 WMS, #460) stays in beacon. This repo supplies the hooks. The node registry (#444) is the first hook. Each later hook comes with its phase plan.
- Versions: match beacon. `datafusion = "^53"`, `arrow = "^58"`, edition 2024, `rust-version = "1.94"`.

Source of the port: `C:\Git\beacon\beacon-db\beacon-datafusion-ext\src\nd\` at beacon `main` (3c48f5e2).

## Workspace layout

```
Cargo.toml                      workspace, shared deps and lints
crates/nd-arrow-array/          input layer: arrow only
crates/datafusion-nd-exec/      plan nodes, registry, boundary rule, test harness
docs/superpowers/specs/2026-09-28-nd-crates-design.md
.github/workflows/ci.yml        fmt, clippy -D warnings, test
.gitignore                      target/
```

`datafusion-nd-exec` depends on `nd-arrow-array` and re-exports it as `datafusion_nd_exec::array`.

## Crate 1: `nd-arrow-array` (the input layer)

Dependencies: `arrow`, `arrow-schema`, `serde`, `serde_json`. Errors use `ArrowError`. `DataFusionError` already has `From<ArrowError>`.

Modules, ported from beacon unless marked new:

| Module | Content |
|---|---|
| `dimensions.rs` | `Dimension`, `Dimensions` (C-order, unique names, rank 0 = scalar). Port as is. |
| `axis.rs` (new, #445) | `AxisOrder { Ascending, Descending, Unordered }` and `AxisMeta { coordinate: Option<Arc<str>>, order: AxisOrder }`. `Dimension` gains `meta: Option<AxisMeta>`. Add `AxisOrder::detect(&ArrayRef)` for readers. |
| `array.rs` | `NdArrowArray { values, dims }`. Port as is. |
| `broadcast.rs` | `BroadcastMap`. Port. Add `gather_indices_for(&Selection)` that fuses `AxisIndices` into the stride walk. Keep `gather_indices_at` for `CellMask`. |
| `selection.rs` (new, #443) | `enum Selection { Full, AxisIndices(Vec<Option<UInt64Array>>), Ragged { axis, lengths: UInt64Array }, CellMask(UInt64Array) }`. `SelectionKind` orders the states coarse to fine. Methods: `kind`, `num_rows(&Dimensions)`, `intersect` (returns the coarsest state that holds the result), `to_cell_indices`, `is_rectangle`. |
| `batch.rs` | `NdRecordBatch`. Replace `selection: Option<UInt64Array>` with `selection: Selection`. Port `with_selection`, `num_rows`, `materialize`, `materialize_with_stats` to the lattice. |
| `extension.rs` (new) | `NdArrayType` implements `arrow_schema::extension::ExtensionType`. `NAME = "nd.array"`. Storage type: `Struct{ values: List<T>, dim_sizes: List<UInt32>, dim_names: List<Utf8> }`, the same layout as beacon. `Metadata = NdArrayMetadata { version: u32, axes: Vec<AxisMetaEntry> }` as JSON. The metadata carries the static axis info, so plans can read it at plan time (needed by #451). |
| `encoding.rs` | Port `encode_nd_array`, `decode_nd_array`, `encode_nd_record_batch`, `decode_nd_record_batch_row`, `nd_batch_count`, `encoded_schema`, `logical_schema`, `infer_target`, `encode_flat_batch_as_nd`. Detect encoded fields with `field.try_extension_type::<NdArrayType>()`. Decode attaches `AxisMeta` from the field metadata to each `Dimension`. |

Port the unit tests of each module. Add tests for each selection state, each state change of `intersect`, the extension type round trip, and the three `AxisOrder` cases.

## Crate 2: `datafusion-nd-exec`

Dependencies: `nd-arrow-array`, `datafusion`, `arrow`, `futures`. Feature `test-utils` exposes the harness.

### Port (from `nd/exec/*` and `nd/optimizer.rs`)
- `NdExecutionPlan: ExecutionPlan` with `execute_nd`, and `SendableNdBatchStream`.
- `NdSourceExec`, `NdProjectionExec`, `NdFilterExec`, `NdBroadcastExec`, `NdExprColumn`, `materialize_nd_stream`, `nd_scan_plan`, `offer_filters_to_child`, `filters_stay_above`, `is_pushable_expr`.
- Add two methods with defaults to `NdExecutionPlan`: `accepts_selection() -> SelectionKind` (default `CellMask`) and `max_output_selection() -> SelectionKind`. The boundary rule uses them.

### #443 in the filter
- `NdFilterExec` checks the footprint of each conjunct. A one-axis conjunct gives `AxisIndices`. Other conjuncts give `CellMask`. Combine with `Selection::intersect`.
- The full footprint routing (spatial box, ragged prefix) stays in #448.

### #444 node registry (`registry.rs`)
- `type NdProbe = Arc<dyn Fn(&Arc<dyn ExecutionPlan>) -> Option<Arc<dyn NdExecutionPlan>> + Send + Sync>`.
- `trait NdSinker: Send + Sync { fn try_sink(&self, parent: &Arc<dyn ExecutionPlan>, child: &Arc<dyn NdExecutionPlan>, registry: &NdNodeRegistry) -> Result<Option<Sunk>>; }` where `Sunk { nd: Arc<dyn NdExecutionPlan>, residual: Option<Arc<dyn ExecutionPlan>> }`. This is the "sink check" of #447.
- `NdNodeRegistry { probes, sinkers }` with `register_probe`, `register_sinker`, `as_nd_plan`. `NdNodeRegistry::default()` holds the built-in probe (the old downcast list) and the built-in sinkers (filter, projection).
- Store the registry in `SessionConfig` as an extension. `execute_nd` finds the child through `TaskContext::session_config().get_extension::<NdNodeRegistry>()`. Fall back to the default registry when the extension is absent.
- `NdBroadcastExec::try_new` takes the registry to validate its child.

### #447 boundary rule (`boundary.rs`)
- `NdBoundaryRule { registry }` implements `PhysicalOptimizerRule`. It replaces `NdFilterPushdown` and `NdProjectionPushdown`.
- Walk the plan bottom up. For each parent of an `NdBroadcastExec`, ask each sinker. On success, put the nd node below the boundary and keep the residual above. Repeat until no sinker accepts.
- A sinker refuses when `child.max_output_selection() > node.accepts_selection()`.
- Port the filter logic (conjunct split with residual `FilterExec`) and the projection logic into the two built-in sinkers.
- `NdBroadcastExec` display shows the nd region: `NdBroadcastExec: region=[NdFilterExec, NdSourceExec]`.

### Session setup (`session.rs`)
- `trait NdSessionStateBuilderExt { fn with_nd_pipeline(self, registry: Arc<NdNodeRegistry>) -> Self; }` for `SessionStateBuilder`.
- It appends `NdBoundaryRule` after the default physical rules and sets the registry extension. Keep the defaults: beacon found that a missing `EnforceDistribution` breaks `count(*)`.

### #446 test harness (`testing/`, feature `test-utils`)
- `NdMemTable`: a `TableProvider` over in-memory `NdRecordBatch` chunks. Its scan returns `nd_scan_plan(MemorySourceConfig)` with encoded batches. It is also the reference provider for users.
- Two corpora:
  - grid: axes `time, lat, lon` with coordinate columns, two chunks, some null cells.
  - profile: axes `N_PROF, N_LEVELS`, no coordinates, fill values as nulls, uneven profile lengths.
- `assert_differential(sql)`: run the query in a flat session and in an nd session, sort both results, compare.
- `assert_plan_nodes(sql, &[...])`: assert the nd nodes in the physical plan.
- The harness in beacon (`beacon-core/tests`, real files) stays in beacon. It can reuse these helpers.

### Tests
- Port the 17 end-to-end tests of `nd/mod.rs` onto the harness.
- Port the unit tests of each exec module and of the optimizer.
- Registry test in `tests/registry_external.rs` (a separate compile unit): a custom `NdExecutionPlan` node plus its probe and sinker. The boundary rule must sink it.
- Boundary rule tests: a sinkable node above an unsinkable node, and the reverse.
- #443 acceptance: a one-axis filter gives `AxisIndices`, and the batch reports `is_rectangle() == true`.
- EXPLAIN test for the region display.

## Steps

1. Write the design doc in `docs/superpowers/specs/`. Commit it.
2. Set up the workspace, CI and `.gitignore`. Check that `cargo build` passes.
3. Port `nd-arrow-array` as is (with its tests). Commit.
4. Add the extension type and change the encoding to use it. Commit.
5. Add `AxisMeta` (#445). Commit.
6. Add the selection lattice (#443) in `selection.rs`, `batch.rs` and `broadcast.rs`. Commit.
7. Port `datafusion-nd-exec` with the old optimizer rules and tests. Commit.
8. Add the registry (#444). Commit.
9. Add the harness (#446) and move the end-to-end tests onto it. Commit.
10. Replace the two rules with the boundary rule (#447). Commit.
11. Add the one-axis `AxisIndices` path in `NdFilterExec` (#443 acceptance). Commit.

Use TDD for each new unit (steps 4 to 11).

## Beacon follow-up (not in this repo)

- `beacon-datafusion-ext::nd` re-exports `datafusion-nd-exec` through a git dependency.
- `beacon-nd-array` builds `nd_arrow_array::NdRecordBatch` and encodes with `NdArrayType`.
- `runtime_builder.rs` calls `with_nd_pipeline`.
- Remove the orphan `beacon-nd-arrow` crate.

## Verification

```bash
cargo fmt --all --check
```
```bash
cargo clippy --workspace --all-targets --all-features -- -D warnings
```
```bash
cargo test --workspace --all-features
```

- Both corpora pass `assert_differential` for the ported queries.
- The EXPLAIN test shows the nd region.
- The external registry test passes.
