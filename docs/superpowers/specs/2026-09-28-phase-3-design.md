# Design: phase 3, structure and read path (beacon #448 to #451)

## Context

Phases 1 and 2 are on `main`. Phase 3 adds the structural nodes, the filter
routing, the sort order and a read-path helper. #460 is beacon-only work.

Decisions from the user:
- #448: defer the spatial box (st_within, st_intersects). Do the one-axis, ragged and cell-mask routing now.
- #450: include `NdRepartitionExec` with its own channels.
- #451: a format opts in to sort order with `NdSourceExec::with_ordered_chunks()`.
- `count(*)`: count every cell of the full grid of each chunk. The test table does this now.

The grid rule stays: the grid of a scan comes from the dimensions of the selected columns only.

## 1. Selection helpers (`nd-arrow-array`)

- `Selection::coarsen(&self, target)`: return the coarsest state that holds the same cells. A cell mask that keeps all cells becomes `Full`. A cell mask that keeps a prefix of the innermost axis for each outer cell becomes `Ragged`. A contiguous cut aligned on the outer axis becomes `AxisIndices`.
- `Selection::slice(&self, target, offset, len)`: keep the retained cells `[offset, offset + len)` in row-major order, then coarsen. A cut aligned on the outer axis keeps the rectangle.

## 2. Boundary rule with several children

- `NdSinker::try_sink` gets `children: &[Arc<dyn ExecutionPlan>]`: the nd children of the boundaries under the parent, one per parent child.
- The rule calls the sinkers only when every child of the parent is an `NdBroadcastExec`.
- The rule no longer looks through a round-robin `RepartitionExec`. The repartition sinks as `NdRepartitionExec` (item 5).

## 3. Filter routing by footprint (#448)

Per batch, `NdFilterExec` routes each conjunct by its footprint:

| Footprint | Result |
|---|---|
| One target axis | `AxisIndices` |
| Other | a mask, evaluated only at the cells that the selection keeps |

- The filter applies the one-axis conjuncts first. The other conjuncts then gather their footprint mask only at the retained cells, through `BroadcastMap::gather_indices_for`. The full-grid mask allocation goes away.
- The result is coarsened. A monotone inner-axis predicate such as `PRES < 100` then gives `Ragged`.
- A metric `cells_evaluated` counts the mask cells.

## 4. Union, limit, empty and axis reorder (#450)

- `NdUnionExec`: the partitions of all children in sequence. Each chunk keeps its own grid. Sinker: a `UnionExec` whose children are all boundaries with the same schema.
- `NdLimitExec { skip, fetch }`: per partition, cut the retained cells in stream order with `Selection::slice`. Sinkers: `LocalLimitExec` and `GlobalLimitExec` over a boundary. A global limit needs a child with one partition.
- `NdEmptyExec`: a leaf with a schema and a partition count. It yields no batches. It is a building block for sinks, with no sinker.
- `NdAxisReorderExec { axes }`: permute the target axes. The columns do not change, because the broadcast aligns them by name. `Full` stays `Full`, `AxisIndices` is permuted, and other states become a cell mask in the new order. No sinker.

## 5. Round-robin repartition (#450)

- `NdRepartitionExec { partitions }`: send whole nd batches round robin to `n` output partitions.
- On the first `execute_nd`, the node spawns one task per input partition. The tasks send batches into one unbounded channel per output partition. Unbounded channels avoid a deadlock when a consumer reads the output partitions in sequence. An error goes to every output.
- Dropping the node's streams aborts the tasks.
- Sinker: a `RepartitionExec` with `RoundRobinBatch` and no order preservation over a boundary.

## 6. Sort order (#451)

- `NdSourceExec::with_ordered_chunks()`: the format declares that each partition yields its chunks in the order of the outer axis.
- The plan-time grid comes from the axis metadata of the encoded fields, with the same rule as `infer_target`. The node reports a lexicographic order over the outer axes of that grid. The order stops at the first axis with no monotone coordinate, or with a coordinate column that is not in the schema.
- `NdFilterExec`, `NdLimitExec` and `NdBroadcastExec` keep the order. `NdProjectionExec` projects the order through a `ProjectionMapping`. Union, repartition and axis reorder report no order.
- `NdMemTable` gets `with_ordered_chunks()` for the tests.

## 7. Axis ranges for readers (#449)

- `axis_ranges(predicates, schema, coordinates)` in a new `pushdown` module: for each one-axis conjunct on a coordinate column, evaluate it on the 1-D coordinate values and return the kept index ranges per axis.
- `AxisRanges::overlaps(chunk_origin, chunk_shape, axis_names)`: true when a chunk holds a kept index on every constrained axis.
- A reader calls these with the filters it already receives as pruning hints. The chunk queue and its metric are beacon work.

## Tests

- Unit tests for `coarsen`, `slice`, each node, the union and limit sinkers, and `axis_ranges`.
- Harness: queries for `UNION ALL`, `LIMIT`, ragged profile filters and `ORDER BY`. A `LIMIT` without `ORDER BY` compares row counts only.
- Plan tests: `NdUnionExec`, `NdLimitExec` and `NdRepartitionExec` sit below the boundary. `ORDER BY time` on the grid plans without a `SortExec`. `ORDER BY` on the profiles keeps it.

## Steps

1. Selection `coarsen` and `slice`.
2. Boundary rule and sinkers with several children.
3. Filter routing (#448).
4. `NdUnionExec`.
5. `NdLimitExec`.
6. `NdRepartitionExec`.
7. `NdEmptyExec` and `NdAxisReorderExec`.
8. Sort order (#451).
9. Axis ranges (#449).

Each step is one commit with its tests. Verification per step: `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test --workspace --all-features`.
