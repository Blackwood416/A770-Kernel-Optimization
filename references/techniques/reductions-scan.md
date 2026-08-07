# Reduction and Scan Selection Reference

Measured on Intel Arc A770, oneAPI 2026.1, driver `32.0.101.8724`. Wall
times are 3-run averages with 20 warmup + 200 timed launches; all outputs
pass the CPU reference. VTune uses `instruction-count`; its sampled wall
times are inflated and only instruction mixes are compared. Source project:
`E:\RiderProjects\Reduction-Opti` (`references\reductions-scan.md`).

## Full-Tensor Reduction on 4096x4096 f32

| Variant | Avg ms | Notes |
|---|---:|---|
| `tree` | 0.16989 | 1024 WGs, SG tree + partials, final tree kernel |
| `split_k` | 0.17476 | 16 K-splits x 64-row blocks, final tree |
| `slm_atomic` | 0.16592 | SG tree, SLM atomic per WG, one global atomic per WG |
| `global_atomic` | 0.16428 | SG tree, one global atomic per sub-group (8 per WG) |
| `global_atomic_thread` | 0.62732 | one global atomic per thread (262144 atomics) |
| `rmsnorm_inline` | 0.18278 | 1 WG per row, SLM row tile, 2 barriers |

## Atomic Contention Evidence

`global_atomic_thread` has the lowest instruction count (122.9M) but is about
3.8x slower than `tree` because all threads serialize on one global address.
The same 64 MB stream with only 8192 atomics (`global_atomic`) hides that cost
under the memory-bound load stream. `slm_atomic` adds a local atomic, an extra
barrier, and one global atomic per WG without a wall-time win at 1024 WGs.

VTune instruction-count (main reduction kernel):

| Variant | Total instr | Send | Sync | Int32 & SP Float | SIMD util |
|---|---:|---:|---:|---:|---:|
| `tree` | 155,160,576 | 22,424,576 | 1,998,848 | 91,072,512 | 100.0% |
| `split_k` | 160,157,696 | 22,424,576 | 1,998,848 | 99,067,904 | 100.0% |
| `slm_atomic` | 176,315,398 | 27,492,558 | 1,998,848 | 93,267,150 | 88.1% |
| `global_atomic` | 135,921,664 | 18,989,056 | 999,424 | 87,949,312 | 96.9% |
| `global_atomic_thread` | 122,929,152 | 18,989,056 | 999,424 | 82,952,192 | 100.0% |
| `rmsnorm_inline` | 362,936,440 | 70,838,296 | 3,997,696 | 178,276,376 | 86.3% |

## Work-Group Scan on 1M f32

Both scans are two-pass: WG-local scan + partial sums, then a second kernel
adds the WG base offset. Times include the scan kernel plus `add_base`.

| Algorithm | WG16 | WG32 | WG64 | WG128 | WG256 | WG512 |
|---|---:|---:|---:|---:|---:|---:|
| Hillis-Steele ms | 0.08613 | 0.06401 | 0.05593 | 0.06250 | 0.06212 | 0.06727 |
| Blelloch ms | 0.09859 | 0.07243 | 0.06266 | 0.06602 | 0.06801 | 0.07684 |

Best measured WG size is 64. Per-work-group Sync cost grows much faster than
the theoretical `2*log2(WG)+1`/`2*log2(WG)+3` counts because each A770 barrier
expands into many SLM/synchronization instructions.

## Selection Rules

### Reduction

1. Use `tree` or `split_k` when deterministic partials, many WGs, or a second
   pass over row/slice sums are needed.
2. Use `global_atomic` when the reduction is memory-bound and atomics per
   output stay small (e.g. one per sub-group). It was the fastest variant.
3. Use `slm_atomic` only when the global atomic count is the bottleneck; at
   1024 WGs it adds instructions and a barrier without winning.
4. Never reduce every thread to one global address at this size;
   `global_atomic_thread` is 3.8x slower despite fewer instructions.
5. Keep the RMSNorm inline SLM-row pattern for row sums that feed a second
   pass over the same row, not for bare full-tensor sums.
6. `split_k` is not a win at 4096x4096 f32 with 1024 WGs; keep it for shapes
   that need more parallelism or per-K-slice partials.

### Scan

1. Start with WG64 for one-element-per-lane work-group scans.
2. Prefer Hillis-Steele for small WG scans: lower barrier and instruction
   counts at every measured WG size.
3. Use Blelloch only when each lane scans multiple elements locally first;
   otherwise its `O(n log n)` work has nothing to amortize.
4. Fewer, larger work-groups do not automatically win; barrier cost per WG
   grows with WG size.

## Reproduction

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1
bench.exe --mode reduce --reduce tree
bench.exe --mode scan --scan hillis --wg 64
```

Copy-ready cores: [reduction tree/atomic](../api/code-snippets.md#reduction-tree-atomic-cores),
[scan Hillis-Steele/Blelloch](../api/code-snippets.md#scan-hillis-blelloch-cores).
