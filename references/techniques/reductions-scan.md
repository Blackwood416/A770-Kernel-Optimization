# Reduction and Work-Group Scan Reference on A770

> Scope: Intel Arc A770 (DG2), Windows 11, oneAPI 2026.1–2026.2, `f32` `4096x4096` full reduction and `1M` work-group scan, verified against CPU reference (`errors=0`).

## 1. Full-Tensor Reduction (`4096x4096 f32`) `[MEASURED]`

| Variant | Wall Avg | Total Instr | Takeaway |
|---|---:|---:|---|
| `global_atomic` (SG tree, 1 global atomic/SG = 8,192 ops) | **`0.164 ms`** | `135.9M` | **Fastest** -- few atomics hide under the 64 MB memory stream |
| `slm_atomic` (SG tree + SLM atomic + 1 global atomic/WG) | `0.166 ms` | `176.3M` | Extra barrier & instructions with no win at 1,024 WGs |
| `tree` (1,024 WGs, SG tree -> partials -> final tree) | `0.170 ms` | `155.2M` | Best deterministic / multi-pass choice |
| `split_k` (16 K-splits x 64-row blocks -> final tree) | `0.175 ms` | `160.2M` | Use only when extra parallelism or per-K partials are needed |
| `global_atomic_thread` (1 global atomic/thread = 262,144 ops) | **`0.627 ms`** | `122.9M` | **`3.8x` slower vs `tree`** despite lowest instruction count -- serializes on one address |

**Selection rules**:
1. Reduce inside each sub-group first; never issue one global atomic per thread (`3.8x` contention cliff vs `tree`).
2. Use `global_atomic` (one atomic per sub-group) for memory-bound scalar reductions when non-deterministic float order is acceptable; use `tree` for deterministic results.
3. Always initialize every local reduction accumulator before `atomic_ref::fetch_add` (uninitialized SLM accumulators produce silent wrong sums `[CORRECTNESS]`).

## 2. Work-Group Scan (`1M f32`, Two-Pass: Local Scan + Add-Base) `[MEASURED]`

| Algorithm | `WG16` | `WG32` | **`WG64`** | `WG128` | `WG256` | `WG512` |
|---|---:|---:|---:|---:|---:|---:|
| **Hillis-Steele (ms)** | `0.0861` | `0.0640` | **`0.0559`** | `0.0625` | `0.0621` | `0.0673` |
| Blelloch (ms) | `0.0986` | `0.0724` | `0.0627` | `0.0660` | `0.0680` | `0.0768` |

**Selection rules**:
1. **`WG=64` is the measured sweet spot** `[HEURISTIC]` for 1-element-per-lane scans; larger work-groups (`WG512`) pay sharply higher A770 barrier expansion costs.
2. **Prefer Hillis-Steele over Blelloch** for single-element-per-lane work-group scans (fewer barriers and instructions at every WG size); reserve Blelloch for multi-element-per-lane local scans that amortize its extra phases.

## 3. Critical Trap: `sycl::reduce_over_group` Compiled SIMD Narrowing `[BUG]` `[CORRECTNESS]`

A source-level 32-lane `sycl::reduce_over_group` silently produces **wrong results** if register pressure causes IGC to compile the kernel at `SIMD16` or `SIMD8`:

- **Mechanism**: IGC lowers `reduce_over_group` to a per-SIMD-pass butterfly with **no cross-pass SLM merge** (no `send.slm`, no barrier), leaving each lane with only half (or a quarter) of the true reduction.
- **How to detect**: Inspect `ocloc disasm` (a true 32-lane float reduction needs an `add (16|M0)` stage; narrowed SIMD16 stops at `(8|`) or check `.ze_info` (`simd_size: 32` vs `16`/`8`). Any unroll that mysteriously gets faster on a reduction kernel is suspect.
- **Correct fix (`SDPA_REDUCE=1`)**: Stage per-lane partials into SLM (one dedicated slot per sub-group/slice), execute `it.barrier(access::fence_space::local_space)`, and read back as `float4` to sum explicitly. Works at any compiled SIMD width (`2.22x` faster than slow serial alternatives; never replace with `[[intel::reqd_sub_group_size(32)]]` which regresses **`6.6x`**, and never hand-write `select_from_group` butterflies which produced `14,559/16,384` errors `[BUG]`).
- **API syntax traps**:
  - Pass `it.get_group()` or `it.get_sub_group()` to `sycl::reduce_over_group` -- never pass `nd_item` (compile error).
  - In ESIMD, `esimd::reduce` **only** matches `std::plus<>{}` / `std::multiplies<>{}`; typed `std::plus<float>{}` compiles and silently returns `0.0f` `[BUG]`.

Copy-ready cores: [Reduction and Scan cores](../api/code-snippets.md#9-reduction-and-hillis-steele-scan-cores).
