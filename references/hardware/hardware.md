# A770 (Xe-HPG / DG2) Hardware, Occupancy, and SLM Reference

> Scope: Intel Arc A770 (DG2-G10, PCI `0x56a0`), Windows 11, oneAPI 2026.2, driver `32.0.101.9150`. Re-measure before transferring hardware constants to any other GPU or driver.

## 1. Terminology and Architecture Specs `[ARCH]`

Unify naming across SYCL, ESIMD, and profiler documentation:

- **1 Xe-core (1 DSS = 2 Subslices)** = `16 XVEs (EUs)`, `192 KB` L1 data cache, `128 KB` Shared Local Memory (SLM).
- **Full A770 GPU** = `32 Xe-cores` = `512 XVEs` = `4,096` HW threads at `128-GRF` or `2,048` HW threads at `256-GRF`.
- **16 MB L3 / LLC**: The shared `16 MB` last-level on-chip cache is labeled `L3` in VTune/hardware docs and `L2` in SYCL/ESIMD `cache_hint_L2` APIs. Both refer to the same `16 MB` cache.

| Spec | Value | Engineering Implication |
|---|---|---|
| Xe-cores / XVEs | `32` Xe-cores / `512` XVEs | Work-groups dispatch round-robin across `32` Xe-cores (`gid` and `gid + 32` share L1) |
| HW threads (`128-GRF` / `256-GRF`) | `8` / `4` per XVE (`4,096` / `2,048` total) | `256-GRF` halves thread slots (`64` HW threads/Xe-core) in exchange for `16 KB` GRF/thread |
| Register width | `256 bits` (`32 B` / GRF) | One `8x8` fp32 DPAS accumulator occupies `8` GRFs (`256 B`) |
| Sub-group sizes | `8, 16, 32` | Standard SYCL may choose `32` even if local dim is `16`; always query `sg.get_local_range()[0]`. In standard SYCL / OpenCL C (`SG=16`), `WG=256` consumes only `16` HW threads (`8 WGs/Xe-core = 128/128` threads) and is optimal for GRF-cached row reductions (`cols <= 4096`) |
| ESIMD max WG size | `128` work-items (`128-GRF`) / `64` (`256-GRF`) | ESIMD-only constraint: one ESIMD work-item = 1 HW thread; ESIMD `WG >= 256` never schedules and hangs silently `[BUG]` (does **not** restrict standard SYCL / OpenCL C `WG=256` at `SG=16`) |
| SLM per Xe-core / per WG | `128 KB` per Xe-core / `64 KB` max per WG | Hard per-WG ceiling is `64 KB`; standard SYCL `local_accessor` at `64 KB` can fail launch |
| L1 / L3 (LLC) | `192 KB` L1 per Xe-core / `16 MB` shared L3 | Single thread maintains L1-speed latency out to `~1 MB`; see [cache-geometry.md](cache-geometry.md) |
| PVC-only features (absent on DG2) | `load_2d`/`prefetch_2d`, `named_barrier`, `512 B` block IO | Do not use on A770 (`load_2d` hangs; `named_barrier` fails JIT; max LSC block is `256 B`) |

## 2. GRF Modes, Work-Group Sizing, and SLM Co-Residency Matrix `[ARCH]` `[MEASURED]`

Frontend capability boundary:

- **Legacy OpenCL `.spv` (`ocloc` runtime recompile)**: locked to `128-GRF` (`r0..r127`). Do not plan 256-GRF kernels on OpenCL C.
- **SYCL/ESIMD (`sycl::ext::intel::experimental::grf_size<256>`) and native Level Zero binaries**: natively execute `256-GRF` (`spill=0 B, priv=0 B`) on oneAPI 2026.2+.

Use this residency matrix to size SLM and work-groups without contradicting mode assumptions:

| GRF Mode | HW Threads / Xe-core | Recommended ESIMD WG | WGs / Xe-core | Safe SLM / WG | Typical Use Case |
|---|---:|---:|---:|---:|---|
| `128-GRF` | `128` | `WG=128` | `1` (full Xe-core) | `<= 56–64 KB` | Single-wave medium GEMM (`BM=256 BN=192 BK=32/64`, `57,344 B` SLM) |
| `128-GRF` | `128` | `WG=32` | `4` | `<= 32 KB` | Standard SYCL / small-tile ESIMD (`> 32 KB` loses resident WGs) |
| `256-GRF` | `64` | **`WG=32`** | **`2` (64/64 threads)** | **`<= 48 KB` (`49,152 B`)** | **Production large GEMM** (`cab4`: `4 x 12 KB = 48 KB`, `92.5 TFLOPS`) |
| `256-GRF` | `64` | `WG=48` or `WG=64` | `1` | `<= 64 KB` | Avoid on RowMajor GEMM (1 WG/Xe-core stalls at barriers; `452 us` vs `290–300 us` at `WG=32 / 48 KB` `[MEASURED]`) |

## 3. Occupancy and XMX Saturation `[MEASURED]`

Measured by scaling a fixed `512`-op `dpas.8x8` body across thread counts:

| Hardware Threads | Added Device Time | Aggregate Rate | Regime |
|---:|---:|---:|---|
| `128` | `5.0 us` | `13.1 Gdpas/s` | Latency-bound (`~10.7 ns`/dpas single-chain floor) |
| `512` | `9.9 us` | `26.5 Gdpas/s` | Filling XVEs |
| `2,048` | `22.6 us` | `46.4 Gdpas/s` | Near saturation |
| `4,096` | `25.6 us` | **`81.9 Gdpas/s`** | **Issue-rate plateau** (`~6.3 ns` = `~12.5–13 cycles` per `dpas.8x8` per EU) |

Occupancy rules:

1. **Compute hardware threads first**: For standard SYCL, `hw_threads = WG_count * (WG_size / SG_size)`. For ESIMD, `hw_threads = total_work_items`.
2. **ILP vs occupancy**: For scalar/vector FMA bodies (`128-GRF`), below `ILP=8` add threads first; at `ILP >= 8`, latency is hidden and capping ILP at `8–16` avoids spills. For XMX `dpas`, adding per-thread ILP does **not** raise the `~82 Gdpas/s` plateau; filling more XVEs and removing wave quantization matter more.
3. **Accumulator GRF ceiling (`256-GRF`)**: Keep DPAS accumulators `<= 128 GRF` (e.g., `acc[4][4]` = `16 x 8 = 128 GRF`, zero spills). Expanding to `acc[4][6]` (`192 GRF`) exhausts staging/pointer registers and emits `35–67` main-loop `send.dc0` scratch spills `[MEASURED]`.
4. **Higher occupancy is not automatically faster**: Lower instruction volume and better latency hiding beat raw occupancy (e.g., `256-GRF` `WG=32` at 50% max thread slots beats `128-GRF` at higher slot utilization).

## 4. SLM Banking, Alignment, and Bandwidth Wall `[MEASURED]`

### 4.1 Read-Side Stride Behaviour (`4 B` word reads, full occupancy)

| Per-Lane Stride | Cost vs Stride 1 (`4 B`) | Reading |
|---|---:|---|
| `0 B` (broadcast) | **`0.69x`** (`43.8 ns`) | Hardware merges same-address reads; never spread out broadcasts |
| `4 B` (consecutive) | `1.00x` (`63.7 ns`) | Baseline |
| `8 B` / `12 B` / `16 B` / `132 B` | `0.85x–0.91x` | Odd strides and `8 B` spacing beat contiguous `4 B` |
| `64 B` / `192 B` | `1.25x` (`79.6 ns`) | First bank-conflict tier |
| `128 B` / `384 B` | `1.67x` (`106.4 ns`) | Second bank-conflict tier |
| **`256 B`** | **`3.29x` (`209.6 ns`)** | **Severe conflict cliff**; avoid all `64 B`-multiple lane strides |

There is no classic 32-bank x 4-byte conflict cliff on DG2. Pad transposed/column-major SLM pitches by one element (odd stride) only when lane strides hit `64 B` multiples; do **not** pad `256 B` LSC block loads (they already span banks and padding breaks alignment).

### 4.2 Write-Side Banking, `16 B` Alignment Trap, and Read Wall

1. **Write-side `256 B` 8-way conflict**: `esimd::slm_scatter` with a `256 B` stride (`0 mod 128 B`) triggers an 8-way SLM store bank conflict (`mov` count jumps `143 -> 218`, slowing `4096^3` GEMM by `+79 us` `[MEASURED]`).
2. **Coalesced `4x64 B` writes (`store.slm.d64x8t`)**: Packing adjacent `uint16_t` pairs into `4x64 B` block writes + `1x128 B` A store cuts per-`BK=16` `store.slm` ops from `9` to `5`, reducing SLM write traffic by `24%–31%` and freeing SBID tokens `[MEASURED]`.
3. **Hard `16 B` (OWord) alignment on `slm_block_store` `[BUG]` `[CORRECTNESS]`**: Passing an `8 B`-aligned offset (e.g., `slm_block_store<uint32_t, 2>`) does not error -- hardware silently truncates down to `16 B` and overwrites adjacent entries. Always align offsets to `16 B` and use `slm_block_store<uint32_t, 4>` or wider.
4. **SLM read bandwidth wall**: Practical/peak SLM read throughput on GEMM kernels sits at **`3.29–3.77 TB/s`** (VTune `full-compute` `[MEASURED]`). Near `3.3+ TB/s`, increase M/N reuse per loaded SLM byte (or use `dpasw` to halve A reads) rather than micro-tuning load instructions.
