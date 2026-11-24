# Cache Geometry, Work-Group Co-Residency, and Cache Hints on A770 (DG2)

> Scope: Intel Arc A770 (DG2), Windows 11, oneAPI 2026.1–2026.2, driver `32.0.101.9150`. Note: the shared `16 MB` last-level cache is called `L3` in hardware/VTune metrics and `L2` in SYCL/ESIMD `cache_hint_L2` APIs.

## 1. Latency Ladder and Effective Capacity `[MEASURED]`

Measured via single-thread dependent pointer-chase (`4 KB` stride):

| Working Set | Latency (ns / access) | Cache Tier |
|---|---:|---|
| `32 KB – 1 MB` | **`52 – 64 ns`** | L1-speed (single thread stays hot out to `1 MB`, ~`5x` wider than `192 KB`/Xe-core) |
| `2 MB – 16 MB` | **`118 – 142 ns`** | Shared `16 MB` L3 (L2) |
| `32 MB` | `169 ns` | Transition |
| `64 MB` | `229 ns` | Mostly DRAM |
| `128 MB` | **`366 ns`** | DRAM plateau |

Planning constants: **`~52 ns` L1, `~120–142 ns` L3 (L2), `~366 ns` DRAM**.

Key empirical cache behaviours:

- **L3 line granularity & scaling**: Stride scans over an `8 MB` L3-resident buffer saturate at `128 B` stride (`2.4 TB/s` at 64 threads; `3.43 TB/s` at full-chip `float4` grid-stride), consistent with `128 B` cache lines. Aggregate near-side capacity scales linearly with active threads (`1` thread `77.7 ns` -> `128` threads `1.6 ns/access` aggregate).
- **Dependent-load floor (`~151 ns`)**: Pointer-chasing loops with cross-iteration address dependencies stall at `~151 ns/hop` (`~2.5x` slower than the `52 ns` L1 hit figure). Unroll independent chains or use tree/array layouts instead of linked chasing.
- **Concurrent streaming penalty (`+16%`)**: One concurrent `64 MB` streamer adds `+16%` latency (`52.5 -> 61.2 ns`) to a hot `64 KB` chaser, and `32` streamers add only `1.0 ns` more -- bandwidth contention, not LRU cliff eviction.
- **Associativity**: Unobservable via software timing because per-thread load serialization (`~152–157 ns`) dominates set conflicts, and VTune does not expose per-set miss counters on DG2.

## 2. DG2 32 Xe-Core Round-Robin Co-Residency and Swizzle Strategy `[ARCH]` `[MEASURED]`

DG2 dispatches consecutive work-groups round-robin across its `32` Xe-cores. Therefore **`gid` and `gid + 32` always land on the same Xe-core and share its L1 cache**.

Choose the work-group swizzle map according to the GRF/wave regime (never mix rules across regimes):

1. **`256-GRF + WG=32` (`64 WGs/wave`, 2 WGs/Xe-core) -- Production GEMM `[DISPATCH]`**:
   - **Single-wave / small grids (`total_wgs <= 64`, e.g., `512x4096x4096` = 64 WGs)**: Use **`swizzle = 1`** (linear N-first) with **both A and B set to `ca.ca`** (`L1=cached, L2=cached`) so `gid` and `gid + 32` reuse B in L1 (`206.0–206.7 us`, `1.05x` faster vs `oneDNN 3.13.2` `[MEASURED]`). Setting B to `L1=streaming` hurts single-wave grids.
   - **Multi-wave large grids (`1024x4096x4096`, `2048x4096x2048`, `4096^3`)**: Use a **`2D 8x8 N-inner` serpentine snake walk** (`N` fast within each `8x8` block, alternating stripe direction) so co-resident WGs share `A` in L1 while spreading `B` across L3 channels.
   - **`K >= 4096` L3 capacity cap (`swizzle <= 4`)**: On `4096^3` (`256x128` tile), `swizzle = 8` expands one wave's `A` working set to `2048 * 4096 * 2 B = 16 MB`, thrashing the `16 MB` L3. Capping **`swizzle <= 4`** saves **`-71 us`** (`1486–1495 us`, `1.10x` faster vs `oneDNN 3.13.2` `[MEASURED]`).
2. **`128-GRF + WG=128` (`32 WGs/wave`, 1 WG/Xe-core) -- Legacy 1D M-Band Swizzle `[MEASURED]`**:
   - 1D `SWZ=2` (`band = gid / (SWZ * GRID_N)`, `bm = band * SWZ + rem % SWZ`, `bn = rem / SWZ`) pays **only as a conjunction**: `> 1 wave` AND topological `GRID_N = N / BN >= 11` AND `total_wgs >= 32` (`+4.3% to +7.2%` vs `SWZ=1` baseline by pairing adjacent Xe-cores onto the same `bn` column, shrinking the 12-WG working set by `35%`).
   - Under `128-GRF`, `SWZ=4` overshoots a `32`-WG wave and loses; `total_wgs < 32` with `SWZ=2` loses `-2.7% to -3.1%`.

## 3. LSC Cache Hint Dispatch Rules `[MEASURED]`

Cache hints are **not** universally neutral -- they matter whenever working sets approach L1 (`192 KB`) or L3 (`16 MB`):

| Operand / Path | Recommended Hint | ISA Code | Measured Effect |
|---|---|---|---|
| Pure streaming DRAM kernel (e.g., axpy) | `L1=streaming, L2=uncached` | `st.uc` | `+14%` faster vs default `ca.ca` baseline |
| GEMM `A` loads (2D snake, shared in L1) | `L1=cached, L2=cached` | `ca.ca` | Preserves shared `A` lines across `gid` & `gid + 32` |
| GEMM `B` loads (single-wave `total_wgs <= 64`) | `L1=cached, L2=cached` | `ca.ca` | Maximizes L1 reuse of `B` |
| GEMM `B` loads (multi-wave 2D snake, `L1StreamB`) | **`L1=streaming, L2=cached`** | **`st.ca`** | Prevents streaming `B` from evicting shared `A` in L1 (`2048x4096x2048` `377.4 -> 373.1–374.7 us`) |
| GEMM `C` store (standard / `Bias` / `SiLU` epilogue) | **`L1=write_back, L2=write_back`** | **`wb.wb`** | Coalesces partial writes in L1/L3 (`LMHd 128x32768x2560` **`-42 us`** vs `st.uc`; `wt.wb` regresses `+17 us` on `wide`) |
| GEMM `C` store (`ResidualAdd` / `BiasResidualAdd`) | **`L1=streaming, L2=uncached`** | **`st.uc`** | Prevents `C` writeback from evicting hot `B`/residual in L3 (`2048x2304+BiRes` `0.98x -> 1.06x` vs `oneDNN 3.13.2`) |

Never use `L1=uncached, L2=streaming` (invalid/unsupported combination on DG2). Specialize `L1StreamB` and epilogue store hints at compile time (`if constexpr`) for zero runtime overhead.
