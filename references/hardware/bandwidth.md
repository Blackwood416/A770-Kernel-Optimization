# A770 Bandwidth Ceilings and Empirical Roofline Reference

> Scope: Intel Arc A770 (DG2), Windows 11, oneAPI 2026.1–2026.2, driver `32.0.101.8860` / `32.0.101.9150`. Always state the **access pattern** alongside any roofline denominator; re-measure before transferring to another GPU.

## 1. Authoritative Bandwidth Roofs `[MEASURED]`

Always distinguish the saturated `float4` streaming roof from narrower single-pattern benchmarks:

| Pattern / Level | Configuration | Measured Roof | Usage |
|---|---|---:|---|
| **DRAM Read (default roof)** | `512 MiB`, `float4`/lane, `8` in-flight, `WG=256` | **`403.1 GB/s`** (`417.2 GB/s` at `4` in-flight) | **Primary denominator for Domain I streaming kernels** (GEMV, RMSNorm, Softmax, SDPA) |
| **DRAM Write** | `512 MiB` store, same shape | **`464.6 GB/s`** | Store-heavy kernels (`+15%` above read because read is latency-limited) |
| **L3 (L2) Read** | `8 MiB` resident, `reps=512` | **`3,429.8 GB/s`** | L3-resident tile denominator (short `~160 us` kernels drop to `~1,650 GB/s`) |
| **SLM Read (practical GEMM wall)** | `256-GRF` / `oneDNN` GEMM (VTune `full-compute`) | **`3.29–3.77 TB/s`** | Practical XMX operand feeding ceiling (microbenchmark `256 B` read peaks at `4.12 TB/s`) |
| `B_DRAM_contiguous` (256 B-msg SYCL) | `256 MiB` contiguous, logical `256 B` `sycl::vec` | `291.1 GB/s` read / `340.0 GB/s` copy | **Narrower pattern only** -- quoting `291.1 GB/s` under a `float4` kernel understates utilization by `~38%` |
| `B_DRAM_strided` (256 B-msg, 1 KB stride) | `64 MiB`, stride `256` floats (`1024 B`) | `277.0 GB/s` read / `328.8 GB/s` copy | Strided DRAM-like baseline (`copy` counts read + write bytes combined) |

## 2. Message Width and Stride Penalties `[MEASURED]`

Measured with logical `sycl::vec` loads/stores over `64 B`-aligned USM (`f32`, `4,096` workers):

| Load Width (at `1024 B` stride) | Read GB/s | Copy GB/s | Reduce GB/s |
|---:|---:|---:|---:|
| `32 B` | `104.0` | `67.9` | `100.1` |
| `64 B` | `206.1` | `247.7` | `203.3` |
| `128 B` | `225.7` | `270.9` | `226.9` |
| `256 B` | **`277.0`** | **`328.8`** | **`273.4`** |

| Stride (at `256 B` width) | Stride Bytes | Read GB/s | Copy GB/s | Notes |
|---:|---:|---:|---:|---|
| `1` float | `4 B` | `854.6` | `1,300.9` | L3-resident contiguous (`B_L2_reuse` SYCL ceiling) |
| `16` floats | `64 B` | `536.4` | `764.1` | Partial cache-line utilization |
| `256` floats | `1,024 B` | `277.0` | `328.8` | Drops to `~32%` of stride-1 rate |

## 3. Compute Ceilings and Empirical Roofline Ridges `[ARCH]` `[MEASURED]`

| Layer | Compute Peak (`bf16` XMX) | Memory Roof | Ridge Point |
|---|---:|---:|---:|
| Hardware theoretical (`~2.1 GHz`) | `137.63 TFLOPS` | `560 GB/s` spec | `245.8 FLOP/B` |
| Pure `dpas.8x8` microbenchmark (`82 Gdpas/s`) | `168 TFLOPS` (instruction limit) / `100.6 TFLOPS` (verified GEMM microkernel) | `403.1 GB/s` (`float4` read) | `249.6 FLOP/B` (at `100.6 TF`) |
| L3-resident XMX tile | `100.6 TFLOPS` | `3,429.8 GB/s` | `29.3 FLOP/B` |

Note: Never use the `39.32 TFLOPS` FP32 vector peak as the roofline denominator for `bf16`/`f16`/`int8` XMX kernels -- it inflates the compute floor by `~4x`.

## 4. Two-Domain Operator Taxonomy `[MEASURED]`

Every A770 compute kernel falls into one of two cleanly separated microarchitectural domains (separated by a `~60` percentage-point gap in DRAM/L3 pass-through):

| Domain | Operators | `XMX active` | `DRAM / L3` Pass-Through | Binding Limiter & Strategy |
|---|---|---:|---:|---|
| **Domain I: DRAM Stream-Through** | GEMV, RMSNorm, Softmax, Decode SDPA, Elementwise DAGs | `0.0%` | **`79%–99.5%`** | Target `403.1 GB/s` read roof (`311–401 GB/s` achieved). Use `vec16` / `float4` coalesced loads, interleaved address streams (`SDPA_ORDER=1`), single-pass SLM staging for multi-pass row ops, and DAG fusion. |
| **Domain II: L3-Resident XMX Matrix** | `bf16` / `int8` / `4-bit` GEMM, `SwiGLU`, Prefill SDPA | **`5%–20%+`** | **`10%–19%`** (`81%–89%` L3 hit) | Bound by instruction volume, SWSB/SBID stalls, barrier exposure, and SLM/L3 locality -- not DRAM peak. Use `GRF=256 + WG=32 + cab4`, `dpasw<8,8>`, `0-Dummy` grouping, `-Xfinalizer`, and 2D swizzle. |

## 5. Practical Rules

1. **Always match the denominator to the pattern**: Judge streaming Domain I kernels against `403.1 GB/s` read / `464.6 GB/s` write; if a kernel hits `>= 85%` of `403.1 GB/s`, stop chasing bandwidth and cut redundant passes (fuse operators) or launch overhead.
2. **Widen transfers up to `256 B` on DG2**: At `1024 B` stride, `256 B` reads beat `32 B` reads by `2.7x` vs `32 B` baseline (and `4.8x` for copy).
3. **Sweep outstanding depth**: `4` in-flight `float4` requests (`417.2 GB/s`) beat `8` in-flight (`403.1 GB/s`) in both runs -- more in-flight requests are not automatically faster.
4. **Ignore unstable microbenchmark cells**: Configs with run-to-run `CV > 5%` (e.g., narrow `32/64 B` stride-1 reads, `256 B` SLM writes, `1 GiB` contiguous trials) must not be promoted as roofline ceilings.
