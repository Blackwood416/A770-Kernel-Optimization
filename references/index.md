# Reference Index

Start with [`SKILL.md`](../SKILL.md) for the 1-minute engineering quick sheet, task router, evidence labels, timing/oneDNN contracts, and safety gates. Every reference page below is `[MEASURED]` on Intel Arc A770 (DG2, Windows 11, oneAPI `2026.2`, driver `32.0.101.9150`, `oneDNN 3.13.2`) unless a narrower scope is noted; re-measure before transferring constants to another GPU or stack.

## Hardware, Bandwidth, and ISA

| Topic | Page | Covers |
|---|---|---|
| Hardware, occupancy, & SLM | [hardware/hardware.md](hardware/hardware.md) | Xe-core/DSS/L3 terminology, `128-GRF vs 256-GRF` residency matrix (`WG=32 + SLM <= 48 KB`), `82 Gdpas/s` plateau, SLM read/write banking & `16 B` alignment |
| Bandwidth & roofline | [hardware/bandwidth.md](hardware/bandwidth.md) | `403.1 GB/s` DRAM read / `464.6 GB/s` write / `3.43 TB/s` L3 roofs, message width & stride penalties, Domain I vs Domain II taxonomy |
| Cache geometry & hints | [hardware/cache-geometry.md](hardware/cache-geometry.md) | L1/L3/DRAM latency ladder, `gid & gid+32` co-residency, 2D `8x8` snake & `swizzle <= 4`, `L1StreamB` (`st.ca`), C store `wb.wb` vs `st.uc` |
| DPAS / XMX & `0-Dummy` | [isa/dpas.md](isa/dpas.md) | `SIMD8` legality, `K=64/32/16` low-bit/bf16 rules, `dpasw<8,8>` (`92.5 TFLOPS`), IGC `dummy_dpasw (null:f)` & `0-Dummy` (`2x8`/`3x4`) theorem |
| SWSB, barriers, & GRF banks | [isa/swsb.md](isa/swsb.md) | 16-SBID saturation & `Short UGM waits (<120 ins)` monotonicity, SLM addr chains, calibrated latencies, `split_barrier`, zero-cost GRF banks |
| LSC messages & addressing | [isa/lsc-messages.md](isa/lsc-messages.md) | `256 B` cap, `load_2d`/prefetch bans, cache policy encodings, stateful BTI vs ExBSO, DG2 A64 `bp_vec` (`-94 ins`), reverse `pack_a`, `send.dc0`, atomics |
| Compiler knobs & `-Xfinalizer` | [codegen/knobs.md](codegen/knobs.md) | SYCL/ESIMD VC `-Xfinalizer` optimal 4-flag set & ICE/regression blacklist, `-O2`/unroll=2, OpenCL `ocloc` flag boundaries |

## Operator Guides

| Topic | Page | Covers |
|---|---|---|
| Dense GEMM, SwiGLU, GEMV, RMSNorm, Softmax | [techniques/techniques.md](techniques/techniques.md) | `GRF=256 cab4` scorecard (`33–35 WIN \| 0 LOSE` vs `oneDNN 3.13.2`), horizontal `SwiGLU`, `INT8` `129.6 TOPS`, `bf16`/`f32` GEMV, RMSNorm dispatch, Softmax |
| Weight-only `INT4` / `NF4` | [weight-only-gemv.md](weight-only-gemv.md) | `BK=gs=64` zero-branch online dequant + SLM LUT `dpasw` (`22.5–24.6 TFLOPS`), `2.5x–3.6x` decode GEMV, accuracy-matched `oneDNN u4` contract |
| Attention / SDPA (Decode & Prefill) | [techniques/sdpa-a770.md](techniques/sdpa-a770.md) | Decode `Q=1` `SDPA_ORDER=1 + SDPA_REDUCE=1` (`386–391 GB/s`), Paged KV, Prefill ESIMD `WG=32`/`RPT=4,8`, stable online-softmax rescale |
| Reductions & scans | [techniques/reductions-scan.md](techniques/reductions-scan.md) | Sub-group-first `global_atomic` vs `tree`, Hillis-Steele `WG=64`, `reduce_over_group` SIMD narrowing fix |
| Irregular shapes, sparse, & conv | [techniques/irregular-shapes.md](techniques/irregular-shapes.md) | Alignment gates, non-16-aligned GEMV `35x–41x` cliff, CSR vs BSR `B4` sparse GEMM, NHWC `OC`-blocked direct conv |
| Precision, math, & tolerances | [techniques/numerics.md](techniques/numerics.md) | Verification tolerance matrix, mandatory `f32` accumulators, `math.exp` throughput, `mad` vs `fma`, free inline `bf16` mov, zero-cost denorms |

## Runtime, Workflow, APIs, and Code

| Topic | Page | Covers |
|---|---|---|
| Execution, L0 queues, & fusion | [workflow/execution.md](workflow/execution.md) | `ordinal 0/1` topology, UR v1/v2 trade-off, 7-path submit ladder (`l0_reg_replay` `0.484 us/edge`), `UncachedHost` doorbells, UMF cliff, Tier-3 `.zebin` fusion |
| Playbook & capability matrix | [workflow/performance-playbook.md](workflow/performance-playbook.md) | Symptom -> ISA -> action lookup table and pure-English DG2 ISA capability matrix |
| Pitfalls & negative results | [workflow/pitfalls.md](workflow/pitfalls.md) | Deduplicated API/hardware gates, `spillMemSize==0` false negative, SROA `priv` spills, WDDM/L0 traps, and correctness hazards |
| Instruments, audit, & safety | [workflow/instruments.md](workflow/instruments.md) | Optimization methodology, VTune `0.05 ms` OA / GT-PIN, `zeModuleGetNativeBinary` + `ocloc disasm` 2-iter audit, Windows TDR safety protocol |
| APIs, baselines, & commands | [api/api-usage.md](api/api-usage.md) | Modern ESIMD/SYCL/Level Zero APIs, `oneDNN` baseline recipes, Windows build/disasm/VTune commands |
| Curated code snippets | [api/code-snippets.md](api/code-snippets.md) | 10 copy-ready production cores (`GRF=256 cab4`, `-Xfinalizer`, Cache Hints, `4-bit`, Domain-I ops, SDPA rescale, Reduction/Scan, Native L0 / Tier-3) |
