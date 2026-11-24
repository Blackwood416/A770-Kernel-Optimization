# A770 Execution Model, Level Zero Queues, Copy Engine, and Graph Reference

This page combines the complete **P14 Windows WDDM + Level Zero Runtime Atlas**
(`docs/99`-`docs/102`, `probes/123`-`probes/126`, claims `C0348`-`C0355`,
measured on Intel Arc A770 PCI `0x56a0`, Windows 11 25H2, oneAPI 2026.2,
driver `32.0.101.9150`, Level Zero SDK `1.33.1` / driver `1.17.39894+4`,
two independent runs per config, all `errors == 0`) with the earlier 2026-09
two-stage submission campaign preserved at the bottom for historical context.

For INT2/INT4 DPAS and update-capability gates, also read
[DG2 capability findings](dg2-lowbit-graph.md). Copy-ready native Level Zero
snippets live in [code-snippets.md](../api/code-snippets.md#p14-native-level-zero-runtime-cores).

## 1. Level Zero Hardware Engine Topology and UR v1 vs v2 Split (`docs/99`, `C0348`)

`[ARCH]` `[MEASURED]` (`probes/123-l0-topology-copy/work/topology.json`):
Arc A770 (512 EUs, 15.559 GiB DDR, 19.2 MHz timer) exposes exactly **2**
hardware command queue groups under Windows 11 WDDM:

| Ordinal | `numQueues` | Hardware Flags | Role and Architectural Consequence |
|---|---:|---|---|
| `ordinal 0` | `1` | `COMPUTE\|COPY\|COOPERATIVE` | Single CCS compute engine; **all multi-queue compute submissions serialize strictly in hardware** (`0.00%` timestamp overlap, `docs/101`) |
| `ordinal 1` | `1` | `COPY` (copy-only) | Single Main Copy Engine (no Link Copy Engine exposed); achieves true hardware overlap with `ordinal 0` compute and faster D2D (`+9.44%` vs `ordinal 0`) |

`[TOOLCHAIN]` `[MEASURED]` **Unified Runtime (UR) v1 vs v2 divergence** (`docs/99`, `docs/100`):

- **Default UR L0 v1**: `sycl::get_native<backend::ext_oneapi_level_zero>(q)` on
  an `in_order` queue returns `ze_command_queue_handle_t` (backed by dynamic
  regular-list batching); `ext_oneapi_register_host_memory` is `false`.
- **UR L0 v2 (`UR_LOADER_USE_LEVEL_ZERO_V2=1`)**: returns
  `ze_command_list_handle_t` (Immediate Command List by default) and unlocks
  `ext_oneapi_register_host_memory = true` (runtime verified; Windows
  `SeLockMemoryPrivilege` + 2 MB `VirtualAlloc(MEM_LARGE_PAGES)` also verified).
- **Critical warning**: UR v2 is a sharp double-edged sword on Windows WDDM — it
  accelerates captured SYCL graphs (`-37.48%` vs v1 graph), yet regresses
  uncaptured `in_order` chains by **`7.93x`** vs v1 `in_order` and collapses
  dual-SYCL-queue Compute+Copy hardware overlap from `99.95%` to `<= 0%`.

## 2. Five-Mode Host Memory PCIe and D2D Bandwidth Matrix (`docs/99`, `C0349`)

`[MEASURED]` (`probes/123-l0-topology-copy/work/bandwidth_matrix.csv`,
two-run conservative worst median, `0` errors across all 90 configs):

| Mode / Engine | Direction & Size | Wall Med (us) | Dev Med (us) | Wall BW (GB/s) | Dev BW (GB/s) | Engineering Takeaway |
|---|---|---:|---:|---:|---:|---|
| `pageable` (`sycl_q`) | H2D / D2H `32 MB` | `5293.0` / `4985.5` | `3380.9` / `3081.7` | `6.339` / `6.730` | `9.925` / `10.888` | Pays `~1710 us` CPU staging overhead |
| `usm_host` (`sycl_q`) | H2D / D2H `32 MB` | `3583.7` / `3278.7` | `3380.8` / `3081.0` | `9.363` / `10.234` | `9.925` / `10.891` | `1.48x` / `1.52x` faster than `pageable` baseline |
| `registered_host` (`ur_v2`) | H2D / D2H `32 MB` | `3552.8` / `3244.2` | `3381.0` / `3081.4` | `9.444` / `10.343` | `9.925` / `10.890` | `4 KB` H2D wall drops to `110.10 us` (`1.68x` vs v1 `usm_host` `184.85 us`) |
| `uncached_host` (`sycl_q`) | H2D / D2H `32 MB` | `3579.6` / `3280.3` | `3380.6` / `3080.8` | `9.374` / `10.229` | `9.926` / `10.891` | `ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED` has zero DMA bandwidth penalty |
| `usm_shared` (`sycl_q`) | H2D `32 MB` | `252.1` | `151.4` | `133.1` | `221.6` | After warmup GPU page migration without CPU re-touch, executes as GDDR6 D2D |
| `d2d` (`l0_compute` / `l0_main_copy`) | D2D `32 MB` | `215.2` / `196.6` | `151.2` / `138.2` | `155.9` / `170.717` | `221.847` / `242.791` | Main Copy Engine beats Compute ordinal by `+9.44%` and PCIe H2D by `22.4x` |
| `usm_host` (`sycl_q` / `l0_main_copy`) | H2D `64 KB` | `186.45` / `59.90` | `10.2` / `9.8` | `0.35` / `1.09` | `6.4` / `6.7` | Native `ordinal 1` Copy Engine cuts small-transfer wall latency `3.11x` vs `sycl_q` |

## 3. Compute + Copy Engine Hardware Overlap and Full-Duplex (`docs/99`, `C0350`)

`[MEASURED]` (`probes/123-l0-topology-copy/work/overlap_matrix.csv`, paired
against a deterministic `~389.6 us` device / `~525 us` wall compute kernel,
`0` compute & copy errors):

| Topology | Payload | Compute Alone (us) | Copy Alone (us) | Concurrent Wall (us) | Hardware Dev Overlap | Notes |
|---|---|---:|---:|---:|---:|---|
| `single_sycl_q` | `2 MB` H2D / `8 MB` D2H | `525.5` / `525.5` | `370.0` / `951.2` | `720.9` / `1453.0` | `-0.78%` / `-20.95%` | Strictly serialized; small-size wall formula shows pseudo-overlap from amortizing 1x `~100 us` WDDM kick |
| `dual_sycl_v1` (2 SYCL `in_order` queues, UR v1) | `2 MB` H2D / `8 MB` D2H | `525.4` / `525.4` | `365.1` / `950.2` | `523.7` / `923.8` | `99.95%` / `91.69%` | True hardware overlap (`89.48%` at `8 MB` H2D, `58.00%-58.63%` at `16 MB`) |
| `dual_sycl_v2` (`UR_LOADER_USE_LEVEL_ZERO_V2=1`) | `2 MB` H2D / `8 MB` D2H | `524.8` / `524.8` | `338.1` / `880.4` | `759.5` / `1351.2` | `-2.01%` / `-29.07%` | **Critical UR v2 regression**: hardware overlap collapses to `<= 0%` at every size |
| `native_l0_main_copy` (`ordinal 0` + `ordinal 1`) | `2 MB` H2D / `8 MB` D2H | `506.0` / `506.0` | `247.1` / `820.3` | `506.2` / `921.3` | `99.95%` / `91.58%` | Lowest copy latency (`86.7 us` @ `256 KB`, `-58.0%` vs `dual_sycl_v1`), `71.29%-72.06%` dev overlap @ `16 MB`, `-36.59%` wall vs `single_sycl_q` |

- **Bidirectional PCIe saturation (`docs/99`)**: because A770 exposes only one
  Copy Engine (`ordinal 1`), concurrent `8 MB` H2D + `8 MB` D2H saturates at
  half-duplex (`10.326 GB/s` dev on `ordinal 1`; adding `ordinal 0` drops to
  `9.744 GB/s` with `-9.89%` dev overlap `[MEASURED]`).

## 4. Seven-Path Submission Ladder, Batch-Closing Cliff, and Graph Replay (`docs/100`, `C0351`-`C0352`)

`[MEASURED]` (`probes/124-submit-graph-tax/accepted.json`, deterministic
`2560`-element hidden-state RAW chain, `10 WGs x 256`, fixed-address
`GraphControl` indirect parameter updates, all `errors == 0`):

| Submission Path | `L=1` (us) | `L=4` (us) | `L=16` (us) | `L=64` (us) | `L=256` (us) | `L=256` / Kernel (us) | `L=256` Host Submit (us) | `L=64` Dev Gap (us/edge) | Setup (us) / Break-even vs `v1_inorder` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `sycl_v1_ooo` | `206.25` | `322.05` | `279.05` | `514.40` | `1534.85` | `5.996` | `1228.55` | `37.435` | — |
| `sycl_v1_inorder` | `94.55` | `194.25` | `269.30` | `468.65` | `1236.70` | `4.831` | `943.90` | `36.008` | baseline |
| `sycl_v1_imm` | `185.35` | `327.10` | `773.95` | `2742.00` | `10858.20` | `42.415` | `3471.35` | `47.080` | — (`8.78x` slower than `v1_inorder`) |
| `sycl_v2_inorder` | `88.90` | `250.10` | `706.35` | `2494.20` | `9810.75` | `38.323` | `3128.15` | `43.746` | — (`7.93x` slower than `v1_inorder`) |
| `sycl_graph_v1` | `174.65` | `173.25` | `153.05` | `256.20` | `682.60` | `2.666` | `43.25` | — | `2180.3` / `4` replays (`-44.80%`) |
| `sycl_graph_v2` | `75.10` | `79.50` | `96.55` | `164.00` | `426.75` | `1.667` | `28.90` | — | `1466.6` / `2` replays (`-65.49%` vs `v1_inorder`, `-37.48%` vs `graph_v1`) |
| `l0_imm_async` | `81.85` | `318.80` | `1235.35` | `4775.40` | `19122.95` | `74.699` | `3114.75` | `67.550` | — (`15.46x` slower than `v1_inorder`) |
| `l0_reg_replay` | `64.20` | `67.80` | `80.65` | `129.55` | `317.15` | `1.239` | `18.10` | `0.484` | `267.2` / `1` replay (`-74.35%` vs `v1_inorder`) |

- **Single-shot advantage of native `l0_reg_replay`**: even for a dynamic shape
  that runs **only once** (`L=256`), recording + closing (`267.2 us`) +
  executing (`317.15 us`) totals **`584.35 us`** — still `2.12x` faster than
  `sycl_v1_inorder` (`1236.70 us`) and `16.79x` faster than `sycl_v2_inorder`
  (`9810.75 us`) `[MEASURED]` (`docs/100`).
- **UR v1 dynamic batch-closing cliff (`L=32`, `docs/100`)**: injecting a single
  mid-chain `4 KB` pageable D2H `memcpy` on the compute queue closes UR v1's
  dynamic batch and inflates total wall time **`+80.45%`** (`390.45 -> 704.55 us`,
  second-half submit `72.55 -> 107.7 us`). Moving that copy to a second
  `in_order` `copy_q` with `compute_q.ext_oneapi_submit_barrier()` recovers
  **`-35.72%`** (`452.85 us`), and a bare `ext_oneapi_submit_barrier()` does not
  close the batch (`320.65 us`). Under UR v2, the clean baseline already sits at
  `1308.25 us` (`3.35x` slower than v1 `clean`) because immediate command lists
  have no dynamic batch left to preserve `[MEASURED]`.
- **Monolithic graph vs subgraph splitting (`L=256`, `docs/100`)**: splitting a
  `1x256` monolithic graph into `8x32` / `32x8` subgraphs inflates wall time
  from `801.8` to `2218.45` (`2.77x`) / `7592.3 us` (**`9.47x`**) under UR v1,
  and from `427.2` to `854.4` (`2.00x`) / `2274.4 us` (**`5.32x`**) under UR v2
  `[MEASURED]`. Always capture whole-token monolithic graphs.
- **Regular vs Immediate `zeEvent` inversion (`L=64`, `docs/100`)**: inside a
  **Regular Command List**, prefer `zeCommandListAppendBarrier` (**`130.85 us`**);
  attaching per-node `device` (`248.8 us`, `+90.14%`), `timestamp` (`259.85 us`,
  `+98.59%`), or `host_visible` (`295.3 us`, `+125.68%`) `zeEvent`s roughly
  doubles dispatch cost. Inside an **Immediate Command List**, the order flips:
  `AppendBarrier` forces a per-step WDDM flush (`4823.15 us`) and is slower than
  `event_device` (`2565.8 us`) `[MEASURED]`.

## 5. Multi-Queue Compute Concurrency, Doorbell Coherency, and Memory/Event Pools (`docs/101`, `C0353`-`C0355`)

`[ARCH]` `[MEASURED]` (`probes/125-concurrency-coherency/accepted.json`):

1. **Zero multi-queue compute concurrency (`0.00%` hardware overlap)**: across
   all 6 queue topologies (`single_inorder`, `single_ooo`, `dual_sycl_v1`,
   `dual_sycl_v2`, `dual_l0_imm`, `dual_l0_reg`) x 5 work-group counts
   (`2/4/8/16/32` WGs, `~497-500 us` pure register FMA, `60` runs, `0` errors),
   hardware timestamp interval overlap is **exactly `0.00%`**. Wall-clock
   overlap formulas show `13.63%-21.15%` false positives solely by amortizing
   `92.60-143.43 us` of WDDM submit/wake tax across back-to-back submissions.
   **Rule**: never split small compute ops across queues expecting Xe-core
   concurrency; horizontally fuse them into one kernel.
2. **GPU-polled host doorbell coherency and TDR calibration**:
   - In 10-round in-kernel polling visibility tests, **only**
     `ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED` (`uncached_host`) passes (`10/10`,
     `0` timeouts, `0` checksum errors); cached `sycl::malloc_host` and
     `sycl::malloc_shared` both stall on round 1 (`0/10`, stale GPU cache).
   - Over 200 ping-pong rounds (`0 B` pure doorbell), `uncached_host` with
     annotated pointer load hits **`6.00 us`** (`atomic_sys`: `8.90 us`),
     beating native L0 event-split (`zeEventHostSynchronize` + `zeEventHostSignal`,
     `78.00 us`) by **`13.00x`**.
   - Single-thread reads of a `4 KB` uncached payload hit a `~75 us` serial PCIe
     latency wall (`81.50 us` scalar / `77.20 us` `vec16`, `1.77x-1.87x` vs
     `144.60 us` event-split); wake with 1 thread on the doorbell, then read
     KB-scale payloads with parallel threads or DMA.
   - **TDR spin budget**: one uncached host spin takes **`1123.12 ns`**
     (`1183.72 ns` `atomic_sys`) vs `154.3 ns` cached. A naive `5,000,000` spin
     cap takes `5.62-5.92 s` and trips Windows 2 s WDDM TDR; a `40 ms` safe
     budget allows only `~35,615` spins.
3. **UMF `>= 16 MB` allocation cliff and static `zeEventPool` reuse**:
   - Under oneAPI 2026.2 (both UR v1 and v2), `sycl::malloc_device + free`
     hits the UMF pool at `4 KB / 64 KB / 1 MB` in **`0.60-0.70 us`**
     (`119.7x-140.9x` faster than bare `zeMemAllocDevice` `83.80-84.85 us`).
   - At **`>= 16 MB`**, allocations spill out of UMF and spike to **`636.70 us`**
     (`16 MB`) and **`1863.35 us`** (`64 MB`) — **`4.4x` / `3.7x` slower** than
     bare `zeMemAllocDevice` (`145.20` / `497.75 us`). `16 MB`
     `zeMemAllocHost_uncached` reaches `8660.90 us` (P90 `10355.0 us`).
   - `zeEventPoolCreate + Destroy` costs **`102.80-232.35 us`**, whereas pooled
     `zeEventCreate + Destroy` takes `0.50-0.60 us` and reusing events via
     **`zeEventHostReset`** takes only **`0.10-0.20 us`** (`1028x-1162x` faster
     than dynamic pool creation).

## 6. Three-Tier Hybrid Fusion and Zero-Tax Runtime Blueprint

Combine P12's two-domain taxonomy (`docs/94`) with P14's runtime measurements
(`docs/99`-`docs/102`) when building a custom A770 operator library:

1. **Tier 1 — Control-plane fusion (Native L0 Regular Command List + Arena)**:
   record logical op graphs into a lightweight CPU IR, run liveness packing into
   one pre-allocated Device Arena (zero `>= 16 MB` hot allocations), route async
   DMA to explicit `ordinal 1` Copy Engine with a static `zeEventPool`, and
   lower compute nodes to a closed Regular Command List using
   `zeCommandListAppendBarrier` + fixed-address `GraphControl` indirect params
   (`0.484 us/edge` gap, `1`-replay break-even).
2. **Tier 2 — Domain II matrix fusion (Static AOT C++20 Prologue/Epilogue)**:
   keep XMX DPAS/DPASW cores hand-tuned and spill-free; absorb adjacent bias,
   activation (`SiLU`/`GELU`), residual add, dual-tile `SwiGLU`, weight
   dequant (`INT4`/`NF4`), or output quantization via compile-time functors
   matched during Tier-1 graph lowering (0 ms JIT tax, zero extra DRAM passes).
3. **Tier 3 — Domain I streaming fusion (Measured In-Process DAG Compiler + `.zebin` Cache) `[MEASURED]`**:
   fuse arbitrary contiguous elementwise, broadcast, and row-reduction chains
   (`Residual + RMSNorm + Quant`, `RoPE + KV-Write`, `Add->Bias->Scale->SiLU->Mul`)
   by hashing the sub-DAG (`SHA-256(source + deviceId + driverVersion)`):
   - **Cold cache**: generate `vec16`/sub-group OpenCL-C source, compile to SPIR-V
     via in-process **`ocloc64.dll` (`oclocInvoke -spv_only -device dg2 -options "-cl-std=CL2.0"`)**,
     create the module via `zeModuleCreate(ZE_MODULE_FORMAT_IL_SPIRV)`, and persist
     **`zeModuleGetNativeBinary`** (`.zebin`) to disk.
   - **Warm / cross-process hit**: load the cached `.zebin` directly with
     **`ZE_MODULE_FORMAT_NATIVE`** (0 ms JIT tax) and inject `ze_kernel_handle_t`
     into the Tier-1 Regular Command List.
   - **Measured payoff (A770, `1024x4096 f32` 5-op chain, 10x-batched L0 Regular Replay, `errors=0`)**:
     collapses `5` unfused kernels (`458.30–458.33 us`) into `1` fused kernel at
     **`167.93–168.00 us` (`399.5–399.6 GB/s` = `99.1%` of DRAM roof, `2.73x` faster vs unfused baseline)**;
     4-op `GELU+RMSNorm` reduction fusion also passes with `0 errors`.

## 7. Historical 2026-09 Two-Stage Submission Campaign (oneAPI 2026.1 / Driver `32.0.101.8724`)

`[MEASURED]` Legacy baseline preserved for provenance: VNNI pack of B +
joint-matrix bf16 GEMM (`1024x1536x512`, `0/1572864` errors, 30 warmup + 200
timed iterations, 3 stable runs; graph node times cross-checked via VTune
`xpu-offload`):

| Form | Wall avg us | Pack us | GEMM us | Launch gap us | Host overhead us |
|---|---:|---:|---:|---:|---:|
| same-queue | 288.23 | 17.62 | 148.65 | 18.36 | 121.96 |
| dual-queue + event | 271.13 | 17.97 | 149.00 | 15.34 | 104.17 |
| fused single kernel | 335.81 | 0 | 241.92 | 0 | 93.88 |
| graph batch submit | 260.64 | 16.92 VTune | 161.25 VTune | n/a | ~93.7 est / ~82.6 VTune |
| host-pack-out | 235.51 | 0 | 153.63 | 0 | 81.86 |
| host-pack-in | 1311.30 | 0 | 153.27 | 0 | 383.31 + CPU 774.73 |

Historical rules that still hold: fuse only when `fused device delta < launch gap + host overhead saved`, keep host preprocessing outside timed loops for static weights, and cross-check SYCL graph node times with VTune `xpu-offload` or native Level Zero timestamps because SYCL graphs do not expose per-node profiling events.
