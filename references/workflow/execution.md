# A770 Execution Model, Level Zero Queues, Copy Engine, and Fusion Reference

> Scope: Intel Arc A770 (DG2, PCI `0x56a0`), Windows 11 25H2 WDDM, oneAPI 2026.2, driver `32.0.101.9150`, Level Zero SDK `1.33.1`. Copy-ready native Level Zero primitives live in [code-snippets.md](../api/code-snippets.md#10-native-level-zero-runtime-copy-engine-doorbell-and-tier-3-zebin-cache).

## 1. Level Zero Engine Topology and UR v1 vs v2 Split `[ARCH]` `[MEASURED]`

A770 exposes exactly **2** hardware command queue groups under Windows WDDM:

| Ordinal | `numQueues` | Flags | Consequence |
|---|---:|---|---|
| `ordinal 0` | `1` | `COMPUTE\|COPY\|COOPERATIVE` | Single CCS compute engine; **multi-queue compute submissions serialize strictly in hardware** (`0.00%` timestamp overlap across `2–32` WGs) |
| `ordinal 1` | `1` | `COPY` (copy-only) | Single Main Copy Engine; achieves true hardware Compute+Copy overlap (`91.6%–99.95%`) and faster D2D (`242.8 GB/s`, `+9.44%` vs `ordinal 0`) |

**Unified Runtime (UR) v1 vs v2 trade-off `[TOOLCHAIN]` `[MEASURED]`**:

- **Default UR L0 v1**: `in_order` queues use dynamic regular-list batching; dual SYCL `in_order` queues (`compute_q` + `copy_q`) achieve `99.95%` (`2 MB`) / `91.7%` (`8 MB`) hardware Compute+Copy overlap.
- **UR L0 v2 (`UR_LOADER_USE_LEVEL_ZERO_V2=1`)**: Switches `in_order` queues to Immediate Command Lists and enables `ext_oneapi_register_host_memory`. It speeds up **captured** SYCL graphs (`-37.5%` vs v1 graph), **but regresses uncaptured `in_order` chains by `7.93x` vs v1 and collapses dual-SYCL Compute+Copy overlap from `99.95%` to `<= 0%`**.
- **Rule**: Never enable `UR_LOADER_USE_LEVEL_ZERO_V2=1` globally on uncaptured loops; bind async DMA explicitly to native Level Zero `ordinal 1`.

## 2. Host Memory, PCIe, and Compute + Copy Overlap `[MEASURED]`

| Transfer Mode (`32 MB` unless noted) | H2D / D2H Wall BW | Dev BW | Engineering Takeaway |
|---|---:|---:|---|
| `pageable` (`sycl_q`) | `6.34 / 6.73 GB/s` | `9.93 / 10.89 GB/s` | Pays `~1,710 us` CPU staging tax |
| `usm_host` (`sycl_q`) | `9.36 / 10.23 GB/s` | `9.93 / 10.89 GB/s` | `1.48x–1.52x` faster wall vs `pageable` baseline |
| `uncached_host` (`BIAS_UNCACHED`) | `9.37 / 10.23 GB/s` | `9.93 / 10.89 GB/s` | Zero DMA bandwidth penalty vs cached `usm_host` |
| `d2d` (`l0_main_copy` `ordinal 1`) | `170.7 GB/s` wall | **`242.8 GB/s`** | `+9.44%` faster vs compute ordinal; `64 KB` H2D wall drops `3.11x` (`59.9` vs `186.5 us` `sycl_q`) |
| `native_l0_main_copy` overlap (`2 / 8 / 16 MB`) | — | **`99.95% / 91.6% / 71.3%–72.1%`** dev overlap | Lowest copy latency (`-58%` at `256 KB` vs `dual_sycl_v1`); bidirectional H2D+D2H saturates at half-duplex (`10.3 GB/s`, single Copy Engine) |

## 3. Seven-Path Submission Ladder and Graph Replay (`L = 1..256` RAW Chain) `[MEASURED]`

| Submission Path | `L=1` | `L=16` | `L=64` | `L=256` | `L=256` / Kernel | `L=64` Dev Gap | Setup / Break-Even vs `v1_inorder` |
|---|---:|---:|---:|---:|---:|---:|---|
| `sycl_v1_inorder` | `94.6 us` | `269.3 us` | `468.7 us` | `1236.7 us` | `4.83 us` | `36.01 us/edge` | baseline |
| `sycl_v2_inorder` | `88.9 us` | `706.4 us` | `2494.2 us` | `9810.8 us` | `38.32 us` | `43.75 us/edge` | **`7.93x` slower vs `v1_inorder`** |
| `sycl_graph_v1` | `174.7 us` | `153.1 us` | `256.2 us` | `682.6 us` | `2.67 us` | — | `2180 us` / `4` replays (`-44.8%`) |
| `sycl_graph_v2` | `75.1 us` | `96.6 us` | `164.0 us` | `426.8 us` | `1.67 us` | — | `1467 us` / `2` replays (`-65.5%` vs `v1_inorder`) |
| `l0_imm_async` | `81.9 us` | `1235.4 us` | `4775.4 us` | `19123.0 us` | `74.70 us` | `67.55 us/edge` | **`15.46x` slower vs `v1_inorder`** |
| **`l0_reg_replay`** | **`64.2 us`** | **`80.7 us`** | **`129.6 us`** | **`317.2 us`** | **`1.24 us`** | **`0.484 us/edge`** | **`267 us` / `1` replay (`-74.4%` vs `v1_inorder`)** |

Critical dispatch laws:

- **Single-shot advantage**: Even for a dynamic shape run **once** (`L=256`), `l0_reg_replay` record + execute (`267 + 317 = 584 us`) beats `sycl_v1_inorder` (`1237 us`) by `2.12x` vs `v1_inorder`.
- **Static vs updatable graphs `[TOOLCHAIN]`**: Query `ext_oneapi_limited_graph` separately from `ext_oneapi_graph` (DG2 omits `KERNEL_ARGUMENTS`/`KERNEL_INSTRUCTION` cmd-list updates). Update dynamic shapes/pointers via a **fixed-address device `GraphControl` parameter struct** without rebuilding the closed command list.
- **Avoid mid-chain pageable D2H on compute queue**: Injecting one `4 KB` pageable D2H closes UR v1's dynamic batch (`+80.5%`, `390 -> 705 us`); move host copies to a separate copy queue (`-35.7%` recovery).
- **Keep graphs monolithic**: Splitting `1x256` into `32x8` subgraphs inflates replay **`9.47x`** (`v1`) / **`5.32x`** (`v2`) vs monolithic graph.
- **`AppendBarrier` vs `zeEvent` inversion**: Inside a **Regular Command List**, use `zeCommandListAppendBarrier` (`130.9 us` vs `248.8–295.3 us` for per-node `zeEvent`s); inside an **Immediate Command List**, `AppendBarrier` forces per-step WDDM flushes (`4823 vs 2566 us`).

## 4. Doorbell Coherency, UMF Allocation Cliff, and Event Pools `[MEASURED]`

1. **GPU-polled host doorbells**: Only **`ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED`** passes visibility (`10/10`, **`6.00 us`** 0 B RTT = `13.0x` faster vs `78.0 us` L0 event-split); cached `malloc_host` and `malloc_shared` stall `0/10`. Calibrate spin caps against **`1.123 us/spin`** (`1.184 us` `atomic_sys`) -- `5,000,000` spins takes `5.6 s` and trips Windows 2 s TDR (`40 ms` budget = `~35,615` spins). Wake with 1 thread on the doorbell, then read KB payloads with parallel threads or DMA.
2. **UMF `>= 16 MB` cliff & static `zeEventPool`**: `sycl::malloc_device` hits the UMF pool at `<= 1 MB` (`0.60–0.70 us`), but spills out at **`>= 16 MB`** (`636.7 us` at `16 MB`, `1863.4 us` at `64 MB`, `3.7x–4.4x` slower than bare `zeMemAllocDevice`). Pre-allocate one **Device Workspace Arena** and one static `zeEventPool` reused via **`zeEventHostReset`** (`0.10–0.20 us`, `> 1,000x` faster vs `zeEventPoolCreate` `103–232 us`).

## 5. Three-Tier Hybrid Fusion Architecture `[MEASURED]`

1. **Tier 1 (Control-Plane Fusion)**: Record whole-token logical graphs into a closed native Level Zero **Regular Command List** (`AppendBarrier` + fixed-address `GraphControl`), backed by a Device Arena, static `zeEventPool`, and `ordinal 1` Copy Engine (`0.484 us/edge`).
2. **Tier 2 (Domain II Matrix Fusion)**: Fuse bias, activation (`SiLU`/`GELU`), residual add, horizontal dual-tile `SwiGLU`, or `4-bit` online dequantization directly inside AOT ESIMD `GRF=256 dpasw` kernels (0 ms JIT tax, zero extra DRAM passes).
3. **Tier 3 (Domain I Streaming DAG Fusion + `.zebin` Cache)**: Hash arbitrary contiguous elementwise / broadcast / row-reduction DAGs (`SHA-256(src + deviceId + driverVersion)`). On cold miss, generate `vec16`/sub-group OpenCL-C -> compile in-process via **`ocloc64.dll` (`-spv_only -device dg2 -options "-cl-std=CL2.0"`)** -> `zeModuleCreate(ZE_MODULE_FORMAT_IL_SPIRV)` -> persist **`zeModuleGetNativeBinary`** (`.zebin`). On warm hit, load via **`ZE_MODULE_FORMAT_NATIVE`** (0 ms JIT) straight into Tier-1 Regular CL. Measured on a `1024x4096 f32` 5-op chain (`Add->Bias->Scale->SiLU->Mul`): **`458.3 -> 168.0 us` (`399.6 GB/s` = `99.1%` of DRAM roof, `2.73x` faster vs unfused baseline)**.
