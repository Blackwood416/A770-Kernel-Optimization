# Performance Playbook and DG2 ISA Capability Matrix

> Scope: Intel Arc A770 (DG2), Windows 11, oneAPI 2026.2, driver `32.0.101.9150`. Measure with [instruments.md](instruments.md) first, then pick the matching symptom row below.

## 1. Symptom -> Hardware/ISA Root Cause -> Action `[MEASURED]`

| Measured Symptom | Hardware / ISA Root Cause | Concrete Action | Expected Effect |
|---|---|---|---|
| High `dummy_dpasw (null:f)` (`15%–18%` XMX loss) in `dpasw` disassembly | DG2 lacks src2 suppression; IGC inserts 8-cycle `null:f` when macro head pairs share `src1` ([dpas.md](../isa/dpas.md)) | Apply `0-Dummy` grouping (`2x8` for 4xB or `3x4` for 3xB, max 1 `size=1` head per macro, rest `size>=3`, anchor on tail SLM load `a23`, cap `<=8`); never revert to `dpas` (`-12.4%` vs `dpasw`) | Drops main-loop `dummy_dpasw` from `19/16/11` to `1` |
| Main-loop `send.dc0 > 0` or `privateMemSize = 576–1088 B` despite `spillMemSize == 0` | `KernelDiag.spillMemSize` misses scratch/SROA spills; `acc[4][6]` (`192 GRF`) overflows 256-GRF; runtime `simd[]` indexing breaks LLVM SROA | Cap accumulators `<= 128 GRF` (`acc[4][4]`), compile-time 2-way unroll with named SSA vars, verify `ocloc disasm` `send.dc0 == 0` | Eliminates scratch stalls; recovers `~2x` vs SROA-spilled build (`661 -> 240 us`) |
| `Short UGM waits (<120 ins) >= 5` or `sync.allrd > 0` in 2-iter wrap | DG2 has only 16 SBID tokens (`$0..$15`); over-issuing `send.ugm + send.slm` steals active UGM tokens ([swsb.md](../isa/swsb.md)) | Switch to `cab4` (`48 KB`) + `split_barrier`, coalesce B loads (`128 B`) & SLM stores (`4x64 B`), embed `-Xfinalizer -SBIDBundle -SWSBDepReduction -nopresched -twoSrcBundleBCR` | Drops `Short UGM waits 5 -> 1`, `sync.allrd 5 -> 0`; cuts `4096^3` `1555 -> 1486 us` (`1.10x` vs `oneDNN`) |
| Large `K >= 4096` GEMM thrashes L3, or `ResidualAdd` epilogue trails `oneDNN` | `gid` & `gid+32` share Xe-core L1; `swizzle=8` hits `16 MB` L3 on `4096^3`; C writeback evicts residual/B ([cache-geometry.md](../hardware/cache-geometry.md)) | Use `2D 8x8 N-inner` snake with `swizzle <= 4` (`K >= 4096`) + `L1StreamB` (`st.ca`); C store `wb.wb` for standard/SiLU and `st.uc` for `ResidualAdd` | Saves `-71 us` on `4096^3`, `-42 us` on LM-Head (`wb.wb`), lifts `BiRes` `0.98x -> 1.06x` vs `oneDNN` |
| `GRF=256` GEMM stalls heavily at barriers (`WG=48 / 56 KB` or `WG=64`) | `SLM > 48 KB` or `WG > 32` forces `1 WG/Xe-core` with no second WG to hide barrier/DRAM waits ([hardware.md](../hardware/hardware.md)) | Target **`WG=32 + SLM <= 48 KB`** (`2 WGs/Xe-core = 64/64` HW threads) | `Btile192` drops `452 us` (`WG=48/56 KB`) -> **`290–300 us`** (`WG=32/48 KB`) |
| High `addc`/`add3`/`mov` count in ESIMD GEMM inner loop | DG2 A64 has no immediate offset folding (`ExDesc[11:0] == 0`) ([lsc-messages.md](../isa/lsc-messages.md)) | Hoist base pointers, pack 4 B-pointers in `simd<uint64_t, 4> bp_vec`, reverse-order `pack_a` | Cuts `-94` instructions/loop (`bp_vec`) and `-5 mov` (`pack_a`) |
| Weight-only 4-bit GEMM (`NF4`/`INT4Sym`) stalls below `2 TFLOPS` | `8 B` fragmented loads, loop division, and dynamic register-array LUT bottleneck unpack ([weight-only-gemv.md](../weight-only-gemv.md)) | Align `BM=64, BN=64, BK=group_size=64, WG=32` + `4x16 B` OWord load + SIMD16 `slm_gather` LUT + `16 B` `slm_block_store` | Reaches `22.49 TFLOPS` (`NF4`) / `24.57 TFLOPS` (`INT4Sym`), `12.95x` vs initial 8 B baseline |
| Streaming Decode SDPA is `~15%–20%` below `403.1 GB/s` roof with minimum bytes | Concurrent sub-group streams hit strided DRAM rows (`328.5 GB/s`); `reduce_over_group` may narrow to SIMD16 ([sdpa-a770.md](../techniques/sdpa-a770.md)) | Split 16 sub-groups within 1 head (`WG=512`), **interleave key-block ownership (`SDPA_ORDER=1`)**, and use per-slice SLM reduction (`SDPA_REDUCE=1`) | Lifts `bf16` Decode SDPA `403 -> 343 us` (`390.7 GB/s` = `96.9%` roof) with `0` errors |
| Domain-I multi-op elementwise/norm chain loses to launch & DRAM round-trips | Single-CCS A770 (`numQueues=1`) serializes separate kernels (`0%` overlap) and repeats DRAM traffic ([execution.md](execution.md)) | Fuse sub-DAG via cached OpenCL-C -> `ocloc64.dll` `.spv` -> `.zebin` (`ZE_MODULE_FORMAT_NATIVE`) into L0 Regular CL | Compresses 5-op `1024x4096 f32` chain `458.3 -> 168.0 us` (`399.6 GB/s` = `99.1%` roof, `2.73x` vs unfused) |
| Deep small-kernel chain (`L=64..256`) regresses `7.93x`, or Copy overlap drops to `0%` | `UR_LOADER_USE_LEVEL_ZERO_V2=1` switches SYCL `in_order` queues to Immediate CLs and breaks dual-SYCL Copy overlap ([execution.md](execution.md)) | Disable global UR v2 on uncaptured loops; use native L0 Regular Command List (`AppendBarrier` + `GraphControl`) + explicit `ordinal 1` Copy Engine | Drops `L=256` `9811 -> 317 us` (`0.484 us/edge`, `1`-replay break-even); restores `91.6%–99.95%` Copy overlap |
| GPU-polled host doorbell stalls `0/10` or triggers Windows 2 s TDR | Cached `malloc_host`/`malloc_shared` return stale GPU cache; uncached PCIe spin takes `1.12 us/spin` ([execution.md](execution.md)) | Allocate with `ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED`, cap spins to `~35,615` (`40 ms`), read KB payloads with parallel threads | `10/10` coherency, `6.00 us` 0 B RTT (`13.0x` vs `78 us` event-split) |
| Hot-path `>= 16 MB` allocs or event creation cause ms spikes | UMF pool stops below `16 MB` (`637–1863 us`); `zeEventPoolCreate` costs `103–232 us` ([execution.md](execution.md)) | Pre-allocate a single Device Workspace Arena and static `zeEventPool` reused via `zeEventHostReset` | Event lifecycle drops to `0.10–0.20 us` (`>1,000x` vs pool create); zero hot alloc spikes |

## 2. DG2 ISA Capability Matrix (What You Can Actually Write) `[ISA]` `[TOOLCHAIN]` `[MEASURED]`

| Feature | OpenCL C / `.spv` | SYCL / ESIMD (2026.2+) | IGA / Asm | How to Reach / Practical Constraint |
|---|---|---|---|---|
| `dpas` (`8x8` bf16/f16/i8) | No (use SYCL) | **Yes** | Yes | `esimd::xmx::dpas<8,8,float>` (`SIMD8` only; `f32`/`i32` acc only) |
| `dpasw` (fused-EU wide) | No | **Yes** | Yes | `esimd::xmx::dpasw<8,8,float>` -- halves `A` SLM reads; production default (`92.5 TFLOPS`) |
| `256-GRF` (large GRF) | No (runtime locked to 128) | **Yes** | Yes | Per-kernel `sycl::ext::intel::experimental::grf_size<256>` (`WG=32`, `SLM <= 48 KB`) |
| SWSB / SBID scheduling | No | **Indirect via `-Xfinalizer`** | Yes | ESIMD VC `-Xfinalizer` 4-flag set optimizes SBID sharing & sync pruning; raw tokens are asm-only |
| `split_barrier` | No | **Yes** | Yes | `exp_esimd::split_barrier<signal/wait>()` + `esimd::fence<local,none,group>()` |
| LSC cache hints | Limited | **Yes** | Yes | `esimd::cache_hint_L1` / `cache_hint_L2` (`ca.ca`, `st.ca`, `wb.wb`, `st.uc`) |
| Bindless / ExBSO | No | **Via Level Zero only** | Yes | OpenCL runtime emits stateful `bti[N]`; ESIMD uses flat `a64`; real bindless requires L0 heap |
| `bf16` / `f16` <-> `f32` | Yes | **Yes** | Yes | Compiles to typed `mov :bf/:hf` (zero overhead inline; avoid separate conversion pass) |
| GRF-indirect `r[a0.0]` | Yes | Yes | Yes | Triggered by dynamically indexed private arrays (warning: runtime-indexed `simd[]` spills to private mem) |
| `{Compacted}` 8 B encoding | Auto | Auto | Yes | Chosen automatically by assembler encoding rules; not a user knob |
| `load_2d` / `prefetch_2d` | No | **Broken (hangs)** | Encodes only | **PVC/Xe2+ only** -- hangs forever on DG2; never use |
| `named_barrier` | No | **Rejected at JIT** | No | **PVC-only** -- device JIT throws unsupported error on DG2 |
| Explicit `prefetch` | Compile error | **Harmful / no-op** | No mnemonic | Do not use on DG2 (`__builtin_prefetch` fails; `esimd::prefetch` regresses GEMM) |
| `tf32` / `fp8` / `dst:hf` DPAS | No | No | No / illegal | Absent or rejected by DG2 VISA validator; accumulate in `f32` |
