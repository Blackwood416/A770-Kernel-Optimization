# LSC Send Messages, Addressing Modes, and A64 Optimization on DG2

> Scope: Intel Arc A770 (DG2 / Xe-HPG), Windows 11, oneAPI 2026.2, driver `32.0.101.9150`.

## 1. LSC Message Capabilities and Hard Gates `[ISA]` `[ARCH]`

| Feature | DG2 Status | Engineering Rule |
|---|---|---|
| Max LSC block width | **`256 B`** (`block_load` / `slm_block_load`) | `512 B` is PVC-only; relayout data instead of widening past `256 B` |
| `load_2d` / `store_2d` / `prefetch_2d` | **Broken / hangs forever `[BUG]`** | PVC/Xe2+ only. Compiles and assembles on DG2 but hardware never completes (`TIMEOUT_KILLED`); never use on A770 |
| Explicit prefetch (`__builtin_prefetch` / `esimd::prefetch`) | **Unusable / harmful `[TOOLCHAIN]` `[MEASURED]`** | OpenCL `__builtin_prefetch` fails compile (`llvm.prefetch.p1i8`); IGA XeHPG has no prefetch mnemonic; ESIMD `esimd::prefetch` either emits no dedicated prefetch or adds `~50` A64 ALU ops + `4–6` SBID stalls in `K=4096` GEMM loops for net regression. Get locality via tiling and `cab4` pipelining instead |
| Immediate offsets `[+imm]` on LSC | **Unsupported (`ExDesc[11:0] == 0`)** | Immediate offsets exist only on `Xe2+`; every `ptr + imm` emits explicit `addc` + `add3` on DG2 |
| `16 B` payload alignment (`d32x4`) | Non-faulting sector split (`+17%–20%` slower `[MEASURED]`) | Keep vector global loads `>= 16 B` (ideally `64 B` / `128 B`) aligned; scalar `u8`/`u16`/`u32` is request-rate bound (`318 / 609 / 1187 GB/s`) |
| `iga64 -Xdsd` decoder | **Wrong table on DG2 `[BUG]`** | Uses E64 layout and prints bogus `AddrSize`/`DataSize`; read `ocloc disasm` comments instead |

## 2. LSC Cache Policy Field (`Desc[19:17]`) `[ISA]`

| Code | Load Policy | Store Policy | Notes |
|---:|---|---|---|
| `1` | `uc.uc` | `uc.uc` | Bypass L1 and L3 (L2) |
| `2` | `uc.ca` | `uc.wb` | Bypass L1, cache in L3 |
| `4` | `ca.ca` **(default)** | `wt.wb` | `wt.wb` regresses GEMM C stores (`+17 us` vs `wb.wb`) |
| `5` | `st.uc` | `st.uc` | Streaming L1, uncached L3 (`+14%` on streaming axpy; best for ResidualAdd C stores) |
| `6` | `st.ca` | `st.wb` | Streaming L1, cached L3 (`L1StreamB` multi-wave GEMM B loads) |
| `7` | `ri.ca` | `wb.wb` **(default)** | Best for standard / Bias / SiLU GEMM C stores (`-42 us` on LM-Head vs `st.uc`) |

See [cache-geometry.md](../hardware/cache-geometry.md#3-lsc-cache-hint-dispatch-rules-measured) for full dispatch rules.

## 3. Addressing Modes: Stateful BTI vs Bindless ExBSO vs A64 `[ISA]` `[MEASURED]`

1. **Runtime OpenCL emits stateful BTI (`a32 + bti[N]`)**: Driver JIT binaries use `Desc[30:29] = 3` (`bti`) and `ExDesc[31:24] = N` (even for SVM pointers). Offline `ocloc` AOT emits bindless `bss[a0.x]` + `{ExBSO}`, which the OpenCL runtime rejects and recompiles -- **never transfer register/descriptor maps directly from AOT to OpenCL JIT**, and never patch BTI to ExBSO without a bindless heap (reads all zeros while looking `1.5x` faster `[BUG]`).
2. **Real bindless requires Level Zero**: Only Level Zero device allocations / native binaries (e.g., `oneDNN`) expose a real bindless surface heap; ESIMD uses flat `a64` stateless messages (`AddrSize = 3`).
3. **DG2 A64 addressing optimizations (critical for ESIMD GEMM) `[MEASURED]`**:
   - **Hoist pointers outside K-loop**: Because DG2 cannot fold immediate offsets into `send.ugm`, computing `base + k * stride` inside the K-loop adds `80–90` `mul/mach/addc/shl` ops per step. Initialize pointers outside the loop and advance by fixed strides.
   - **Vectorize 4x64-bit pointers (`bp_vec`: `esimd::simd<uint64_t, 4>`)**: Advance 4 B-row pointers in one vector add using `static_cast<uint64_t>(static_cast<uint32_t>(BK * N)) * 2ULL` (cast via `uint32_t` to avoid signed 32-bit sign-extension overflow). Cuts **`-94` instructions** from the `128x256` main loop (`796 -> 702`: `addc 40->16`, `add3 44->16`, `mov 303->270`, `sync.nop 26->16` `[MEASURED]`).
   - **Reverse-order `pack_a`**: Populate uninitialized `simd<uint16_t, 64>` from top lane down (`select<16,1>(48) -> (32) -> (16) -> (0)`) to prevent LLVM/IGC dead-init `mov(32)` (`-5 mov` ops `[MEASURED]`).
   - **Avoid non-transposed `lsc_gather` for GEMM**: Replacing contiguous `block_load` with `SIMD8` `lsc_gather` + register transpose inflates `mov` (`187 -> 380`) and regresses `LMHd (128x32768x2560)` by **`-20%` vs contiguous block_load** (`406 -> 506 us` `[MEASURED]`).

## 4. Message Issue Cost, `send.dc0` Prologue, and Atomics `[MEASURED]`

| Operation | Measured Marginal Cost | Takeaway |
|---|---:|---|
| `64 B` `store.ugm` (distinct `128 B`-spaced addrs) | **`169 ns`/msg/thread** | `~9.6x` costlier than same-address coalesced stores (`17.6 ns`) |
| `64 B` `load.ugm` (distinct addrs) | **`51 ns`/msg/thread** | `~3.3x` cheaper than distinct-address stores; linear up to `250` msgs/thread |
| Prologue `send.dc0` (arg block `> 16 B`) | **`~212–227 ns`/128 B** (`1.77 ns/B`) | `<= 16 B` args cost `0` `dc0`; per byte `dc0` is `~1.9x` slower than `load.ugm` (`0.91 ns/B`). Put large read-only tables in global memory, not kernel args. **Any `send.dc0` inside a hot loop indicates a register scratch spill!** |
| Contended `fp32` `atomic_add` (4,096 threads, 1 addr) | **`2.09 ns`/op** | Linear and cheap; still reduce inside sub-group/WG first |
| Contended `uint` CAS loop (4,096 threads, 1 addr) | **`1.2–1.5 us`/op** | **`~600x` slower** than `fp32` atomic add; never use CAS loops as contended counters |

Atomics default to `wt.wb` (bypass L1). Always verify atomic microbenchmarks scale linearly with op count (compilers fold redundant integer `fetch_add`).
