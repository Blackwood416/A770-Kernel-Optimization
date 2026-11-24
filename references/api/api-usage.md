# API Usage, Build Commands, and Library Baselines on A770

> Scope: Intel Arc A770 (DG2), Windows 11, oneAPI 2026.2 (`icx` / `icx-cl`), Level Zero SDK `1.33.1`, `oneDNN 3.13.2`, VTune `2026.6`. Copy-ready kernel cores live in [code-snippets.md](code-snippets.md).

## 1. Modern ESIMD APIs (`GRF=256`, `dpasw`, `split_barrier`, Cache Hints) `[TOOLCHAIN]`

Headers:
- `<sycl/ext/intel/esimd.hpp>`
- `<sycl/ext/intel/esimd/xmx/dpas.hpp>`
- `<sycl/ext/intel/experimental/esimd/memory.hpp>`
- `<sycl/ext/intel/experimental/grf_size_properties.hpp>`

Key rules:

- **Enable native `256-GRF` per kernel**: Pass `sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}` to `h.parallel_for(...)` with `WG=32` and `SLM <= 48 KB` (`2 WGs/Xe-core`). Never pass global `-ze-opt-large-register-file`.
- **Use `esimd::xmx::dpasw<8, 8, float>(acc, b, a)`**: `SystolicDepth` is fixed at `8`; `ExecutionSize` must be `8` (`16` compiles and produces wrong results `[BUG]`). `B` is VNNI-packed (`word = (k/2)*N + n`); pair `dpasw` with bit-swapped SLM `A` layout (`slm_tid = (tid & ~6) | ((tid & 2) << 1) | ((tid & 4) >> 1)`).
- **SLM & global block IO**: `slm_init<BYTES>()`, `slm_block_load<T, N>(off)`, `slm_block_store<uint32_t, 4+>(off_16b_aligned, val)` (**strict 16 B alignment -- 8 B offsets silently overwrite neighbors `[BUG]`**), `block_load<T, N>(ptr, props)` (max `256 B` on DG2).
- **Named SSA variables only**: Never index `simd[]` with a runtime loop variable (breaks LLVM SROA -> `privateMemSize = 576–1088 B`, `2x` slower vs named SSA). Never assign `.select<16,1>() = block_load(...)` (emits blocking `mov {$dst}`). Pass non-`const` references to helpers calling `.bit_cast_view<uint32_t>()`.
- **Cache hints**: Apply `esimd::cache_hint_L1<...>` + `esimd::cache_hint_L2<...>` via `constexpr esimd::properties` (`ca.ca` for shared `A` and single-wave `B`; `st.ca` for multi-wave `B`; `wb.wb` for standard `C` stores; `st.uc` for `ResidualAdd` `C` stores).
- **Compile-time dispatch & reductions**: Use `if constexpr` (runtime `if/else` in ESIMD can hang on A770). Pass `std::plus<>{}` / `std::multiplies<>{}` to `esimd::reduce` (typed `std::plus<float>{}` silently returns `0` `[BUG]`).

## 2. Standard SYCL Sub-Group Rules `[CORRECTNESS]`

- **Dynamic sub-group width**: Always compute `per_lane = nvec / sg.get_local_range()[0]` -- never hardcode `16` lanes (compiler may select `32`).
- **1D `nd_range` for row-per-WG kernels**: 2D local ranges can map sub-groups along dim 1 instead of dim 0; prefer 1D `nd_range` and `sycl::aligned_alloc_device<float>(64, ...)` for `vec<float,16>` fast paths.
- **Barriers & reductions**: Use `it.barrier(sycl::access::fence_space::local_space)`. Pass `it.get_group()` or `sg` (never `nd_item`) to `sycl::reduce_over_group`, and verify compiled SIMD width (`simd_size: 32`) on 32-lane reductions (or use explicit SLM reduction; see [reductions-scan.md](../techniques/reductions-scan.md#3-critical-trap-syclreduce_over_group-compiled-simd-narrowing-bug-correctness)).

## 3. Native Level Zero Interop `[TOOLCHAIN]`

- Extract handles via `sycl::get_native<sycl::backend::ext_oneapi_level_zero>(obj)` (context, device, kernel_bundle, kernel).
- Enumerate `zeDeviceGetCommandQueueGroupProperties` to pin async DMA to the **`COPY`-only `ordinal 1`** Main Copy Engine (`zeCommandListCreateImmediate`).
- Record hot chains once into a closed **Regular Command List** (`zeCommandListAppendBarrier`, never per-node `zeEvent` inside regular lists) with a fixed-address `GraphControl` parameter struct (`0.484 us/edge` gap, `1`-replay break-even).
- Allocate GPU-polled host doorbells with `ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED` (calibrate spin cap to `~1.12 us/spin` to avoid Windows 2 s TDR). Reuse a static `zeEventPool` via `zeEventHostReset` (`0.10–0.20 us`).
- Export native binaries via `zeModuleGetNativeBinary` for disassembly auditing and Tier-3 `.zebin` disk caching (`ZE_MODULE_FORMAT_NATIVE`).

## 4. `oneDNN` Baseline Recipes `[CORRECTNESS]`

Always enable `ONEDNN_VERBOSE=profile,dispatch`, record the implementation string (`jit:gemm:any` vs `ocl:ref:any`), and verify output against the CPU reference before quoting a speedup:

- **GEMM (`bf16` / `f32` / `int8`)**: `dnnl::matmul` with `ab`/`ab` tags. For `f32` GEMV-N1 (`A*x`), use `{M,K} x {K,1}` (never `{1,K}`, which computes `x*A`).
- **RMSNorm**: `dnnl::layer_normalization_forward` with `prop_kind::forward_inference` and `normalization_flags::use_scale | normalization_flags::rms_norm`. **Scale (`gamma`) memory must always be `f32` even when src/dst are `f16`/`bf16`**.
- **Softmax**: `dnnl::softmax_forward` with `algorithm::softmax_accurate`, `axis=1`; re-measure per shape (long rows like `1024x16384` can spike sharply).
- **Direct Conv (`f32`)**: `dnnl::convolution_forward` (`convolution_direct`, `nchw` / `oihw`).
- **Weight-only `u4` Matmul (accuracy-matched)**: `src={M,K} f16 ab`, `wei={K,N} u4 ba` (`[N, K/2]` low nibble first), `scales={K/gs, N} f16 ab` (`mask=(1<<0)|(1<<1)`, `groups={gs,1}`), `zp={1} u8 = 8`, **`attr.set_fpmath_mode(fpmath_mode::strict, true)`** (or `any` with `apply_to_int=true`; unset is rejected on GPU).

## 5. Build, Disassembly, and VTune Commands (Windows) `[TOOLCHAIN]`

Standard ESIMD + Level Zero build with optimal `-Xfinalizer` 4-flag link embedding:

```powershell
# Inside a oneAPI setvars.bat environment (PowerShell):
icx-cl -fsycl -O2 /Qstd=c++20 /EHsc -fsycl-device-code-split=per_kernel `
  -I"<L0_SDK_ROOT>\include" kernel.cpp "<L0_SDK_ROOT>\lib\ze_loader.lib" `
  -Xs "-Xfinalizer -SBIDBundle -Xfinalizer -SWSBDepReduction -Xfinalizer -nopresched -Xfinalizer -twoSrcBundleBCR" `
  /Fe:kernel.exe
```

Rapid `-Xfinalizer` sweep without relinking (PowerShell -- never use `cmd /c 'set VAR=val && ...'` which appends a trailing space):

```powershell
$env:SYCL_CACHE_PERSISTENT = "0"
$env:SYCL_PROGRAM_APPEND_COMPILE_OPTIONS = "-Xfinalizer -SBIDBundle -Xfinalizer -SWSBDepReduction -Xfinalizer -nopresched -Xfinalizer -twoSrcBundleBCR"
```

Native binary disassembly (`ocloc disasm` emits dot-prefixed `.text.*.asm`; walk output dirs with `os.walk`):

```powershell
New-Item -ItemType Directory -Force disasm_out | Out-Null
ocloc disasm -file dumped_kernel.zebin -device dg2 -dump disasm_out
```

VTune GPU profiling (launch the executable **directly**, never wrapped in `cmd /c`, or the GPU plugin silently disables itself):

```powershell
# Dynamic instruction mix (6 exhaustive classes via CLI; no admin required):
vtune -collect gpu-hotspots -knob characterization-mode=instruction-count -result-dir vt_ic -- .\kernel.exe
vtune -report hotspots -r vt_ic -group-by computing-task -format csv

# High-frequency 0.05 ms (50 us) OA counters (eliminates cross-launch dilution on 100-400 us kernels):
vtune -collect gpu-hotspots -knob characterization-mode=overview -knob gpu-sampling-interval=0.05 -result-dir vt_ov -- .\kernel.exe

# Full-compute / LSC-SLM (admin required, must pass -allow-multiple-runs; eu-stall-sampling is unsupported on DG2 Windows):
vtune -collect gpu-hotspots -allow-multiple-runs -knob characterization-mode=full-compute -result-dir vt_fc -- .\kernel.exe
```
