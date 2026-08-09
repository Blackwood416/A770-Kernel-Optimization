# API Usage Guide

> Baseline contract: every oneDNN baseline must record the implementation
> string (`jit:gemm:any`, `ocl:ref:any`, ...), format tags, dtypes, post-ops,
> fpmath mode, dims, reorder/preprocessing, device time, wall time, and CPU
> reference correctness. See the oneDNN Baseline Contract in SKILL.md.

## Table of Contents

- SYCL joint_matrix
- ESIMD dpas and memory APIs
- Kernel structure rules
- Standard SYCL sub-group kernels
- oneDNN RMSNorm baseline
- oneDNN Softmax baseline
- Build and run commands
- VTune commands
- Verification methodology
- Automation and experiment records

Cross-reference: every API below is exercised by an embedded snippet in [code-snippets.md](code-snippets.md); measured context for when to use each one is in [techniques.md](../techniques/techniques.md) and [pitfalls.md](../workflow/pitfalls.md).

## SYCL joint_matrix

Namespace: `sycl::ext::oneapi::experimental::matrix`.

Types for A770 bf16 (sub-group size 16):

```cpp
joint_matrix<sub_group, bf16, use::a, TM, TK, layout::row_major> sub_a;      // TM=8, TK=16
joint_matrix<sub_group, bf16, use::b, TK, TN, layout::ext_intel_packed> sub_b; // TN=8
joint_matrix<sub_group, float, use::accumulator, TM, TN> sub_c;
```

Rules:

- A stays `layout::row_major`; B must be `layout::ext_intel_packed` (VNNI). A 16x16 register block is built from 2x2 8x8 accumulators with 2 A slices and 2 B slices and 4 `joint_matrix_mad` calls per K block.
- `joint_matrix_load(sg, m, ptr, stride)` takes stride in elements. For packed B, cast the uint32 pointer to bf16 and pass `BN * 2` so the loader walks packed rows correctly.
- Convert USM pointers to the right address space with `sycl::address_space_cast<access::address_space::global_space, access::decorated::no>(ptr)`; SLM slices use `local_space`.
- `joint_matrix_store(sg, acc, ptr, stride, layout::row_major)` writes C.
- Software prefetch exists as `sycl::ext::oneapi::experimental::prefetch(ptr, count, properties(prefetch_hint_L2))`, but on DG2 it is unsupported for joint_matrix paths and measured negative; do not use it for GEMM-scale data that fits L2.

## ESIMD dpas and Memory APIs

Headers: `sycl/ext/intel/esimd/xmx/dpas.hpp`, `sycl/ext/intel/esimd/memory.hpp`, `sycl/ext/intel/esimd/memory_properties.hpp`.

DPAS (A770 bf16):

```cpp
simd<bf16, M * K> a(...);   // A: RepeatCount x 16 bf16, row-major
simd<bf16, K * N> b(...);   // B: VNNI-packed, uint32 view
simd<float, M * N> c0(0.0f);
simd<float, M * N> c1 = dpas<8, M, float>(c0, b, a); // accumulate
simd<float, M * N> c2 = dpas<8, M, float>(b, a);     // no source C
```

Rules:

- `SystolicDepth` must be 8; `ExecutionSize` 16 is unsupported on A770 despite compiling.
- B VNNI packing: `word = (k/2)*N + n`, low 16 bits `B[2k][n]`, high 16 bits `B[2k+1][n]`.
- `dpasw` is ExecutionSize-8 only and does not reduce instruction count for a 16x16 tile; skip it.
- SLM: `slm_init<SLM_TOTAL_BYTES>()`, `slm_block_load<T, N>(offset, overaligned_tag<16>{})`, `slm_block_store(offset, value, overaligned_tag<16>{})`, `barrier()`.
- Global block loads: `block_load<T, N>(ptr, overaligned_tag<16>{})`; max 256 B on DG2.
- Load SLM slices into a named `simd` lvalue, then `bit_cast_view<bf16>()`; calling `bit_cast_view` on a temporary fails to compile.
- `prefetch<bf16, N>(...)` exists but measured negative on A770 GEMM; only consider it when the working set exceeds 16 MB L2.
- Cache hints such as `properties(alignment<16>, cache_hint_L1<cache_hint::cached>)` measured neutral; do not expect gains.
- `esimd::reduce` only recognizes `std::plus<>` and `std::multiplies<>`; pass `std::plus<>{}` (or `std::multiplies<>{}`). Typed functors such as `std::plus<float>{}` compile but fall through an empty branch and silently return `T{}`.

## Kernel Structure Rules

- ESIMD kernels: `q.single_task([=]() SYCL_ESIMD_KERNEL { ... })`.
- Never branch on runtime values inside an ESIMD kernel; use host-side template dispatch (`if constexpr`) and instantiate the kernel twice.
- Hoist A/B/SLM base pointers outside the K loop and advance them with fixed strides.
- Write C back in 64 B blocks, e.g. `block_store<float, 16>` per output row.
- For operand-layout loads, read exactly 256 B per DPAS operand slice from global or SLM so no `select` is needed.
- On oneAPI 2026.1, `sycl::group_barrier(it, sycl::access::fence_space::local_space)` no longer compiles; use `it.barrier(sycl::access::fence_space::local_space)` or pass a `memory_scope`.

## Standard SYCL Sub-Group Kernels

- For memory-bound GEMV (`y = A*x`, row-major A), the measured best shape is a 2D `nd_range` with one row per sub-group, `vec<float,16>` loads for both A and x, and `sycl::reduce_over_group` to combine lane partials:

```cpp
constexpr size_t SUB = 16;       // local dimension; actual SG may be 32
constexpr size_t SG_PER_WG = 32; // work-group size 512
const size_t nvec = N / 16;

h.parallel_for(nd_range<2>(range<2>(SUB, M), range<2>(SUB, SG_PER_WG)),
               [=](nd_item<2> it) {
    auto sg = it.get_sub_group();
    const size_t lane = sg.get_local_linear_id();
    const size_t sg_id = sg.get_group_linear_id();
    const size_t sg_size = sg.get_local_range()[0];
    const size_t per_lane = nvec / sg_size;
    const size_t row = it.get_group(1) * SG_PER_WG + sg_id;

    vec<float, 16> acc(0.0f);
    for (size_t k = 0; k < per_lane; ++k) {
        const size_t col = (lane * per_lane + k) * 16;
        acc += *reinterpret_cast<const vec<float, 16> *>(a + row * N + col) *
               *reinterpret_cast<const vec<float, 16> *>(x + col);
    }
    float sum = 0.0f;
    for (size_t e = 0; e < 16; ++e) sum += acc[e];
    const float s = reduce_over_group(sg, sum, plus<float>());
    if (lane == 0) y[row] = s;
});
```

- Do not hardcode `per_lane` from `N/16`: A770 may compile the kernel with 32-lane sub-groups, which would cover only half of each row. Always derive `per_lane` from `sg.get_local_range()[0]`, or pin the size with `properties{sub_group_size<16>}` and accept that pinning was measured slower for this GEMV.
- Do not assume the sub-group dimension is local dim 0. On oneAPI 2026.1, a 2D local range `(16, 32)` formed 32-lane sub-groups along dim 1 (`sg.get_group_linear_id()` tracked `lid0`, `sg.get_local_linear_id()` tracked `lid1`). For row-per-work-group kernels, prefer a 1D `nd_range` so sub-groups are contiguous linear blocks, or probe the mapping first.
- For row-reduction operators (RMSNorm, layer norm, softmax), stage the row in SLM and normalize from SLM. Copy-ready patterns: [f32 RMSNorm core](code-snippets.md#f32-rmsnorm-core-slm-row-tile) and [f32 Softmax core](code-snippets.md#f32-softmax-core-slm-row-tile). For softmax, shrink the WG size for short rows and keep a scalar generic path for odd column counts.
- Allocate A/x/y with `sycl::aligned_alloc_device<float>(64, ...)` so 64 B `vec<float,16>` loads are aligned.

## oneDNN RMSNorm Baseline

oneDNN 3.11.2 exposes RMSNorm as a layer-normalization flag, not a separate primitive kind. Use `layer_normalization_forward` with `use_scale | rms_norm`; with `forward_inference` no stats are required. SYCL shared USM pointers can back the oneDNN memory objects on the same GPU engine.

```cpp
dnnl::engine eng(dnnl::engine::kind::gpu, 0);
dnnl::stream stream(eng);
dnnl::memory::desc src_md({M, N}, dnnl::memory::data_type::f32,
                          dnnl::memory::format_tag::ab);
dnnl::memory::desc scale_md({N}, dnnl::memory::data_type::f32,
                            dnnl::memory::format_tag::a);

auto flags = dnnl::normalization_flags::use_scale |
             dnnl::normalization_flags::rms_norm;
dnnl::layer_normalization_forward::primitive_desc pd(
    eng, dnnl::prop_kind::forward_inference, src_md, src_md, eps, flags);
dnnl::layer_normalization_forward ln(pd);

dnnl::memory src_mem(src_md, eng, x);
dnnl::memory dst_mem(src_md, eng, y);
dnnl::memory scale_mem(scale_md, eng, gamma);
ln.execute(stream, {{DNNL_ARG_SRC, src_mem},
                    {DNNL_ARG_DST, dst_mem},
                    {DNNL_ARG_SCALE, scale_mem}});
stream.wait();
```

Compile with `/EHsc` when using the C++ API exceptions (`icx-cl /fsycl /EHsc ...`). Always verify the oneDNN output against the CPU RMSNorm reference; it measured `errors: 0/4194304` for 1024x4096 f32.

For f16/bf16 src, the scale memory is still `f32`. Keep a separate `float*`
scale array filled from the converted gamma values; passing the `T* gamma`
array to the f32 scale descriptor corrupts every output (`max_abs_err ~2.68`
at 64x256 f16/bf16). The measured oneDNN implementation on A770 for all swept
RMSNorm shapes is `ocl:reusable:vectorized`, not a GEMM JIT path.

## oneDNN Softmax Baseline

oneDNN 3.11.2 has a dedicated softmax primitive. For row-major f32 `{M, N}` and softmax along the last axis, use `softmax_forward` with `prop_kind::forward_inference`, `algorithm::softmax_accurate`, and `axis=1`. Create the engine and stream from the same SYCL GPU queue and pass shared USM pointers.

```cpp
dnnl::engine eng = dnnl::sycl_interop::make_engine(
    q.get_device(), q.get_context());
dnnl::stream stream = dnnl::sycl_interop::make_stream(eng, q);
dnnl::memory::desc md({M, N}, dnnl::memory::data_type::f32,
                      dnnl::memory::format_tag::ab);
dnnl::softmax_forward::primitive_desc pd(
    eng, dnnl::prop_kind::forward_inference,
    dnnl::algorithm::softmax_accurate, md, md, 1);
dnnl::softmax_forward softmax(pd);

auto src_mem = dnnl::sycl_interop::make_memory(
    md, eng, dnnl::sycl_interop::memory_kind::usm, x);
auto dst_mem = dnnl::sycl_interop::make_memory(
    md, eng, dnnl::sycl_interop::memory_kind::usm, y);
softmax.execute(stream, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
stream.wait();
```

Compile with `/EHsc` and link `dnnl.lib`:

```bat
icx-cl /fsycl /EHsc softmax_opt.cpp /I "%ONEAPI_ROOT%\dnnl\latest\include" "%ONEAPI_ROOT%\dnnl\latest\lib\dnnl.lib" /Fe:softmax_opt.exe
```

Always verify the oneDNN output against the CPU softmax reference. The 1024x4096 baseline measured 0.217 ms while 1024x16384 measured 2.64 ms, so re-measure the baseline per shape instead of extrapolating.

## oneDNN Conv Baseline

For an f32 convolution baseline on A770, use `dnnl::convolution_forward` with
`prop_kind::forward_inference` and the same SYCL USM interop as the softmax
baseline:

```cpp
dnnl::memory::desc src_md({N, IC, IH, IW}, dnnl::memory::data_type::f32,
                          dnnl::memory::format_tag::nchw);
dnnl::memory::desc wei_md({OC, IC, KH, KW}, dnnl::memory::data_type::f32,
                          dnnl::memory::format_tag::oihw);
dnnl::memory::desc dst_md({N, OC, OH, OW}, dnnl::memory::data_type::f32,
                          dnnl::memory::format_tag::nchw);
auto pd = dnnl::convolution_forward::primitive_desc(
    eng, dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
    src_md, wei_md, dst_md, {1, 1}, {0, 0}, {0, 0});
```

On the measured `4x32x64x64 -> 4x64x62x62` f32 shape, oneDNN was
`0.0935 ms` while the best SYCL cache-blocked NHWC kernel was `0.1605 ms`.

## oneDNN INT4 Matmul Baseline (f16 x u4 -> bf16)

Use this configuration to avoid oneDNN's `ocl:ref` fallback. With `ab` u4 weights and no scales, oneAPI 2026.1 selected `ocl:ref:any` (~110 ms for 1024x1536x512); with `ba` + group scales + zero point it selected `jit:gemm:any` (0.033 to 0.034 ms).

```cpp
dnnl::memory::desc src_md(
    {M, K}, dnnl::memory::data_type::f16, dnnl::memory::format_tag::ab);
dnnl::memory::desc wei_md(
    {K, N}, dnnl::memory::data_type::u4, dnnl::memory::format_tag::ba);
dnnl::memory::desc dst_md(
    {M, N}, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::ab);
dnnl::memory::desc scale_md(
    {K / group_size, N}, dnnl::memory::data_type::f16,
    dnnl::memory::format_tag::ab);
dnnl::memory::desc zp_md(
    {1}, dnnl::memory::data_type::u8, dnnl::memory::format_tag::a);

dnnl::primitive_attr attr;
attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1),
                {group_size, 1}, dnnl::memory::data_type::f16);
attr.set_zero_points(DNNL_ARG_WEIGHTS, 0, {},
                     dnnl::memory::data_type::u8);
attr.set_fpmath_mode(dnnl::fpmath_mode::any, true);

dnnl::matmul::primitive_desc pd(eng, src_md, wei_md, dst_md, attr);
dnnl::matmul prim(pd);
```

Pass the packed weights buffer as `[N, K/2]` bytes, low nibble first along K. Pass `DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS` for the f16 scale memory and `DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS` for the u8 zero-point memory. Signed int4 data stored as `[-8,7]` maps to u4 by adding 8, so the zero-point tensor is `{8}`.

SYCL `dpas` cannot run this mixed f16/u4 shape directly. The stable SYCL fallback is host-dequantized u4 -> bf16 followed by the bf16 ESIMD DPAS kernel; that lands at 0.060 to 0.061 ms versus oneDNN's 0.033 to 0.034 ms.

## Build and Run Commands

Windows oneAPI (user-validated):

```bat
cmd /c '"%ONEAPI_ROOT%\setvars.bat" && icx-cl /fsycl gemm.cpp /Fe:gemm.exe'
cmd /c '"%ONEAPI_ROOT%\setvars.bat" && gemm.exe'
```

Large GRF option (negative for joint_matrix/XMX and ESIMD kernels at JIT;
measured positive only for AOT + unroll4 simple non-XMX bf16 loops; see
[codegen.md](../workflow/codegen.md). Keep only for verification):

```bat
icx-cl /fsycl -Xsycl-target-backend "-options -ze-opt-large-register-file" gemm.cpp /Fe:gemm.exe
```

## IR, SPIR-V, and GEN Assembly Dump

Text device LLVM IR is not supported (`-S -emit-llvm` reports
`IR output is not supported`). Preserve these artifacts instead:

```powershell
icpx -fsycl -fsycl-device-obj=spirv -O2 -DUNROLL=1 bench.cpp
llvm-spirv -r -o module.spv bench-sycl-spir64-unknown-unknown.bc
llvm-spirv --to-text module.spv -o module.spv.txt
ocloc compile -device dg2-g10-a0 -spirv-input module.spv -output gen
```

The `ocloc` device name for A770 is `dg2-g10-a0` (PCI ID `0x56a0`).

## VTune Commands

Instruction-count profile (no admin needed):

```bat
vtune -collect gpu-hotspots -knob characterization-mode=instruction-count -result-dir <dir> -- <exe>
```

Full-compute profile (admin, requires `-allow-multiple-runs`):

```bat
vtune -collect gpu-hotspots -allow-multiple-runs -knob characterization-mode=full-compute -result-dir <dir> -- <exe>
```

Launch-overhead check:

```bat
vtune -collect xpu-offload -result-dir <dir> -- <exe>
```

Report:

```bat
vtune -report hotspots -r <dir> -group-by computing-task ...
```

Read these metrics: GPU time per kernel, ALU0/ALU1 instructions, Send instructions, XMX instructions, GPU Barriers, Occupancy, XMX pipeline active, XVE Active/Stalled/Idle, L3 Bandwidth Bound.

## Execution Model Submission Forms

- Same-queue two-kernel submission is the baseline; dual-queue + event saved
  about 17 us per invocation on the measured pack+GEMM shape.
- SYCL graph: build and `finalize()` once, then call `execute_graph` in the
  timed loop. Graph does not expose per-node SYCL profiling events on oneAPI
  2026.1; use VTune `xpu-offload` for node times.
- Fuse only when the extra device time is smaller than the launch gap plus
  host overhead it removes. On the measured shape, fusing pack into GEMM lost
  about 47 us per iteration.
- Host preprocessing is only profitable when it can run once outside the
  timed loop. Repeating host VNNI pack inside the loop cost 775 us of CPU work
  plus about 301 us of USM visibility overhead.

Copy-ready forms: [execution graph/dual-queue cores](code-snippets.md#execution-graph-dual-queue-cores).

## Robustness Harness

Before running new ESIMD kernel shapes, isolate each variant in its own
process and let a watchdog capture exit code, timeout, minidumps, and Windows
Event Log:

```powershell
python scripts\watchdog.py --exe build\risk_probe.exe `
    --iterations 1 --timeout 15 --label risk_probe --out artifacts\watchdog_risk
python scripts\watchdog.py --exe build\risk_batch_esimd.exe `
    --iterations 10 --timeout 30 --label risk_batch_esimd --out artifacts\watchdog_stress
```

Details and the full safety checklist are in
[robustness.md](../workflow/robustness.md).

## Automation and Experiment Records

The reusable harness provides environment probe, build, correctness compare,
unified benchmark, oneDNN baseline probe, VTune parse, watchdog, and
`record_experiment.py` with the unified JSON schema:

```powershell
python scripts\record_experiment.py --operator gemv --shape 4096x4096 `
    --dtype f32 --variant sycl_subgroup_direct_l2 --exe build\f32_gemv.exe `
    --probe-onednn --out artifacts\records\f32_gemv.json
```

Every record keeps `operator`, `shape`, `dtype`, `variant`, driver/oneAPI,
`device_median_us`, `wall_median_us`, `pipeline_median_us`, `max_abs_err`,
`errors`, `vtune`, baseline implementation, and status. It also writes a
Markdown evidence file with the `[MEASURED]` validity domain. CLI contract and
full examples:
[automation.md](../workflow/automation.md).

## Verification Methodology

1. Run 100 warmup iterations, then 1000 timed iterations with `q.wait()`, and report the average.
2. Repeat at least 3 times and require stable values before accepting a change.
3. Compare against a CPU reference and require `errors: 0/<total>`.
4. When `beta != 0`, restore the initial C0 after timing, run one extra kernel, and verify that single-shot result.
5. Keep failed variants as standalone files and record the negative result with baseline numbers; they are the controls for future claims.
6. Verify library baselines with the same CPU reference before trusting their timings; oneDNN `1xK` matmul is `x*A`, not row-major `A*x`.
7. For steady-state GEMV timing, copy inputs into aligned USM once, warm up, then time the launch loop; a buffer-backed first launch can include host-to-device transfer.
8. Record the oneDNN implementation string by setting `ONEDNN_VERBOSE=profile,dispatch` before the run; save `jit:gemm:any` vs `ocl:ref:any` with the result.
9. Follow the unified Benchmark Protocol in SKILL.md: report median, p10/p90 or MAD, flag CV, and separate device time, wall time, and pipeline time.
