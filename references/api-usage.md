# API Usage Guide

## Table of Contents

- SYCL joint_matrix
- ESIMD dpas and memory APIs
- Kernel structure rules
- Standard SYCL sub-group kernels
- Build and run commands
- VTune commands
- Verification methodology

Cross-reference: every API below is exercised by an embedded snippet in [code-snippets.md](code-snippets.md); measured context for when to use each one is in [techniques.md](techniques.md) and [pitfalls.md](pitfalls.md).

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
- Allocate A/x/y with `sycl::aligned_alloc_device<float>(64, ...)` so 64 B `vec<float,16>` loads are aligned.

## Build and Run Commands

Windows oneAPI (user-validated):

```bat
cmd /c '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && icx-cl /fsycl gemm.cpp /Fe:gemm.exe'
cmd /c '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && gemm.exe'
```

Large GRF option (measured negative on A770 for joint_matrix and ESIMD; keep only for verification):

```bat
icx-cl /fsycl -Xsycl-target-backend "-options -ze-opt-large-register-file" gemm.cpp /Fe:gemm.exe
```

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

## Verification Methodology

1. Run 100 warmup iterations, then 1000 timed iterations with `q.wait()`, and report the average.
2. Repeat at least 3 times and require stable values before accepting a change.
3. Compare against a CPU reference and require `errors: 0/<total>`.
4. When `beta != 0`, restore the initial C0 after timing, run one extra kernel, and verify that single-shot result.
5. Keep failed variants as standalone files and record the negative result with baseline numbers; they are the controls for future claims.
6. Verify library baselines with the same CPU reference before trusting their timings; oneDNN `1xK` matmul is `x*A`, not row-major `A*x`.
7. For steady-state GEMV timing, copy inputs into aligned USM once, warm up, then time the launch loop; a buffer-backed first launch can include host-to-device transfer.
