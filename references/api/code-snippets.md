# Production Code Snippets for Intel Arc A770 (DG2)

> Scope: Curated, modern production building blocks verified on Intel Arc A770 (DG2), Windows 11, oneAPI 2026.2, driver `32.0.101.9150` `[MEASURED]`. Legacy naive and superseded `joint_matrix` teaching examples have been removed to keep this reference lean. Adapt tile bounds and strides to your shape and validate `errors == 0`, `spill == 0 B`, `priv == 0 B`, and main-loop `send.dc0 == 0`.

## Table of Contents

- 1. GRF=256 cab4 (48 KB) + split_barrier + 0-Dummy 2x8 + bp_vec GEMM Core
- 2. ESIMD VC -Xfinalizer Optimal 4-Flag CMake & Debug Recipe
- 3. DG2 Cache-Hint Dispatch & SROA-Safe Named SSA Registers
- 4. 4-Bit (NF4 / INT4Sym) BK=gs=64 Zero-Branch Dequant + SLM LUT Core
- 5. Domain I: GEMV Core (Sub-Group Per Row, Direct L2)
- 6. Domain I: RMSNorm and Register-Cached Fused LayerNorm Cores
- 7. Domain I: Softmax Core (SLM Row Tile + Dynamic WG)
- 8. Stable Online-Softmax Rescale Core
- 9. Reduction and Hillis-Steele Scan Cores
- 10. Native Level Zero Runtime, Copy Engine, Doorbell, and Tier-3 .zebin Cache

## 1. GRF=256 cab4 (48 KB) + split_barrier + 0-Dummy 2x8 + bp_vec GEMM Core

Core primitives behind `92.5 TFLOPS` (`4096^3` bf16 GEMM, `1.10x` vs `oneDNN 3.13.2` `[MEASURED]`):

```cpp
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

namespace esimd = sycl::ext::intel::esimd;
namespace exp_esimd = sycl::ext::intel::experimental::esimd;

// Launch with native 256-GRF, WG=32, SLM <= 48 KB (2 WGs/Xe-core = 64/64 HW threads):
// sycl::ext::oneapi::experimental::properties kprops{
//     sycl::ext::intel::experimental::grf_size<256>};
// h.parallel_for(nd_range, kprops, [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL { ... });

SYCL_ESIMD_FUNCTION inline void slm_fence_group() {
    esimd::fence<esimd::memory_kind::local, esimd::fence_flush_op::none,
                 esimd::fence_scope::group>();
}
SYCL_ESIMD_FUNCTION inline void sbar_signal() {
    exp_esimd::split_barrier<exp_esimd::split_barrier_action::signal>();
}
SYCL_ESIMD_FUNCTION inline void sbar_wait() {
    exp_esimd::split_barrier<exp_esimd::split_barrier_action::wait>();
}

// cab4 4-buffer circular SLM (4 * 12 KB = 48 KB <= 49,152 B at WG=32, BK=16):
// Beat order:
//   compute_ops(r_buf) -> sbar_wait() -> slm_fence_group() -> sbar_signal()
//   -> store_buf(w_buf) -> load_slm_ops(next_r_buf) -> load_b() -> load_a()
// Interleaving store_buf and load_slm_ops cuts SLM ld addr chains from 45/48 to 8/48.

// 0-Dummy 2x8 DPASW grouping (4xB: b0..b3, 4xA: a0..a3 from a01 & a23; macro cap <= 8):
// Anchor each macro head on tail SLM load (a3) with max ONE size=1 node per macro:
acc00 = esimd::xmx::dpasw<8, 8, float>(acc00, b0, a3); // size=1 lead (b0)
acc10 = esimd::xmx::dpasw<8, 8, float>(acc10, b1, a0); // size=4 group (b1)
acc11 = esimd::xmx::dpasw<8, 8, float>(acc11, b1, a1);
acc12 = esimd::xmx::dpasw<8, 8, float>(acc12, b1, a2);
acc13 = esimd::xmx::dpasw<8, 8, float>(acc13, b1, a3);
acc01 = esimd::xmx::dpasw<8, 8, float>(acc01, b0, a0); // size=3 group (b0)
acc02 = esimd::xmx::dpasw<8, 8, float>(acc02, b0, a1);
acc03 = esimd::xmx::dpasw<8, 8, float>(acc03, b0, a2);
// Repeat symmetrically for (b2, b3): [b2(a3)](1) -> [b3(a0..a3)](4) -> [b2(a0..a2)](3).
// For 3xB (Btile192), use 3x4 macros: [b0(a3)](1)->[b1(a0..a2)](3), etc.

// Vectorized 64-bit B pointers (bp_vec) remove DG2 A64 addc/add3 bloat (-94 ins):
esimd::simd<uint64_t, 4> bp_vec;
const uint64_t b_k_step_bytes = static_cast<uint64_t>(static_cast<uint32_t>(16 * N)) * 2ULL;
bp_vec += b_k_step_bytes;
auto b_row0 = esimd::block_load<uint16_t, 64>(
    reinterpret_cast<const uint16_t*>(static_cast<uint64_t>(bp_vec[0])), props_b);

// Reverse-order pack_a prevents LLVM dead-init mov(32) (-5 mov ops):
esimd::simd<uint16_t, 64> ak;
ak.template select<16, 1>(48) = r3.template select<16, 1>(ah);
ak.template select<16, 1>(32) = r2.template select<16, 1>(ah);
ak.template select<16, 1>(16) = r1.template select<16, 1>(ah);
ak.template select<16, 1>(0)  = r0.template select<16, 1>(ah);

// Bit-swapped cooperative A write for fused-EU dpasw (128-GRF 16x8 thread grid):
const int slm_tid = (tid & ~6) | ((tid & 2) << 1) | ((tid & 4) >> 1);
```

## 2. ESIMD VC -Xfinalizer Optimal 4-Flag CMake & Debug Recipe

Static AOT link embedding (zero runtime environment variables required):

```cmake
set(A770_XFINALIZER_OPTS
    "-Xfinalizer -SBIDBundle -Xfinalizer -SWSBDepReduction -Xfinalizer -nopresched -Xfinalizer -twoSrcBundleBCR")
target_compile_options(my_target PUBLIC -fsycl -O2 /Qstd=c++20)
target_compile_options(my_target PRIVATE -fsycl-device-code-split=per_kernel)
target_link_options(my_target PUBLIC -fsycl "SHELL:-Xs \"${A770_XFINALIZER_OPTS}\"")
```

PowerShell runtime sweep without relinking (bypass persistent IGC cache):

```powershell
$env:SYCL_CACHE_PERSISTENT = "0"
$env:SYCL_PROGRAM_APPEND_COMPILE_OPTIONS = "-Xfinalizer -SBIDBundle -Xfinalizer -SWSBDepReduction -Xfinalizer -nopresched -Xfinalizer -twoSrcBundleBCR"
```

## 3. DG2 Cache-Hint Dispatch & SROA-Safe Named SSA Registers

```cpp
// A (shared across gid & gid+32 on 2D 8x8 N-inner snake walk): L1=cached, L2=cached
constexpr esimd::properties props_a_ca{
    esimd::alignment<64>,
    esimd::cache_hint_L1<esimd::cache_hint::cached>,
    esimd::cache_hint_L2<esimd::cache_hint::cached>};
// B (multi-wave 2D walk, L1StreamB=true): L1=streaming, L2=cached (use ca.ca if total_wgs <= 64)
constexpr esimd::properties props_b_st{
    esimd::alignment<128>,
    esimd::cache_hint_L1<esimd::cache_hint::streaming>,
    esimd::cache_hint_L2<esimd::cache_hint::cached>};
// C store: standard / Bias / SiLU epilogue uses wb.wb (-42 us on LM-Head vs st.uc)
constexpr esimd::properties props_c_wb{
    esimd::alignment<64>,
    esimd::cache_hint_L1<esimd::cache_hint::write_back>,
    esimd::cache_hint_L2<esimd::cache_hint::write_back>};
// C store: ResidualAdd / BiasResidualAdd epilogue uses st.uc to preserve hot residual/B in L3
constexpr esimd::properties props_c_res_st{
    esimd::alignment<64>,
    esimd::cache_hint_L1<esimd::cache_hint::streaming>,
    esimd::cache_hint_L2<esimd::cache_hint::uncached>};

// SROA-safe named 1-GRF SSA registers + compile-time 2-way unroll:
// NEVER index simd[cur_buf] with a runtime loop variable (breaks SROA -> priv=576..1088 B).
// NEVER write vec.select<16,1>(0) = block_load(...) (emits blocking mov {$dst}).
esimd::simd<uint16_t, 32> a0_0, a0_1, a0_2, a0_3;
esimd::simd<uint16_t, 32> a1_0, a1_1, a1_2, a1_3;
a0_0 = esimd::block_load<uint16_t, 32>(ap0, props_a_ca);
```

## 4. 4-Bit (NF4 / INT4Sym) BK=gs=64 Zero-Branch Dequant + SLM LUT Core

Reaches `22.49 TFLOPS` (`NF4`) / `24.57 TFLOPS` (`INT4Sym`) on `512x4096x4096, gs=64` (`12.95x` vs naive 8 B baseline `[MEASURED]`):

```cpp
// BM=64, BN=64, BK=64, WG=32 (8x4): BK == group_size == 64 -> zero loop division/branches.
// Stage 16-entry float NF4 codebook at SLM tail (total SLM = 32,832 B <= 48 KB -> 2 WGs/Xe-core).
esimd::simd<uint32_t, 4> w_pk = esimd::block_load<uint32_t, 4>(w_ptr, esimd::alignment<16>);
esimd::simd<uint32_t, 16> nibble_byte_offs = extract_nibbles_byte_offsets(w_pk); // nibble * 4
esimd::simd<float, 16> w_f32 = esimd::slm_gather<float, 16>(lut_slm_base, nibble_byte_offs) * tile_scale;
esimd::simd<sycl::ext::oneapi::bfloat16, 16> w_bf16 = w_f32;
// Offset MUST be 16 B-aligned; never call slm_block_store<uint32_t, 2> on 8 B offsets:
esimd::slm_block_store<uint32_t, 4>(slm_b_off_16b_aligned, w_bf16.template bit_cast_view<uint32_t>());
```

## 5. Domain I: GEMV Core (Sub-Group Per Row, Direct L2)

`215 us` on `4096x4096 f32` (`2.08x` faster vs `oneDNN Kx1` `450 us` `[MEASURED]`). Derives `per_lane` dynamically from `sg.get_local_range()[0]` so `SG=32` compilation never drops half the row:

```cpp
constexpr size_t VEC = 16;
constexpr size_t WG = 256;
const size_t nvec = N / VEC;
const size_t n_wg = (M * 32 + WG - 1) / WG; // conservative for SG<=32

q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(n_wg * WG, WG), [=](sycl::nd_item<1> it) {
        auto sg = it.get_sub_group();
        const size_t lane = sg.get_local_linear_id();
        const size_t sg_size = sg.get_local_range()[0];
        const size_t row = it.get_group(0) * (WG / sg_size) + sg.get_group_linear_id();
        if (row >= M) return;

        const size_t per_lane = nvec / sg_size;
        sycl::vec<float, VEC> acc(0.0f);
        for (size_t k = 0; k < per_lane; ++k) {
            const size_t col = (lane * per_lane + k) * VEC;
            acc += *reinterpret_cast<const sycl::vec<float, VEC>*>(a + row * N + col) *
                   *reinterpret_cast<const sycl::vec<float, VEC>*>(x + col);
        }
        float sum = 0.0f;
        for (size_t e = 0; e < VEC; ++e) sum += acc[e];
        const float total = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
        if (lane == 0) y[row] = total;
    });
});
```

## 6. Domain I: RMSNorm and Register-Cached Fused LayerNorm Cores

### 6.1 Register-Cached Fused Row LayerNorm (`WG=256, SG=16, cols <= 4096`)

`249.5 us` (`403.4 GB/s` = `100%` read roof) on `2048x4096 f32` `Add+LayerNorm` and `257.8 us` (`390.5 GB/s`, `2.79x` vs `oneDNN 3.13.2`) on `Add+GeluErf+LayerNorm` (`spill=0 B, priv=0 B` `[MEASURED]`). All `reg_vals[_i]` accesses use compile-time constant indices under `#pragma unroll` so LLVM SROA promotes the array to `<= 32 GRF` (`WG=256` uses only `16` HW threads at `SG=16`):

```c
// OpenCL C (identical pattern applies in SYCL with intel::reqd_sub_group_size(16)):
// Compile-time constants: WG=256 (cols in 1025..4096), PER=(cols + WG - 1) / WG <= 16, N_SGS=16.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void fused_add_gelu_layernorm_f32(
    __global const float* restrict a, __global const float* restrict b,
    __global const float* restrict gamma, __global const float* restrict beta,
    __global float* restrict out, int rows, int cols, float eps) {
    __local float smem[16];
    __local float row_stats[2];
    const int row = (int)get_group_id(0);
    if (row >= rows) return;
    const int lid = (int)get_local_id(0), sg_id = (int)get_sub_group_id(), lane = (int)get_sub_group_local_id();
    const size_t row_off = (size_t)row * (size_t)cols;

    float reg_vals[16]; // PER <= 16 -> <= 32 GRF out of 128 (0 spill, 0 private mem)
    float lsum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int c = lid + i * 256;
        if (c < cols) {
            float x = a[row_off + c] + b[row_off + c];
            x = 0.5f * x * (1.0f + erf(x * 0.7071067811865475f)); // single prologue eval
            reg_vals[i] = x;
            lsum += x;
        }
    }
    float sg_sum = sub_group_reduce_add(lsum);
    if (lane == 0) smem[sg_id] = sg_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg_id == 0) {
        float tot = sub_group_reduce_add((lane < 16) ? smem[lane] : 0.0f);
        if (lane == 0) row_stats[0] = tot / (float)cols;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const float row_mean = row_stats[0];

    // Exact two-pass variance in registers (0 DRAM reads, 0 loop divs, 0 catastrophic cancellation):
    float lm2 = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int c = lid + i * 256;
        if (c < cols) {
            float d = reg_vals[i] - row_mean;
            lm2 = fma(d, d, lm2);
        }
    }
    float sg_m2 = sub_group_reduce_add(lm2);
    if (lane == 0) smem[sg_id] = sg_m2;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg_id == 0) {
        float tot = sub_group_reduce_add((lane < 16) ? smem[lane] : 0.0f);
        if (lane == 0) row_stats[1] = rsqrt(tot / (float)cols + eps);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const float inv_std = row_stats[1];

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int c = lid + i * 256;
        if (c < cols) {
            float y = (reg_vals[i] - row_mean) * inv_std * gamma[c] + beta[c];
            out[row_off + c] = y;
        }
    }
}
```

### 6.2 Standalone RMSNorm Core (SLM Row Tile, 2 Barriers)

`90.5–99 us` on `1024x4096 f32` (`1.25x` faster vs `oneDNN` `112.9 us` `[MEASURED]`). Reads `x` from global memory once; keeps `gamma` in L2 (also serves as the Tier-2 SLM staging template for `4096 < cols <= 8192`):

```cpp
constexpr size_t WG = 128; // tune 64-128 for hidden=4096
constexpr size_t VEC = 16;
const size_t chunks = cols / VEC;

q.submit([&](sycl::handler& h) {
    sycl::local_accessor<sycl::vec<float, VEC>, 1> tile(sycl::range<1>(chunks), h);
    sycl::local_accessor<float, 1> row_sum(sycl::range<1>(1), h);
    h.parallel_for(sycl::nd_range<1>(rows * WG, WG), [=](sycl::nd_item<1> it) {
        const size_t tid = it.get_local_linear_id();
        const size_t row = it.get_group(0);
        auto sg = it.get_sub_group();
        if (tid == 0) row_sum[0] = 0.0f;

        sycl::vec<float, VEC> sq_acc(0.0f);
        for (size_t c = tid; c < chunks; c += WG) {
            auto v = *reinterpret_cast<const sycl::vec<float, VEC>*>(x + row * cols + c * VEC);
            tile[c] = v;
            sq_acc += v * v;
        }
        it.barrier(sycl::access::fence_space::local_space);

        float partial = 0.0f;
        for (size_t e = 0; e < VEC; ++e) partial += sq_acc[e];
        const float psum = sycl::reduce_over_group(sg, partial, sycl::plus<float>());
        if (sg.get_local_linear_id() == 0) {
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::work_group,
                             sycl::access::address_space::local_space>(row_sum[0]).fetch_add(psum);
        }
        it.barrier(sycl::access::fence_space::local_space);

        const float inv_rms = sycl::rsqrt(row_sum[0] / static_cast<float>(cols) + eps);
        for (size_t c = tid; c < chunks; c += WG) {
            auto g = *reinterpret_cast<const sycl::vec<float, VEC>*>(gamma + c * VEC);
            *reinterpret_cast<sycl::vec<float, VEC>*>(y + row * cols + c * VEC) = tile[c] * (inv_rms * g);
        }
    });
});
```

## 7. Domain I: Softmax Core (SLM Row Tile + Dynamic WG)

`89–108 us` on `1024x4096 f32` (`2.0x–2.4x` faster vs `oneDNN` `[MEASURED]`). Dispatch `Threads = (cols <= 256) ? 16 : (cols <= 1024 ? 32 : 128)`; stage up to `8,192` floats (`32 KB` SLM) in one shot:

```cpp
template <int Threads>
void softmax_slm_aligned(sycl::queue& q, const float* x, float* y, int rows, int cols) {
    using float16 = sycl::vec<float, 16>;
    const size_t chunks = static_cast<size_t>(cols) / 16;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float16, 1> tile(sycl::range<1>(chunks), h);
        h.parallel_for(sycl::nd_range<1>(rows * Threads, Threads), [=](sycl::nd_item<1> it) {
            const size_t row = it.get_group(0), tid = it.get_local_linear_id();
            const float* in = x + row * cols;
            float* out = y + row * cols;

            float pmax = -std::numeric_limits<float>::infinity();
            for (size_t c = tid; c < chunks; c += Threads) {
                float16 v = *reinterpret_cast<const float16*>(in + c * 16);
                tile[c] = v;
                for (int e = 0; e < 16; ++e) pmax = sycl::max(pmax, v[e]);
            }
            it.barrier(sycl::access::fence_space::local_space);
            const float rmax = sycl::reduce_over_group(it.get_group(), pmax, sycl::maximum<float>());

            float16 svec(0.0f);
            for (size_t c = tid; c < chunks; c += Threads) svec += sycl::exp(tile[c] - rmax);
            float psum = 0.0f;
            for (int e = 0; e < 16; ++e) psum += svec[e];
            const float inv_sum = 1.0f / sycl::reduce_over_group(it.get_group(), psum, sycl::plus<float>());

            for (size_t c = tid; c < chunks; c += Threads) {
                *reinterpret_cast<float16*>(out + c * 16) = sycl::exp(tile[c] - rmax) * inv_sum;
            }
        });
    });
}
```

## 8. Stable Online-Softmax Rescale Core

Guarantees every exponential scale factor is `<= 1.0` so large attention score spans cannot overflow fp32 to `inf`/`NaN` `[CORRECTNESS]`:

```cpp
constexpr float LOG2E = 1.4426950408889634f;
const float oldM = mArr[r];
float rescalePrev = 1.0f;
float rescaleCur = 1.0f;
if (mTile > oldM) {
    rescalePrev = std::exp2((oldM - mTile) * LOG2E); // scale previous acc and l
} else if (mTile < oldM) {
    rescaleCur = std::exp2((mTile - oldM) * LOG2E);  // scale current tile
}
lArr[r] = lArr[r] * rescalePrev + lTile * rescaleCur;
mArr[r] = std::max(mTile, oldM);
acc = acc * rescalePrev; // multiply current tile p by rescaleCur before SxV
```

## 9. Reduction and Hillis-Steele Scan Cores

Sub-group-first global atomic reduction (`0.164 ms` on `4096x4096 f32`, `3.8x` faster vs per-thread atomic `[MEASURED]`):

```cpp
float partial = 0.0f;
for (size_t i = gid; i < N; i += grid_size) partial += in[i];
float sg_sum = sycl::reduce_over_group(it.get_sub_group(), partial, sycl::plus<float>());
if (it.get_sub_group().get_local_linear_id() == 0) {
    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>(out[0]).fetch_add(sg_sum);
}
```

Hillis-Steele work-group inclusive scan (`WG=64` sweet spot, `0.0559 ms` on `1M f32` `[MEASURED]`):

```cpp
// local_accessor<float, 1> buf(WG, h); load buf[lid] = x[gid];
for (size_t offset = 1; offset < WG; offset <<= 1) {
    it.barrier(sycl::access::fence_space::local_space);
    float val = (lid >= offset) ? (buf[lid] + buf[lid - offset]) : buf[lid];
    it.barrier(sycl::access::fence_space::local_space);
    buf[lid] = val;
}
```

## 10. Native Level Zero Runtime, Copy Engine, Doorbell, and Tier-3 .zebin Cache

```cpp
// 1) Native Regular Command List replay (0.484 us/edge, 1-replay break-even):
// Use zeCommandListAppendBarrier inside Regular CLs (NOT per-node zeEvent: 130.9 vs 248.8-295.3 us).
ze_command_list_desc_t cl_desc{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, compute_ord, 0};
ze_command_list_handle_t reg_cl = nullptr;
zeCommandListCreate(ze_ctx, ze_dev, &cl_desc, &reg_cl);
for (int step = 0; step < L; ++step) {
    zeCommandListAppendLaunchKernel(reg_cl, ze_kernel, &dispatch, nullptr, 0, nullptr);
    if (step + 1 < L) zeCommandListAppendBarrier(reg_cl, nullptr, 0, nullptr);
}
zeCommandListClose(reg_cl);
zeCommandQueueExecuteCommandLists(ze_compute_q, 1, &reg_cl, nullptr);
zeCommandQueueSynchronize(ze_compute_q, UINT64_MAX);

// 2) Explicit ordinal 1 (COPY-only) Main Copy Engine for true Compute+Copy overlap (91.6%-99.95%):
uint32_t ngroups = 0;
zeDeviceGetCommandQueueGroupProperties(ze_dev, &ngroups, nullptr);
std::vector<ze_command_queue_group_properties_t> gprops(ngroups);
for (auto& p : gprops) p.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
zeDeviceGetCommandQueueGroupProperties(ze_dev, &ngroups, gprops.data());
uint32_t copy_ord = 0;
for (uint32_t i = 0; i < ngroups; ++i) {
    if ((gprops[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) &&
        !(gprops[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE)) {
        copy_ord = i; break;
    }
}
ze_command_queue_desc_t copy_desc{
    ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, copy_ord, 0, 0,
    ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
ze_command_list_handle_t copy_imm = nullptr;
zeCommandListCreateImmediate(ze_ctx, ze_dev, &copy_desc, &copy_imm);
// Reuse static zeEventPool via zeEventHostReset (0.10-0.20 us vs 103-232 us pool create).

// 3) TDR-safe UncachedHost GPU-polled doorbell (10/10 coherent, 6.00 us RTT; ~1.12 us/spin):
ze_host_mem_alloc_desc_t hdesc{
    ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED};
void* doorbell = nullptr;
zeMemAllocHost(ze_ctx, &hdesc, 64, 64, &doorbell);

// 4) Tier-3 Domain-I DAG Fusion (.zebin SHA-256 disk cache; 399.6 GB/s = 99.1% roof, 2.73x vs unfused):
// Key: SHA-256(opencl_src + device_id + driver_version).
// Cache hit  -> zeModuleCreate(ctx, dev, &native_desc /* ZE_MODULE_FORMAT_NATIVE */, &mod, nullptr).
// Cache miss -> ocloc64.dll oclocInvoke({"ocloc","-file","k.cl","-spv_only","-device","dg2","-options","-cl-std=CL2.0"})
//            -> zeModuleCreate(ZE_MODULE_FORMAT_IL_SPIRV) -> zeModuleGetNativeBinary -> write <cache>/<sha256>.zebin.
```
