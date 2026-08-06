# Reusable Code Snippets

## Table of Contents

- Ladder to snippet map
- Naive baseline
- SLM tiling
- Register tiling
- SIMD vectorized copies
- Host-side VNNI packing for bf16 B
- Host-side A operand-layout packing (ESIMD 16x16)
- joint_matrix kernel core
- ESIMD dpas smoke test
- ESIMD SLM double-buffer skeleton
- ESIMD 16x16 GEMM core (A/B SLM relay, zero-select loads)
- Fewer barriers: 4-buffer B pipeline
- Complete GEMM alpha/beta with runtime dimensions
- f32 GEMV core (sub-group per row, direct L2)
- f32 RMSNorm core (SLM row tile)
- f32 Softmax core (SLM row tile)
- Correctness and timing harness

These snippets are taken from kernels that compiled and ran on Arc A770. Each section names the compilable source file it was extracted from, so you can recover the full original implementation when this skill is used away from the campaign workspace. They are building blocks, not a drop-in operator: adapt the tile constants, address math, and host packing to your shape, and keep the tile-divisibility constraints from the linked variants.

## Ladder to Snippet Map

Every step of the measured ladder in [techniques.md](techniques.md) maps to embedded code in this file, so the skill stays self-contained without the campaign workspace.

| Ladder step | Technique | Embedded snippet |
|---|---|---|
| 1 | Naive | [Naive baseline](#naive-baseline) |
| 2 | SLM tiling | [SLM tiling](#slm-tiling) |
| 3 | Register tiling | [Register tiling](#register-tiling) |
| 4 | SIMD vectorized loads/stores | [SIMD vectorized copies](#simd-vectorized-copies) |
| 5-6 | joint_matrix direct global / basic SLM | [joint_matrix kernel core](#joint_matrix-kernel-core) |
| 7 | software prefetch | Negative; see [pitfalls.md](pitfalls.md) and [api-usage.md](api-usage.md) |
| 8 | double buffer (small tile) | [ESIMD SLM double-buffer skeleton](#esimd-slm-double-buffer-skeleton) |
| 9-13 | joint_matrix tuned path | [joint_matrix kernel core](#joint_matrix-kernel-core), [VNNI packing](#host-side-vnni-packing-for-bf16-b) |
| 14 | ESIMD dpas, BK32, 16x16 | [dpas smoke test](#esimd-dpas-smoke-test), [SLM skeleton](#esimd-slm-double-buffer-skeleton) |
| 15 | wide SLM->GRF loads | [16x16 GEMM core](#esimd-16x16-gemm-core-ab-slm-relay-zero-select-loads) |
| 16 | host-side operand layout | [A operand-layout packing](#host-side-a-operand-layout-packing-esimd-16x16) + [16x16 GEMM core](#esimd-16x16-gemm-core-ab-slm-relay-zero-select-loads) |
| 17 | address slimming + 4-buffer B | [4-buffer B pipeline](#fewer-barriers-4-buffer-b-pipeline) |
| 18 | A SLM relay | [16x16 GEMM core](#esimd-16x16-gemm-core-ab-slm-relay-zero-select-loads) |

The f32 RMSNorm ladder lives in [techniques.md](techniques.md#f32-rmsnorm-ladder-1024x4096-row-major-x-f32); its final kernel is [f32 RMSNorm core](#f32-rmsnorm-core-slm-row-tile) and its oneDNN baseline API is in [api-usage.md](api-usage.md#onednn-rmsnorm-baseline).

The f32 Softmax ladder lives in [techniques.md](techniques.md#f32-softmax-ladder); its final kernel is [f32 Softmax core](#f32-softmax-core-slm-row-tile) and its oneDNN baseline API is in [api-usage.md](api-usage.md#onednn-softmax-baseline).

## Naive Baseline

Source pattern: `gemm.cpp` from the campaign workspace.

Per-element global reads. Measured 1.95174 ms; use it only as a correctness baseline, never as a performance target.

```cpp
q.parallel_for(range{M, N}, [=](id<2> idx) {
    const int row = idx[0], col = idx[1];
    float sum = 0.0f;
    for (int k = 0; k < K; ++k)
        sum += static_cast<float>(A[row * K + k]) *
               static_cast<float>(B[k * N + col]);
    C[row * N + col] = sum;
});
```

## SLM Tiling

Source pattern: `gemm_tiled.cpp`.

One 16x16 tile per work-item staged through SLM. Measured 1.49231 ms; the win comes from replacing scattered global reads with cooperative block copies, but the tile is still too small.

```cpp
constexpr size_t TILE = 16;
local_accessor<bf16, 2> tileA(range<2>{TILE, TILE}, h);
local_accessor<bf16, 2> tileB(range<2>{TILE, TILE}, h);

h.parallel_for(nd_range<2>(range<2>{M, N}, range<2>{TILE, TILE}),
               [=](nd_item<2> item) {
    const int row = item.get_global_id(0);
    const int col = item.get_global_id(1);
    const int lr = item.get_local_id(0);
    const int lc = item.get_local_id(1);
    float sum = 0.0f;
    for (size_t bk = 0; bk < K; bk += TILE) {
        tileA[lr][lc] = (row < M && bk + lc < K)
                            ? A[row * K + bk + lc] : bf16(0.0f);
        tileB[lr][lc] = (bk + lr < K && col < N)
                            ? B[(bk + lr) * N + col] : bf16(0.0f);
        item.barrier(access::fence_space::local_space);
        for (int k = 0; k < TILE; ++k)
            sum += static_cast<float>(tileA[lr][k]) *
                   static_cast<float>(tileB[k][lc]);
        item.barrier(access::fence_space::local_space);
    }
    if (row < M && col < N) C[row * N + col] = sum;
});
```

## Register Tiling

Source pattern: `gemm_rgt.cpp`.

Each thread accumulates a TM x TN block in registers and reuses A/B slices loaded from SLM. Measured 0.430648 ms; this is where the compute/load ratio starts to matter.

```cpp
constexpr int BM = 64, BN = 64, BK = 16;
constexpr int TM = 4, TN = 4;
float acc[TM][TN] = {0.0f};
const int tid = local_row * (BN / TN) + local_col;
const int threads_per_wg = (BM / TM) * (BN / TN); // 256

for (size_t bk = 0; bk < K; bk += BK) {
    for (int i = tid; i < BM * BK; i += threads_per_wg) {
        const int r = i / BK, c = i % BK;
        const int g_r = wg_row * BM + r, g_c = bk + c;
        tileA[r][c] = (g_r < M && g_c < K) ? A[g_r * K + g_c] : bf16(0.0f);
    }
    for (int i = tid; i < BK * BN; i += threads_per_wg) {
        const int r = i / BN, c = i % BN;
        const int g_r = bk + r, g_c = wg_col * BN + c;
        tileB[r][c] = (g_r < K && g_c < N) ? B[g_r * N + g_c] : bf16(0.0f);
    }
    item.barrier(access::fence_space::local_space);
    for (int k = 0; k < BK; ++k) {
        float regA[TM], regB[TN];
        for (int m = 0; m < TM; ++m)
            regA[m] = static_cast<float>(tileA[local_row * TM + m][k]);
        for (int n = 0; n < TN; ++n)
            regB[n] = static_cast<float>(tileB[k][local_col * TN + n]);
        for (int m = 0; m < TM; ++m)
            for (int n = 0; n < TN; ++n)
                acc[m][n] += regA[m] * regB[n];
    }
    item.barrier(access::fence_space::local_space);
}
```

## SIMD Vectorized Copies

Source pattern: `gemm_simd.cpp`.

Load global -> SLM as `vec<bf16, 4>` words and store C as `vec<float, 4>`. Measured 0.273737 ms; instruction count drops even though the algorithm shape is unchanged.

```cpp
constexpr size_t VEC_SIZE = 4; // 4 x bf16 = 8 B; 4 x float = 16 B
// Global -> SLM as vector words.
for (int i = tid * VEC_SIZE; i < BM * BK;
     i += threads_per_wg * VEC_SIZE) {
    const int r = i / BK, c = i % BK;
    const int g_r = wg_row * BM + r, g_c = bk + c;
    if (g_r < M && g_c + VEC_SIZE - 1 < K) {
        auto vec_a = *reinterpret_cast<const vec<bf16, VEC_SIZE> *>(
            &A[g_r * K + g_c]);
        *reinterpret_cast<vec<bf16, VEC_SIZE> *>(&tileA[r][c]) = vec_a;
    }
}
// C write-back: one 4-float vectorized store per output row.
*reinterpret_cast<vec<float, VEC_SIZE> *>(&C[g_r * N + g_c]) = vec_out;
```

## Host-Side VNNI Packing for bf16 B

Source: `gemm_esimd_v10.cpp`, `gemm_esimd_smoke.cpp`, `gemm_esimd_slm.cpp`.

VNNI packs two consecutive K rows of B into one uint32. The kernel then copies uint32 words and feeds DPAS its exact operand layout.

```cpp
void pack_bp(bf16 *Braw, uint32_t *Bp, int K, int N) {
    for (int k2 = 0; k2 < K / 2; k2++)
        for (int j = 0; j < N; j++) {
            uint16_t lo = static_cast<uint16_t>(
                sycl::bit_cast<uint16_t>(Braw[(k2 * 2) * N + j]));
            uint16_t hi = static_cast<uint16_t>(
                sycl::bit_cast<uint16_t>(Braw[(k2 * 2 + 1) * N + j]));
            Bp[k2 * N + j] =
                static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
        }
}
```

Verify against the raw B array, not the packed array: word index is `(k/2)*N + n`, low 16 bits are `B[2k][n]`, high 16 bits are `B[2k+1][n]`.

## Host-Side A Operand-Layout Packing (ESIMD 16x16)

Source: `gemm_esimd_tile16_wide_v8b.cpp`.

For a BM=128, BK=32, 32-thread work-group with 8 work-item rows, pack A so each DPAS operand becomes one contiguous 256 B block. This is what lets the kernel load A with four `block_load<bf16, 128>` calls and no `select`.

```cpp
for (int wg_row = 0; wg_row < M / BM; wg_row++)
    for (int kb = 0; kb < K / BK; kb++)
        for (int wi = 0; wi < 8; wi++)
            for (int o = 0; o < 4; o++)
                for (int rr = 0; rr < 8; rr++)
                    for (int cc = 0; cc < 16; cc++) {
                        const int m = wg_row * BM + wi * 16 + rr + (o / 2) * 8;
                        const int k = kb * BK + cc + (o % 2) * 16;
                        Ap[((((wg_row * 16 + kb) * 8 + wi) * 4 + o) * 8 +
                            rr) *
                               16 +
                           cc] = A[m * K + k];
                    }
```

Measured: A direct global read after this packing took the kernel from 0.0836 ms to 0.0734 ms (v5), and staging the same layout through SLM later reached 0.0614 ms (v8b).

## joint_matrix Kernel Core

Source: `gemm_jm_walk_n.cpp`.

Measured best joint_matrix variant: 0.11831 ms (45.4% of oneMKL). BM=128, BN=64, BK=16, 512-thread work-group, VNNI B, vectorized A copy, double buffer, N-first walk, K first/last split.

```cpp
constexpr int TM = 8, TN = 8, TK = 16;    // A770 bf16 DPAS shape
constexpr int TM_SG = 16, TN_SG = 16;     // per sub-group C block
constexpr int BM = 128, BN = 64, BK = 16; // work-group tile
constexpr size_t VEC_SIZE = 4;
constexpr size_t SG_SIZE = 16;

range<2> global_size{(M / BM) * (BM / TM_SG),
                     (N / BN) * (BN / TN_SG) * SG_SIZE};
range<2> local_size{BM / TM_SG, (BN / TN_SG) * SG_SIZE};

q.submit([&](handler &h) {
    local_accessor<bf16, 3> tileA(range<3>{2, BM, BK}, h);
    local_accessor<uint32_t, 3> tileB(range<3>{2, BK / 2, BN}, h);

    h.parallel_for(nd_range<2>(global_size, local_size), [=](nd_item<2> item) {
        auto sg = item.get_sub_group();
        // N-first walk: swap M/N group indices for L2 reuse.
        int wg_row = item.get_group(1);
        int wg_col = item.get_group(0);
        int sg_row_in_wg = item.get_local_id(0);
        int sg_col_in_wg = item.get_local_id(1) / SG_SIZE;
        int local_tid = item.get_local_id(0) * (BN / TN_SG * SG_SIZE) +
                        item.get_local_id(1);
        int threads_per_wg = (BM / TM_SG) * (BN / TN_SG) * SG_SIZE; // 512

        auto pA_global = sycl::address_space_cast<
            access::address_space::global_space,
            access::decorated::no>(A);
        auto pC_global = sycl::address_space_cast<
            access::address_space::global_space,
            access::decorated::no>(C);

        auto load_block = [&](int buf, size_t bk) {
            // A: vec<bf16,4> vectorized global -> SLM.
            for (int i = local_tid * VEC_SIZE; i < BM * BK;
                 i += threads_per_wg * VEC_SIZE) {
                int r = i / BK, c = i % BK;
                int g_r = wg_row * BM + r, g_c = bk + c;
                if (g_r < M && g_c + VEC_SIZE - 1 < K) {
                    auto vec_a = *reinterpret_cast<
                        const vec<bf16, VEC_SIZE> *>(&A[g_r * K + g_c]);
                    *reinterpret_cast<vec<bf16, VEC_SIZE> *>(
                        &tileA[buf][r][c]) = vec_a;
                }
            }
            // B: already VNNI-packed on host, copy uint32 words.
            for (int i = local_tid; i < (BK / 2) * BN;
                 i += threads_per_wg) {
                int r = i / BN, c = i % BN;
                int g_r = bk / 2 + r, g_c = wg_col * BN + c;
                tileB[buf][r][c] =
                    (g_r < K / 2 && g_c < N) ? B_packed[g_r * N + g_c] : 0u;
            }
        };

        joint_matrix<sub_group, bf16, use::a, TM, TK, layout::row_major>
            sub_a0, sub_a1;
        joint_matrix<sub_group, bf16, use::b, TK, TN,
                     layout::ext_intel_packed> sub_b0, sub_b1;
        joint_matrix<sub_group, float, use::accumulator, TM, TN>
            sub_c00, sub_c01, sub_c10, sub_c11;
        joint_matrix_fill(sg, sub_c00, 0.0f);
        joint_matrix_fill(sg, sub_c01, 0.0f);
        joint_matrix_fill(sg, sub_c10, 0.0f);
        joint_matrix_fill(sg, sub_c11, 0.0f);

        load_block(0, 0);
        item.barrier(access::fence_space::local_space);

        auto compute_block = [&](int cur) {
            auto pA_slm0 = sycl::address_space_cast<
                access::address_space::local_space,
                access::decorated::no>(&tileA[cur][sg_row_in_wg * TM_SG][0]);
            auto pA_slm1 = sycl::address_space_cast<
                access::address_space::local_space,
                access::decorated::no>(
                &tileA[cur][sg_row_in_wg * TM_SG + TM][0]);
            auto pB_slm0 = sycl::address_space_cast<
                access::address_space::local_space,
                access::decorated::no>(
                reinterpret_cast<bf16 *>(
                    &tileB[cur][0][sg_col_in_wg * TN_SG]));
            auto pB_slm1 = sycl::address_space_cast<
                access::address_space::local_space,
                access::decorated::no>(
                reinterpret_cast<bf16 *>(
                    &tileB[cur][0][sg_col_in_wg * TN_SG + TN]));
            joint_matrix_load(sg, sub_a0, pA_slm0, BK);
            joint_matrix_load(sg, sub_a1, pA_slm1, BK);
            joint_matrix_load(sg, sub_b0, pB_slm0, BN * 2);
            joint_matrix_load(sg, sub_b1, pB_slm1, BN * 2);
            joint_matrix_mad(sg, sub_c00, sub_a0, sub_b0, sub_c00);
            joint_matrix_mad(sg, sub_c01, sub_a0, sub_b1, sub_c01);
            joint_matrix_mad(sg, sub_c10, sub_a1, sub_b0, sub_c10);
            joint_matrix_mad(sg, sub_c11, sub_a1, sub_b1, sub_c11);
        };

        // K split: all but the last block prefetch unconditionally.
        size_t bk = 0;
        for (; bk + BK < K; bk += BK) {
            int cur = (bk / BK) % 2, nxt = cur ^ 1;
            load_block(nxt, bk + BK);
            compute_block(cur);
            item.barrier(access::fence_space::local_space);
        }
        compute_block((bk / BK) % 2);

        int global_r = wg_row * BM + sg_row_in_wg * TM_SG;
        int global_c = wg_col * BN + sg_col_in_wg * TN_SG;
        joint_matrix_store(sg, sub_c00,
                           pC_global + global_r * N + global_c, N,
                           layout::row_major);
        joint_matrix_store(sg, sub_c01,
                           pC_global + global_r * N + global_c + TN, N,
                           layout::row_major);
        joint_matrix_store(sg, sub_c10,
                           pC_global + (global_r + TM) * N + global_c, N,
                           layout::row_major);
        joint_matrix_store(sg, sub_c11,
                           pC_global + (global_r + TM) * N + global_c + TN, N,
                           layout::row_major);
    });
});
```

Key details: packed B rows are loaded with stride `BN * 2` bf16; the 16x16 C block is four 8x8 accumulators; the work-group is 32 sub-groups x 16 lanes = 512 threads.

## ESIMD dpas Smoke Test

Source: `gemm_esimd_smoke.cpp`.

Smallest proof that `esimd::dpas` works on A770 with correct operand layout. Single 8x8x16 tile, both accumulate and non-accumulate forms.

```cpp
q.single_task([=]() SYCL_ESIMD_KERNEL {
    simd<bf16, M * K> a(A, overaligned_tag<16>{}); // 8x16 row-major
    simd<bf16, K * N> b(B, overaligned_tag<16>{}); // VNNI-packed view
    simd<float, M * N> c0(0.0f);

    simd<float, M * N> c1 = dpas<8, M, float>(c0, b, a); // C += A*B
    c1.copy_to(Cd);

    simd<float, M * N> c2 = dpas<8, M, float>(b, a);     // C = A*B
    c2.copy_to(C2);
}).wait();
```

Measured: both forms 0/64 errors. Use this pattern before building any larger ESIMD XMX kernel; the A/B layout bugs show up here first.

## ESIMD SLM Double-Buffer Skeleton

Source: `gemm_esimd_slm.cpp`.

The smallest working pipelined GEMM skeleton (32x32x32, 16 work-items, 0/1024 errors). Copy this structure when adding SLM to a new ESIMD kernel.

```cpp
slm_init<SLM_TOTAL_BYTES>();
const uint32_t lid = it.get_local_id(0);
const int tr = lid / 4, tc = lid % 4; // dpas tile row/col

auto load_block = [=](int buf, size_t bk) SYCL_ESIMD_FUNCTION {
    const uint32_t offA = buf * SLM_BUF_BYTES;
    const uint32_t offB = offA + 1024;
    for (int rr = 0; rr < 2; rr++) {
        const int r = lid + rr * 16;
        simd<bf16, 16> av = block_load<bf16, 16>(
            A + r * Ktot + bk, overaligned_tag<16>{});
        slm_block_store(offA + r * 32, av, overaligned_tag<16>{});
    }
    simd<uint32_t, 16> bv = block_load<uint32_t, 16>(
        Bp + (bk / 2) * BN + lid * 16, overaligned_tag<16>{});
    slm_block_store(offB + lid * 64, bv, overaligned_tag<16>{});
};

auto load_grf = [=](int buf, simd<bf16, 128> &a,
                    simd<bf16, 128> &b) SYCL_ESIMD_FUNCTION {
    const uint32_t offA = buf * SLM_BUF_BYTES;
    const uint32_t offB = offA + 1024;
    for (int r = 0; r < DPAS_M; r++)
        a.select<16, 1>(r * 16) = slm_block_load<bf16, 16>(
            offA + (tr * DPAS_M + r) * 32, overaligned_tag<16>{});
    for (int k2 = 0; k2 < 8; k2++) {
        simd<uint32_t, 8> w = slm_block_load<uint32_t, 8>(
            offB + (k2 * BN + tc * DPAS_N) * 4, overaligned_tag<16>{});
        b.select<16, 1>(k2 * 16) = w.bit_cast_view<bf16>();
    }
};

simd<float, DPAS_M * DPAS_N> c(0.0f);
load_block(0, 0);
barrier();
for (size_t bk = 0; bk < Ktot; bk += BK) {
    const int cur = (bk / BK) % 2, nxt = cur ^ 1;
    if (bk + BK < Ktot) load_block(nxt, bk + BK);
    simd<bf16, 128> a, b;
    load_grf(cur, a, b);
    c = dpas<8, DPAS_M, float>(c, b, a);
    barrier();
}
```

Note the `bit_cast_view` call on the named `simd` lvalue `w`; doing this on a temporary fails to compile.

## ESIMD 16x16 GEMM Core (A/B SLM Relay, Zero-Select Loads)

Source: `gemm_esimd_tile16_wide_v8b.cpp`.

Final champion structure (v8b, 0.0613 to 0.0615 ms, about 90% of oneDNN): 32-thread work-group, BM=128, BN=64, BK=32, A and B both staged through SLM in host-packed operand layout, one barrier per K block.

```cpp
// Per-thread constants hoisted out of the K loop (address slimming).
const int wi_row = lid / 4; // 16 rows each
const int wi_col = lid % 4; // 16 cols each
const uint32_t b_op_base = (uint32_t)(wi_col * 256);
const bf16 *apb = Ap + (size_t)((wg_row * (K / BK)) * 8) * 512;
const uint32_t *brow = Bp + (size_t)r2 * N + wg_col * BN + hp * 32;

auto load_block = [=](int buf, int abuf, const bf16 *apb,
                      const uint32_t *brow) SYCL_ESIMD_FUNCTION {
    const uint32_t offA = abuf * SLM_A_BYTES;
    const uint32_t offB = SLM_B_BASE + buf * SLM_B_BYTES;

    // A: 8 KB block, 256 B per work-item.
    simd<bf16, 128> av = block_load<bf16, 128>(
        apb + lid * 128, overaligned_tag<16>{});
    slm_block_store(offA + lid * 256, av, overaligned_tag<16>{});

    // B: 128 B global read, scattered into four SLM operand slices.
    simd<uint32_t, 32> bv = block_load<uint32_t, 32>(
        brow, overaligned_tag<16>{});
    slm_block_store(offB + b_slm0, bv.select<8, 1>(0).read(),
                    overaligned_tag<16>{});
    slm_block_store(offB + b_slm1, bv.select<8, 1>(8).read(),
                    overaligned_tag<16>{});
    slm_block_store(offB + b_slm2, bv.select<8, 1>(16).read(),
                    overaligned_tag<16>{});
    slm_block_store(offB + b_slm3, bv.select<8, 1>(24).read(),
                    overaligned_tag<16>{});
};

auto load_grf = [=](int buf, int abuf, simd<bf16, 128> &a0,
                    simd<bf16, 128> &a1, simd<bf16, 128> &a2,
                    simd<bf16, 128> &a3, simd<bf16, 128> &b0,
                    simd<bf16, 128> &b1, simd<bf16, 128> &b2,
                    simd<bf16, 128> &b3) SYCL_ESIMD_FUNCTION {
    const uint32_t offA = abuf * SLM_A_BYTES + wi_row * 1024;
    const uint32_t offB = SLM_B_BASE + buf * SLM_B_BYTES;

    a0 = slm_block_load<bf16, 128>(offA + 0 * 256,
                                   overaligned_tag<16>{});
    a1 = slm_block_load<bf16, 128>(offA + 1 * 256,
                                   overaligned_tag<16>{});
    a2 = slm_block_load<bf16, 128>(offA + 2 * 256,
                                   overaligned_tag<16>{});
    a3 = slm_block_load<bf16, 128>(offA + 3 * 256,
                                   overaligned_tag<16>{});

    simd<uint32_t, 64> wb0 = slm_block_load<uint32_t, 64>(
        offB + b_op_base + 0 * 1024, overaligned_tag<16>{});
    simd<uint32_t, 64> wb2 = slm_block_load<uint32_t, 64>(
        offB + b_op_base + 1 * 1024, overaligned_tag<16>{});
    simd<uint32_t, 64> wb1 = slm_block_load<uint32_t, 64>(
        offB + b_op_base + 2 * 1024, overaligned_tag<16>{});
    simd<uint32_t, 64> wb3 = slm_block_load<uint32_t, 64>(
        offB + b_op_base + 3 * 1024, overaligned_tag<16>{});
    b0 = wb0.bit_cast_view<bf16>();
    b2 = wb2.bit_cast_view<bf16>();
    b1 = wb1.bit_cast_view<bf16>();
    b3 = wb3.bit_cast_view<bf16>();
};

simd<bf16, 128> a0, a1, a2, a3;
simd<bf16, 128> b0, b1, b2, b3;
simd<float, 64> c00(0.0f), c01(0.0f), c10(0.0f), c11(0.0f);

load_block(0, 0, apb, brow);
barrier();
for (int b = 0; b < K / BK; b++) {
    const int cur = b & 1, nxt = cur ^ 1;
    if (b + 1 < K / BK)
        load_block(nxt, nxt, apb + 4096, brow + 16 * N);
    load_grf(cur, cur, a0, a1, a2, a3, b0, b1, b2, b3);
    c00 = dpas<8, 8, float>(c00, b0, a0);
    c00 = dpas<8, 8, float>(c00, b1, a1);
    c01 = dpas<8, 8, float>(c01, b2, a0);
    c01 = dpas<8, 8, float>(c01, b3, a1);
    c10 = dpas<8, 8, float>(c10, b0, a2);
    c10 = dpas<8, 8, float>(c10, b1, a3);
    c11 = dpas<8, 8, float>(c11, b2, a2);
    c11 = dpas<8, 8, float>(c11, b3, a3);
    barrier();
    apb += 4096;
    brow += 16 * N;
}
```

Write C back in 64 B rows (two 16-float rows per accumulator pair):

```cpp
const int gr = wg_row * BM + wi_row * 16;
const int gc = wg_col * BN + wi_col * 16;
for (int r = 0; r < 8; r++) {
    simd<float, 16> row0;
    row0.select<8, 1>(0) = c00.select<8, 1>(r * 8);
    row0.select<8, 1>(8) = c01.select<8, 1>(r * 8);
    block_store<float, 16>(C + (size_t)(gr + r) * N + gc, row0,
                           overaligned_tag<16>{});

    simd<float, 16> row1;
    row1.select<8, 1>(0) = c10.select<8, 1>(r * 8);
    row1.select<8, 1>(8) = c11.select<8, 1>(r * 8);
    block_store<float, 16>(C + (size_t)(gr + 8 + r) * N + gc, row1,
                           overaligned_tag<16>{});
}
```

## Fewer Barriers: 4-Buffer B Pipeline

Source: `gemm_esimd_tile16_wide_v6b4.cpp`.

With A read directly from global (v6b4), four B buffers of 4 KB and one barrier per pair of K blocks cut barriers from 109M to 57M and measured -14%.

```cpp
load_block(0, brow);
load_block(1, brow + 16 * N);
barrier();

for (int b = 0; b < K / BK; b += 2) {
    if (b + 2 < K / BK) load_block((b + 2) & 3, brow + 32 * N);
    if (b + 3 < K / BK) load_block((b + 3) & 3, brow + 48 * N);

    load_grf(b & 3, ap, a0, a1, a2, a3, b0, b1, b2, b3);
    // 8 dpas for block b (same chain as v8b).
    load_grf((b + 1) & 3, ap + 4096, a0, a1, a2, a3, b0, b1, b2, b3);
    // 8 dpas for block b+1.
    barrier();

    ap += 2 * 4096;
    brow += 32 * N;
}
```

## Complete GEMM Alpha/Beta with Runtime Dimensions

Source: `gemm_esimd_v10.cpp`.

Runtime values inside an ESIMD kernel hang A770, so dispatch the fast path with a template and `if constexpr`, and pass dimensions as kernel parameters.

```cpp
template <bool Plain>
void gemm_esimd_v10_impl(bf16 *Ap, uint32_t *Bp, float *C, int M, int N,
                         int K, float alpha, float beta, queue q) {
    // v8b kernel body; Plain selects the write-back path below.
}

void gemm_esimd_v10(bf16 *Ap, uint32_t *Bp, float *C, int M, int N, int K,
                    float alpha, float beta, queue q) {
    if (alpha == 1.0f && beta == 0.0f)
        gemm_esimd_v10_impl<true>(Ap, Bp, C, M, N, K, alpha, beta, q);
    else
        gemm_esimd_v10_impl<false>(Ap, Bp, C, M, N, K, alpha, beta, q);
}
```

General write-back recomputes the old C row in the kernel instead of keeping a second copy of the accumulator:

```cpp
if constexpr (Plain) {
    // alpha=1, beta=0: plain 64B stores.
} else {
    c00 *= alpha;
    c01 *= alpha;
    c10 *= alpha;
    c11 *= alpha;
    for (int r = 0; r < 8; r++) {
        simd<float, 16> old0 = block_load<float, 16>(
            C + (size_t)(gr + r) * N + gc, overaligned_tag<16>{});
        simd<float, 16> row0;
        row0.select<8, 1>(0) = c00.select<8, 1>(r * 8).read() +
                               old0.select<8, 1>(0).read() * beta;
        row0.select<8, 1>(8) = c01.select<8, 1>(r * 8).read() +
                               old0.select<8, 1>(8).read() * beta;
        block_store<float, 16>(C + (size_t)(gr + r) * N + gc, row0,
                               overaligned_tag<16>{});
        // Same pattern for the c10/c11 row pair with old1.
    }
}
```

## f32 GEMV Core (Sub-Group Per Row, Direct L2)

Source pattern: `gemv.cpp` from the GEMV-Opti campaign.

Measured 0.215 ms for `y = A*x`, `4096x4096`, row-major f32 A on A770; the same-operation baselines were oneMKL GPU gemv 0.329 ms and oneDNN GPU matmul `Kx1` 0.380 ms. Keep the per-lane trip count dynamic because the compiler may select 32-lane sub-groups.

```cpp
constexpr size_t SUB = 16;       // local dimension 0
constexpr size_t SG_PER_WG = 32; // local dimension 1; 512 threads/WG
constexpr size_t VEC = 16;

void gemv_fast(queue& q, float* a, float* x, float* y,
               size_t M, size_t N) {
    const size_t nvec = N / VEC;
    q.submit([&](handler& h) {
        h.parallel_for(
            nd_range<2>(range<2>(SUB, M), range<2>(SUB, SG_PER_WG)),
            [=](nd_item<2> it) {
                auto sg = it.get_sub_group();
                const size_t lane = sg.get_local_linear_id();
                const size_t sg_id = sg.get_group_linear_id();
                const size_t sg_size = sg.get_local_range()[0];
                const size_t per_lane = nvec / sg_size;
                const size_t row = it.get_group(1) * SG_PER_WG + sg_id;

                vec<float, VEC> acc(0.0f);
                for (size_t k = 0; k < per_lane; ++k) {
                    const size_t col = (lane * per_lane + k) * VEC;
                    acc += *reinterpret_cast<const vec<float, VEC>*>(
                               a + row * N + col) *
                           *reinterpret_cast<const vec<float, VEC>*>(x + col);
                }
                float sum = 0.0f;
                for (size_t e = 0; e < VEC; ++e) sum += acc[e];
                const float s =
                    reduce_over_group(sg, sum, plus<float>());
                if (lane == 0) y[row] = s;
            });
    });
}
```

This kernel requires `M % SG_PER_WG == 0` and `N % (VEC * SG_PER_WG) == 0`; keep the naive scalar row kernel as a fallback for arbitrary dimensions. Do not use oneDNN `1xK` matmul as the GEMV baseline: it computes `x*A`, which is `A^T*x` for row-major A.

## u4->bf16 GEMV Core (Sub-Group Per Row, bf16 Vector + Float Accumulator)

Source pattern: `gemv_u4_bf16.cpp` from the QuantizedGEMV-Opti campaign.

Measured 0.0883 ms for `y = A*x`, `4096x4096`, bf16 A, u4-packed x,
group_size=128, f16 scales, zero-point 8, bf16 y; oneDNN `Kx1` measured
0.1456 ms. The u4 vector is dequantized to bf16 on the host once, then this
kernel runs. Use a float vector accumulator; a bf16 accumulator fails large K.

```cpp
void dequant_x_u4_host(const std::uint8_t* xp, const sycl::half* scales,
                       bf16* xd, std::size_t n, std::size_t group_size) {
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t g = k / group_size;
        const std::uint8_t byte = xp[k / 2];
        const float wv = (k & 1)
                             ? static_cast<float>((byte >> 4) & 0x0F) - 8.0f
                             : static_cast<float>(byte & 0x0F) - 8.0f;
        xd[k] = static_cast<bf16>(
            static_cast<float>(scales[g]) * wv);
    }
}

void gemv_bf16_x_usm(sycl::queue& q, const bf16* a, const bf16* xd,
                     bf16* y, std::size_t m, std::size_t n) {
    constexpr std::size_t VEC = 16;
    constexpr std::size_t WG = 128;
    const std::size_t nvec = n / VEC;
    const std::size_t n_wg = (m * 32 + WG - 1) / WG;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(n_wg * WG),
                              sycl::range<1>(WG)),
            [=](sycl::nd_item<1> it) {
                auto sg = it.get_sub_group();
                const std::size_t lane = sg.get_local_linear_id();
                const std::size_t sg_id = sg.get_group_linear_id();
                const std::size_t sg_size = sg.get_local_range()[0];
                const std::size_t num_sg = WG / sg_size;
                const std::size_t row =
                    it.get_group(0) * num_sg + sg_id;
                if (row >= m) return;

                const std::size_t per_lane = nvec / sg_size;
                const std::size_t rem = nvec % sg_size;
                sycl::vec<float, VEC> acc(0.0f);

                const auto accumulate = [&](std::size_t col) {
                    const auto av =
                        *reinterpret_cast<const sycl::vec<bf16, VEC>*>(
                            a + row * n + col);
                    const auto xv =
                        *reinterpret_cast<const sycl::vec<bf16, VEC>*>(
                            xd + col);
                    acc += av.template convert<float>() *
                           xv.template convert<float>();
                };

                for (std::size_t k = 0; k < per_lane; ++k) {
                    accumulate((lane * per_lane + k) * VEC);
                }
                if (lane < rem) {
                    accumulate((sg_size * per_lane + lane) * VEC);
                }

                float partial = 0.0f;
                for (std::size_t e = 0; e < VEC; ++e) partial += acc[e];
                const float total =
                    sycl::reduce_over_group(sg, partial, sycl::plus<float>());
                if (lane == 0) y[row] = static_cast<bf16>(total);
            });
    });
}
```

Requires 64 B aligned USM and `n % 16 == 0`; keep a scalar u4 fallback for
arbitrary column counts. oneDNN baseline descriptor:

```cpp
src_md = memory::desc({M, K}, data_type::bf16, format_tag::ab);
wei_md = memory::desc({K, 1}, data_type::u4, format_tag::ba); // [1, K/2] bytes
dst_md = memory::desc({M, 1}, data_type::bf16, format_tag::ab);
scale_md = memory::desc({groups, 1}, data_type::f16, format_tag::ab);
zp_md = memory::desc({1}, data_type::u8, format_tag::a);
attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1),
                {group_size, 1}, data_type::f16);
attr.set_zero_points(DNNL_ARG_WEIGHTS, 0, {}, data_type::u8);
attr.set_fpmath_mode(fpmath_mode::any, true);
```

`format_tag::ba` for `{K,1}` u4 is the same packed-vector layout as the GEMM
campaign: byte `k/2`, low nibble first along K. Both f16 and bf16 src select
`jit:gemm:any`; plain `ab` u4 weights fail to create the primitive descriptor.

## f32 RMSNorm Core (SLM Row Tile)

Source pattern: `rmsnorm.cpp` from the RMSNorm-Opti campaign.

Measured 0.09797 to 0.1000 ms for 1024x4096 f32 on A770; the oneDNN RMSNorm baseline was 0.1235 to 0.1242 ms. One work-group per row, 128 threads, 2 barriers, x read from global once and normalized from SLM.

```cpp
constexpr size_t WG_THREADS = 128;
constexpr size_t VEC = 16;
constexpr size_t CHUNKS_PER_ROW = 256;  // 4096 / 16
constexpr size_t CHUNKS_PER_THREAD = 2; // 128 * 2 = 256 chunks

void rmsnorm_fast(queue& q, const float* x, const float* gamma, float* y,
                  size_t rows, size_t cols, float eps) {
    q.submit([&](handler& h) {
        local_accessor<vec<float, VEC>, 2> tile(
            range<2>(1, CHUNKS_PER_ROW), h);
        local_accessor<float, 1> row_sum(range<1>(1), h);

        h.parallel_for(nd_range<1>(range<1>(rows * WG_THREADS),
                                   range<1>(WG_THREADS)),
                       [=](nd_item<1> it) {
            const size_t tid = it.get_local_linear_id();
            const size_t row = it.get_group(0);
            auto sg = it.get_sub_group();

            if (tid == 0) row_sum[0] = 0.0f;

            for (size_t p = 0; p < CHUNKS_PER_THREAD; ++p) {
                const size_t cv =
                    (tid * CHUNKS_PER_THREAD + p) % CHUNKS_PER_ROW;
                tile[0][cv] =
                    *reinterpret_cast<const vec<float, VEC>*>(
                        x + row * cols + cv * VEC);
            }

            it.barrier(access::fence_space::local_space);

            vec<float, VEC> acc(0.0f);
            for (size_t p = 0; p < CHUNKS_PER_THREAD; ++p) {
                const size_t cv =
                    (tid * CHUNKS_PER_THREAD + p) % CHUNKS_PER_ROW;
                acc += tile[0][cv] * tile[0][cv];
            }
            float partial = 0.0f;
            for (size_t e = 0; e < VEC; ++e) partial += acc[e];
            const float psum =
                reduce_over_group(sg, partial, plus<float>());
            if (sg.get_local_linear_id() == 0) {
                atomic_ref<float, memory_order::relaxed,
                           memory_scope::work_group,
                           access::address_space::local_space>
                    ref(row_sum[0]);
                ref.fetch_add(psum);
            }

            it.barrier(access::fence_space::local_space);

            const float inv_rms =
                1.0f / sqrt(row_sum[0] / static_cast<float>(cols) + eps);

            for (size_t p = 0; p < CHUNKS_PER_THREAD; ++p) {
                const size_t cv =
                    (tid * CHUNKS_PER_THREAD + p) % CHUNKS_PER_ROW;
                const auto g =
                    *reinterpret_cast<const vec<float, VEC>*>(gamma + cv * VEC);
                *reinterpret_cast<vec<float, VEC>*>(
                    y + row * cols + cv * VEC) =
                    tile[0][cv] * (inv_rms * g);
            }
        });
    });
}
```

Requires `cols % (VEC * CHUNKS_PER_THREAD) == 0` (32 for this shape); keep the naive scalar row kernel as a fallback for arbitrary dimensions. Staging gamma in SLM is not useful here because gamma is already read once per row and stays in L2. Replacing the `vec` square accumulator with a scalar `v[e]*v[e]` loop was measured neutral: the compiler already vectorizes it.

## f32 Softmax Core (SLM Row Tile)

Source pattern: `softmax_opt.cpp` from the Softmax-Opti campaign.

Measured 0.107 to 0.109 ms for 1024x4096 f32 on A770; the oneDNN softmax baseline was 0.217 ms. One work-group per row, dynamic work-group size by column count, `vec<float,16>` SLM chunks, one barrier after the SLM load, and `sycl::reduce_over_group(it.get_group(), ...)` for both max and sum.

```cpp
constexpr std::size_t WG = 128;          // used for cols > 1024
constexpr std::size_t VEC = 16;
constexpr std::size_t TILE_CHUNKS = 128; // 2048 floats, 32 KB SLM
using float16 = sycl::vec<float, VEC>;

template <bool Aligned, int Threads>
void softmax_slm_impl(sycl::queue& q, const float* x, float* y, int rows,
                      int cols) {
    const std::size_t chunks =
        Aligned ? static_cast<std::size_t>(cols) / VEC
                : (static_cast<std::size_t>(cols) + VEC - 1) / VEC;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float16, 1> tile(sycl::range<1>(chunks), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(rows * Threads),
                              sycl::range<1>(Threads)),
            [=](sycl::nd_item<1> it) {
                const std::size_t row = it.get_group(0);
                const std::size_t tid = it.get_local_linear_id();
                const float* row_in = x + row * static_cast<std::size_t>(cols);
                float* row_out = y + row * static_cast<std::size_t>(cols);

                for (std::size_t c = tid; c < chunks; c += Threads) {
                    if constexpr (Aligned) {
                        tile[c] = *reinterpret_cast<const float16*>(
                            row_in + c * VEC);
                    } else {
                        const std::size_t off = c * VEC;
                        const std::size_t remain =
                            static_cast<std::size_t>(cols) - off;
                        float16 v(0.0f);
                        for (std::size_t e = 0; e < VEC && e < remain; ++e) {
                            v[e] = row_in[off + e];
                        }
                        tile[c] = v;
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                float partial_max = -std::numeric_limits<float>::infinity();
                for (std::size_t c = tid; c < chunks; c += Threads) {
                    const std::size_t off = c * VEC;
                    const std::size_t limit =
                        Aligned ? VEC
                                : std::min<std::size_t>(
                                      VEC,
                                      static_cast<std::size_t>(cols) - off);
                    for (std::size_t e = 0; e < limit; ++e) {
                        partial_max = sycl::max(partial_max, tile[c][e]);
                    }
                }
                const float row_max = sycl::reduce_over_group(
                    it.get_group(), partial_max, sycl::maximum<float>());

                float partial_sum = 0.0f;
                if constexpr (Aligned) {
                    float16 sum_vec(0.0f);
                    for (std::size_t c = tid; c < chunks; c += Threads) {
                        sum_vec += sycl::exp(tile[c] - row_max);
                    }
                    for (std::size_t e = 0; e < VEC; ++e) {
                        partial_sum += sum_vec[e];
                    }
                } else {
                    for (std::size_t c = tid; c < chunks; c += Threads) {
                        const std::size_t off = c * VEC;
                        const std::size_t limit = std::min<std::size_t>(
                            VEC, static_cast<std::size_t>(cols) - off);
                        for (std::size_t e = 0; e < limit; ++e) {
                            partial_sum += sycl::exp(tile[c][e] - row_max);
                        }
                    }
                }
                const float row_sum = sycl::reduce_over_group(
                    it.get_group(), partial_sum, sycl::plus<float>());
                const float inv_sum = 1.0f / row_sum;

                if constexpr (Aligned) {
                    for (std::size_t c = tid; c < chunks; c += Threads) {
                        const float16 out =
                            sycl::exp(tile[c] - row_max) * inv_sum;
                        *reinterpret_cast<float16*>(row_out + c * VEC) = out;
                    }
                } else {
                    for (std::size_t c = tid; c < chunks; c += Threads) {
                        const std::size_t off = c * VEC;
                        if (off < static_cast<std::size_t>(cols)) {
                            const std::size_t limit = std::min<std::size_t>(
                                VEC, static_cast<std::size_t>(cols) - off);
                            for (std::size_t e = 0; e < limit; ++e) {
                                row_out[off + e] =
                                    sycl::exp(tile[c][e] - row_max) * inv_sum;
                            }
                        }
                    }
                }
            });
    });
}

void softmax_slm(sycl::queue& q, const float* x, float* y, int rows,
                 int cols) {
    const bool aligned = cols % static_cast<int>(VEC) == 0;
    const int threads = cols <= 256 ? 16 : (cols <= 1024 ? 32 : 128);
    // Dispatch the template for aligned/generic x 16/32/128. For cols > 8192
    // use the tiled variant below instead of the one-shot kernel.
    if (aligned) {
        if (threads == 16) {
            softmax_slm_impl<true, 16>(q, x, y, rows, cols);
        } else if (threads == 32) {
            softmax_slm_impl<true, 32>(q, x, y, rows, cols);
        } else {
            softmax_slm_impl<true, 128>(q, x, y, rows, cols);
        }
    } else {
        if (threads == 16) {
            softmax_slm_impl<false, 16>(q, x, y, rows, cols);
        } else if (threads == 32) {
            softmax_slm_impl<false, 32>(q, x, y, rows, cols);
        } else {
            softmax_slm_impl<false, 128>(q, x, y, rows, cols);
        }
    }
}
```

Requires 64 B aligned USM pointers for the `Aligned` path (`sycl::aligned_alloc_shared<float>(64, ...)`). The generic `Aligned=false` path handles odd column counts with scalar loads; it still beat the naive row-per-item kernel by a wide margin.

For rows larger than one SLM tile, do not stage the max pass. Reduce it directly from global first, then stage 32 KB tiles only for the sum and normalize passes:

```cpp
// cols > 8192: same nd_range and SLM tile as above, but the max pass is:
float partial_max = -std::numeric_limits<float>::infinity();
for (std::size_t c = tid; c < chunks; c += Threads) {
    const std::size_t off = c * VEC;
    const std::size_t limit = std::min<std::size_t>(
        VEC, static_cast<std::size_t>(cols) - off);
    for (std::size_t e = 0; e < limit; ++e) {
        partial_max = sycl::max(partial_max, row_in[off + e]);
    }
}
const float row_max = sycl::reduce_over_group(
    it.get_group(), partial_max, sycl::maximum<float>());
// Then for each tile: load into `tile`, barrier, accumulate exp(x - row_max),
// reduce the sum, and in a second tile pass normalize from SLM with a barrier
// after each tile store. This reduced 1024x16384 from 0.66 ms to 0.59 ms.
```

## Correctness and Timing Harness

Source: `gemm_esimd_v10.cpp`.

Benchmark and verify the same way for every variant. With `beta != 0`, the timed loop accumulates C, so restore C0 and run one extra kernel for the correctness check.

```cpp
constexpr size_t warmup_iters = 30;
constexpr size_t run_iters = 200;
for (size_t i = 0; i < warmup_iters; i++)
    gemm_esimd_v10(Ap, Bp, C, M, N, K, tc.alpha, tc.beta, q);
q.wait();

auto start = std::chrono::high_resolution_clock::now();
for (size_t i = 0; i < run_iters; i++)
    gemm_esimd_v10(Ap, Bp, C, M, N, K, tc.alpha, tc.beta, q);
q.wait();
auto end = std::chrono::high_resolution_clock::now();
double total =
    std::chrono::duration<double, std::milli>(end - start).count();

for (int i = 0; i < M * N; i++)
    C[i] = C0[i];
gemm_esimd_v10(Ap, Bp, C, M, N, K, tc.alpha, tc.beta, q);
q.wait();

long long errs = 0;
for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
        float ref = 0.0f;
        for (int k = 0; k < K; k++)
            ref += static_cast<float>(A[(size_t)i * K + k]) *
                   static_cast<float>(Braw[(size_t)k * N + j]);
        ref = tc.alpha * ref + tc.beta * C0[(size_t)i * N + j];
        if (C[(size_t)i * N + j] != ref)
            errs++;
    }
```

Report `errors=0/<total>` and the per-run average, and repeat at least 3 times to confirm stability before accepting a change.
