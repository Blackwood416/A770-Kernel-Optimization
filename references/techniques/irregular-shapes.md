# Irregular Shapes, Sparse GEMM, and Direct Convolution on A770

> Scope: Intel Arc A770 (DG2), Windows 11, oneAPI 2026.1–2026.2, `f32`, `64 B`-aligned USM. Rules are `[HEURISTIC]` outside the measured shapes below; re-measure before generalizing.

## 1. Alignment Gates and Odd-Shape Fallbacks `[MEASURED]`

### 1.1 Softmax (`rows=64, cols in 256..4096`)

| `cols` | Fast `vec16` SLM (`us`) | Scalar-Chunk SLM (`us`) | Naive Row (`us`) | `oneDNN` (`us`) |
|---:|---:|---:|---:|---:|
| `256` | `17.8` | **`14.7`** | `136.0` | `28.3` |
| `512` | **`10.5`** | `15.2` | `266.3` | `14.1` |
| `1025` (odd) | N/A | **`18.0`** | `1099.0` | `18.4` |
| `2011` (odd) | N/A | **`23.3`** | `2205.6` | `21.6` |
| `4096` | **`17.9`** | `37.4` | `2201.1` | `31.8` |

- **Gate**: Enable the `vec<float,16>` fast path only when `cols % 16 == 0 && cols >= 512` on `64 B`-aligned USM. For `cols <= 256` or odd column counts, use the **scalar-chunk SLM fallback** (matches `oneDNN` and beats naive by `60x–95x` vs naive baseline).
- **Dynamic WG**: `16` threads for `cols <= 256`, `32` for `cols <= 1024`, `128` otherwise.

### 1.2 GEMV-N1 (`M=64, f32`) -- The Non-16-Aligned Cliff

- **Fast-path gate**: Requires `M % 32 == 0 && N % 16 == 0` with `vec<float,16>` and dynamic `sg.get_local_range()[0]` trip counts (`10.9–51.4 us` for `N = 512..4096`).
- **Severe scalar fallback penalty**: Non-16-aligned columns collapse from `16.9 us` (`N=1024`) to `586.2 us` (`N=1025`, **`34.7x` slower** vs `N=1024` baseline) and from `28.2 us` (`N=2048`) to `1159.2 us` (`N=2011`, **`41.1x` slower** vs `N=2048` baseline). Whenever odd `N` is hot, write a **vectorized body + scalar tail** (or padded buffer) instead of dropping the whole row to scalar.
- **Gather/scatter caution**: Small arrays (`1K–2K` elements) sit at `17–20 us` (launch-bound) with no measurable random-vs-sequential gap; benchmark irregular memory with `>= 100K` elements.

## 2. Sparse GEMM (`M=K=512, N=8, f32`) `[MEASURED]` `[HEURISTIC]`

| Sparsity | CSR Direct (`us`) | Dense Filter (`us`) | BSR `B4` (`us`) | BSR `B8` (`us`) | BSR `B16` (`us`) |
|---:|---:|---:|---:|---:|---:|
| `50%` | `298.0` | `246.0` | **`168.3`** | `260.4` | `1359.1` |
| `90%` | **`65.1`** | `246.6` | `143.9` | `256.0` | `1359.0` |
| `99%` | **`30.9`** | `153.4` | `38.0` | `167.0` | `1402.8` |

- **`sparsity >= 90%`**: Use **CSR direct** traversal.
- **`sparsity ~50%`**: Use **BSR `B4`** with `vec<float,8>` output accumulation.
- **Occupancy trap**: Large BSR blocks (`B16` at `M=512` leaves only `32` row-block work-items) starve the GPU (`8x` slower vs `B4`); split large row blocks across work-groups. Never extrapolate `N=8` rules to large `N` without re-measuring.

## 3. Direct Convolution vs `im2col + GEMM` (`N=4, IC=32, 64x64, OC=64, 3x3, f32`) `[MEASURED]`

| Variant | Wall Avg (`ms`) | Notes |
|---|---:|---|
| `onednn_conv_f32` (`convolution_direct`) | **`0.0935`** | Strongest library baseline |
| `direct_nhwc_cacheblock_oc64` | **`0.1605`** | Best custom SYCL (`[KH][KW][IC][OC]` weights, full `OC=64` cache block) |
| `direct_nchw_naive` | `0.2234` | Baseline NCHW |
| `direct_nhwc_cacheblock_oc16` | `0.2976` | Smaller OC block cuts input reuse (`1.85x` slower vs `oc64`) |
| `im2col_gemm_tile32` | `0.3330` | Beats `tile64` (`2.33 ms`, SLM/occupancy loss), still trails OC64 direct |
| `direct_nhwc_naive` | `1.3352` | **`~6x` slower vs NCHW direct** without OC blocking & HWIO weights |

**Rules & correctness traps**:
1. Default to `oneDNN` for `f32` conv; if writing custom NHWC conv, pair **`OC` cache blocking** with **`[KH][KW][IC][OC]`** weight layout.
2. In `im2col + GEMM`, the GEMM width is `OC` (not batch `N`) and `K = KH * KW * IC`; output channel stride is `OC` for both NCHW and NHWC outputs `[CORRECTNESS]`.
