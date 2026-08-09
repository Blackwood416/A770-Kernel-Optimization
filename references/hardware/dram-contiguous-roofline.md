# A770 Contiguous DRAM and Validated Roofline

> `[MEASURED]` Intel Arc A770 / oneAPI 2026.1.0 / graphics driver
> `32.0.101.8860` / Level Zero `1.15.38308` / Windows 11. f32 standard-SYCL
> kernels over 64 B aligned USM. Every executable emitted the correctness JSON
> contract; all reported `errors: 0` and no contract mismatch occurred.

## Protocol

- Contiguous source is 256 MiB, far above the 16 MiB L2.
- Warmup 32 launches, 3 campaign repeats x 20 samples, batch of 32
  back-to-back launches; SLM uses batch 1 because launch gaps dominate.
- Copy GB/s counts read plus write bytes.
- Sample-level CV > 5% is preserved as unstable and not promoted to a
  `[MEASURED]` ceiling.

## Named Bandwidth Patterns

| Name | Definition | Value |
|---|---|---:|
| `B_DRAM_contiguous` read | 256 MiB contiguous read, 256 B messages | 291.11 GB/s |
| `B_DRAM_contiguous` copy | 256 MiB read + write, 256 B messages | 340.03 GB/s |
| `B_DRAM_contiguous` reduce | 256 MiB read + WG reduce, 256 B messages | 290.58 GB/s |
| `B_DRAM_strided` | historical stable, stride 256 el, 256 B messages | 277.0 GB/s |
| `B_L2_reuse` | historical stable, stride 1 el, 256 B messages | 854.6 GB/s |
| `B_SLM` | historical stable read, 256 B messages | 4115.8 GB/s |

The 1 GiB contiguous trial measured about `228 GB/s` and was retained as a
negative result; the stable 256 MiB result is used because it is reproducible
within the requested range.

## Theoretical, Validated, and Empirical Layers

| Layer | Compute | Memory | Ridge |
|---|---:|---:|---:|
| Hardware/theoretical | 137.63 TFLOPS bf16 XMX peak | 560 GB/s spec | 245.8 FLOP/B |
| Validated vs theoretical memory | 100.627 TFLOPS bf16 DPAS peak | 560 GB/s spec | 179.7 FLOP/B |
| Empirical DRAM | 100.627 TFLOPS | 291.11 GB/s | 345.7 FLOP/B |
| Empirical L2 | 100.627 TFLOPS | 854.6 GB/s | 117.7 FLOP/B |

The measured bf16 DPAS peak uses a compact logical GEMM microkernel
(`C[32768 x 128] = A[32768 x 32768] * B[32768 x 128]`, DPAS8, 16 accumulator
chains, 2048 K iterations), CPU-reference verified and stable at CV 3.23%.
Theoretical XMX peak efficiency is 73.1%.

## Roofline Classification

Use `B_DRAM_contiguous` for fully coalesced streaming kernels and
`B_L2_reuse` for L2-resident tiles. Streaming GEMV/RMSNorm/Softmax can
partially reuse L2, so classify with both patterns before choosing a ceiling.
The empirical DRAM ridge is `345.7 FLOP/B`; the empirical L2 ridge is
`117.7 FLOP/B`.

## Negative Results

- The 1 GiB contiguous read trial measured about `228 GB/s` and was not
  promoted.
- Contiguous 32/64/128 B read and 64 B copy/reduce configs had CV > 5% and
  were retained as unstable.
- Current-host strided median (`287.6 GB/s`) had CV above the promotion
  threshold; the historical `277 GB/s` remains the retained stable value.

Full measurement tables and curves are maintained in the campaign; this page
keeps the named constants used for roofline decisions.
