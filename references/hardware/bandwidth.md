# A770 Bandwidth and Roofline Reference

Measured on Intel Arc A770 (DG2), oneAPI 2026.1, driver `32.0.101.8724`,
Windows 11. All numbers are A770/Windows measurements; re-measure before
transferring them to another GPU, driver, or compiler.

> Evidence: `[MEASURED]` f32 standard-SYCL kernels over 64 B aligned USM,
> 4096 workers, message widths 32-256 B, strides 1/16/256 float elements.

## Contents

- Environment and method
- Measured bandwidth patterns
- Message-width cost
- Stride cost
- Roofline classification
- Rules
- Unstable negative results
- Reproduction

## Environment and Method

- f32 SYCL kernels over 64 B aligned USM: global read, global copy, global
  reduce, and SLM read/write.
- Each config is CPU-reference verified, warmed up, timed in a loop, and
  recorded three times; median GB/s is reported. Configs with >5% run-to-run
  CV are preserved as unstable.
- Load width is the logical message width expressed with `sycl::vec`
  loads/stores (32/64/128/256 B), not a single LSC `block_load` instruction.
  Treat these as practical standard-SYCL ceilings, not hardware instruction
  peaks.
- Stride is in float elements: 1/16/256 elements = 4/64/1024 B. The stride-256
  footprint spans 64 MiB, above the 16 MiB L2, so it represents the DRAM-like
  regime.

## Measured Bandwidth Patterns

| Path | Configuration | Median GB/s | Stable |
|---|---|---:|---|
| Strided DRAM-like global read | read, 256 B, stride 256 el | 277.0 | True |
| Strided DRAM-like global copy | copy, 256 B, stride 256 el | 328.8 | True |
| Strided DRAM-like global reduce | reduce, 256 B, stride 256 el | 273.4 | True |
| L2-resident global read | read, 256 B, stride 1 el | 854.6 | True |
| L2-resident global copy | copy, 256 B, stride 1 el | 1300.9 | True |
| SLM read | 256 B messages | 4115.8 | True |
| SLM write | 256 B messages | 5413.4 | False |

Copy GB/s counts read plus write bytes (the benchmark doubles the source
payload), so `328.8 GB/s` is already the aggregate bidirectional traffic; do
not double it again.

`277 GB/s` is a strided-pattern effective bandwidth (`B_DRAM_strided`), not a
universal hardware DRAM ceiling. A dedicated contiguous streaming campaign on
the current Windows/oneAPI stack (driver `32.0.101.8860`) measured
`B_DRAM_contiguous` at `291.1 GB/s` read, `340.0 GB/s` copy, and `290.6 GB/s`
reduce (256 MiB source, 256 B messages, 3 repeats x 20 samples, CV below 5%).
Use the pattern names
`B_DRAM_contiguous`, `B_DRAM_strided`, `B_L2_reuse`, and `B_SLM` when choosing
a roofline model.

## Message Width Cost (stride 256 el = 1024 B)

| Load width B | read GB/s | copy GB/s | reduce GB/s |
|---:|---:|---:|---:|
| 32 | 104.0 | 67.9 | 100.1 |
| 64 | 206.1 | 247.7 | 203.3 |
| 128 | 225.7 | 270.9 | 226.9 |
| 256 | 277.0 | 328.8 | 273.4 |

## Stride Cost (256 B load width)

| Stride el | Stride bytes | read GB/s | copy GB/s | reduce GB/s |
|---:|---:|---:|---:|---:|
| 1 | 4 | 854.6 | 1300.9 | 841.4 |
| 16 | 64 | 536.4 | 764.1 | 533.5 |
| 256 | 1024 | 277.0 | 328.8 | 273.4 |

## Roofline Classification

This is an empirical campaign roofline, not a hardware/theoretical roofline.
Empirical ridge constants from the contiguous campaign:

- DRAM ridge: `100.6 TFLOPS / 291.1 GB/s = 345.7 FLOP/B`.
- L2 ridge: `100.6 TFLOPS / 854.6 GB/s = 117.7 FLOP/B`.

The older `26.2 TFLOPS / 277.0 GB/s = 94.7 FLOP/B` ridge is retained as the
original campaign number but is not the current roofline baseline. Use the
DRAM ridge for streaming DRAM-bound f32 kernels and the L2 ridge for
L2-resident tiles. Full tables:
[dram-contiguous-roofline.md](dram-contiguous-roofline.md).

| Operation | FLOP/B | Achieved | Class |
|---|---:|---|---|
| bf16 GEMM 1024x1536x512 | 180.7 | 26.2 TFLOPS | memory-bound by the DRAM ridge; see note |
| f32 GEMV 4096x4096 | 0.5 | 312 GB/s | memory-bound |
| f32 RMSNorm 1024x4096 | 0.5 | 342 GB/s | memory-bound |
| f32 Softmax 1024x4096 | 0.6 | 314 GB/s | memory-bound |

Class is judged against the current empirical ridges, not the superseded
`94.7 FLOP/B` ridge. The bf16 GEMM row (`180.7 FLOP/B`) is below the DRAM
ridge (`345.7 FLOP/B`) and above the L2 ridge (`117.7 FLOP/B`), so it is
memory-bound by the DRAM pattern; but its achieved bandwidth is about
`145 GB/s` (`26.2 TFLOPS / 180.7`), roughly half of `B_DRAM_contiguous`
(`291.1 GB/s`), so the practical limit is instruction/issue rate rather than
the bandwidth ceiling. The historical "compute-bound" label came from the old
ridge and does not transfer to the current constants.

## Rules

1. Run a bandwidth microbenchmark before restructuring a kernel: if the
   arithmetic intensity is far below the matching empirical ridge and
   achieved bandwidth is near the access-pattern baseline (`B_DRAM_strided`
   for strided, `B_DRAM_contiguous` for fully coalesced, `B_L2_reuse` for
   L2-resident), stop chasing bandwidth and look at instruction volume,
   redundant reads, and launch overhead.
2. Wider messages win: at the 1024 B stride, 256 B loads are about `2.7x`
   faster than 32 B loads for read and about `4.8x` faster for copy. Stage
   global-to-SLM traffic with 256 B messages.
3. Stride is an explicit cost: for 256 B reads, stride 256 el drops bandwidth
   to about `32%` of the stride-1 value. Quantify transposed, padded, or
   interleaved layouts with the microbenchmark before writing the kernel.
4. SLM is the fastest on-chip path measured: keep repeatedly-read rows/tiles
   in SLM, but verify with wall time because barriers and occupancy can erase
   the SLM advantage.

## Negative Results (Unstable Configs)

The following configs had run-to-run CV >5% and must not be used as
performance conclusions:

- read 32/64 B at stride 1
- copy 64 B at stride 1
- reduce 64 B at stride 1
- SLM write at 256 B

The benchmark preserves them in `results/failures.csv` instead of deleting
them.

## Reproduction

Build and run the microbenchmark with the standard oneAPI Windows commands
in [api-usage.md](../api/api-usage.md); the benchmark reports GB/s and keeps
unstable configs in its own failure log.

The strided `277 GB/s` number must not be promoted to a universal hardware
ceiling. The current campaign preserved both the historical stable value and
its own unstable strided median (`287.6 GB/s`, CV above 5%) in the failure
log; neither replaces the pattern-specific `B_DRAM_contiguous` value.

Copy-ready microbenchmark core: [bandwidth read/copy/reduce cores](../api/code-snippets.md#bandwidth-read-copy-reduce-cores).
