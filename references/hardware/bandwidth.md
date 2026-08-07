# A770 Bandwidth and Roofline Reference

Measured on Intel Arc A770 (DG2), oneAPI 2026.1, driver `32.0.101.8724`,
Windows 11. All numbers are single-machine conclusions; re-measure before
transferring them to another GPU, driver, or compiler.

Source project: `E:\RiderProjects\BandWidth-Opti` (`references\bandwidth.md`).

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

## Bandwidth Ceilings

| Path | Configuration | Median GB/s | Stable |
|---|---|---:|---|
| DRAM-like global read | read, 256 B, stride 256 el | 277.0 | True |
| DRAM-like global copy | copy, 256 B, stride 256 el | 328.8 | True |
| DRAM-like global reduce | reduce, 256 B, stride 256 el | 273.4 | True |
| L2-resident global read | read, 256 B, stride 1 el | 854.6 | True |
| L2-resident global copy | copy, 256 B, stride 1 el | 1300.9 | True |
| SLM read | 256 B messages | 4115.8 | True |
| SLM write | 256 B messages | 5413.4 | False |

Copy GB/s counts read plus write bytes, so `328.8 GB/s` is about `657.6 GB/s`
of bidirectional DRAM traffic.

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

Ridge point: `26.2 TFLOPS / 277.0 GB/s = 94.7 FLOP/B`.

| Operation | FLOP/B | Achieved | Class |
|---|---:|---|---|
| bf16 GEMM 1024x1536x512 | 180.7 | 26.2 TFLOPS | compute-bound |
| f32 GEMV 4096x4096 | 0.5 | 312 GB/s | memory-bound |
| f32 RMSNorm 1024x4096 | 0.5 | 342 GB/s | memory-bound |
| f32 Softmax 1024x4096 | 0.6 | 314 GB/s | memory-bound |

## Rules

1. Run a bandwidth microbenchmark before restructuring a kernel: if the
   arithmetic intensity is far below the ridge and achieved bandwidth is near
   the DRAM-like ceiling, stop chasing bandwidth and look at instruction
   volume, redundant reads, and launch overhead.
2. Wider messages win: at the 1024 B stride, 256 B loads are about `2.7x`
   faster than 32 B loads for read and about `4.8x` faster for copy. Stage
   global-to-SLM traffic with 256 B messages.
3. Stride is an explicit cost: for 256 B reads, stride 256 el drops bandwidth
   to about `32%` of the stride-1 value. Quantify transposed, padded, or
   interleaved layouts with the microbenchmark before writing the kernel.
4. SLM is the fastest on-chip path measured: keep repeatedly-read rows/tiles
   in SLM, but verify with wall time because barriers and occupancy can erase
   the SLM advantage.

## Unstable Configs

The following configs had run-to-run CV >5% and must not be used as
performance conclusions:

- read 32/64 B at stride 1
- copy 64 B at stride 1
- reduce 64 B at stride 1
- SLM write at 256 B

The benchmark preserves them in `results/failures.csv` instead of deleting
them.

## Reproduction

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1
cmd /c '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul && build\bandwidth.exe'
python scripts/analyze.py
```

Copy-ready microbenchmark core: [bandwidth read/copy/reduce cores](../api/code-snippets.md#bandwidth-read-copy-reduce-cores).
