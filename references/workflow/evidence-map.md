# Evidence map: operator / technique -> doc -> probe -> claim

Purpose: this skill states a lot of measured facts, and the first question a reader asks is *where does
that number come from*. Every row below points at the committed artifact in the **Arc-ISA-Research** repo
(paths are repo-relative; the repo is authoritative, this page is the map).

Two kinds of entry:

- **operator** rows: the kernel, its pinned numbers, the doc that carries the full table and the probe
  directory that holds the raw CSV/record files;
- **technique** rows: an ISA or scheduling fact, the probe that established it and the skill page it
  lives on.

`docs/69-probe-index.md` in the repo is the machine-generated inverse of this page (probe -> claims ->
docs); if the two disagree, the repo is right.

## 1. Operator families

| Family | Pinned number (worst run) | Opponent | Doc | Probe |
|---|---|---|---|---|
| f32 GEMV | 216.56 us, 0/4096 [MEASURED] | oneDNN 449.97 us (2.08x ahead) | `docs/51-gemv.md` | `probes/077-gemv/` |
| f32 RMSNorm | 90.53 us [MEASURED] | oneDNN 112.90 us (1.25x ahead) | `docs/52-rmsnorm.md` | `probes/078-rmsnorm/` |
| f32 Softmax | 100.52 us [MEASURED] | oneDNN 205.44 us (2.04x ahead) | `docs/54-softmax.md` | `probes/080-softmax/` |
| bf16 GEMM | 95.18 us [MEASURED] | oneDNN 72.81 us (1.31x behind) | `docs/61-gemm-latency.md` | `probes/086-gemm-latency/` |
| int8 GEMM | 92.49 us [MEASURED] | oneDNN 68.50 us (1.35x behind) | `docs/64-int8-ablation.md` | `probes/089-int8-ablation/` |
| bf16 SDPA decode | 557.26 us [MEASURED] | no oneDNN primitive (roofline only) | `docs/60-sdpa-fix.md` | `probes/085-sdpa-fix/` |

Each operator doc is generated from the probe's CSV plus two `bench/records/<stamp>/record.json` files, so
the number can be re-derived without the GPU: `python probes/<NNN>-<name>/make_doc.py`.

## 2. Gap pricing (why a kernel is where it is)

| Question | Answer | Doc | Probe |
|---|---|---|---|
| Why is int8 1.35x behind and what can be removed? | Only the B VNNI pack is removable: 12.43 us = 51.7% of the 24.03 us gap; barrier/global/feed/dpas are one dependency chain and cannot be summed [MEASURED] | `docs/64-int8-ablation.md` | `probes/089-int8-ablation/` |
| Does paying that pack up front help? | Yes, but only at 32-column (full-sector) pass width: 15.09 -> 11.10 us, end to end 89.94 us (-1.95) [MEASURED] | `docs/67-int8-prepack.md` | `probes/092-int8-prepack/` |
| Why does bf16 GEMM miss 90 us? | Per-K-tile barrier 22.42 us = 4.33x the 5.18 us gap; one 128-thread group per DSS, XVE stalled 92.4% [MEASURED] | `docs/61-gemm-latency.md` | `probes/086-gemm-latency/` |
| Why is SDPA at 59.7% of the roof and not closer? | The cross-pass reduction barrier: 35.31 us when deleted, and deleting it is wrong (2,864 bad values) [MEASURED] | `docs/60-sdpa-fix.md` | `probes/085-sdpa-fix/` |
| Is any unroll/prefetch setting faster? | No: nine grid points all 0 errors, best is still the adopted one [MEASURED] | `docs/65-sdpa-grid.md` | `probes/090-sdpa-grid/` |
| Can the barrier be hidden by hoisting the next load? | No: 2.82-3.17x slower, the extra buffers spill (0/0 -> 11/5, 55/30) [MEASURED] | `docs/66-sdpa-pipeline.md` | `probes/091-sdpa-pipeline/` |

## 3. Roofline denominators

| Pattern | Number | Where |
|---|---|---|
| DRAM read, float4 grid-stride, 8 in flight | 403.1 GB/s [MEASURED] | `probes/087-roof/work/roof.csv` |
| DRAM read, 4 in flight | 417.2 GB/s (faster than 8) [MEASURED] | same CSV |
| DRAM write | 464.6 GB/s [MEASURED] | same CSV |
| L2 read, 8 MiB resident | 3429.8 GB/s [MEASURED] | same CSV |
| DRAM read, 256 B messages (the old 291 GB/s) | 291.1 GB/s, a *different* pattern [MEASURED] | `references/hardware/dram-contiguous-roofline.md` |

Quote the pattern, not just the number: using 291 under a float4 kernel understates utilisation by ~38%.

## 4. ISA facts and where they were established

| Fact | Probe | Skill page |
|---|---|---|
| bindless ExBSO vs stateful BTI addressing, and what the runtime emits | `probes/056-bindless-vs-stateless/` | `isa/addressing.md` |
| `dpas` accepted shapes/encodings on XeHPG, operand rules | `probes/022-dpas/`, `047-dpas-forms/` | `isa/dpas.md` |
| LSC message shapes, cache policy fields, oword/block reads | `probes/035-lsc/`, `055-oword-block-read/` | `isa/lsc-messages.md` |
| SWSB / scoreboard tokens and their cost | `probes/020-swsb/`, `051-swsb-token/` | `isa/swsb.md` |
| region addressing, GRF bank conflicts, indirect operands | `probes/036-cache-policy/`, `049-grf-bank/`, `064-indirect/` | `isa/encoding.md` |
| dual-issue behaviour, loop vehicles, divergence stacks | `probes/063-dual-issue/`, `067-loop-vehicle/`, `065-divergence-stack/` | `codegen/asm-control.md` |
| AOT vs JIT codegen, large-GRF, flag effects | `probes/071-aot-vs-jit/`, `058-large-grf/` | `codegen/knobs.md` |
| `reduce_over_group` narrowed per SIMD pass (wrong values) | `probes/085-sdpa-fix/` | `techniques/reductions-scan.md` |
| instrument availability (what can and cannot be measured here) | `probes/084-gtpin-types/`, `076-eu-metrics/` | `workflow/instruments.md` |

## 5. How to re-check a row

Repo-relative, no GPU needed for the doc side:

```bat
python probes\make_probe_index.py          REM rebuild docs/69 (probe -> claims -> docs)
python probes\086-gemm-latency\make_doc.py REM any operator doc, from its committed CSV
python scripts\claim_ledger.py             REM validate every claim's evidence path
```

With the GPU, `bench\run_baselines.ps1 -Only <tag> -Batch 50` regenerates the `bench/records/<stamp>/`
file a pinned number is quoted from. A number without a stamp or a CSV is not a number this skill should
be trusted on — see `workflow/methodology.md`.
