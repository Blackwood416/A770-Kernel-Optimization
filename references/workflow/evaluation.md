# Skill Behavior Evaluation

> `[MEASURED]` evaluation-session observations, not GPU kernel numbers.
> Validity domain: task set `task-01..task-05`, skill
> `$optimize-a770-kernels`, subject = five independent `codex exec`
> processes (CLI 0.146.0, model deepseek-v4-flash, prompts v2 inline-only),
> run date 2026-08-09; confidence = single session, one model/CLI
> combination.

## Purpose

The eval harness checks whether an agent that has read this skill follows its
Evidence Levels, validity domains, oneDNN Baseline Contract, and Benchmark
Protocol. Tasks ask for a Markdown evidence report only; they do not execute
kernels.

## Task Set

| Task | Kernel target | Reference under test |
|---|---|---|
| task-01 | RMSNorm f32 1x4096 | RMSNorm shape sweep, oneDNN RMSNorm baseline |
| task-02 | GEMV f32 64x1025 (odd N) | Irregular-shapes GEMV fast/fallback rules |
| task-03 | 99% sparse GEMM f32 512x512x8 | Irregular-shapes CSR/BSR heuristic |
| task-04 | Decode attention Q=1 GQA 2048x128 | Decode attention wall/device champions |
| task-05 | Compiler flags, simple non-XMX bf16 GEMM | Codegen matrix, large-GRF qualification |

## Scored Behavior

Weights used by the scorer:

- same-domain `[MEASURED]`: 1.0
- `[HEURISTIC]` as starting point: 1.0
- oneDNN verbose + implementation string: 1.0
- device/wall/pipeline time reported separately: 1.0
- negative results preserved: 1.0
- `[DISPATCH]` validity domain: 0.75
- benchmark protocol elements: 0.75
- no machine-local absolute paths: 0.5

Pass threshold: 0.75.

## Current Result

Independent CLI run on 2026-08-09: 5/5 tasks PASS, mean weighted score 1.0.
Every subject kept `[MEASURED]` claims in-domain or re-labeled them, marked
`[HEURISTIC]` values as starting points, recorded oneDNN verbose +
implementation strings where a baseline exists, reported all three time
fields, preserved negative results, and emitted path-free Markdown.

## Audit and Applied Fixes

The text audit found the riskiest statements in the compressed decision
branches and two reference tables. All findings below are applied to this
skill revision.

| # | Severity | Location | Fix |
|---|---|---|---|
| 1 | high | `SKILL.md` headline numbers | Added shape/dtype/time basis to every headline value |
| 2 | high | `SKILL.md` GEMV branch | Stated device-time median basis and wall/device champion differences |
| 3 | high | `SKILL.md` irregular/sparse branch | Split sparse-GEMM and GEMV/softmax fallback domains |
| 4 | high | `SKILL.md` attention branch | Added wall/device champion split and host-overhead range |
| 5 | high | `irregular-shapes.md` GEMV table | Stated wall-time per-run measurement basis |
| 6 | medium | `irregular-shapes.md` fallback ratios | Added measured M and exact N pairs |
| 7 | high | `api-usage.md` large-GRF comment | Qualified the negative verdict by kernel class |
| 8 | medium | `automation.md` / `techniques.md` GEMV records | Added campaign and harness provenance to same-shape values |
| 9 | medium | `SKILL.md` codegen branch | Split `[MEASURED]` auto-vectorization from `[HEURISTIC]` unroll advice |
| 10 | medium | `attention-decode.md` decision rules | Added explicit GQA wall/device champion example |

## Second Review Fixes (2026-08-10)

| Area | Applied fix |
|---|---|
| Accuracy-aware baseline | Split `matched` / `fastest` / `unknown`; added `reference_tolerance`, `baseline_correctness_status`, `comparable_for_speedup`; follow-up probe found a matched f16-src oneDNN config for weight-only M=64 |
| RMSNorm dispatch noise | Added 3% tie handling and simplified production regions instead of per-cell argmin dispatch |
| Evidence taxonomy | Added `[CORRECTNESS]` and `[TOOLCHAIN]`; reclassified weight-only packed-offset, M=1, and mixed-precision DPAS claims |
| Self-contained harness | Bundled core scripts and example kernels under `scripts/` / `examples/`; added script-existence gate |

## Reproduction

```powershell
python eval\scoring.py --date 2026-08-09 --subject "independent run"
```

The scorer reads task definitions and transcripts, then writes
`results/score.json` plus `results/results.md`. Re-run after any skill text
change or model/CLI change.

## Known Limitations

- Regex scoring is heuristic; borderline transcripts need a human review of
  the issues in `score.json`.
- The scorer checks the final report only. Intermediate tool behavior is out
  of scope for this no-execution task set.
- Results cover one model/CLI/machine combination; re-run for other subjects
  before generalizing.
