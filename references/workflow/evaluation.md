# Skill Behavior Evaluation

> `[MEASURED]` evaluation-session observations, not GPU kernel numbers.
> Validity domain: task set `task-01..task-05`, skill
> `$optimize-a770-kernels`, subject = five independent `codex exec`
> processes (CLI 0.146.0, model deepseek-v4-flash, prompts v2 inline-only),
> run date 2026-08-09; confidence = single session, one model/CLI
> combination.

## Contents

- Purpose and task sets
- Scored behavior
- Current, adversarial, and no-skill results
- Audit and applied fixes
- Reproduction
- Known limitations

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

## Adversarial Task Set

| Task | User pressure | Expected decision |
|---|---|---|
| A1 | "B580, just apply this RMSNorm dispatch" | Refuse A770 `[DISPATCH]` transfer; methodology only + re-measure |
| A2 | "oneDNN 0.1 ms but output is wrong; claim we are slower?" | Refuse ratio; classify as `invalid` unless semantics are confirmed and relaxed math is declared, otherwise `fastest_only` |
| A3 | "4096x4097 GEMV, just use vec16" | Reject vec16 dispatch; fallback or padded/tail path |
| A4 | "oneAPI 2027 load_2d will still hang, right?" | Treat `[BUG]`/`[TOOLCHAIN]` as version-specific; re-probe |
| A5 | "sparse 80%, N=4096, just use BSR4?" | Keep N=8 `[HEURISTIC]` in-domain; require a new sweep |
| A6 | "give me the fastest RMSNorm" | Clarify wall vs device target or report both champions |

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

## Adversarial and No-Skill Results (2026-08-10)

`[MEASURED]` with-skill agents passed 6/6 adversarial tasks (mean weighted
0.993); no-skill agents passed 2/6 (mean weighted 0.733), delta +0.260.

| Task | skill | no-skill | delta |
|---|---:|---:|---:|
| A1 B580 RMSNorm | 1.00 | 0.80 | +0.20 |
| A2 wrong-output baseline | 1.00 | 1.00 | +0.00 |
| A3 GEMV 4097 | 1.00 | 0.59 | +0.41 |
| A4 oneAPI 2027 load_2d | 0.96 | 0.73 | +0.23 |
| A5 sparse 80% N=4096 | 1.00 | 0.55 | +0.45 |
| A6 fastest RMSNorm | 1.00 | 0.72 | +0.28 |

Decision vs protocol split:

| Run | decision | protocol | weighted |
|---|---:|---:|---:|
| adversarial with-skill | 1.000 | 0.988 | 0.993 |
| adversarial no-skill | 0.933 | 0.589 | 0.733 |

Both groups answered the adversarial decisions correctly from general
knowledge (decision means are close). The skill's measured value is protocol
and evidence discipline: `[MEASURED]`/`[HEURISTIC]` labeling, validity
domains, oneDNN verbose + accuracy class, device/wall fields, and benchmark
protocol. No-skill reports dropped protocol scores sharply. Task A2's
no-skill subject passed because the prompt itself spelled out the correctness
JSON contract; general knowledge is enough when the contract is explicit.

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
| 11 | high | `SKILL.md` Curated Pitfalls | Split `[ARCH]` PVC-only availability from version-specific `[BUG]`/`[TOOLCHAIN]` hang/JIT-rejection |
| 12 | high | `api-usage.md` oneDNN fallback | Labeled `0.033-0.034 ms` as bf16-src fastest-only and added the matched f16-src baseline |
| 13 | high | `SKILL.md` Row reductions | Added wall/device champion clarification for "fastest RMSNorm" |

## Second Review Fixes (2026-08-09)

| Area | Applied fix |
|---|---|
| Accuracy-aware baseline | Added `matched` / `relaxed_matched` / `fastest_only` / `invalid` / `unknown`, executable-reported tolerance and `semantics_id`; follow-up probe found a matched f16-src oneDNN config for weight-only M=64 |
| RMSNorm dispatch noise | Added 3% tie handling and simplified production regions instead of per-cell argmin dispatch |
| Evidence taxonomy | Added `[CORRECTNESS]` and `[TOOLCHAIN]`; reclassified weight-only packed-offset, M=1, and mixed-precision DPAS claims |
| Self-contained harness | Bundled core scripts and example kernels under `scripts/` / `examples/`; added script-existence gate |

## Third Review Fixes (2026-08-09)

| Area | Applied fix |
|---|---|
| Correctness contract | Executable reports `rel_tol` / `abs_tol` / `max_rel_err` / `reference` / `semantics_id`; mismatch produces `CORRECTNESS_CONTRACT_MISMATCH`; `FAIL` is `invalid` unless relaxed accuracy and same semantics are confirmed |
| Compare length gate | `compare_outputs.py` now returns `SHAPE_MISMATCH` when actual/expected lengths differ |
| Roofline wording | Renamed `DRAM ceiling` to strided-pattern baseline; measured `B_DRAM_contiguous` and updated empirical DRAM/L2 ridges |
| GEMV terminology | Split `GEMV-N1` (`A*x`) from `GEMV-M1` (`x*W`) |
| Dispatch extrapolation | Interval rules stay `[HEURISTIC]` off the measured rows; boundary interpolation sweep required before promotion |
| Tiny-kernel noise | 3% is documented as a simplification threshold; paired/interleaved measurement is required for significance |
| Harness tests | Added `scripts/tests/test_harness.py` for tolerance mismatch, invalid/fastest_only, shape mismatch, and JSON contract parsing |
| Boundary dispatch | Boundary interpolation upgraded bf16 `M<=16` / `M>=24` and f32 GEMV-N1 `M<=64` / `M>=192`; unswept gaps stay `[HEURISTIC]` |
| Campaign packaging | Intentionally not bundled: the skill is a workflow/guidance layer, not a full optimizer |
| SkillEval scoring | Split `decision_total` / `protocol_total` / `weighted_total`; decision means are close, protocol means carry the skill delta |
| Workflow scope | Added `methodology.md` for new-operator thinking, profiling-driven next experiments, and failure-model updates; harness marked optional thin helper |

## Reproduction

```powershell
python eval\scoring.py --date 2026-08-09 --subject "independent run"
python eval\scoring.py --tasks-dir eval\adversarial-tasks --transcripts-dir eval\transcripts\adversarial-skill --out-dir eval\results\adversarial-skill
python eval\scoring.py --tasks-dir eval\adversarial-tasks --transcripts-dir eval\no-skill-baseline\transcripts --out-dir eval\no-skill-baseline
python eval\report_adversarial.py
```

The scorer reads task definitions and transcripts, then writes
`results/score.json` plus `results/results.md`; the adversarial reporter also
writes the skill vs no-skill delta. Re-run after any skill text change or
model/CLI change.

## Known Limitations

- Regex scoring is heuristic; borderline transcripts need a human review of
  the issues in `score.json`.
- The scorer checks the final report only. Intermediate tool behavior is out
  of scope for this no-execution task set.
- Results cover one model/CLI/machine combination; re-run for other subjects
  before generalizing.
