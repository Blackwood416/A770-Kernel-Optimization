# Evidence integrity: how a page of numbers stays true

This page is about the *other* failure mode. `methodology.md` says how to find the bottleneck;
this page says how to keep a measurement page from quietly becoming fiction, and what to run to prove
it has not. Every rule below was paid for by an audit rejection in the campaign repo — the incidents
are named with the artifact that carries the numbers, because the incident is the argument.

## The rules

| Rule | It prevents | The incident that produced it |
|---|---|---|
| Any source that cannot interpolate must not restate a derived number: source comments, README index rows, notes | a number that goes stale with no generator to catch it | a kernel header said the interleaved slices own "8 KB blocks, 8 KB apart" while the kernel gives `UNROLL` keys = 2 KB, and the same file's other comment had already been corrected; separately a README index row still printed 1.68 % after the generator moved to unrounded means and the page printed 1.71 %. Both were fixed by **removing the number and pointing at the artifact**, not by updating it |
| A threshold printed on a page must print the constant that scores it | a claim that contradicts its own scorer | prediction claims typed "within 5%" while `ROOF_TOLERANCE` decided the verdict; moving the constant would have left the claim lying. The strings now interpolate the constant |
| A derived percentage is computed from unrounded quantities | a percentage nobody can reproduce from the printed columns | a spacing effect divided two column means rounded to 1 decimal and printed 1.71 % where the raw-rate value is 1.70 %, i.e. recomputing from the table gave a different answer than the sentence |
| A pre-registration is citable, not editable; corrections are append-only and dated | rewriting the prediction after seeing the result | the falsification branch quoted an interval formed from the two shapes known when it was written; after four shapes were measured the low bound excluded one of them. The branch now points at the pages, and an `## Amendment <date>` block records exactly what was reworded and why, with predictions and thresholds untouched |
| A page asserts the artifact it pinned, never the live code | a correct historical page going red the moment a default moves | `docs/70-sdpa-occupancy.md`'s generator checked its pinned `split4` records against the *current* `#define`; when the default became `SDPA_SPLIT=16` the assertion reported "re-pin the records" for records that were still exactly right. Fixed by reading the macro out of the page's own `accepted.json` and using the live default only to say what superseded it |
| One derived value, one home; restate at most once, with provenance | two artifacts disagreeing on one measurement, and the reader not knowing which is current | a gap was computed in a sweep script, restated in `accepted.json`, and typed again in prose: three homes, one of which went stale |
| A superseding result names itself on the predecessor's page | the same conclusion quoted with two different numbers as "current" on two pages | the adoption gate literally requires the predecessor summary to contain the superseding task id (`grep T-114 docs/68-p6-summary.md`); before that, the P5 summary still called 557 us the adopted build while the new default measured 427 |
| Never type a value the renderer already holds | a page that looks interpolated and is not — the single most common audit blocker | a template that printed `{len(correct_points)} points` four lines below a paragraph that typed "9 points"; a section that printed `{acc['budget_us']:.0f}` while typing "536 us" two lines earlier; a SIMD width interpolated from a CSV in the table and typed as `SIMD8` in the prose beside it |
| Prove "no typed numbers" by perturbing the artifacts, and account for every field that does not move | a plausible-looking claim of interpolation that nobody checked | a page claimed every number was derived; a harness of 383 single-value perturbations moved 208 of them, and all 169 non-movers were then individually classified as: a field the generator never reads, a value outside the printed `max()`/range selection (perturbing roof 403.1 → 400.0 does not move a printed maximum), or a change below the printed precision |
| Make an acceptance criterion an executable predicate, not a sentence | a task closed on "the doc exists" or on a threshold a human interpreted | a check written with POSIX single quotes passed under bash and went RED in the gate, because the runner is `cmd.exe` and `'0.70'` arrived as a literal token; and a doc-shape predicate with a minimum line count plus required markers is what stops a one-line summary being accepted as a write-up |
| Enforce table column shape with one shared scanner (`G8`, `ops/table_shape.py`) | unescaped pipes inside code spans (`add (16\|M0)`) silently splitting markdown table rows into extra columns | `ops/consistency.py` G8 imports `scan` directly from `ops/table_shape.py` over `docs/*.md` instead of keeping a second copy of the rule, and `--strict` fails on any row whose cell count disagrees with its header |
| Ratchet digits in generator-free carriers (`G9`, `ops/carrier_digits.py`) | stale numbers typed into source comments, `ops/*.py` docstrings, `README.md`, or `ops/tasks.json` board notes outside G7 | every digit run in a carrier must be a `NAME` (token with a letter, e.g. `docs/84`, `f32`, `2x2`), a `CITATION` naming an owning artifact on the same line (or paragraph for prose blocks) whose own text contains that token, or a documented `ALLOW` rule; `--strict` ratchets against `ops/carrier_digits_baseline.json` |
| Lock falsifiable predictions in `prediction.md` before collection and report partial/full falsifications verbatim | post-hoc rationalization that hides unexpected compiler or hardware mechanisms when headlines still hold | P12 pre-registered falsifications (`probes/115`–`probes/118`, `docs/91`–`docs/94`) each caught real structural facts (`B_TASKS_PER_THREAD=1` vs `3`, extra `Synchronization` per block, `M0` co-residency `+9.42%`, intra-kernel `XVE Idle`) that a post-hoc pass would have glossed over |

- **An artifact may not read a constant from the current world.** A measurement record that derives a
  parameter from the live source tree is a cache, not a record: it silently re-states itself whenever the
  source moves. This cost a full round here — two probes derived "the default swizzle factor" by re-reading
  `#define GEMM_SWIZZLE` from the kernel, which was harmless while the default never changed, and then
  rewrote both frozen artifacts (and three generated pages) the moment it did. Freeze the value the probe
  MEASURED, with the commit it was measured at, and make a missing declaration a hard refusal — never a
  fallback to the live tree, because the fallback is what hides the defect. Same family as "a predecessor
  figure comes from the predecessor's own artifact", one level down: the predecessor's constants too.

- **Sample the worktree twice.** A verification pass that only checks for a clean tree *before* running its
  checks cannot see a check that writes to that tree: it reports all-green and leaves the repository
  changed, which is how the defect above stayed invisible across many closeouts. Re-sample after the run
  and fail if the run itself dirtied anything.

## Corollaries that came up twice or more

- **A generator that scans its own inputs must exclude itself.** The probe index listed the probes
  that no page mentions; its own page then mentioned them, so the second run un-orphaned them and the
  byte-identical-regeneration check failed. Skip the output file, and exclude the input classes that
  are legitimately uncited (summary-only directories).
- **A comparison needs its own twin per shape.** The claim "the win is not a cheaper rendezvous" was
  made about a shape that had no barrier-removed twin, so it was unproven even though it was true.
  The fix was to measure a twin for every compared shape *and* to make the renderer exit non-zero when
  one is missing — a generator invariant, not a resolution to be careful.
- **A prose claim should become a code invariant.** If the page says "prefetch is not an axis because
  both branches are the same loop", have the generator exit 1 when that stops being true. If it says
  "the amortisation is exact", assert the identity `pass(n) == pass(1)/n + fixed` on the loaded rows.
  A checker cannot be talked into believing a stale sentence.
- **State which tree you read.** An auditor must reproduce a blocker at `HEAD` (`git show HEAD:path`),
  not inside the task's diff — a rework that replaces a wrong number legitimately leaves the old
  number in the diff, and judging that line rejects work that is already correct. The converse bit
  harder: an orchestrator who grepped `HEAD`, found the old number gone, and declared the blocker
  stale was wrong, because the fix was sitting **uncommitted** in the worktree — so `git status` comes
  before any "that is stale" verdict.
- **Numbers in source comments are numbers on a page.** A kernel header comment that quoted the
  pre-re-run medians was caught by the same rule as a doc paragraph; the artifact it must quote is the
  CSV it names, worst-of-runs, and it should name the file so a reader can check.
- **Parallel CPU build, FIFO-serialized GPU lease (`ops/gpu_lease.py`).** CPU compilation, static analysis,
  and doc generation run concurrently across parallel workers, while all GPU measurements serialize through
  `python ops/gpu_lease.py --owner <probe> -- cmd ...`. Before and after each lease (and on timeout or
  abnormal exit), the broker checks A770 PnP device status (`pnputil`), scans Windows System events for
  display driver resets, and dispatches a sub-second OpenCL canary via `tools/genrun/genrun.exe`; on a hang
  or WDDM TDR it escalates through Tier-A process-tree purge (`taskkill /F /T`), Tier-B `pnputil` device
  restart, and Tier-C circuit breaker (`ops/out/gpu_circuit_breaker.json`, exit 75).
- **Pre-registered falsifications pay for themselves when they expose hidden mechanisms (P12, `docs/91`–`docs/94`).**
  Treating a missed pre-registered predicate as a diagnostic clue rather than a failure to hide uncovered four
  concrete mechanisms across P12:
  - `probes/116-sdpa-counters` `P1` (`docs/92`): expecting `Synchronization` to be invariant across `SDPA_ORDER`
    caught IGC emitting `1` extra non-gateway `Synchronization` instruction per loop block in `SDPA_ORDER=0`
    (`33` vs `17` on `s2048`, `9` vs `5` on `s512` `[MEASURED]`), while `P2` caught that dividing `~97%`
    captured `>= 335` us kernels by `duty_cycle` over-reports `150` MB vs `134.2` MB theoretical (`[TOOLCHAIN]`).
  - `probes/117-sdpa-stream-mech` `P1` (`docs/93`): predicting `M0` would match `probes/099-dram-streams`'s flat
    band (`+/-3.0%` vs `ORDER=0` baseline) falsified that assumption on impact and proved that `WG=512` `16`-slice
    co-residency plus `K`/`V` dual-buffer alternation already generates **`+9.42%`** (`367.3 -> 401.9` GB/s vs
    `ORDER=0` baseline `[MEASURED]`), explaining `43.2%` of the full SDPA order win.
  - `probes/118-operator-anatomy` `P1` (`docs/94`): predicting `int8` `GEMM_PREPACK=1` would cut `96` `Send`/thread
    (`2.0x` `bf16`) was falsified (`-48` `Send`/thread = `1.0x` `bf16` `[MEASURED]`), exposing the exact source
    duality: `int8` has `B_TASKS_PER_THREAD=1` (`BN=128, BK=32, WG=128`, VNNI `k4`) saving `3` loads/task, whereas
    `bf16` S1 has `B_TASKS_PER_THREAD=3` (`BN=192, BK=32, WG=128`, VNNI `k2`) saving `1` load/task (`3*1*16 == 1*3*16 == 48` `[MEASURED]`).
  - `probes/115-bf16-prepack` `P1`/`P2` (`docs/91`): missing the `S1` `R=1` break-even (`95.62` vs `94.27` us
    `adopted` baseline `[MEASURED]`) and the `>= 15.0%` total-instruction threshold (`-8.31%` actual `[MEASURED]`)
    quantified both the `GRID_M` break-even gate (`S1` `GRID_M=4` needs `R>=2` vs `S2` `GRID_M=8` wins at `R=1`)
    and why eliminating `48` strided half-sector `Send`s cuts latency (`-13.66%`) more than instruction count (`-8.31%`).
  - `probes/121-gemm-closing-gap` `P1`/`P2`/`P3` (`docs/97`): while all `S1` clauses met and surpassed `oneDNN`
    (`bf16` `68.30` / `66.08` us vs `73.16` us `oneDNN` baseline; `int8` `64.14` / `57.31` us vs `68.71` us `oneDNN`
    baseline `[MEASURED]`), partial falsification on `S2` (`126.13` vs `102.03` us `bf16` `oneDNN` baseline; `108.80`
    vs `104.39` us `int8` `oneDNN` baseline `[MEASURED]`) and the `Synchronization` count (`96` / `41` vs `33` / `17`
    barrier-only estimate `[TOOLCHAIN]` `[MEASURED]`) exposed the single-wave (`WG=128`) vs multi-wave (`grf256`
    `WG=32`) residency crossover and Xe-HPG SWSB scoreboard `sync.allrd`/`sync.nop` counting inside GT-PIN `Synchronization`.

## What to run before believing a page

1. Regenerate twice; the bytes must be identical (this also catches self-scanning bugs).
2. `grep -n` the generator for every number the page prints. A hit outside an f-string expression
   field, a format spec, or a commented constant is a violation.
3. Perturb each artifact value the generator reads; confirm the printed line moves. For the ones that
   do not move, write down why in the commit message.
4. Re-run the acceptance predicate as the gate runs it (same shell), not the way it reads best.

The cost of skipping this is not a wrong sentence somewhere: it is that a *later* task reports a
number nobody can trace, and the whole ladder above it becomes untrustworthy.
