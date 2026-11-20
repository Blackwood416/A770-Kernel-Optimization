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

## What to run before believing a page

1. Regenerate twice; the bytes must be identical (this also catches self-scanning bugs).
2. `grep -n` the generator for every number the page prints. A hit outside an f-string expression
   field, a format spec, or a commented constant is a violation.
3. Perturb each artifact value the generator reads; confirm the printed line moves. For the ones that
   do not move, write down why in the commit message.
4. Re-run the acceptance predicate as the gate runs it (same shell), not the way it reads best.

The cost of skipping this is not a wrong sentence somewhere: it is that a *later* task reports a
number nobody can trace, and the whole ladder above it becomes untrustworthy.
