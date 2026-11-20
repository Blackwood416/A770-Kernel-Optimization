# DG2 Low-Bit DPAS and Static Graph Capability

Use for INT2/INT4 DPAS, missing graph aspects, fixed-address parameter blocks,
or evaluating claims about native submission overhead. These are capability
and correctness findings, not new performance dispatch rules.

## Evidence and validity

`[MEASURED]` functional evidence: A770, Windows, driver `32.0.101.8974`,
oneAPI `2026.1.0.20260617`, Level Zero driver string `1.15.39183+1`,
2026-09-06. The A770-DeepResearch campaign's W1/W3 source and archived stdout
were audited; this skill update did not rerun GPU tests. Campaign files are
not bundled with the skill. Read its `experiments/skill-backfill-audit.md`
and the corrective review `experiments/skill-backfill-audit-v2.md` when the
campaign checkout is available; do not assume its runners are skill-root
commands. The 2026-09-07 corrective review inspected updated source and
archived samples; it did not independently rerun GPU binaries.
The subsequent `experiments/skill-backfill-audit-v3.md` closes specific
runner and broadcast source defects after 31 passing host-side tests;
it does not establish a successful, time-calibrated polling protocol.
The 2026-10-03 review `experiments/breakthrough-review.md` audits archived
W7-W9 runs dated 2026-09-07, recomputes W7/W8 samples and tests runner failure
handling with host-only mocks. It did not execute GPU kernels.
The same day's corrective review `experiments/breakthrough-review-v2.md`
rechecks new W8/W9 archives and closes the earlier input/NaN and specific
runner false-PASS defects; remaining malformed-metric and validator-test
issues are documented there. No GPU kernels were run by either review.

Source checkpoints: intel/llvm `1cd007682727c3efadefa0b1ea6483bb0e771fe7`;
intel/intel-graphics-compiler `21ed964360185656844701b4452be5e8bd760aa3`;
intel/compute-runtime `85d89a5ea6c2e73ddbdf618cd9f55e596b9eeb27`.
These checkpoints are not a verified source match for installed binaries.

## Low-bit DPAS

- `[ARCH]` DG2's IGA `ModelXeHPG.hpp` contains integer U2/S2/U4/S4 DPAS.
  ESIMD `xmx/dpas.hpp` can express these through explicit precision template
  arguments and packed containers. A is Src2, B is Src1; pack B for DPAS's
  layout rather than assuming ordinary row-major nibble packing is enough.
- `[ARCH]` Use SIMD8 on DG2. With SD=8, both integer inputs <=4 bits give
  OPS_PER_CHAN=8 and K=64; either input at 8 bits gives 4 and K=32.
  FP16/BF16 gives 2 and K=16. K is not SystolicDepth.
- `[ARCH]` INT2 does not double INT4's DPAS arithmetic throughput. Lower
  storage size may reduce traffic; end-to-end benefit still needs measurement.
  W4A8 is not the same compute-throughput class as W4A4.
- `[MEASURED]` W1 archived 84 passing tile invocations against an independent
  logical-matrix int64 CPU reference: M=1/8, N=8, K=32/64; representative
  signed/unsigned 2/4/8-bit combinations and C0 cases. This did not test a
  complete GEMM, odd tails, application quantization, or steady-state throughput.
  Its three named random-seed patterns repeat the same generated data.
- `[TOOLCHAIN]` Generic vISA u1/s1 entries do not establish DG2 INT1 support.
  Integer INT4 also does not establish FP4/NF4 or INT4-by-FP16 DPAS support.

Intel's [Xe-HPG architecture table](https://www.intel.com/content/www/us/en/developer/articles/technical/introduction-to-the-xe-hpg-architecture.html)
labels 512 INT4/INT2 ops/clock **per EU/XVE**, not per Xe-core. Keep this
theoretical unit separate from measured kernel throughput.

## Static versus updatable graph

- `[TOOLCHAIN]` Query `ext_oneapi_limited_graph` separately from
  `ext_oneapi_graph`. At the cited llvm checkpoint, limited needs command
  buffers; full also needs all required update capabilities. Missing full
  support does not reject ordinary static finalize/replay.
- `[TOOLCHAIN]` compute-runtime's Xe-HPG-specific
  `getPlatformCmdListUpdateCapabilities()` omits KERNEL_ARGUMENTS and
  KERNEL_INSTRUCTION, although generic implementations may include them.
  This is the immediate capability-gate explanation, not proof of why the
  driver omitted them. Do not infer a silicon erratum from inline walker data.
- `[MEASURED]` W3's single-node graph passed 100 sequential parameter-block
  replays: shared Params pointer fixed, device input/output pointer pairs
  alternate, n cycles through 1/255/256/257/4093 with upper bound 4096,
  alpha/bias change, and output canaries remain intact. This establishes a
  useful functional pattern, not zero-cost updates or multi-graph concurrency.
- `[CORRECTNESS]` Finish the previous use before host mutation, or establish
  a separately verified transfer/dependency protocol. Keep pointed-to
  allocations alive. Fixed launch geometry still needs guards or graph variants.
  Plain captured scalar values do not update when the host variable changes.
- `[CORRECTNESS]` Raw USM accesses on a default queue do not infer dependency
  edges. Use explicit events/edges or an in-order queue for a dependent chain;
  retain explicit branch/join edges for a parallel diamond. Validate the exact
  benchmark path, not a different verification-only kernel.

## Environment and rejected performance promotions

`[TOOLCHAIN]` On the cited Windows stack, compiler-only environment setup
failed to expose Level Zero, while complete oneAPI `setvars.bat` restored it.
Check adapter dependencies, including UMF, and actual loaded DLL paths before
concluding that the device lacks a feature. Do not assume a DLL is a zero-byte
stub based on an earlier environment's symptoms.

## Corrected campaign experience

- `[MEASURED, NOISY]` The corrected W6 benchmark uses Shared USM, explicit
  G0 event dependencies/G1 graph edges, per-batch output and canary checks,
  and same-process rotation of G0/G1/P0. It supports trying static replay
  and bounded single-workgroup fusion on small affine chains. N=256,
  L=1/2/8/32/128, three sessions of 20 samples per variant; all archived
  samples passed. Every variant remained above the campaign's 5% CV gate.
  These observations establish candidates, not reusable speedup constants
  or a graph crossover depth. P0 changes intermediate storage as well as
  launch count; compare against ordinary single-kernel fusion on real work.
- `[MEASURED, NOISY]` W4's native immediate/replay/rebuild paths were
  interleaved within its process. Try regular-list batching or replay for
  small dependent kernels, but benchmark the actual immediate queue mode.
  Its SYCL comparisons still reference separate worker runs; matching
  Shared USM alone does not make them paired or isolate framework costs.
- `[MEASURED, INCONCLUSIVE]` W2's four paired comparisons at 4/128 MiB
  had confidence intervals crossing 1. No clear benefit was detected in
  these settings; this is not equivalence or a general prefetch prohibition.
- `[CORRECTNESS]` A shared-local action broadcast needs synchronization
  both before readers consume it and before its next overwrite. A barrier
  before reading alone can race across iterations on a spin-only path.
  The campaign's third revision added the missing second barrier. Validate
  shared-USM atomic capabilities, system scope and host/device interoperability
  separately from P0 fusion; the polling probe still lacks this evidence and
  an archived device-time calibration. An iteration cap that resets after
  work is not a whole-kernel time budget, and a host timeout followed by
  event.wait() does not guarantee GPU cancellation by that deadline.

## W7-W9 follow-up: paired paths and atomic capabilities

- `[MEASURED]` W7 compares SYCL graph replay and native L0 replay in the same
  process/context on the same shared buffers, with AB/BA batches and actual
  device-handle validation. For (L,N)=(2,256),(8,256),(32,256),(8,65536),
  three sessions of 20 pairs reproduce L0 replay speedups of 1.24-1.47x
  (ratios of medians), all NOISY_PERF_WIN. Median paired differences were
  79-95 us; each case also had negative differences. This is a comparison
  of these implementations, not a fixed overhead constant: SYCL range
  kernels and precompiled OpenCL kernels do not have verified matching
  codegen, launch geometry or queue/barrier implementations.
- `[TOOLCHAIN]` Atomic scope and allocation concurrency are separate
  capabilities. At the cited llvm checkpoint, UR exposes SYSTEM scope,
  but separately maps each L0 allocation capability set. SYCL's
  `usm_atomic_shared_allocations` tests SINGLE_SHARED's
  ATOMIC_CONCURRENT_ACCESS; `usm_system_allocations` tests SYSTEM_SHARED
  access. Do not equate `sharedSystemAllocCapabilities` with system atomic
  scope or infer a runtime contradiction from their differing values.
  W9 archived single-shared=0x3 (RW|ATOMIC), without concurrent-atomic bits.
  Query the actual relevant SYCL aspects before host/device atomic polling;
  raw ATOMIC support alone does not authorize concurrent host access.
- `[MEASURED]` W9's corrected 2026-10-03 archive directly queries
  usm_atomic_shared_allocations=false, usm_atomic_host_allocations=false,
  usm_system_allocations=false and usm_shared_allocations=true on this stack.
  Its device-scope, two-workgroup normal/abort examples report exact
  comparison of all 256 uint32_t payload elements after 10/3 steps. These
  are device-internal communication observations, not host cancellation
  or a guarantee of workgroup forward progress.
- `[CORRECTNESS]` W9's calibration and timeout loops differ: the latest
  archive records 1M calibration iterations at 177.867 ms host wall and
  the actual 200k timeout loop at 98.953 ms. Do not transfer a per-iteration
  estimate across loops or present it as a TDR safety bound. A subsequent
  empty kernel returning from wait() does not independently prove that no
  driver reset occurred.
- `[CORRECTNESS]` W8's corrected generator converts to signed before
  subtraction; its output comparator rejects NaN/Inf. New measured results
  below supersede the old defective-input campaign. Its local_accessor
  intermediate is SLM, not demonstrated
  register residency; global address-space traffic is not measured DRAM
  traffic. Differences between three implementation medians do not isolate
  physical percentages for launch versus memory, especially when workgroup
  geometry changes.

### Corrected W8 fusion observations

`[MEASURED]` 2026-10-03, same A770/driver/compiler domain; f32 RMSNorm
(epsilon=1e-5) + row-major GEMV+bias, fixed periodic signed inputs, Shared
USM, host resets inputs/weights/output before each measured submit-to-wait.
Three sessions of 20 batches per variant, interleaved. Comparison is against
the campaign's simple two-kernel G0, not an optimized library baseline.
Observed max error metric 4.38432e-5; the actual criterion is
abs(actual-reference)/(abs(reference)+1e-5) < 1e-4, with finite checks.

| K x M | G0 / fused SLM median us | G0/fused ratio (95% CI) | Status |
|---|---|---|---|
| 256 x 256 | 608 / 509 | 1.1945 (1.1823-1.2141) | NOISY_PERF_WIN |
| 256 x 64 | 630.5 / 498.5 | 1.2648 (1.2048-1.2955) | NOISY_PERF_WIN |
| 256 x 1024 | 698.5 / 736 | 0.9490 (0.9413-0.9592) | LOSS, noisy |
| 1024 x 256 | 818 / 740 | 1.1054 (1.0984-1.1111) | NOISY_PERF_WIN |

Use single-workgroup SLM fusion as a candidate; do not infer an M threshold
or choose a production winner from four points. The M=1024 slowdown is
consistent with reduced workgroup parallelism, but its cause was not
isolated by profiling. Compare optimized GEMV/library combinations and
real target shapes before selection.

The campaign's new fixed performance constants remain **not admitted**:

- The original W3 timed chains lacked dependency edges and output comparison;
  the original W6 reused that G0 baseline. The corrected campaign supersedes
  those numbers, but its noisy results do not establish stable dispatch rules.
- Original cross-worker SYCL/L0 comparisons used different allocation kinds;
  corrected runs use Shared USM but still different binaries and sample runs.
  Subtracting their wall times does not isolate a fixed UR
  CPU tax or establish a WDDM physical latency floor.
- W2 computed input bytes divided by host wall time. This is neither measured
  DRAM traffic nor proof of bus saturation; no benefit at one setting does
  not establish a general prefetch prohibition.
- The original host/GPU polling timeout involved invalid atomic object usage
  or potentially divergent group barriers; it cannot identify a hardware
  cache-coherence cause. Those specific source defects were subsequently
  repaired, but bounded loops are not automatically TDR-proof. Do not treat
  the revised poll probe as a validated reusable implementation yet.

For promotion, repair correctness/dependencies first, then use matched memory,
binary/launch configuration and actual paired sampling; keep noisy data and
device/host timing separate. Do not use worker PASS labels as substitutes for
these artifacts. Existing older execution measurements retain their own domains.
