# Synchronization on DG2 (XeHPG): Barriers, Fences, and Gateway Messages

**Scope**: Intel Arc A770 (DG2), ocloc/IGC `32.0.101.8857`, driver `32.0.101.9150`, IGA
`1.1.0-640a706f`. Established by compiling eight kernels that differ only in their
synchronization statement and reading the ISA (`probes/037-sync/sync_variants.cl`), then
bit-flipping the fence/gateway descriptors (`probes/037-sync/fence_bits.csv`).

## 1. There is no barrier or fence instruction on DG2 `[ISA]`

`iga64 -Xlist-ops -p XeHPG` lists only `sync` (plus `nop`, `movi`) in that area. Real
synchronization is expressed as **send messages**:

```asm
; barrier(CLK_LOCAL_MEM_FENCE)  ->  three instructions plus payload setup
send.slm (1|M0)  r6   r3  null:0 0x0 0x0210001F        ; fence.slm.none.group
mov (1|M0)  r14.2<1>:f   0.0:f                          ; payload: barrier kind/count
mov (2|M0)  r14.10<1>:ub r3.11<0;1,0>:ub                ; payload: participant id
mov (8|M0)  null<1>:ud   r6.0<1;1,0>:ud                 ; wait for the fence
send.gtwy (1|M0) null r14 null:0 0x0 0x02000004         ; signal barrier
sync.bar
```

A work-group barrier costs **three instructions** (fence message, gateway signal, `sync.bar`)
plus two `mov`s that build the gateway payload.

## 2. Source -> ISA map `[ISA]`

| Source | Emitted ISA |
|---|---|
| `barrier(CLK_LOCAL_MEM_FENCE)` | `fence.slm.none.group` + `signal barrier` + `sync.bar` |
| `barrier(CLK_GLOBAL_MEM_FENCE)` | `fence.ugm.none.group` + `signal barrier` + `sync.bar` |
| `barrier(LOCAL \| GLOBAL)` | `fence.slm` **then** `fence.ugm`, then signal + `sync.bar` |
| `barrier(0)` | **only** `signal barrier` + `sync.bar` — **no fence at all** |
| `mem_fence(CLK_LOCAL_MEM_FENCE)` | `fence.slm` only — **no barrier, no `sync.bar`** |
| `mem_fence(CLK_GLOBAL_MEM_FENCE)` | `fence.ugm` only |
| two `barrier(LOCAL)` | two complete sequences |

Two consequences worth internalizing:

- `barrier(0)` is an **execution barrier only**; it orders no memory traffic because no fence
  message is emitted for it.
- `mem_fence()` is a **memory fence only**; it does not wait for other threads, because there is
  no gateway signal and no `sync.bar`.

## 3. Fence message descriptor (`0x0210001F`, SFID `slm` or `ugm`) `[ISA]`

| Field | Bits | Encoding |
|---|---|---|
| Opcode | `[4:0]` | `0x1F` = fence. Sibling opcodes on the same SFIDs: `0x1E` read state, `0x1D` compression-state control, `0x1B` load status, `0x17` atomic float CAS, `0x0F` atomic smax |
| Scope | `[11:9]` | 0 `group`, 1 `local`, 2 `tile`, 3 `gpus` |
| Type | `[14:12]` | 0 `none`, 1 `evict`, 2 `invalidate`, 3 `clean` |

Printed as `fence.<space>.<type>.<scope>`. `slm` and `ugm` use the *same* descriptor — the address
space comes from the SFID, not from the descriptor bits.

## 4. Gateway message descriptors `[ISA]`

| Descriptor | Prints as |
|---|---|
| `0x02000000`, `0x02000010` | `end of thread` (EOT; the kernel epilogue uses `0x02000010` with `{EOT}`) |
| `0x02000002` | `wait for event` |
| `0x02000004`, `0x02000014` | `signal barrier` |
| `0x0200000C` | `system routine barrier` |
| `0x02000018` | `save barrier` |

## 5. Hazard: do not hand-inject a barrier `[BUG]`

Injecting `send.gtwy … 0x02000004` + `sync.bar` into a driver kernel **deadlocks it** (8/8
variants spun until killed by the watchdog; no TDR, GPU stays healthy). Rebuilding the payload the
way the compiler does (`r14.2 = 0.0f`, `r14.10 = work-group thread id`) still deadlocks, so the
payload's remaining fields (participant count / barrier id) are not yet understood.
Until they are, treat hand-written barriers as off-limits; use `sync.bar` only in code the compiler
already produced.

## 6. What this means for kernels

1. A barrier is ~3 instructions plus a fence. On a memory-bound kernel the fence dominates; on a
   compute-bound kernel the `sync.bar` wait dominates. Neither is free, so prefer fewer, larger
   phases over many small barriers.
2. `barrier(0)` (or SYCL's `group_barrier` without a fence) is the cheap option when you only need
   ordering of execution, not visibility of prior writes.
3. `mem_fence` does not synchronize threads; using it where a barrier is required is a correctness
   bug that no amount of timing will reveal.

## 7. What synchronization costs `[MEASURED]`

Source-level, 64 work-groups x 64 lanes = 4096 hardware threads, K=512, baseline subtracted:

| form | per operation | emits |
|---|---|---|
| `group_barrier(group)` (local) | **84.0 ns** | fence.slm + signal barrier + sync.bar |
| `group_barrier(group, device)` | **109.3 ns** | fence.ugm + signal barrier + sync.bar |
| `atomic_fence(acq_rel, device)` | 97.7 ns | not a single message - do not use as a fence price |

Switching the barrier's fence from slm to ugm costs about **25 ns**: that is the price of device-wide
visibility. Note that the SYCL `atomic_fence` is *more* expensive than a complete local barrier, so it
must not be treated as one fence message.

Message-level, fence messages injected into the driver's axpy kernel (128 hardware threads):

| variant | n=0 | n=1 | n=8 | n=32 | marginal |
|---|---|---|---|---|---|
| fence.slm, scope=group | 4.1 us | 4.1 | 4.3 | 4.7 | ~19 ns |
| fence.ugm, scope=group | 4.1 | 4.1 | 4.2 | 4.6 | ~19 ns |
| fence.ugm, scope=gpus (bit 11) | 4.1 | 4.1 | 4.3 | 4.6 | ~19 ns |
| fence.ugm, type=clean (bit 14) | 4.2 | 4.1 | 4.2 | 4.6 | ~19 ns |

One fence message costs about **19 ns**, and the scope/type/SFID bits do not change that: they change
*what* is guaranteed visible, not the throughput. A single fence is below measurement noise.

Budget guidance:

1. A barrier is ~84 ns (local) or ~109 ns (device scope) at full occupancy. One barrier per 1K
   elements is cheap; barriers that scale with element count are not.
2. Only pay the device-scope premium (25 ns) when cross-work-group visibility is actually needed.
3. Injected fence messages are safe to hand-write (16/16 variants ran correctly); injected gateway
   barriers deadlock (8/8), so the barrier remains compiler-only.

## Thread exit: what `send.gtwy {EOT}` actually costs

Every OpenCL kernel compiles to exactly **one** `send.gtwy … {EOT}` (verified by counting the AOT
disassembly of two different kernels in `probes/062-eot`), so "natural end of kernel" and "one EOT per
thread" are the same thing - there is no source-level way to run a thread without it.

What that costs, measured on A770 (`eot_cost.csv`, 15 repeats, median):

| knob | measurement |
|---|---|
| work-group count (local=64, 1024 -> 262144 groups) | 2.29 -> 124.27 us, slope **~0.47 ns per work-group**, intercept ~1.54 us (least squares over the 5-point ladder) |
| work-group width at fixed 262144 threads (local 1 -> 64) | 67.29 -> 3.33 us (20x), slope **~0.25 ns per work-group** |
| 8 extra fma per thread | +0.21 us at both 65536 and 1048576 threads |

Practical reading: the launch/retire floor is set by the number of work-groups, not by per-thread
instructions, and **local=32 is the knee** - going 32 -> 64 buys nothing, while 1 -> 32 is a 20x
difference. Splitting a kernel into many tiny work-groups is the expensive mistake; adding work per
thread is nearly free at these sizes. Isolating the EOT instruction from thread setup would require
patching the EOT bit of a runtime-dumped binary, which risks a non-retiring thread; it stays an open
instrument item.
