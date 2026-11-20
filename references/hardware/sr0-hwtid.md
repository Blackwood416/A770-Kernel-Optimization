# sr0: the per-thread status register on A770 (DG2) `[MEASURED]` `[ISA]`

Scope: measured and source-backed facts about `sr0` on Arc A770 (DG2), read while running real
kernels. `sr0` is not documented by Intel for DG2; the field layout below comes from IGC's own
decoder plus a hardware read-back experiment.

## 0. How to read sr0 from a kernel at all `[TOOLCHAIN]`

You cannot patch an AOT binary: an `ocloc compile` zebin keeps a `.spv` section, and the compute
runtime recompiles from that IR, so ISA edits are silently discarded (zeroing the `.spv` section
gives `CL_INVALID_BINARY`). Get the *runtime* binary instead:

```powershell
tools\genrun\genrun.exe kernel_dg2.bin axpy --n 64 --local 64 --repeat 1 --warmup 0 --dumpbin jit.bin
```

That dump is authoritative for its ISA: patching it changes behaviour (verified by rewriting a
stored constant and reading `0xcccc0003` back). Then disassemble, replace a constant `mov` with
`mov (16|M0) r15.0<1>:ud sr0.<k><0;1,0>:ud`, `ocloc asm`, run with `--dumpy N` for bit-exact
readback. Full recipe: `probes/050-sr0/read_sr0.ps1`.

## 1. Measured distribution `[MEASURED]`

Grid `local=64`, `n=4096` (64 work-groups = 128 hardware threads, SIMD32):

| dword | bits that vary | distinct values | reading |
|---|---|---|---|
| `sr0.0` | `0x00000FB0` (4,5,7,8,9,10,11) | 128, one per thread | slice / subslice / EU / TID |
| `sr0.1` | `0x01E00000` (21..24) | 16, steps every 8 work-groups | tile / dispatch slot counter |
| `sr0.2` | none | always `0xFFFFFFFF` | **not implemented on DG2** |
| `sr0.3` | none | always `0xFFFFFFFF` | **not implemented on DG2** |

`sr0.0` bit 27 is always 1 (`0x08000000`). Two threads of the same work-group differ **only** in
bit 7.

## 2. Field map `[ISA]`

IGC's decoder (`IGC/Compiler/CISACodeGen/CShader.cpp`, `CShader::GetHWTID()`) for XeHPG/DG2:

```
m_HW_TID = sr0 & 0x7FFF        // sr0.0 low 15 bits only
remove bit[7:6]                // reserved
remove bit[3]                  // reserved
```

The PVC branch of the same file spells the layout out, and it matches this generation:

```
[14:12] Slice ID    [11:9] SubSlice ID    [8] EUID[2]
[7:6]  Reserved     [5:4]  EUID[1:0]      [3] Reserved MBZ    [2:0] TID
```

The measured varying bits `{4,5,7,8,9,10,11}` fall exactly on `EUID[1:0]`, the subslice/slice
field, and the reserved bit 7 - which is why IGC has to strip bit 7 before the value is a usable id.

## 2b. Verified against the get_* queries [MEASURED]

Same grid (4096 work-items, local 64) run twice: once writing `get_local_id(0) | (get_group_id(0)<<8)`,
once writing `sr0.<k>` from a patched runtime-dumped binary. Checked over all 4096 work-items:

| hypothesis | holds |
|---|---|
| `sr0.0` bit7 == `(local_id>>5) & 1` | 4096/4096 |
| `sr0.0`[10:4] constant inside one hardware thread | 4096/4096 |
| `sr0.0`[31:16] constant across the launch | 4096/4096 |
| `sr0.0`[15:4] gives 128 distinct labels for 128 hardware threads | yes |
| `sr0.1`[24:21] == `2*(group_id>>3) + ((local_id>>5)&1)` | 4096/4096 |

So `sr0.0` is a **verified unique hardware-thread label**. An earlier guess
`sr0.1[22:21] == (thread & 3)` was wrong (holds for only half the items); the correct relation is the
one in the table.

## 3. Rules

1. **Never hand-roll hardware ids from sr0.** Bits 7:6 and 3 are reserved yet get set by hardware;
   use the IGC builtin path that applies the strip.
2. `sr0.1` is *not* part of HWTID; treat it as tile/dispatch information
   (`ForceZeroTileID` in `igc_flags.h` says tile id comes from `sr0.1`).
3. `sr0.2` / `sr0.3` read as all-ones on DG2: ported code that relies on them is broken, not
   merely slow.
4. `get_local_id` / `get_global_id` come from the thread payload `r0`, not from `sr0`.

## 4. Work-group to hardware-thread mapping `[MEASURED]`

OpenCL kernels on this box: `local=32` -> 1 hardware thread, `local=64` -> 2, `local=128` -> 4,
each SIMD32 thread covering 32 work-items. Budget occupancy from this, not from the SYCL
`max_work_group_size` number.
