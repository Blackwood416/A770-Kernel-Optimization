# Driving GEN Assembly Directly (edit-and-run loop on DG2)

**Scope**: Intel Arc A770 (DG2), Windows 11, driver IGC `32.0.101.9150`, ocloc/IGC
`32.0.101.8857`, native OpenCL host `tools/genrun`. This is the method behind the `[ISA]` and
`[MEASURED]` facts on the ISA pages; it is also the only way found so far to execute hand-written
GEN asm on this box.

## 1. Why the obvious paths fail `[BUG][ISA]`

| Attempt | Result |
|---|---|
| `ocloc compile` then `clCreateProgramWithBinary` | driver answers `module got recompiled from IR because provided native binary is incompatible with underlying device and/or driver [-Wrecompiled-from-ir]` and silently uses the embedded SPIR-V — your asm never runs |
| strip the `.spv` section so it cannot recompile | `CL_INVALID_BINARY (-42)` |
| `IGC_ShaderDumpEnable=1` / `IGC_DumpToCustomDir` | the DriverStore `igc64.dll` has the dump code compiled out — the dump directory stays empty |
| swap a matching `igc64.dll` into ocloc | no 9150-matching ocloc exists locally; `ocloc.exe` imports its own bundled compiler |

Root cause `[SRC]`: NEO's `Zebin::validateTargetDevice`
(`compute-runtime/shared/source/device_binary_format/zebin/zebin_decoder.cpp`) compares the
binary's IntelGT notes (productConfig, gfxCore, targetMetadata, zebinVersion, vISAAbiVersion,
stepping range) against the device. ocloc 8857 emits ze_info **1.70**, the runtime driver emits
**1.73**.

## 2. The working loop

```
genrun.exe kernel.bin --dumpbin rt_native.bin      # 1. harvest the DRIVER-native zebin
ocloc disasm -file rt_native.bin -device dg2 -dump dump_native
#                                                  2. edit dump_native\.text.<kernel>.asm
ocloc asm -file edited.bin -device dg2 -dump dump_edit   # 3. reassemble
genrun.exe dump_edit\edited.bin ...                # 4. run it — no recompile warning
```

- `[ISA]` Step 4 runs the edited instructions: prefixing a `mad` source with `-` changes the
  result, and re-disassembling shows the edit verbatim, so nothing silently rewrites the text.
- `[ISA]` `ocloc asm` writes its output **inside the `-dump` directory**, not the cwd.
- `[ISA]` A round-trip with no edits reproduces the driver binary **byte-identically**
  (`v00-baseline` md5 `c399f6b1…` in `probes/020-swsb`), and every edited variant still passes
  `ocloc validate` (“Binary is VALID”) while keeping `.spv`, `.ze_info` (v73) and
  `.note.intelgt.compat`.
- `[TOOLCHAIN]` `ocloc asm` syntax is `ocloc asm -file <name> -dump <dir> -device dg2 [-v]`;
  `disasm` rejects `-output` and defaults to `./dump/`.

## 3. Rules that keep this safe

1. **No control flow.** A hand-injected backward `goto` with a predicate hangs the GPU (no TDR
   event, ~300 s of spinning); recovery is kill-the-process plus
   `pnputil /restart-device`. Straight-line repetition is safe — a body duplicated up to 2048
   instructions (~32 KB of code) runs fine.
2. **Always run under a watchdog** (a campaign script: hard timeout, task-kill of the
   process tree, optional device restart). A missing bench line means "not measured", never "0 µs".
3. **Compatibility is per driver.** Conclusions about encodings transfer, but a binary must always
   descend from the runtime driver's own zebin; an ocloc-compiled binary will be recompiled away.
4. **Asm text is not a specification.** `mad D,A,B,C` is `D = A + B*C` (the printed `src0` is the
   addend) and `mul D,A,B` is `A*B`, established by patch-and-run, not by reading the disassembly.

## 4. Toolchain

- `[TOOLCHAIN]` Intel IGA (bundled at `intel-graphics-compiler/visa/iga`) builds standalone with
  MSVC: `cmake -G Ninja` + `cmake --build`, producing `iga64.exe`
  (`asm`/`disasm`/`-Xlist-ops`/`-Xprint-json`/`-Xprint-bits`/`-Xdecode-fields`).
- `[TOOLCHAIN]` IGA's CMake files call `add_compile_options(/WX)`, which lands **after**
  `CMAKE_CXX_FLAGS`; the only way to disable warnings-as-errors is the environment variable
  `_CL_=/WX- /W3` (that variable is appended last).
- `[TOOLCHAIN]` IGA's platform name for DG2 is `XeHPG`; it needs the raw `.text.<kernel>` code
  section, not the zebin ELF (feeding the ELF gives `error decoding instruction` at offset 0x0).
