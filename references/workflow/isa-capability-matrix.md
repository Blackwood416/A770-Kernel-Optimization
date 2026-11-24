# ISA capability matrix (what you can actually write on DG2)

For every ISA feature this project has studied: can a plain kernel make the compiler emit it, is it
asm-only, or is it unreachable? Generated from `probes/072-capability-matrix/capability_matrix.csv`; the AOT/LSC rows come from committed dumps by `run_matrix.py` (OpenCL C and ESIMD sources under that probe's `src/`).

## Matrix

| feature | source | asm | how to write it from source | measured |
|---|---|---|---|---|
| GRF-indirect 寻址 r[a0.0] | ✅ 源码可达 | ✅ | 动态下标访问私有数组 float t[8]; t[n&7] | CL 源码 -> ISA 中 r[a0 出现 1 次 |
| call（函数调用） | ✅ 源码可达 | ✅ | __attribute__((noinline)) 设备函数被调用两次 | CL 源码 -> call 2 条 / join 0 条（IGC 默认仍内联） |
| prefetch（OpenCL __builtin_prefetch） | ❌ | ❌ | __builtin_prefetch(&x[64]) | CL 源码编译即失败（rc=6）：InvalidFunctionCall: Unexpected llvm intrinsic: llvm.prefetch.p1i8 |
| prefetch（ESIMD esimd::prefetch） | ❌ | ❌ | esimd::prefetch(x + k)（2026.2 memory.hpp -> __esimd_lsc_prefetch_stateless） | SYCL/ESIMD 源码可编译（需显式 L1/L2 cache hint），但全部 kernel section 里 prefetch 指令 0 条 |
| dpas（8x8 bf16/f32 acc） | ✅ 源码可达 | ✅ | ESIMD xmx::dpas 或 dpas 内建 | GEMM ISA 中 dpas 逐条可数（probes/070 gemm_isa.csv） |
| dpasw（2x2 双宽） | ✅ SYCL/ESIMD 可达（OpenCL C ❌） | ✅ | SYCL/ESIMD `sycl::ext::intel::esimd::xmx::dpasw<8,8>` (`oneAPI 2026.2`) | `[MEASURED]` IGA 编码面及 ESIMD `dpasw<8,8>` 均已验证；`4096x4096x4096` bf16 GEMM 实测 `92.5 TFLOPS`（较 `dpas` 节省一半 A SLM 读） |
| bf16/f16 <-> f32 转换 | ✅ 源码可达 | ✅ | sycl::bfloat16 / sycl::half 构造与转换 | mov :bf/:hf，见 docs/44 |
| SWSB 令牌（@1/$2/A@1） | 🟡 VC `-Xfinalizer` 间接可控 | ✅ | 手写 asm 可加；ESIMD VC `-Xfinalizer`（`-SBIDBundle`/`-SWSBDepReduction`/`-nopresched`/`-twoSrcBundleBCR`） | `[MEASURED]` OpenCL `.spv` 不可控，但 SYCL/ESIMD VC `-Xfinalizer` 可改写 SBID 复用与同步距离（`sync.allrd 5->0`, `Short UGM waits 5->1`） |
| {Compacted} 压缩编码 | ❌ | ✅ | 源码不可控；汇编器按编码约束自动选择 | IGC 输出里大面积出现；asm 侧可手写（C3 的触发条件另测） |
| join / call 栈 | 🟡 部分 | ✅ | IGC 生成 join；call 默认被内联 | 发散档位 1..128 只有 1 条 join |
| 256 GRF（大寄存器文件） | ✅ SYCL/ESIMD 可达（OpenCL `.spv` ❌） | ✅ | ESIMD `sycl::ext::intel::experimental::grf_size<256>` + `WG=32` + `SLM <= 48 KB`（`I-002` 仅限旧 OpenCL `.spv`） | `[MEASURED]` ESIMD + L0 native binary 稳定下发 `256-GRF`（`spill=0 B, priv=0 B`），解锁 `2 WGs/Xe-core = 64/64` 双驻留 |
| bindless / ExBSO 表面寻址 | 🟡 部分 | ❌ | OpenCL 无 bindless 堆；L0 才有 | B7 已收口，不在 OpenCL 重试 |
| send.dc0（常量/参数读） | ✅ 源码可达 | ✅ | 3 条由 IGC 从普通 OpenCL 源码发射（probes/001-ocloc-isa/dump/.text.axpy.asm） | (W) send.dc0 (8\|M0) r1 r127 null:0 0x0 0x024844FD {A@1,$0} // wr:1h+0, rd:4; oword aligned block read x8 |
| send.ugm 读（全局 load） | ✅ 源码可达 | ✅ | 4 条由 IGC 从普通 OpenCL 源码发射（probes/001-ocloc-isa/dump/.text.axpy.asm） |  send.ugm (16\|M0) r19 r15 null:0 a0.0 0x24280500 {ExBSO,A@2,$3} // wr:2+0, rd:2; load.ugm.d32.a32.ca.ca.bss[a0 |
| send.ugm 写（全局 store） | ✅ 源码可达 | ✅ | 2 条由 IGC 从普通 OpenCL 源码发射（probes/001-ocloc-isa/dump/.text.axpy.asm） |  send.ugm (16\|M0) null r11 r27:2 a0.0 0x240E0504 {ExBSO,A@2,$7} // wr:2+2, rd:0; store.ugm.d32.a32.wb.wb.bss[a |
| send.ugm 原子（atomic_inc） | ✅ 源码可达 | ✅ | 3 条由 IGC 从普通 OpenCL 源码发射（probes/074-p2-cleanup/work/atomic_dump/.text.k_order.asm） | (W) send.ugm (1\|M0) r15 r6 r10:1 0x0 0x0410058C {A@1,$2} // wr:2+1, rd:1; atomic_iadd.ugm.d32.a64 |

## How to use it

- **Prefer source (SYCL/ESIMD & OpenCL C).** `dpas`, `dpasw<8,8>` (ESIMD), `256-GRF` (ESIMD `grf_size<256>`), `bf16/f16` conversion, GRF-indirect (`r[a0.0]` from a dynamically indexed private array), noinline `call`, `join` and `{Compacted}` all come out of the compiler; you do not need asm injection for them. Note that legacy OpenCL `.spv` (`ocloc`) still cannot reach `dpasw` or `256-GRF` (I-002 boundary), whereas `SYCL/ESIMD` VC (oneAPI 2026.2+) emits both cleanly `[MEASURED]`.
- **VC `-Xfinalizer` for SWSB scheduling**: raw hand-written SWSB tokens remain asm-only, while ESIMD VC `-Xfinalizer` flags (`-SBIDBundle`, `-SWSBDepReduction`, `-nopresched`, `-twoSrcBundleBCR`) directly optimize SBID allocation and sync placement without asm patching.
- **Unreachable paths**: prefetch (OpenCL `__builtin_prefetch` fails to compile with
  `Unexpected llvm intrinsic: llvm.prefetch.p1i8`; ESIMD `esimd::prefetch` compiles only with explicit L1/L2 cache
  hints and emits zero prefetch instructions / hurts GEMM with ~50 extra ALU ops and SBID thrashing), and
  bindless/ExBSO outside L0. Do not design around them.
- `{Compacted}` is decided by the assembler's encoding constraints, not a knob: you can neither force nor forbid it.

Numbers behind the rows live in the CSVs cited in the probe (dpas/dpasw matrices, conversion cost, SWSB class table,
divergence stack, and this matrix itself) as well as Aurex v0.2 ESIMD `GRF=256` verification runs (`commit bf81332`).
