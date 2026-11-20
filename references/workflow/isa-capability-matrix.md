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
| dpasw（2x2 双宽） | ❌ | ✅ | 无 SYCL/ESIMD 入口 | 只验证了 IGA 编码面；源码侧无入口 |
| bf16/f16 <-> f32 转换 | ✅ 源码可达 | ✅ | sycl::bfloat16 / sycl::half 构造与转换 | mov :bf/:hf，见 docs/44 |
| SWSB 令牌（@1/$2/A@1） | ❌ | ✅ | IGC 自行发射，源码不可控 | 手写 asm 可加；三类指令各接受不同拼写 |
| {Compacted} 压缩编码 | ❌ | ✅ | 源码不可控；汇编器按编码约束自动选择 | IGC 输出里大面积出现；asm 侧可手写（C3 的触发条件另测） |
| join / call 栈 | 🟡 部分 | ✅ | IGC 生成 join；call 默认被内联 | 发散档位 1..128 只有 1 条 join |
| 256 GRF（大寄存器文件） | ❌ | ❌ | 驱动未开 256-GRF 编译 | 不可达（I-002），写入决策页 |
| bindless / ExBSO 表面寻址 | 🟡 部分 | ❌ | OpenCL 无 bindless 堆；L0 才有 | B7 已收口，不在 OpenCL 重试 |
| send.dc0（常量/参数读） | ✅ 源码可达 | ✅ | 3 条由 IGC 从普通 OpenCL 源码发射（probes/001-ocloc-isa/dump/.text.axpy.asm） | (W) send.dc0 (8\|M0) r1 r127 null:0 0x0 0x024844FD {A@1,$0} // wr:1h+0, rd:4; oword aligned block read x8 |
| send.ugm 读（全局 load） | ✅ 源码可达 | ✅ | 4 条由 IGC 从普通 OpenCL 源码发射（probes/001-ocloc-isa/dump/.text.axpy.asm） |  send.ugm (16\|M0) r19 r15 null:0 a0.0 0x24280500 {ExBSO,A@2,$3} // wr:2+0, rd:2; load.ugm.d32.a32.ca.ca.bss[a0 |
| send.ugm 写（全局 store） | ✅ 源码可达 | ✅ | 2 条由 IGC 从普通 OpenCL 源码发射（probes/001-ocloc-isa/dump/.text.axpy.asm） |  send.ugm (16\|M0) null r11 r27:2 a0.0 0x240E0504 {ExBSO,A@2,$7} // wr:2+2, rd:0; store.ugm.d32.a32.wb.wb.bss[a |
| send.ugm 原子（atomic_inc） | ✅ 源码可达 | ✅ | 3 条由 IGC 从普通 OpenCL 源码发射（probes/074-p2-cleanup/work/atomic_dump/.text.k_order.asm） | (W) send.ugm (1\|M0) r15 r6 r10:1 0x0 0x0410058C {A@1,$2} // wr:2+1, rd:1; atomic_iadd.ugm.d32.a64 |

## How to use it

- **Prefer source.** dpas, bf16/f16 conversion, GRF-indirect (`r[a0.0]` from a dynamically indexed private array),
  noinline `call`, `join` and `{Compacted}` all come out of the compiler; you do not need asm injection for them.
- **asm-only**: dpasw (2x2) and SWSB tokens. IGA encodes both; no source construct reaches them, so they are for
  hand-scheduled kernels on a runtime-dumped binary.
- **Neither path works**: prefetch (OpenCL `__builtin_prefetch` fails to compile with
  `Unexpected llvm intrinsic: llvm.prefetch.p1i8`; ESIMD `esimd::prefetch` compiles only with explicit L1/L2 cache
  hints and then emits zero instructions in the ISA; IGA's XeHPG model has no prefetch mnemonic), 256-GRF, and
  bindless/ExBSO outside L0. Do not design around them.
- `{Compacted}` is decided by the assembler's encoding constraints, not a knob: you can neither force nor forbid it.

Numbers behind the rows live in the CSVs cited in the probe (dpas/dpasw matrices, conversion cost, SWSB class table,
divergence stack, and this matrix itself).
