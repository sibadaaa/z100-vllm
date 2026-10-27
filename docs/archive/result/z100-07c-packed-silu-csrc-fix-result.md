# S7c csrc 根修 fp16 packed SiLU（gfx906+DTK 关闭 `use_vec`）+ 重建 `_C` + 复验 — 执行结果

> 归档时间：2026-10-08
> 执行者：子 agent（S7c，csrc 根修 + 重建 + 复验）
> 对应设计文档：[docs/design/z100-07-memory-kvcache-graph.md](../design/z100-07-memory-kvcache-graph.md)（S7c 无独立设计文档，由 S7b §5.1 派生）
> 前一档：[z100-07b-eager-nan-result.md](z100-07b-eager-nan-result.md)（S7b：根因定位 + Python 侧绕过 + §5.1 的 csrc patch 草案）
> 状态：✅ **csrc 根修已落地、`_C_stable_libtorch` 已增量重建并安装、逐项复验全过**
> ⚠️ 两处与任务书原文的偏差必须让 Lead 知道：**(a)** 被修的 TU 属于 `_C_stable_libtorch` 而**不是** `_C`
> （`_C.abi3.so` md5 逐字节未变）；**(b)** 守卫**不能**用 `__gfx906__`（host pass 里没有这个宏），
> 实际用的是 `defined(__DTK_ARCH__)` + 运行时 `gcnArchName` 前缀判定（§2.2 / §3.1）。

**一句话结论**：在 `csrc/libtorch_stable/activation_kernels.cu` 增加一个**只在 DTK（`__DTK_ARCH__`）+ 运行时
gcnArchName=gfx906 + fp16** 为真的 host 侧判定 `dtk_gfx906_fp16_no_packed()`，把它接在该文件**全部 3 处**
`const bool use_vec = (d % vec_size == 0)` 上，使 DTK+gfx906 的 fp16 激活改走**标量分支**（社区 ROCm / CUDA /
其它架构逐字节不变，§3.1 有 host 汇编级负向控制）。增量重建+安装后（只重编 1 个 TU + 1 次链接，TU 9.5 s）：
`_C.silu_and_mul` fp16 同一输入 **20 次非有限 = 0/0/0**（改前 424–455/次、20 次合计 8880）、
clamp 变体 0（改前 107/24576 nonfinite）、MoE 直调点 0/98304（改前 1907）、逐层扫描 396 模块 0 非有限、
**eager ids == graph ids == HF transformers 逐 token 完全一致**（并在**关掉 Python 兜底**的树上单独验证，
证明 csrc 修复既充分又必要——改前同配置是 `lock×16`）；5 个 gfx906 测试文件 39/12/13/5/5 全过，
新增 gfx906+DTK 专属单测 3 过（在旧 `.so` 上 3 败）。

---

## 1. 目标与验收标准

### 1.1 本步任务书门禁（原文要点）

1. **落地 csrc 根修**：在 `csrc/libtorch_stable/activation_kernels.cu`（以及同名孪生文件
   `csrc/activation_kernels.cu`，若存在同样代码也要一起改）让 **DTK + gfx906 的 fp16 实例化关闭 `use_vec`**，
   走标量分支；**必须精确限定**，社区 ROCm / CUDA / 其它架构行为**逐字节不变**，并给出负向控制证据；
2. **重建并安装**：在 `z100-build-211` 里增量重建（应只重编该 TU + 链接，~40 s），装进 canonical 副本
   `/mnt/game/port/gfx906-vllm-211`；装前粗查 S8（容器 `z100-s8`）是否正在启动引擎；记录新的尺寸与 md5；
3. **复验**：`_C.silu_and_mul` fp16 20 次非有限=0 且与 torch 参考误差合理；`silu_and_mul_with_clamp` 一次；
   `apply_moe_activation(SILU)` 非有限=0（改前 1917/98304）；`tools/dcu/s7_gate.sh` 的 **eager 与 graph**
   与 HF 逐 token 一致（参考 ids `[12095,13,576,6722,315,15344,374,21718,13,576,6722,315,17689,374,24081,13]`）；
   逐层扫描非有限=0；5 个 gfx906 测试文件全过；`gate-211.sh` 的 `IMPORT_OK_ALL4` + 四个 `.so` 只含 gfx906；
4. **锁死回归**：在 `tests/kernels/` 下加一个**只在 gfx906+DTK 生效**（其余平台 skip）单测，直接断言
   `_C.silu_and_mul` 的 fp16 输出有限且等于参考值；
5. **Python 侧修复去留**：保留 `activation.py` 的回退作为兜底，注释说明"csrc 已根修，此处为兜底"，
   且第 4 步单测**直接测 `_C` 算子**。

### 1.2 硬约束（本步遵守情况见 §3.9）

禁止执行任何自编译 HIP 独立可执行程序（本步只做 `hipcc -E/-S`、`ninja`、`python`/`vllm`/`triton`、`docker exec`）；
不 start/kill/exec 生产容器 `qwen38-vllm`；不动 `z100-build`（2.10 基线）；不替换 DTK 库；产物只落 `/mnt/game`；
容器 `--memory=8g`（本步新建的任何容器都没有，只复用既有容器）、`MAX_JOBS≤4`；
不改 `docs/design/*`、`docs/z100-todolist.md`、`docs/result/README.md`。

**写作用域**：`csrc/libtorch_stable/activation_kernels.cu`、`vllm/model_executor/layers/activation.py`
（仅注释）、`tests/kernels/test_gfx906_packed_silu_fp16.py`、`tools/dcu/s7c_direct_check.py`、本归档。

---

## 2. 实际改动

### 2.1 改动清单

| 文件 | 位置 | 改动摘要 |
|---|---|---|
| `csrc/libtorch_stable/activation_kernels.cu` | **11** | `#include <cstring>`（`std::strncmp`） |
| 同上 | **240-274** | 新增 host 侧判定 `dtk_gfx906_fp16_no_packed(torch::headeronly::ScalarType)`：`#if defined(__DTK_ARCH__)` 真实实现（`dtype == Half` 且 `gcnArchName` 前缀 `gfx906`），`#else` 返回常量 `false` |
| 同上 | **296-297** | `LAUNCH_ACTIVATION_GATE_KERNEL`（silu_and_mul / clamp / mul_and_silu / gelu_and_mul / gelu_tanh_and_mul）的 `use_vec` 加守卫 |
| 同上 | **1003-1004** | `LAUNCH_ACTIVATION_GATE_KERNEL_WITH_PARAM`（fatrelu_and_mul / swigluoai_and_mul / situ_and_mul）的 `use_vec` 加守卫 |
| 同上 | **1407-1408** | `LAUNCH_ACTIVATION_KERNEL`（gelu_new / gelu_fast / gelu_quick / relu_squared 等逐元素激活）的 `use_vec` 加守卫 |
| `csrc/activation_kernels.cu` | **无改动** | 见 §2.3：该孪生文件**根本没有向量化分支**（`grep -cE "use_vec\|PackedVec\|__half2\|packed_" = 0`），且其 `silu_and_mul` **未被任何 `TORCH_LIBRARY` 注册** |
| `vllm/model_executor/layers/activation.py` | **127-135**、**188-189** | **仅注释**：说明 csrc 已根修、此处是兜底、不能拦直调点、回归单测必须打裸算子（S7b 的 32 行逻辑逐字未动；本步该文件 `git diff --stat` = 43 insertions，其中 32 行来自 S7b、11 行是本步注释） |
| `tests/kernels/test_gfx906_packed_silu_fp16.py` | 新增 | DTK+gfx906 专属回归单测，3 个用例（见 §2.4） |
| `tools/dcu/s7c_direct_check.py` | 新增 | 裸 `torch.ops._C` 三段探针 + bf16/fp32/mul_and_silu 控制组（§3.3 的原始输出来源） |

`git diff --stat`：`csrc/libtorch_stable/activation_kernels.cu | 46 ++++-`、`vllm/.../activation.py | 43 ++++`
（2 files changed, 86 insertions(+), 3 deletions(-)）。

确切 csrc 改动（`git diff -- csrc/libtorch_stable/activation_kernels.cu`，删节）：

```diff
 #include <cmath>
+#include <cstring>
 #include <type_traits>
@@ -236,6 +237,42 @@ packed_gelu_tanh_kernel(const packed_t& val, const float /*alpha*/) {
 }  // namespace vllm
 
+// Precision guard for the gfx906 + DTK 26.04 fp16 packed-activation miscompile.
+// ... (守卫理由，见 §2.2) ...
+#if defined(__DTK_ARCH__)
+static inline bool dtk_gfx906_fp16_no_packed(
+    torch::headeronly::ScalarType dtype) {
+  if (dtype != torch::headeronly::ScalarType::Half) {
+    return false;
+  }
+  return std::strncmp(get_device_prop()->gcnArchName, "gfx906", 6) == 0;
+}
+#else
+static inline bool dtk_gfx906_fp16_no_packed(
+    torch::headeronly::ScalarType dtype) {
+  (void)dtype;
+  return false;
+}
+#endif
+
 // Launch activation and gating kernel.
@@ -256,7 +293,8 @@
   int vec_size = support_vec / input.element_size();                           \
-  const bool use_vec = (d % vec_size == 0);                                    \
+  const bool use_vec =                                                         \
+      (d % vec_size == 0) && !dtk_gfx906_fp16_no_packed(dtype);                \
（同样两处：`:1000`、`:1404` 附近，共 3 处）
```

### 2.2 守卫为什么精确（以及为什么**不能**照抄 S7b §5.1 的 `is_gfx906()` 占位符）

S7b §5.1 的草案写的是 `!(input.element_size() == 2 && is_gfx906())`。本步落地时改成
**`dtype == Half`（只 fp16，不含 bf16）+ 两层限定**，理由是：

1. **`__gfx906__` 在 host pass 里不存在**（任务书原文建议的 `defined(__gfx906__) && defined(__DTK_ARCH__)`
   对 host 代码恒为假）。实测（§3.1(1)）：`hipcc -x hip --cuda-host-only -S` 下
   `#ifdef __gfx906__ return 906; #else return 0;` 编出的是 `xorl %eax,%eax`（返回 0）；
   同一次编译的 `#ifdef __DTK_ARCH__` 编出 `movl $1,%eax`（返回 1）。
   而 `use_vec` 是 **host** 侧的 launch 决策，所以只能靠 `__DTK_ARCH__` + 运行时 `gcnArchName`。
2. **`__DTK_ARCH__` 是 DTK 独有的**：它由 `ATen/dtk_macros.h` 定义，而该头是 **DTK 的 hipify 注入**到每个
   生成文件里的（本仓库 `csrc/quantization/gptq/compat.cuh:28-36` 已经依赖同一事实）。社区 ROCm 没有这个
   头、CUDA 更不会有 ⇒ 社区构建里 `#else` 分支生效，`!false` 被优化掉，
   **host 汇编与改动前逐指令相同**（§3.1(2) 的负向控制）。
3. **运行时 `gcnArchName` 前缀判定**把 DTK 构建的其它架构（例如 DTK 上编 gfx928）排除在外，只有
   `gfx906*` 关向量化。用 `dtype == Half` 而不是 `element_size() == 2`，是为了**不动 bf16**
   （S7b §3.4(a) 已证 bf16 实例正确，且 gfx906 上 bf16 是软件模拟、本就慢）。
4. **为什么不"修 packed 路径本身"**：S7b §3.4(d)/§5.2 已证缺陷**非确定性**（同输入同排布，93% 稳定 +
   7% 抖动）且 live kernel 的 ISA 级机理**未定性**（只到"与 S6 的 VALU→SFU hazard 同类"的可疑级）。
   在机理未明的情况下改 packed 数学/加等待槽属于盲改；**关掉该分支走已被证正确的标量实现**是当前
   唯一有同等强度验证的等价修法。代价是 fp16 激活在 gfx906 上不用 128-bit 向量读，见 §3.8 的性能数据。

### 2.3 孪生文件 `csrc/activation_kernels.cu` 为什么不用改（证据）

```console
$ grep -cE "use_vec|PackedVec|__half2|packed_compute|packed_" csrc/activation_kernels.cu
0                                   # 该文件是标量-only 的旧实现
$ grep -n "silu_and_mul" csrc/libtorch_stable/torch_bindings.cpp csrc/torch_bindings.cpp
csrc/libtorch_stable/torch_bindings.cpp:605:  ops.def("silu_and_mul(Tensor! result, Tensor input) -> ()");
csrc/libtorch_stable/torch_bindings.cpp:876:  ops.impl("silu_and_mul", TORCH_BOX(&silu_and_mul));
                                    # 旧 _C 的 torch_bindings.cpp 里没有任何 silu_and_mul 注册
```

即：`torch.ops._C.silu_and_mul` **只**由 `_C_stable_libtorch` 注册（与 S7b §3.5 的 profiler 结论一致），
根目录的 `csrc/activation_kernels.cu` 里那个标量 `vllm::silu_and_mul` 已无注册点（死代码），且它本来就
没有向量化分支 ⇒ **无需改动**。

### 2.4 新增回归单测（门禁 4）

`tests/kernels/test_gfx906_packed_silu_fp16.py`：

- `pytestmark = pytest.mark.skipif(not _dtk_gfx906(), ...)`，`_dtk_gfx906()` = `torch.cuda` 可用 +
  `current_platform.is_rocm()` + `is_hygon_dtk()` + `on_gfx906()` ⇒ **只在 DTK + gfx906 生效，其余平台 skip**；
- 三个用例全部打**裸算子**（不经 `CustomOp`，因此不受 Python 兜底掩盖）：
  1. `_C.silu_and_mul` fp16、`d=3072`（8 的倍数 ⇒ 正常会选向量分支）、`|gate|≤5.9`、**连打 8 次**，
     每次都断言 `torch.isfinite(out).all()` 且 `assert_close(out.float(), silu(gate)*up, atol=5e-2, rtol=2e-2)`；
  2. `_C.silu_and_mul_with_clamp(out, x, 5.9, 1.0, 0.0)` 同上；
  3. `apply_moe_activation(MoEActivation.SILU, ...)`（`fused_moe/activation.py` 的直调点）同上。

**门禁 5 的处理**：Python 回退保留，但注释明确写"csrc 已根修、此处是兜底、它拦不到直调点、
回归单测必须打裸算子"（`activation.py:127-135`、`:188-189`）。

---

## 3. 执行过程与原始证据

### 3.1 守卫的编译期证据与负向控制（`hipcc`，只编译/预处理，不执行）

**(1) host pass 里 `__gfx906__` 不存在、`__DTK_ARCH__` 存在**（决定性的宏可用性实验）：

```console
$ docker exec z100-build-211 bash -lc 'cd /mnt/game/port/s7c-probe && cat > p1.cu <<"EOF"
#include <ATen/dtk_macros.h>
extern "C" int host_probe() {
#ifdef __gfx906__
  return 906;
#else
  return 0;
#endif
}
extern "C" int dtk_probe() {
#ifdef __DTK_ARCH__
  return 1;
#else
  return 0;
#endif
}
EOF
hipcc -x hip --cuda-host-only -DUSE_ROCM -I$TORCH_INC -S -O0 -o p1.s p1.cu'
$ grep -A8 "host_probe:" p1.s
host_probe:
	xorl	%eax, %eax          # <= __gfx906__ 未定义：返回 0
$ grep -A8 "dtk_probe:" p1.s
dtk_probe:
	movl	$1, %eax            # <= __DTK_ARCH__ 已定义：返回 1
```

同一次编译的 **device pass** 里 `__gfx906__` 是定义的（上一轮 `-dM` 实测）：

```console
$ hipcc -x hip --cuda-device-only -E -dM --offload-arch=gfx906 m.cu | grep -iE "gfx|HIP_DEVICE"
#define __GFX9__ 1
#define __HIP_DEVICE_COMPILE__ 1
#define __amdgcn_processor__ "gfx906"
#define __amdgcn_target_id__ "gfx906"
#define __gfx906__ 1
```

**(2) 负向控制：把 `__DTK_ARCH__` 关掉时，host 汇编与改动前逐指令相同**（社区 ROCm/CUDA 路径不变）：

```console
# p3.cu: `use_vec_orig(d,vec) = (d%vec==0)`；`use_vec_new(d,vec,dtype) = (d%vec==0) && !guard(dtype)`
# guard() 就是本步落地的那段（#if defined(__DTK_ARCH__) ... #else return false; #endif）
$ hipcc -x hip --cuda-host-only -O2            -S -o p3-nonDTK.s p3.cu     # 不定义 __DTK_ARCH__
$ hipcc -x hip --cuda-host-only -O2 -D__DTK_ARCH__=520 -S -o p3-DTK.s p3.cu
$ norm() { awk "/^$1:/{f=1;next} f&&/^\t.size\t$1/{exit} f{print}" "$2" \
             | grep -vE "^\s*\.(Lfunc_end|cfi_startproc|cfi_endproc)"; }
$ norm use_vec_orig p3-nonDTK.s > n_orig.txt; norm use_vec_new p3-nonDTK.s > n_new.txt
$ diff n_orig.txt n_new.txt && echo "IDENTICAL: non-DTK host codegen of use_vec is unchanged"
IDENTICAL: non-DTK host codegen of use_vec is unchanged
# 规范化后的指令体（两边完全一样）：
	movl	%edi, %eax
	cltd
	idivl	%esi
	xorl	%eax, %eax
	testl	%edx, %edx
	sete	%al
	retq
# 而 -D__DTK_ARCH__=520 的版本里出现：
	callq	_Z8gcn_namev@PLT
	leaq	.L.str(%rip), %rsi
	movl	$6, %edx
	callq	strncmp@PLT
```

**(3) 真实构建里 `__DTK_ARCH__` 确实为定义**（不是靠我手工 `-D`）：

```console
$ head -6 /mnt/game/port/build-211/temp/csrc/libtorch_stable/activation_kernels.hip
// !!! This is a file automatically generated by hipify!!!
// Replaces the #include <ATen/dtk_macros.h> that DTK's patched
// torch.utils.hipify injects; see cmake/hipify.py. ...
#if defined(USE_ROCM)
#define __DTK_ARCH__ 520
#define DTK_VERSION 11800
...
$ hipcc -x hip --cuda-host-only -E -DUSE_ROCM=1 -dM <generated .hip> | grep -E "DTK_ARCH|USE_ROCM|gfx906"
#define USE_ROCM 1
#define __DTK_ARCH__ 520                    # host pass 可见；__gfx906__ 不出现
$ grep "^  DEFINES\|^  FLAGS" build.ninja（该 TU 边）
  DEFINES = ... -DUSE_ROCM -D__HIP_PLATFORM_AMD__=1 ...
  FLAGS = ... --offload-arch=gfx906 ... -DUSE_ROCM=1 ...
```

**(4) 落地后 `.so` 的静态证据：guard 真的编进了 launch 路径**（老 `.so` 为 0，新 `.so` 为 1）：

```console
$ objdump -d -C --disassemble="silu_and_mul(torch::stable::Tensor&, torch::stable::Tensor&)" \
      vllm/_C_stable_libtorch.abi3.so | grep -c "call.*strncmp@plt"
1                                    # NEW
$ 同上，对 /mnt/game/port/s7c/_C_stable_libtorch.abi3.so.installed-before（旧安装件）
0                                    # OLD
# 新 .so 里含该 guard 的 host 函数（共 10 个激活入口；top_k_per_row_decode 的 strncmp 是既有代码，旧 .so 也有）：
fatrelu_and_mul / gelu_and_mul / gelu_fast / gelu_new / gelu_quick / gelu_tanh_and_mul /
mul_and_silu / relu_squared / silu_and_mul_clamp / silu_and_mul
# 旧 .so 的同名列表只有：top_k_per_row_decode（本步未触及）
```

### 3.2 增量重建与安装

**(a) 同步源码到构建副本**（`/src` 是工作区只读挂载）：

```console
$ docker exec z100-build-211 bash -lc 'cp /src/csrc/libtorch_stable/activation_kernels.cu \
    /mnt/game/port/gfx906-vllm-211/csrc/libtorch_stable/activation_kernels.cu; md5sum ...'
5793025ff0165fe8a342bc04de097da2  /src/csrc/libtorch_stable/activation_kernels.cu
5793025ff0165fe8a342bc04de097da2  /mnt/game/port/gfx906-vllm-211/csrc/libtorch_stable/activation_kernels.cu
1ac933298c58c72598311b0800b6b801  （回滚副本 /mnt/game/port/s7c/activation_kernels.cu.s7c-orig）
```

**(b) 增量 `ninja`（目标 `_C_stable_libtorch`，`MAX_JOBS=4`）——只重跑 hipify + 1 TU + 1 链接**：

```console
$ MAX_JOBS=4 LOG=/mnt/game/port/s7c/build-s7c.log bash /mnt/game/port/build-211.sh _C_stable_libtorch
[1/30] Running hipify on all extension source files.
[2/3] Building HIP object CMakeFiles/_C_stable_libtorch.dir/csrc/libtorch_stable/activation_kernels.hip.o
[3/3] Linking HIP shared module _C_stable_libtorch.abi3.so
----- ninja rc=0 2026-10-08T21:25:02+08:00 -----
# .ninja_log 里的真实耗时（ms）：
   hipify_all.stamp                              27 -> 6430    （~6.4 s）
   libtorch_stable/activation_kernels.hip.o    6432 -> 15926   （TU 9.49 s）
   _C_stable_libtorch.abi3.so                 15926 -> 16262   （链接 0.34 s）
$ ninja -n
ninja: no work to do.                # 已 clean
```

**(c) S8 抢占检查 + 原子安装**。安装前实测 S8 的引擎**正在启动**（已 up 2–3 分钟、8003 尚未监听），
因此**不用** `install-211.sh` 的"就地 copy + 事后 `RPATH_CHANGE`"两步（那会留下一个约 1 s 的窗口），
改为**先 `DESTDIR` 安装到 staging、再用 `mv` 原子替换**（`cmake_install.cmake` 显示 `NEW_RPATH` 为
**空串**，所以 `DESTDIR` 前缀不影响最终字节，与原流程产物一致）：

```console
$ docker exec z100-s8 bash -lc 'ps -o pid,etime,cmd -p 6453'
   6453  02:13  python -c import torch; ... serve ... --port 8003 ...
$ docker exec z100-build-211 bash -lc 'DESTDIR=/mnt/game/port/s7c/stage cmake --install \
     /mnt/game/port/build-211/temp --prefix /mnt/game/port/gfx906-vllm-211 --component _C_stable_libtorch'
-- Installing: /mnt/game/port/s7c/stage/.../vllm/_C_stable_libtorch.abi3.so
-- Set non-toolchain portion of runtime path of "..." to ""
$ # 同一文件系统内 mv = rename(2)，读者只会看到旧 inode 或新 inode，永远不会看到半个文件
$ mv -f <stage>/vllm/_C_stable_libtorch.abi3.so /mnt/game/port/gfx906-vllm-211/vllm/_C_stable_libtorch.abi3.so
$ md5sum /mnt/game/port/s7c/_C_stable_libtorch.abi3.so.installed-before   # 回滚件
cc2aca69ca9a9afcb67b7428ede181b1
$ md5sum /mnt/game/port/gfx906-vllm-211/vllm/_C_stable_libtorch.abi3.so   # 新安装件
93bf7b4a391ac97e2f67cc7141633cb1   （40,691,216 B，原 cc2aca69…/40,688,656 B）
```

**S8 未受影响**（安装后 21:39 复查：引擎仍活着，且 `/proc/6453/maps` 里**还没有** `_C_stable_libtorch`，
说明它尚未 dlopen 该库 ⇒ 它之后拿到的是修好的一份）：

```console
$ docker exec z100-s8 bash -lc 'grep -E "_C_stable_libtorch|_C\.abi3" /proc/6453/maps'
（无输出；只映射了 _moe_C_stable_libtorch.abi3.so）
$ ps -o pid,etime -p 6453  →  15:57   （未崩溃）
```

**(d) 产物身份（新/旧、raw/installed 全表）**：

| 产物 | raw 旧（S4c 基线） | raw 新 | installed 旧 | **installed 新** |
|---|---|---|---|---|
| `_C.abi3.so` | `048736b1…` | **`048736b1…`（未变）** | `6fba0596…` | **`6fba0596…`（未变）** |
| `_rocm_C.abi3.so` | `019b327b…` | `019b327b…` | `5d1e50e1…` | `5d1e50e1…` |
| `_C_stable_libtorch.abi3.so` | `7068229e…` | **`ad6b1d66…`** | `cc2aca69…` | **`93bf7b4a391ac97e2f67cc7141633cb1`** |
| `_moe_C_stable_libtorch.abi3.so` | `af01ed9b…` | `af01ed9b…` | `996b282c…` | `996b282c…` |

> ⚠️ **任务书原文的预期不成立**：它写"`_C` 的旧基线 `048736b1…` 会变"。实际被修的 TU
> （`csrc/libtorch_stable/activation_kernels.cu`）编进的是 **`_C_stable_libtorch`**，
> `_C.abi3.so` **逐字节不变**（S7b §3.5 的 profiler 已经指出 live kernel 来自 libtorch_stable，
> 本步 §2.3 又从注册面复证）。S9 若要记"已修 `_C`"，应记 **`_C_stable_libtorch` 的新 md5**。

### 3.3 复验 1：裸 `_C` 算子（改前 / 改后，同一脚本 `tools/dcu/s7c_direct_check.py`）

改前的对照不是靠换回源码重建，而是**用旧安装件做了一棵独立的前缀树**
`/mnt/game/port/s7c/pre-fix`（整树 `cp -a` 后把 `vllm/_C_stable_libtorch.abi3.so` 换回
`cc2aca69…`），因此**没有对 canonical 副本做第二次替换**，两份证据互不污染。

```console
$ # 改前（VLLM_DIR=/mnt/game/port/s7c/pre-fix，_C_stable_libtorch md5=cc2aca69…）
[s7c] device=Z100 gcnArchName=gfx906:sramecc+:xnack-
[s7c] _C=/mnt/game/port/s7c/pre-fix/vllm/_C.abi3.so md5=6fba05966e153b3d04771d6960c782bb size=78102728
[s7c] _C_stable_libtorch=.../vllm/_C_stable_libtorch.abi3.so md5=cc2aca69ca9a9afcb67b7428ede181b1 size=40688656
=== _C.silu_and_mul fp16 8x6144 (d=3072), 20 launches ===
[s7c-rep] trial= 0 nonfinite=449    nbad>1e-2=1938   maxerr=inf
[s7c-rep] trial= 1 nonfinite=455    nbad>1e-2=1953   maxerr=inf
[s7c-rep] trial= 2 nonfinite=446    nbad>1e-2=1940   maxerr=inf
[s7c-rep] trial= 3 nonfinite=443    nbad>1e-2=1909   maxerr=inf
[s7c-rep] trial= 4 nonfinite=424    nbad>1e-2=1868   maxerr=inf
[s7c-rep] trial= 5 nonfinite=429    nbad>1e-2=1864   maxerr=inf
[s7c-rep] trial= 6 nonfinite=429    nbad>1e-2=1874   maxerr=inf
[s7c-rep] trial= 7 nonfinite=453    nbad>1e-2=1934   maxerr=inf
[s7c-rep] trial= 8 nonfinite=445    nbad>1e-2=1926   maxerr=inf
[s7c-rep] trial= 9 nonfinite=451    nbad>1e-2=1937   maxerr=inf
[s7c-rep] trial=10 nonfinite=445    nbad>1e-2=1923   maxerr=inf
[s7c-rep] trial=11 nonfinite=443    nbad>1e-2=1911   maxerr=inf
[s7c-rep] trial=12 nonfinite=447    nbad>1e-2=1934   maxerr=inf
[s7c-rep] trial=13 nonfinite=449    nbad>1e-2=1929   maxerr=inf
[s7c-rep] trial=14 nonfinite=446    nbad>1e-2=1914   maxerr=inf
[s7c-rep] trial=15 nonfinite=443    nbad>1e-2=1909   maxerr=inf
[s7c-rep] trial=16 nonfinite=451    nbad>1e-2=1937   maxerr=inf
[s7c-rep] trial=17 nonfinite=443    nbad>1e-2=1925   maxerr=inf
[s7c-rep] trial=18 nonfinite=442    nbad>1e-2=1931   maxerr=inf
[s7c-rep] trial=19 nonfinite=447    nbad>1e-2=1929   maxerr=inf
[s7c-rep] masks identical to trial 0: 0/19; nonfinite across 20 runs: min=424 max=455 sum=8880
=== _C.silu_and_mul_with_clamp fp16 (limit=5.9), 1 launch ===
[s7c-clamp] nonfinite=107/24576 nbad>1e-2=719 maxerr=inf
=== apply_moe_activation(SILU) fp16, 1 launch ===
[s7c-moe] nonfinite=1907/98304 nbad>1e-2=7612 maxerr=inf
=== control: bf16 / fp32 / mul_and_silu must stay exact ===
[s7c-ctl] silu_and_mul torch.bfloat16 nonfinite=0 nbad>1e-2=5360 maxerr=0.18663
[s7c-ctl] silu_and_mul torch.float32 nonfinite=0 nbad>1e-2=0 maxerr=0
[s7c-ctl] mul_and_silu fp16 (half-swapped) nonfinite=0 nbad>1e-2=303 maxerr=0.0247459
[s7c] DONE

$ # 改后（canonical /mnt/game/port/gfx906-vllm-211，_C_stable_libtorch md5=93bf7b4a…）
[s7c] _C=/mnt/game/port/gfx906-vllm-211/vllm/_C.abi3.so md5=6fba05966e153b3d04771d6960c782bb size=78102728
[s7c] _C_stable_libtorch=.../vllm/_C_stable_libtorch.abi3.so md5=93bf7b4a391ac97e2f67cc7141633cb1 size=40691216
=== _C.silu_and_mul fp16 8x6144 (d=3072), 20 launches ===
[s7c-rep] trial= 0 nonfinite=0      nbad>1e-2=303    maxerr=0.0247459
[s7c-rep] trial= 1 nonfinite=0      nbad>1e-2=303    maxerr=0.0247459
[s7c-rep] trial= 2 nonfinite=0      nbad>1e-2=303    maxerr=0.0247459
[s7c-rep] trial= 3 nonfinite=0      nbad>1e-2=303    maxerr=0.0247459
[s7c-rep] trial= 4 nonfinite=0      nbad>1e-2=303    maxerr=0.0247459
[s7c-rep] trial= 5 nonfinite=0      nbad>1e-2=303    maxerr=0.0247459
[s7c-rep] trial= 6 nonfinite=0      nbad>1e-2=303    maxerr=0.0247459
[s7c-rep] trial= 7 nonfinite=0      nbad>1e-2=303    maxerr=0.0247459
[s7c-rep] trial= 8 nonfinite=0      nbad>1e-2=303    maxerr=0.0247459
[s7c-rep] trial= 9 nonfinite=0      nbad>1e-2=303    maxerr=0.0247459
[s7c-rep] trial=10 nonfinite=0      nbad>1e-2=303    maxerr=0.0247459
[s7c-rep] trial=11 nonfinite=0      nbad>1e-2=303    maxerr=0.0247459
[s7c-rep] trial=12 nonfinite=0      nbad>1e-2=303    maxerr=0.0247459
[s7c-rep] trial=13 nonfinite=0      nbad>1e-2=303    maxerr=0.0247459
[s7c-rep] trial=14 nonfinite=0      nbad>1e-2=303    maxerr=0.0247459
[s7c-rep] trial=15 nonfinite=0      nbad>1e-2=303    maxerr=0.0247459
[s7c-rep] trial=16 nonfinite=0      nbad>1e-2=303    maxerr=0.0247459
[s7c-rep] trial=17 nonfinite=0      nbad>1e-2=303    maxerr=0.0247459
[s7c-rep] trial=18 nonfinite=0      nbad>1e-2=303    maxerr=0.0247459
[s7c-rep] trial=19 nonfinite=0      nbad>1e-2=303    maxerr=0.0247459
[s7c-rep] masks identical to trial 0: 19/19; nonfinite across 20 runs: min=0 max=0 sum=0
=== _C.silu_and_mul_with_clamp fp16 (limit=5.9), 1 launch ===
[s7c-clamp] nonfinite=0/24576 nbad>1e-2=303 maxerr=0.0247459
=== apply_moe_activation(SILU) fp16, 1 launch ===
[s7c-moe] nonfinite=0/98304 nbad>1e-2=1218 maxerr=0.0235214
=== control: bf16 / fp32 / mul_and_silu must stay exact ===
[s7c-ctl] silu_and_mul torch.bfloat16 nonfinite=0 nbad>1e-2=5360 maxerr=0.18663
[s7c-ctl] silu_and_mul torch.float32 nonfinite=0 nbad>1e-2=0 maxerr=0
[s7c-ctl] mul_and_silu fp16 (half-swapped) nonfinite=0 nbad>1e-2=303 maxerr=0.0247459
[s7c] DONE
```

要点：**改后 20 次全部 0 非有限、且 20 次的坏元素掩码 19/19 完全一致（退化为确定性）**，
`maxerr` 从 `inf` 变成 **0.0247**（fp16 舍入量级：`|silu(gate)*up|` 上界约 35，0.0247 属正常）。
bf16 / fp32 / `mul_and_silu` 三组对照**改前后逐位相同**（本步没有把关向量化的范围扩到 fp16 之外）。

> 过程备注：首版脚本把 clamp 的实参顺序写错（schema 是
> `silu_and_mul_with_clamp(result, input, limit, alpha=1.0, beta=0.0)`，我误传成 `(1.0, 0.0, 5.9)`），
> 得到一份无意义的 clamp 数字；已修正为 `(5.9, 1.0, 0.0)` 并**重跑改前/改后两份**，
> 上表全部来自修正后的 `before-fix2.log` / `after-fix.log`。

### 3.4 复验 2：端到端 A/B 矩阵（`tools/dcu/s7_gate.sh`，单卡 2，Qwen3-0.6B fp16，greedy，`max_tokens=16`）

为了把"csrc 修复本身"与"Python 兜底"分开，本步造了 4 棵前缀树（都在 `/mnt/game/port/s7c/`）：

| 前缀树 | `_C_stable_libtorch` | `activation.py` 兜底 | eager 输出 |
|---|---|---|---|
| `pre-fix-csrc-only` | 旧 `cc2aca69…` | **无**（HEAD 版） | `[1023,1023,…,1023]`（16 个）❌ `'locklocklocklock…'` |
| `csrc-only` | 新 `93bf7b4a…` | **无**（HEAD 版） | `[12095, 13, …24081, 13]` ✅ |
| `gfx906-vllm-211`（canonical） | 新 `93bf7b4a…` | 有（S7b） | 同上 ✅ |
| `pre-fix` | 旧 `cc2aca69…` | 有（S7b） | （未跑 e2e，仅用于 §3.3 的算子级改前） |

```console
$ # (1) 改前：旧 kernel + 无 Python 兜底
[s7] engine ready in 23.6s mode=eager
[s7] S7_IDS_EAGER=[1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023]
[s7] S7_TEXT_EAGER='locklocklocklocklocklocklocklocklocklocklocklocklocklocklocklock'
[s7] S7_DECODE_EAGER TOKENS=128 SECONDS=4.6696 MS_PER_TOKEN=36.482
S7_GATE_RC=0

$ # (2) 改后：新 kernel + 无 Python 兜底（证明 csrc 修复本身充分）
INFO [gpu_worker.py:719] Available KV cache memory: 27.18 GiB
[s7] engine ready in 23.0s mode=eager
[s7] S7_IDS_EAGER=[12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315, 17689, 374, 24081, 13]
[s7] S7_TEXT_EAGER=' Paris. The capital of Italy is Rome. The capital of Spain is Madrid.'
[s7] S7_DECODE_EAGER TOKENS=128 SECONDS=4.4682 MS_PER_TOKEN=34.908
S7_GATE_RC=0

$ # (3) canonical（新 kernel + Python 兜底）
INFO [gpu_worker.py:719] Available KV cache memory: 27.14 GiB
[s7] engine ready in 23.6s mode=eager
[s7] S7_IDS_EAGER=[12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315, 17689, 374, 24081, 13]
[s7] S7_DECODE_EAGER TOKENS=128 SECONDS=4.4569 MS_PER_TOKEN=34.820
S7_GATE_RC=0

$ # (4) graph（canonical）
INFO [gpu_worker.py:719] Available KV cache memory: 26.75 GiB
[s7] engine ready in 63.1s mode=graph
[s7] S7_IDS_GRAPH=[12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315, 17689, 374, 24081, 13]
[s7] S7_TEXT_GRAPH=' Paris. The capital of Italy is Rome. The capital of Spain is Madrid.'
[s7] S7_DECODE_GRAPH TOKENS=128 SECONDS=1.6855 MS_PER_TOKEN=13.168
S7_GATE_RC=0

$ # (5) HF transformers 参考
[s7c-hf] HF_IDS= [12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315, 17689, 374, 24081, 13]
```

⇒ **eager(新 kernel，无兜底) == eager(新 kernel，有兜底) == graph == HF**，逐 token 相同；
改动前的同一 eager 配置则是 `'lock'×16`。**csrc 修复既充分（无兜底也正确）又必要（旧 kernel 无兜底必错）。**

### 3.5 复验 3：逐层扫描非有限 = 0（用 `csrc-only` 树，即**没有 Python 兜底**）

```console
$ S7B_SCAN=/mnt/game/port/s7c/layer_scan-csrc-only.jsonl \
  python -u tools/dcu/s7b_scan.py /mnt/game/models/Qwen3-0.6B csrc-only "enforce_eager=True"
[s7b] engine ready in 23.3s tag=csrc-only
[s7b] worker probe=[{'modules': 373, 'info': {'mode': 'CompilationMode.NONE', 'custom_ops': ['+sparse_attn_indexer', 'all'],
        'enabled_custom_ops': ['apply_rotary_emb', 'rms_norm', 'rotary_embedding', 'silu_and_mul'], ...}}]
[s7b] hooks installed on 373 modules
[s7b] S7B_IDS[csrc-only]=[12095, 13]
[s7b] S7B_TEXT[csrc-only]=' Paris.'
[s7b] SCAN=/mnt/game/port/s7c/layer_scan-csrc-only.jsonl modules=396 nonfinite_modules=0
[s7b] S7B_SCAN_DONE
```

关键旁证：`enabled_custom_ops` 里**确实包含 `silu_and_mul`** ⇒ eager 路径真的在用 `_C` 算子，
而这一轮**没有 Python 兜底**（`csrc-only` 树的 `activation.py` 是 HEAD 版），
所以"0 非有限"只能由 csrc 修复解释。（对照 S7b §3.2：改前同一扫描 385/396 模块非有限。）

### 3.6 复验 4：回归测试计数（canonical 副本，卡 2，先 `import torch` + `--noconftest`）

```console
##### tests/kernels/test_gfx906_packed_silu_fp16.py            （本步新增）
3 passed, 17 warnings in 18.89s                                rc=0
##### tests/kernels/attention/test_triton_unified_attention_gfx906.py
39 passed, 1 warning in 16.31s                                 rc=0
##### tests/kernels/attention/test_triton_unified_attn_gfx906_v1.py
12 passed, 1 warning in 16.44s                                 rc=0
##### tests/kernels/quantization/test_gguf_fixed_cols_gfx906.py
13 passed, 3 warnings in 132.90s (0:02:12)                     rc=0
##### tests/kernels/mamba/test_fused_sigmoid_gating.py
5 passed, 1 warning in 14.46s                                  rc=0
##### tests/kernels/mamba/test_fused_recurrent_packed_decode.py   （S5/S6 基线 5 的那个）
5 passed, 1 warning in 14.96s                                  rc=0
##### tests/kernels/test_fused_recurrent_packed_decode.py         （同名但另一个文件，额外跑）
8 passed, 1 warning in 18.44s                                  rc=0
```

**新增单测的负向控制（证明它真的能锁住回归）**：对 **旧 `.so`**（`pre-fix` 树）跑同一文件：

```console
##### new regression test against PRE-FIX _C (must FAIL)
>           assert torch.isfinite(out).all(), f"launch {trial} produced inf/NaN"
E           AssertionError: launch 0 produced inf/NaN
E       AssertionError: assert tensor(False, device='cuda:0')
3 failed, 17 warnings in 18.89s
```

⇒ 同一个测试文件：旧 `.so` **3 failed**、新 `.so` **3 passed**。因为用例打的是**裸算子**，
Python 兜底无法掩盖 csrc 回退，符合门禁 5 的要求。

### 3.7 复验 5：`gate-211.sh` 与 `IMPORT_OK_ALL4` + 四个 `.so` 只含 gfx906

```console
$ docker exec z100-build-211 bash /mnt/game/port/gate-211.sh      # 完整日志 /mnt/game/port/s7c/s7c-gate-211.log
triton  : 3.5.0                       # 与 S4c 基线 gate-211-s4c.log 逐字相同（gate 环境不含 S6 自编 triton）
TORCH_PIN_OK  True
TRITON_PIN_OK True                    # importlib.metadata: 3.6.0+das185.dtk2604
rc=0
IMPORT_OK /mnt/game/port/gfx906-vllm-211/vllm/_C.abi3.so
rc=0
rc=134                                # 2b) "import vllm 先于 torch" 的已知 SIGABRT（硬约束 8），S4c 基线同值
_C.abi3.so                             archs: gfx906
_rocm_C.abi3.so                        archs: gfx906
vendor _C (reference)                  archs: gfx906 gfx926 gfx928 gfx936
ROCM_SKINNY_GEMM_ENABLED = False
===== gate done 2026-10-08T21:38:30+08:00 =====

$ . /mnt/game/port/setenv-211.sh; cd "$REPO_DIR"; \
  python -c "import torch, vllm, vllm._C, vllm._rocm_C, vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch; print('IMPORT_OK_ALL4')"
IMPORT_OK_ALL4     rc=0
$ for so in vllm/*.so（四个）; do strings -a $so | grep -oE "amdgcn-amd-amdhsa--gfx[0-9a-f]+" | sed 's/.*--//' | sort -u; done
_C.abi3.so                             gfx906 | n_distinct=1
_rocm_C.abi3.so                        gfx906 | n_distinct=1
_C_stable_libtorch.abi3.so             gfx906 | n_distinct=1
_moe_C_stable_libtorch.abi3.so         gfx906 | n_distinct=1
```

### 3.8 性能口径（顺带记录，供 S9 用）

| 路径 | 本步实测（Qwen3-0.6B fp16，单卡 2） | 参照（S7/S7b） |
|---|---|---|
| eager（canonical，新 kernel + 兜底） | **34.82 ms/token** | S7b 修复后 37.4（Python `F.silu`+mul 兜底） |
| eager（`csrc-only`，新 kernel 走 csrc 标量分支，无兜底） | **34.91 ms/token** | — |
| eager（旧 kernel，无兜底） | 36.48 ms/token（数值是垃圾） | — |
| graph（canonical，`VLLM_COMPILE` + graph） | **13.17 ms/token** | S7：11.6 / S7b：32.7（不捕 graph） |
| KV cache | eager 27.14–27.18 GiB / graph 26.75 GiB | S7b：eager 27.18 |

即：**标量分支没有让 eager 变慢**（34.9 vs S7b 的 37.4，甚至更快，因为不用再走 Python 的
`F.silu`+乘法两步）；graph 路径本来就不走这些 `_C` 激活（`custom_ops=none`），13.17 ms/token 与 S7 同量级。
本机 4 卡共享、S8 同时在加载 27B INT8，绝对数字有噪声，量级结论可靠。

### 3.9 资源与硬约束自检

```console
$ docker ps -a --format '{{.Names}}\t{{.Status}}'
z100-eager Up 42 minutes     z100-s8 Up 43 minutes     z100-serve Up 2 hours
z100-triton Up 14 hours      z100-build-211 Up 15 hours  z100-build Up 21 hours
qwen38-vllm Exited (137) 14 hours ago      # 生产容器：全程只 docker ps/inspect，未 start/kill/exec
$ uptime
 13:39:06 up 1 day, 3:00,  load average: 1.12, 1.03, 1.09
$ docker exec z100-build-211 cat /sys/fs/cgroup/memory.events
low 0 high 0 max 2250 oom 0 oom_kill 0      # z100-build-211（既有 8g 上限）
$ docker exec z100-eager cat /sys/fs/cgroup/memory.current   → 2.64 GiB（上限 10 GiB，S7b 建的容器）
```

- **未执行任何自编译 HIP 独立可执行程序**：全程只有 `hipcc -E/-dM/-S`（预处理/汇编）、`ninja`、
  `objdump`、`cmake --install`、`python`/`vllm`/`triton`（GPU 内核经 torch 派发）、`docker exec`；
  0 挂死、0 SIGABRT（除 gate 里那条**既有**的 `import` 顺序 SIGABRT）、loadavg 全程 <1.2。
- 未 start/kill/exec `qwen38-vllm`（一直 `Exited (137)`）；未动 `z100-build`/`build-gfx906`（2.10 基线）；
  未替换任何 DTK 库；`MAX_JOBS=4`；本步新建产物只在 `/mnt/game/port/{s7c,build-211/temp}`；
  未改 `docs/design/*`、`docs/z100-todolist.md`、`docs/result/README.md`。

---

## 4. 门禁判定

| # | 门禁（本步任务书） | 判定 | 证据指向 |
|---|---|---|---|
| 1 | csrc 根修落地：DTK+gfx906 的 fp16 关闭 `use_vec` 走标量 | ✅ | §2.1（`activation_kernels.cu:240-274` + 3 处 `use_vec`）；§3.2(b) 重编只有该 TU；§3.7 静态 `strncmp` 证据 |
| 2 | **精确限定**（只有 DTK+gfx906 关；社区 ROCm/CUDA/其它架构逐字节不变）+ 负向控制 | ✅ | §3.1(1) `__gfx906__` 在 host pass 不存在、`__DTK_ARCH__` 存在；§3.1(2) 关掉宏时 host 汇编**逐指令相同**；§3.1(3) 真实构建确带 `__DTK_ARCH__ 520` 且 `--offload-arch=gfx906`；运行时 `gcnArchName` 前缀判定排除其它架构；bf16/fp32 对照组前后逐位相同（§3.3） |
| 3 | 增量重建 `_C`（只重编该 TU + 链接，~40 s） | ✅（**但目标是 `_C_stable_libtorch`，见偏差 a**） | §3.2(b)：hipify 6.4 s + TU 9.5 s + 链接 0.34 s，共 3 步；`ninja -n` → no work |
| 4 | 装进 canonical 副本；装前查 S8；记录新尺寸/md5 | ✅ | §3.2(c)：S8 引擎在启动 ⇒ 改**原子安装**（DESTDIR staging + `mv`）；`93bf7b4a…` / 40,691,216 B；旧件留档 `cc2aca69…` |
| 5 | `_C.silu_and_mul` fp16 20 次非有限=0 且误差合理 | ✅ | §3.3：20×`nonfinite=0`，`masks identical 19/19`，`maxerr=0.0247`（改前 424–455/次、sum 8880、`maxerr=inf`） |
| 6 | `_C.silu_and_mul_with_clamp` 一次 | ✅ | §3.3：`nonfinite=0/24576`，`maxerr=0.0247`（改前 107/24576、`inf`） |
| 7 | `apply_moe_activation(SILU)` 非有限 → 0（改前 1917/98304） | ✅ | §3.3：`[s7c-moe] nonfinite=0/98304`（改前 1907–1924/98304） |
| 8 | eager == graph == HF 逐 token 一致（参考 ids） | ✅ | §3.4：四路 ids 全等于参考序列；且**无 Python 兜底**的 `csrc-only` 树同样一致 |
| 9 | 逐层扫描非有限 = 0 | ✅ | §3.5：`modules=396 nonfinite_modules=0`（用无兜底树，`enabled_custom_ops` 含 `silu_and_mul`） |
| 10 | 5 个 gfx906 测试文件全过（39/12/13/5/5） | ✅ | §3.6：39/12/13/5/5，rc 全 0（另附同名 `tests/kernels/test_fused_recurrent_packed_decode.py` 8 过） |
| 11 | `gate-211.sh` 的 `IMPORT_OK_ALL4` + 四个 `.so` 只含 gfx906 | ✅ | §3.7：`IMPORT_OK_ALL4` rc=0；四个 `n_distinct=1` 均 `gfx906`；gate-211 与 S4c 基线逐字一致 |
| 12 | 锁死回归：gfx906+DTK 专属单测直接断言裸 `_C` | ✅ | §2.4 + §3.6：新 `.so` 3 passed / 旧 `.so` 3 failed（其余平台 skipif） |
| 13 | Python 兜底保留 + 注释说明；单测不打 Python 回退 | ✅ | `activation.py:127-135`、`:188-189`；单测直接 `torch.ops._C.*` 与 `apply_moe_activation` |
| 14 | 硬约束（不跑自编译 HIP 可执行程序 / 不碰生产容器与 2.10 树 / 不换 DTK 库 / 产物只落 `/mnt/game` / MAX_JOBS≤4 / 8 GiB 不 OOM） | ✅ | §3.9：`oom_kill 0`、loadavg 1.12、`qwen38-vllm` 仍 `Exited (137)`、`ninja` 全在 `z100-build-211` |

**S7c 总体：✅ 全过**（含两处必须上报的偏差：被修目标是 `_C_stable_libtorch` 而非 `_C`；守卫不能用
`__gfx906__` 而用 `__DTK_ARCH__` + 运行时 `gcnArchName`）。

---

## 5. 未完成项 / 阻塞与原因

1. **无阻塞项**。但有两处**任务书原文与实测不符**，需要 Lead 在 S9 的"已修产物清单"里更正：
   - (a) 修的是 `_C_stable_libtorch.abi3.so`（`cc2aca69…` → `93bf7b4a…`），
     **`_C.abi3.so` 逐字节未变**（raw `048736b1…`、installed `6fba0596…`）。任务书写"旧基线
     `048736b1…` 会变"是不成立的；`_C` 里没有本函数的注册点（§2.3）。
   - (b) 守卫无法用 `defined(__gfx906__)`（host pass 不定义该宏，§3.1(1)）。
     现方案用 `#if defined(__DTK_ARCH__)` + 运行时 `strncmp(gcnArchName, "gfx906", 6)`。
2. **`csrc/activation_kernels.cu`（孪生文件）未改**：它没有向量化分支，其 `silu_and_mul` 也没有注册点
   （§2.3 的 grep 证据）。若 Lead 要求"S5 的 legacy `_C` 也动一行"，那将是**无效果的改动**，本步不做。
3. **ISA 级机理仍未定性**（承接 S7b §5.2）：本步是**等价规避**（关分支走已验证正确的标量实现），
   不是"修好 `__half2` 紧凑路径"。若要真正闭合，仍需从 `.so` 抽 gfx906 code object 反汇编定位调度缺陷
   （不在本步门禁内）。
4. **未做 CUDA/社区-ROCm 的实编译验证**：本机只有 DTK+gfx906。"非 DTK 路径编译产物逐字节不变"的论证 =
   `#else` 常量 false（§2.2）+ host 汇编级负向控制（§3.1(2)）；没有 CUDA 工具链可做全 TU 对照。
5. **未测量 fp16 packed 激活在 gfx906 上的纯内核开销**：只给了端到端 ms/token（§3.8）。
   若 S9 要做严谨性能对拍，可用 `benchmarks/kernels/` 单算子测（本步未做，避免与 S9 范围重叠）。
6. **S8 的既有进程**：它 21:22 启动、安装发生在 21:26；21:39 复查 `/proc/6453/maps` 里**尚未**映射
   `_C_stable_libtorch`，所以它之后拿到的是修复版；但**如果**它在更早阶段已经 dlopen 过该库，
   其引擎内存里仍是旧代码——S8/S9 若要引用 21:26 之前的 MoE 数值结果，需重启引擎。
7. **新增单测文件被本 fork 的 `.gitignore` 忽略，未 `git add`（写作用域外）**：
   `.gitignore:301` 有一条宽泛规则 `test_*.py`（仅两处白名单），因此新建的
   `tests/kernels/test_gfx906_packed_silu_fp16.py` 在 `git status` 里是**不可见**的（文件确实在磁盘上、
   pytest 直接指定路径或目录收集都能跑到，§3.6 的 3 passed 就是证据）。**文件本身已落盘、
   已同步进 canonical 副本**；但 S9 固化/提交时必须显式处理，二选一：
   `git add -f tests/kernels/test_gfx906_packed_silu_fp16.py`，或在 `.gitignore` 里加一行
   `!tests/kernels/test_gfx906_packed_silu_fp16.py`（后者改的是 `.gitignore`，不在本步写作用域内，
   故本步未做）。前面的 S5/S6 gfx906 测试文件（`tests/kernels/attention/test_triton_unified_*_gfx906*.py`、
   `tests/kernels/quantization/test_gguf_fixed_cols_gfx906.py`）都是 **tracked** 的，说明历史上是
   用 `git add -f` 或规则之前就入库的。

---

## 6. 对后续步骤的影响

1. **S9 用哪个副本**：继续用 canonical 副本 **`/mnt/game/port/gfx906-vllm-211`**（本步已就地升级），
   环境脚本 `tools/dcu/s7_env.sh` / `s8_env.sh` 不用改。**`_C` 的 md5 不再是判别依据**，要记：
   - `vllm/_C.abi3.so` = `6fba05966e153b3d04771d6960c782bb`（78,102,728 B，**未变**）
   - `vllm/_rocm_C.abi3.so` = `5d1e50e1a77de74f2d308519f8120745`（未变）
   - **`vllm/_C_stable_libtorch.abi3.so` = `93bf7b4a391ac97e2f67cc7141633cb1`（40,691,216 B，本步改变）**
   - `vllm/_moe_C_stable_libtorch.abi3.so` = `996b282c4c6a3d3b46b5e8ba4922af39`（未变）
   回滚件：`/mnt/game/port/s7c/_C_stable_libtorch.abi3.so.installed-before`（`cc2aca69…`）。
2. **硬约束 13（gfx906 禁用 `--enforce-eager`）可以放宽到"fp16 稠密模型"**：本步证明
   **仅靠 csrc**（无 Python 兜底）eager 就与 graph/HF 逐 token 一致、逐层 0 非有限。
   但**不要**整体删除该约束：MoE/GGUF/MiniMax 的**其它**激活仍可能踩同类 hazard（本步只把该 TU 的
   3 条 `use_vec` 全部关掉，覆盖了 `_C` 侧激活；triton/inductor 侧未动），建议改为
   "gfx906 上 `--enforce-eager` 仅限 fp16 稠密模型，且必须在带本修复的 `_C_stable_libtorch` 上"。
3. **S10（MoE / GGUF / 多模态）**：`apply_moe_activation(SILU)` 与 GGUF 直调点**现在已被覆盖**
   （§3.3 `[s7c-moe] nonfinite=0/98304`），S7b §5.1/§6.3 的"这些路径仍会崩"前提**已解除**。
   但 MiniMax 的 `swiglu_oai` 依赖 packed 算子的注释（S7b §5.1）值得 S10 再用真实模型复跑确认——
   本步只到 `_C` 算子级。
4. **Python 兜底保留但已降级为冗余**：它不会掩盖 csrc 回退（新单测打裸算子，§3.6），
   留着可以保护"用旧 `_C` 前缀树"的场景；若将来 S9 决定清理该 Python 分支，
   必须同时保留新单测，否则根修无人守。
5. **构建习惯不变**：只改 csrc 单 TU 仍走
   `MAX_JOBS=4 bash /mnt/game/port/build-211.sh _C_stable_libtorch`（~17 s 全流程）；
   注意 S4d 的 mtime 边界（用 `cp -p` 恢复旧源码后要 `touch` 再 `ninja`）。
6. **被本步推翻的前提**：任务书"重建 `_C`（`048736b1…` 会变）"、"守卫用 `__gfx906__`"两条。
7. **S9 固化清单里要补一条**：新单测被 `.gitignore` 的 `test_*.py` 规则忽略（§5.7），
   提交时用 `git add -f` 或加白名单；否则将来"根修回退"时无人守（Python 兜底会掩盖它）。

---

## 7. 复现方式

```bash
# ---------- 0) 前置：容器 z100-build-211（构建）与 z100-eager（跑，卡 2）都在；
#      canonical 副本 /mnt/game/port/gfx906-vllm-211 已是 S7c 之后的版本。

# ---------- 1) 守卫可用性 / 负向控制（只编译，不执行；~10 s） ----------
docker exec z100-build-211 bash -lc 'cd /mnt/game/port/s7c-probe && \
  hipcc -x hip --cuda-host-only -DUSE_ROCM -I/usr/local/lib/python3.10/dist-packages/torch/include \
        -S -O0 -o p1.s p1.cu && grep -A2 "host_probe:" p1.s'   # 期望：xorl %eax,%eax（__gfx906__ 不存在）
docker exec z100-build-211 bash -lc 'cd /mnt/game/port/s7c-probe && \
  hipcc -x hip --cuda-host-only -O2 -S -o p3-nonDTK.s p3.cu && \
  hipcc -x hip --cuda-host-only -O2 -D__DTK_ARCH__=520 -S -o p3-DTK.s p3.cu'
# 期望：p3-nonDTK.s 的 use_vec_new 与 use_vec_orig 指令体逐条相同；p3-DTK.s 里出现 strncmp@PLT

# ---------- 2) 改源码 → 同步到构建副本 → 增量重建（~17 s） ----------
#     仓库改动见 §2.1（csrc/libtorch_stable/activation_kernels.cu:11,240-274,296-297,1003-1004,1407-1408）
docker exec z100-build-211 bash -lc 'cp -p /mnt/game/port/gfx906-vllm-211/csrc/libtorch_stable/activation_kernels.cu \
     /mnt/game/port/s7c/activation_kernels.cu.s7c-orig
  cp /src/csrc/libtorch_stable/activation_kernels.cu /mnt/game/port/gfx906-vllm-211/csrc/libtorch_stable/activation_kernels.cu
  MAX_JOBS=4 LOG=/mnt/game/port/s7c/build-s7c.log bash /mnt/game/port/build-211.sh _C_stable_libtorch'
# 期望：[1/30] hipify + [2/3] 该 TU + [3/3] Linking _C_stable_libtorch.abi3.so，rc=0

# ---------- 3) 原子安装（先查 S8 有没有在启动引擎） ----------
docker exec z100-s8 bash -lc 'pgrep -af "vllm|python" | head'
docker exec z100-build-211 bash -lc 'set -e; . /mnt/game/port/setenv-211.sh >/dev/null 2>&1
  T=/mnt/game/port/gfx906-vllm-211/vllm/_C_stable_libtorch.abi3.so
  cp -p "$T" /mnt/game/port/s7c/_C_stable_libtorch.abi3.so.installed-before
  rm -rf /mnt/game/port/s7c/stage; mkdir -p /mnt/game/port/s7c/stage
  DESTDIR=/mnt/game/port/s7c/stage cmake --install /mnt/game/port/build-211/temp \
      --prefix /mnt/game/port/gfx906-vllm-211 --component _C_stable_libtorch
  mv -f /mnt/game/port/s7c/stage/mnt/game/port/gfx906-vllm-211/vllm/_C_stable_libtorch.abi3.so "$T"
  md5sum "$T"'
# 期望：93bf7b4a391ac97e2f67cc7141633cb1

# ---------- 4) 裸算子改前/改后（~1 min/次；改前用旧件前缀树，不动 canonical） ----------
docker exec z100-build-211 bash -lc 'rm -rf /mnt/game/port/s7c/pre-fix
  cp -a /mnt/game/port/gfx906-vllm-211 /mnt/game/port/s7c/pre-fix
  cp /mnt/game/port/s7c/_C_stable_libtorch.abi3.so.installed-before /mnt/game/port/s7c/pre-fix/vllm/_C_stable_libtorch.abi3.so
  cp /src/tools/dcu/s7c_direct_check.py /mnt/game/port/s7c/'
docker exec z100-eager bash -lc 'source /opt/dtk/env.sh >/dev/null 2>&1; export S7_HIP_VISIBLE_DEVICES=2
  export VLLM_DIR=/mnt/game/port/s7c/pre-fix; source $VLLM_DIR/tools/dcu/s7_env.sh >/dev/null 2>&1
  cd $VLLM_DIR; python -u /mnt/game/port/s7c/s7c_direct_check.py | tee /mnt/game/port/s7c/before-fix2.log'
# 期望（坏）：20 次 nonfinite 424–455、maxerr=inf；clamp 107/24576；moe 1907/98304
docker exec z100-eager bash -lc 'source /opt/dtk/env.sh >/dev/null 2>&1; export S7_HIP_VISIBLE_DEVICES=2
  source /mnt/game/port/gfx906-vllm-211/tools/dcu/s7_env.sh >/dev/null 2>&1; cd $VLLM_DIR
  python -u /mnt/game/port/s7c/s7c_direct_check.py | tee /mnt/game/port/s7c/after-fix.log'
# 期望（好）：20 次 nonfinite=0、maxerr≈0.0247；clamp 0/24576；moe 0/98304

# ---------- 5) 端到端 eager / graph / HF（每条约 1–2 min） ----------
docker exec z100-eager bash -lc 'source /opt/dtk/env.sh >/dev/null 2>&1; export S7_HIP_VISIBLE_DEVICES=2
  source /mnt/game/port/gfx906-vllm-211/tools/dcu/s7_env.sh >/dev/null 2>&1; cd $VLLM_DIR
  bash tools/dcu/s7_gate.sh eager /mnt/game/models/Qwen3-0.6B /mnt/game/port/s7c/s7c-gate-eager.log
  bash tools/dcu/s7_gate.sh graph /mnt/game/models/Qwen3-0.6B /mnt/game/port/s7c/s7c-gate-graph.log'
# 期望两条的 S7_IDS_* 都等于 [12095,13,576,6722,315,15344,374,21718,13,576,6722,315,17689,374,24081,13]
# 想复现"仅 csrc 也够"：见 §7-6 的 csrc-only 前缀树 + 同样两条命令（其中一条去掉兜底）

# ---------- 6) 无 Python 兜底的前缀树（证明 csrc 修复充分/必要） ----------
cd /home/lyh/z100-vllm && git show HEAD:vllm/model_executor/layers/activation.py > /tmp/act_head.py
docker exec z100-build-211 bash -lc 'rm -rf /mnt/game/port/s7c/csrc-only
  cp -a /mnt/game/port/gfx906-vllm-211 /mnt/game/port/s7c/csrc-only'
docker exec -i z100-build-211 bash -c 'cat > /mnt/game/port/s7c/csrc-only/vllm/model_executor/layers/activation.py' < /tmp/act_head.py
docker exec z100-eager bash -lc 'source /opt/dtk/env.sh >/dev/null 2>&1; export S7_HIP_VISIBLE_DEVICES=2
  export VLLM_DIR=/mnt/game/port/s7c/csrc-only; source $VLLM_DIR/tools/dcu/s7_env.sh >/dev/null 2>&1; cd $VLLM_DIR
  bash tools/dcu/s7_gate.sh eager /mnt/game/models/Qwen3-0.6B /mnt/game/port/s7c/s7c-gate-eager-csrc-only.log'
# 期望：与 HF 一致（改前同配置是 [1023]×16）——把 .so 换回旧件即得反例

# ---------- 7) 逐层扫描（用无兜底树） ----------
docker exec z100-eager bash -lc 'source /opt/dtk/env.sh >/dev/null 2>&1; export S7_HIP_VISIBLE_DEVICES=2
  export VLLM_DIR=/mnt/game/port/s7c/csrc-only; source $VLLM_DIR/tools/dcu/s7_env.sh >/dev/null 2>&1
  export S7B_SCAN=/mnt/game/port/s7c/layer_scan-csrc-only.jsonl; rm -f $S7B_SCAN; cd $VLLM_DIR
  python -u tools/dcu/s7b_scan.py /mnt/game/models/Qwen3-0.6B csrc-only "enforce_eager=True"'
# 期望：modules=396 nonfinite_modules=0

# ---------- 8) 回归测试（6 个文件）+ 新增单测的负向控制 ----------
docker exec z100-eager bash -lc 'source /opt/dtk/env.sh >/dev/null 2>&1; export S7_HIP_VISIBLE_DEVICES=2
  source /mnt/game/port/gfx906-vllm-211/tools/dcu/s7_env.sh >/dev/null 2>&1; cd $VLLM_DIR
  cp /src/tests/kernels/test_gfx906_packed_silu_fp16.py tests/kernels/
  for f in tests/kernels/test_gfx906_packed_silu_fp16.py \
           tests/kernels/attention/test_triton_unified_attention_gfx906.py \
           tests/kernels/attention/test_triton_unified_attn_gfx906_v1.py \
           tests/kernels/quantization/test_gguf_fixed_cols_gfx906.py \
           tests/kernels/mamba/test_fused_sigmoid_gating.py \
           tests/kernels/mamba/test_fused_recurrent_packed_decode.py; do
    python -c "import torch, pytest, sys; sys.exit(pytest.main([\"-q\",\"--no-header\",\"--noconftest\",\"-p\",\"no:cacheprovider\",\"$f\"]))" | tail -1
  done'
# 期望：3 / 39 / 12 / 13 / 5 / 5 passed
# 负向控制（旧 .so 上必须失败）：
docker exec z100-eager bash -lc '... VLLM_DIR=/mnt/game/port/s7c/pre-fix ... \
  cp /src/tests/kernels/test_gfx906_packed_silu_fp16.py tests/kernels/ && python -c "...pytest.main([...])"'
# 期望：3 failed（"launch 0 produced inf/NaN"）

# ---------- 9) 门禁 ----------
docker exec z100-build-211 bash /mnt/game/port/gate-211.sh        # 期望 IMPORT_OK / TORCH_PIN_OK / TRITON_PIN_OK / 两个 so 只 gfx906
docker exec z100-build-211 bash -lc '. /mnt/game/port/setenv-211.sh >/dev/null 2>&1; cd "$REPO_DIR"
  python -c "import torch, vllm, vllm._C, vllm._rocm_C, vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch; print(\"IMPORT_OK_ALL4\")"
  for so in vllm/_C.abi3.so vllm/_rocm_C.abi3.so vllm/_C_stable_libtorch.abi3.so vllm/_moe_C_stable_libtorch.abi3.so; do
    printf "%-34s " "$(basename $so)"; strings -a $so | grep -oE "amdgcn-amd-amdhsa--gfx[0-9a-f]+" | sed "s/.*--//" | sort -u | tr "\n" " "; echo; done'
# 期望：IMPORT_OK_ALL4；四行各只有 gfx906

# ---------- 10) 回滚 ----------
docker exec z100-build-211 bash -lc 'cp -p /mnt/game/port/s7c/_C_stable_libtorch.abi3.so.installed-before \
    /mnt/game/port/gfx906-vllm-211/vllm/_C_stable_libtorch.abi3.so  # 注意：它是 install 后的形态，可直接用
  cp -p /mnt/game/port/s7c/activation_kernels.cu.s7c-orig /mnt/game/port/gfx906-vllm-211/csrc/libtorch_stable/activation_kernels.cu
  touch /mnt/game/port/gfx906-vllm-211/csrc/libtorch_stable/activation_kernels.cu
  MAX_JOBS=4 bash /mnt/game/port/build-211.sh _C_stable_libtorch'
```

---

## 8. 参考

- 前一档（本步直接依据）：[z100-07b-eager-nan-result.md](z100-07b-eager-nan-result.md) §4（根因边界）、
  **§5.1（`use_vec` patch 草案）**、§5.2（ISA 机理未定性）、§5.3（直调点清单）、§3.5（live kernel 身份）
- 构建增量与 mtime 边界：[z100-04d-build-incrementality-result.md](z100-04d-build-incrementality-result.md)
  §3.5（1 TU + 1 链接 ≈ 40 s）、§3.8（`cp -p` 旧源码后要 `touch`）
- 构建环境来源：[z100-04b-stack-upgrade-result.md](z100-04b-stack-upgrade-result.md) §6
  （容器 `z100-build-211`、副本 `/mnt/game/port/gfx906-vllm-211`、构建树 `/mnt/game/port/build-211`、
  `setenv-211.sh`/`build-211.sh`/`install-211.sh`/`gate-211.sh`）
- 同族缺陷：[z100-06-triton-gfx906-result.md](z100-06-triton-gfx906-result.md) §3.5（gfx906 VALU→SFU hazard）
- 硬约束与模板：[docs/z100-todolist.md](../z100-todolist.md) 硬约束 1–13、
  [docs/result/README.md](README.md)
- 关键源码位置（本步）：
  - `csrc/libtorch_stable/activation_kernels.cu:11`（`<cstring>`）
  - `csrc/libtorch_stable/activation_kernels.cu:240-274`（**守卫** `dtk_gfx906_fp16_no_packed`）
  - `csrc/libtorch_stable/activation_kernels.cu:296-297 / :1003-1004 / :1407-1408`（**3 处 `use_vec`**）
  - `csrc/libtorch_stable/activation_kernels.cu:119-159`（向量化分支）、`:61-90`（`packed_compute`）、
    `:172-179`（`packed_silu_kernel`，即被编坏的那段）
  - `vllm/model_executor/layers/activation.py:127-135`（注释：csrc 已根修、此处兜底）、`:188-189`
  - `csrc/libtorch_stable/torch_bindings.cpp:605 / :876`（**唯一的** `_C::silu_and_mul` 注册点）、
    `:613`（`silu_and_mul_with_clamp` schema）
  - `csrc/quantization/gptq/compat.cuh:28-36`（本仓库既有"`__DTK_ARCH__` 由 DTK hipify 注入"的先例）
- 本步脚本/产物：
  - [`tools/dcu/s7c_direct_check.py`](../../tools/dcu/s7c_direct_check.py)（裸算子探针）
  - [`tests/kernels/test_gfx906_packed_silu_fp16.py`](../../tests/kernels/test_gfx906_packed_silu_fp16.py)（回归单测）
  - `/mnt/game/port/s7c/`：`before-fix2.log`、`after-fix.log`、`build-s7c.log`、`s7c-gate-graph.log`、
    `s7c-gate-eager.log`、`s7c-gate-eager-csrc-only.log`、`s7c-gate-eager-prefix-csrc-only.log`、
    `s7c-gate-211.log`、`s7c-scan-csrc-only.log`、`layer_scan-csrc-only.jsonl`、
    `_C_stable_libtorch.abi3.so.installed-before`（回滚件）、`activation_kernels.cu.s7c-orig`（回滚件）
  - `/mnt/game/port/s7c/{pre-fix,csrc-only,pre-fix-csrc-only}/`（三棵 A/B 前缀树，各 455 MB）
  - `/mnt/game/port/s7c-probe/`（`p1.cu`/`p1.s`、`p3.cu`/`p3-nonDTK.s`/`p3-DTK.s` 宏与负向控制探针）
