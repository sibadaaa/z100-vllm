# S7b 定位 `--enforce-eager` 在 gfx906 上的 NaN 根因 — 执行结果

> 归档时间：2026-10-08
> 执行者：子 agent（S7b，`--enforce-eager` NaN 根因定位）
> 对应设计文档：[docs/design/z100-07-memory-kvcache-graph.md](../design/z100-07-memory-kvcache-graph.md)（S7b 无独立设计文档，由 S7 归档 §5.1 派生）
> 前一档：[z100-07-memory-kvcache-graph-result.md](z100-07-memory-kvcache-graph-result.md)（S7：发现 eager 全 NaN、A/B 矩阵收敛到"非 inductor 的 vllm_c 路径"）
> 状态：✅ **根因定位完成 + 最小修复已落地并端到端验证**；
> ⚠️ 修复只覆盖 `CustomOp` 路径，`fused_moe` 直接调 `torch.ops._C.silu_and_mul` 的路径仍受影响（§5.1），需 csrc 侧修复或 Lead 决策。

**一句话结论**：`--enforce-eager` 的 NaN **不是 attention / KV cache / rms_norm / 编译模式的问题**，而是
`csrc/libtorch_stable/activation_kernels.cu` 里 **fp16 `__half2` 向量化分支的 gate 侧 SiLU
（`packed_compute<...,act_first=true>` → `packed_silu_kernel`）在 gfx906 + DTK 26.04 上被编坏**：
`_C.silu_and_mul` 对 |x|≤5.9 的 fp16 输入有约 1.8% 元素返回 inf/NaN 或 fp16 饱和值（最大 65376），
而**同一文件**的标量分支（`d % 8 != 0`）、`_C.mul_and_silu`（`act_first=false`）、`_C.gelu_and_mul`
以及 bf16/fp32 全部正确；数值错在 `model.layers.0.mlp.act_fn` 第一次出现（6/15360 元素非有限），
随后 `down_proj` 把 NaN 扩散到整个 hidden state。修复方式：`SiluAndMul` / `SiluAndMulWithClamp`
在 gfx906 + fp16 下改走 `forward_native`（`vllm/model_executor/layers/activation.py:114,178,286`），
修复后 eager 逐 token 与 `VLLM_COMPILE` 路径和 HF transformers **完全一致**。

---

## 1. 目标与验收标准

### 1.1 本步任务书门禁（原文）

1. **稳定复现**：用 `s7_compare.sh` 的矩阵拿到"`CompilationMode.NONE` → NaN / `VLLM_COMPILE` → 与 HF
   transformers 逐 token 一致"的最小可复现命令（固定 prompt/greedy/`max_tokens`）；
2. **定位到具体算子/路径**（本步核心）：逐层对比找出**第一个**出现 NaN 或数量级异常的位置；
   二分替换可疑组件，跟踪 gfx906 特有分支；
3. **给出结论与处置**：根因明确且修复小且安全 ⇒ 落地最小修复 + 修复前后证据；根因在 DTK/工具链侧
   ⇒ 给出精确规避方案并评估"是否应该改默认值"；
4. 无论结果如何都要给出 `file:line` 级证据与 **6–20 行最小复现脚本**（放 `tools/dcu/`）。

### 1.2 追加的硬约束（本步遵守情况见 §3.9）

- 固定 `HIP_VISIBLE_DEVICES=2`（单卡；Lead 用 0、S8 用 1）；
- 新建容器 `z100-eager`（同镜像、设备/group 照抄 `z100-build-211`、`--memory=10g`、`--shm-size=16g`、
  `--ipc=host`、不加 `--rm`），挂 `/mnt/game` 与只读工作区 `/src`；
- **禁止执行自编译 HIP 独立可执行程序**（只做 `hipcc -S` 出汇编）；
- 不 start/kill/exec 生产容器 `qwen38-vllm`；不动 `z100-build`/`z100-build-211` 构建树；不替换 DTK 库；
  产物与缓存只落 `/mnt/game`；不改 `docs/design/*`、`docs/z100-todolist.md`、`docs/result/README.md`；
- 写作用域：`vllm/**`、`csrc/**`（仅当明确需要）、`tools/dcu/**`、本归档。

---

## 2. 实际改动

### 2.1 仓库源码改动：`vllm/model_executor/layers/activation.py`（+32 行，唯一源码改动）

| 位置 | 改动摘要 |
|---|---|
| `vllm/model_executor/layers/activation.py:114-137` | 新增模块级判定 `_broken_gfx906_fp16_packed_silu(x)`：`fp16 + cuda + is_rocm + on_gfx906()` 才返回 True（非 DTK/非 gfx906/非 fp16 一律走原路径，满足硬约束 6） |
| `vllm/model_executor/layers/activation.py:178-181` | `SiluAndMul.forward_hip()`：命中上述条件时返回 `forward_native`（`F.silu(gate) * up`），否则 `forward_cuda`（原 `_C.silu_and_mul`） |
| `vllm/model_executor/layers/activation.py:282-289` | `SiluAndMulWithClamp.forward_hip()`：在既有 "非默认 alpha/beta 才回退" 的判定里再插入同一条 gfx906+fp16 判定 |

```diff
@@ -111,6 +111,31 @@ class FatreluAndMul(CustomOp):
+def _broken_gfx906_fp16_packed_silu(x: torch.Tensor) -> bool:
+    """Whether ``x`` would take the miscompiled gfx906 fp16 packed SiLU path.
+    ... csrc/libtorch_stable/activation_kernels.cu (act_and_mul_kernel with
+    use_vec, :106/:250) ... packed_compute<..., act_first=true> ->
+    packed_silu_kernel, :171 ... See docs/result/z100-07b-eager-nan-result.md.
+    """
+    if x.dtype != torch.float16 or not x.is_cuda:
+        return False
+    if not current_platform.is_rocm():
+        return False
+    try:
+        from vllm.platforms.rocm import on_gfx906
+    except ImportError:  # pragma: no cover - non-DTK builds
+        return False
+    return on_gfx906()
@@ class SiluAndMul(CustomOp):
+    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
+        if _broken_gfx906_fp16_packed_silu(x):
+            return self.forward_native(x)
+        return self.forward_cuda(x)
@@ class SiluAndMulWithClamp(CustomOp):
         if self.alpha == 1.0 and self.beta == 0.0:
+            if _broken_gfx906_fp16_packed_silu(x):
+                return self.forward_native(x)
             return self.forward_cuda(x)
```

`csrc/**` **未改动**（csrc 侧修复方案见 §5.1，因需要重建 `_C` 并替换 S4c 已验证产物，留待 Lead 决策）。
`git diff --stat`：`vllm/model_executor/layers/activation.py | 32 ++++++++++++++++++++++++`（1 file changed, 32 insertions）。

### 2.2 新增脚本（`tools/dcu/**`，任务书允许的写作用域）

| 文件 | 作用 |
|---|---|
| `tools/dcu/s7b_min_repro.py` | **18 行最小复现**：fp16 `_C.silu_and_mul` vs `F.silu(gate)*up`（§3.3） |
| `tools/dcu/s7b_hooks.py` | 逐模块 forward pre/post hook（`torch.isfinite` 统计），经 `LLM.collective_rpc` 注入 worker |
| `tools/dcu/s7b_scan.py` | 逐层扫描驱动：打印 `mode/custom_ops/enabled_custom_ops` + 扫描结果 |
| `tools/dcu/s7b_activation_probe.py` | 合成输入上重跑 `_C.silu_and_mul` 4 次（发现非确定性） |
| `tools/dcu/s7b_activation_matrix.py` | dtype × 算子 × 形状矩阵（定位到 fp16 + act_first=true） |
| `tools/dcu/s7b_activation_detail.py` | 逐元素对比 kernel 输出 / 精确值 / `exp(-gate)` |
| `tools/dcu/s7b_activation_localize.py` | 同数学经 `mul_and_silu` 交换半区验证（排除"silu 函数本身"） |
| `tools/dcu/s7b_activation_repeat.py` | 同输入 20 次重复（非确定性证据）+ gate 网格扫描 |
| `tools/dcu/s7b_activation_vecpath.py` | 用 `d % 8` 切换 `use_vec` 分支（**关键定位实验**） |
| `tools/dcu/s7b_activation_scope.py` | fp16/bf16 × 各激活算子 × 10 次的相对误差矩阵（缺陷边界） |
| `tools/dcu/s7b_kernel_sweep.py` | `_C.rms_norm`/`fused_add_rms_norm`/`silu_and_mul` fp16/bf16/fp32 正确性 |
| `tools/dcu/s7b_kernel_identity.py` | torch profiler 取**真实 Kernel 名**（揭示 live TU 是 libtorch_stable）+ 列置换实验 |
| `tools/dcu/s7b_moe_activation_check.py` | 证明 `apply_moe_activation(SILU)` 直调坏 kernel（修复覆盖缺口） |
| `tools/dcu/s7b_repro_act_sfu.cu` | `hipcc -S` 只编汇编的模板复现（**注意**：对应非 stable 的 TU，见 §3.6 与 §5.2） |

### 2.3 环境（容器 `z100-eager`，未加 `--rm`，保留）

```console
$ docker inspect z100-eager --format '...'
ImageID=sha256:d1c08bf30382f6d8f49e0ad7130ceb4e3a31484ce4bdd3b1aeb7787b5d2fb14e   # 与 z100-build-211 同镜像
Devices=[{"PathOnHost":"/dev/kfd",...},{"PathOnHost":"/dev/mkfd",...},{"PathOnHost":"/dev/dri","CgroupPermissions":"rwm"}]
GroupAdd=["video"]  SecurityOpt=["label=disable"]  IpcMode=host  ShmSize=17179869184
Memory=10737418240  MemorySwap=10737418240  AutoRemove=false  WorkingDir=/mnt/game/port
Mounts=/home/lyh/z100-vllm:/src:ro /mnt/game:/mnt/game:rw /opt/hyhal:/opt/hyhal:ro
```

创建命令（与 S7 的 `z100-serve` 同构，仅内存 10g / 名字不同）：

```bash
docker run -d --name z100-eager \
    --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
    --group-add video --security-opt label=disable --ipc host \
    --memory=10g --memory-swap=10g --shm-size=16g \
    -v /opt/hyhal:/opt/hyhal:ro -v /home/lyh/z100-vllm:/src:ro -v /mnt/game:/mnt/game \
    -w /mnt/game/port \
    harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 \
    bash -c 'sleep infinity'
```

容器内补装（复用 S7 已缓存的轮子，全部来自 `/mnt/game/port/{wheels-211,s7/pip-wheels}`，未新增下载）：

```console
$ pip install --no-deps /mnt/game/port/wheels-211/torch-2.11.0+das.opt1.dtk2604.*.whl
$ pip install --no-deps $W/mistral_common-1.11.5*.whl $W/openai-2.54.0*.whl $W/xgrammar-0.2.8*.whl \
                     $W/apache_tvm_ffi-*.whl $W/torchvision-0.26.0-*.whl
$ cp $D/s7_usercustomize.py /root/.local/lib/python3.10/site-packages/usercustomize.py
SHIM_OK True 0.26.0+cu130 InterpolationMode.BILINEAR
torch 2.11.0+das.opt1.dtk2604.2608241728.gfb4aa7 ; triton 3.6.0+git7c56a5e4 /mnt/game/triton-gfx906/install/triton/__init__.py
```

> `z100-eager` 全程只用**卡 2**（`S7_HIP_VISIBLE_DEVICES=2`，`devcount=1`、`name0=Z100`）；未 exec/start/kill 任何其它容器。

---

## 3. 执行过程与原始证据

### 3.1 稳定复现（门禁 1）

最小可复现命令（卡 2、单 prompt、greedy、`max_tokens=16`、`ignore_eos`）：

```console
$ docker exec z100-eager bash -lc '
    source /opt/dtk/env.sh >/dev/null 2>&1
    export S7_HIP_VISIBLE_DEVICES=2
    source /mnt/game/port/gfx906-vllm-211/tools/dcu/s7_env.sh >/dev/null
    cd $VLLM_DIR
    # 修复前（等价于 S7 的 enforce_eager 矩阵项）
    bash tools/dcu/s7_compare.sh /mnt/game/models/Qwen3-0.6B pre-fix "enforce_eager=True" \
        /mnt/game/port/s7b/s7b-eager-prefix.log'
```

修复前（沿用 S7 已归档的原始日志，本步用 `s7b_scan.py` 复现同样现象）：输出 `lock×16`、logits 全 `nan`
（S7 §3.7 原文 `S7_IDS_EAGER=[1023, ...]`、`PROBE ... naN=6`）。本步的逐层扫描（§3.2）在同一配置下复现。

修复后（同一命令，`tag=eager-fixed`）：

```console
[s7] S7_IDS[eager-fixed]=[12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315, 17689, 374, 24081, 13]
[s7] S7_TEXT[eager-fixed]=' Paris. The capital of Italy is Rome. The capital of Spain is Madrid.'
[s7] S7_DECODE[eager-fixed] TOKENS=128 SECONDS=4.7882 MS_PER_TOKEN=37.408
```

**A/B 收敛（本步实测，单变量）**：

| 路径 | 配置 | token ids | 与 HF 是否一致 |
|---|---|---|---|
| 修复前 eager | `enforce_eager=True`（`mode=NONE, custom_ops=all`） | `[1023 × 16]` ❌ | ✗ |
| 修复后 eager | 同上 | `[12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315, 17689, 374, 24081, 13]` ✅ | ✅ |
| 编译（不捕获 graph） | `mode=VLLM_COMPILE, cudagraph=NONE` | 同上 ✅ | ✅ |
| HF transformers | `AutoModelForCausalLM` fp16 | 同上 ✅ | — |

### 3.2 逐层定位：**第一个非有限值出现在 `model.layers.0.mlp.act_fn`（SiluAndMul）**

方法：`s7b_hooks.py` 给 373 个模块挂 forward pre/post hook（`nn.Module.__call__` 在 `mode=NONE` 下未被
`torch.compile` 绕过，因此 hook 必触发），对每个模块的输入/输出统计 `numel / 非有限元素数 / |absmax|`，
经 `LLM.collective_rpc(s7b_hooks.worker_probe)` 注入到 worker 的模型上；再跑一次 greedy 生成。

**修复前（`/mnt/game/port/s7b/layer_scan-eager.jsonl`，pass 1，首 17 行原文）**：

```console
  i module                                   IN                                                   OUT
  1 <root>                                   in[0] n=5 bad=0 amax=9625 ; in[1] n=5 bad=0 amax=4  out n=5120 bad=5120 amax=nan
  2 model.embed_tokens                         in[0] n=5 bad=0 amax=9625                            out n=5120 bad=0 amax=0.09131
  3 model.layers.0                             in[0] n=5 bad=0 amax=4 ; in[1] n=5120 bad=0 amax=0.09131 out[0] n=5120 bad=4096 amax=459.2 ; out[1] n=5120 bad=0 amax=2.932
  4 model.layers.0.input_layernorm             in[0] n=5120 bad=0 amax=0.09131                      out n=5120 bad=0 amax=2.658
  5 model.layers.0.self_attn                   in[0] n=5 bad=0 amax=4 ; in[1] n=5120 bad=0 amax=2.658 out n=5120 bad=0 amax=2.984
  6 model.layers.0.self_attn.qkv_proj          in[0] n=5120 bad=0 amax=2.658                        out[0] n=20480 bad=0 amax=5.25
  7 model.layers.0.self_attn.q_norm            in[0] n=10240 bad=0 amax=5.25                        out n=10240 bad=0 amax=28.86
  8 model.layers.0.self_attn.k_norm            in[0] n=5120 bad=0 amax=3.779                        out n=5120 bad=0 amax=419.5
  9 model.layers.0.self_attn.rotary_emb        ... out[0] n=10240 bad=0 amax=28.86 ; out[1] n=5120 bad=0 amax=419.5
 10 model.layers.0.self_attn.attn              ... out n=10240 bad=0 amax=1.145
 11 model.layers.0.self_attn.o_proj            ... out[0] n=5120 bad=0 amax=2.984
 12 model.layers.0.post_attention_layernorm    in[0] n=5120 bad=0 amax=2.984 ; in[1] n=5120 bad=0 amax=0.09131 out[0] n=5120 bad=0 amax=9.828 ; out[1] n=5120 bad=0 amax=2.932
 13 model.layers.0.mlp                         in[0] n=5120 bad=0 amax=9.828                        out n=5120 bad=4096 amax=459.2
 14 model.layers.0.mlp.gate_up_proj            in[0] n=5120 bad=0 amax=9.828                        out[0] n=30720 bad=0 amax=5.922
 15 model.layers.0.mlp.act_fn                  in[0] n=30720 bad=0 amax=5.922                       out n=15360 bad=6 amax=5320   <-- 第一个非有限值
 16 model.layers.0.mlp.down_proj               in[0] n=15360 bad=6 amax=5320                        out[0] n=5120 bad=4096 amax=459.2
 17 model.layers.1                             ... out[0] n=5120 bad=5120 amax=nan ; out[1] n=5120 bad=5120 amax=nan
```

（Qwen3-0.6B：hidden 1024、intermediate 3072、prompt 5 token ⇒ `15360 = 5×3072`、`5120 = 5×1024`。）

传播链完全自洽：`act_fn` 输出 6/15360 个 inf/NaN 与一个 5320（远超输入 `absmax=5.922` 的合法上界）⇒
`down_proj` 是"每行对所有 3072 个 intermediate 求和"的 matmul，任一元素 NaN 即使该 token 整个 hidden
NaN（4/5 token 中招 = 4096/5120）⇒ layer1 `input_layernorm` 的 RMS 统计被 NaN 污染 ⇒ 此后全 NaN。

（记录数 396 略多于模块数 373：共享模块——如被 28 层复用的 `rotary_emb`——在一次 forward 里被调用多次，
`_name()` 按首次出现命名，故 `i=23` 显示的仍是 `layers.0.self_attn.rotary_emb`。）

**统计口径**：修复前 pass 1 的 396 个模块记录中 **385 个输出含非有限值**（pass 2 共 376 条全部非有限）；
修复后同一脚本 pass 1 的 396 个模块记录中 **0 个非有限**：

```console
[s7b] SCAN=/mnt/game/port/s7b/layer_scan-eager.jsonl modules=772 nonfinite_modules=761   # 修复前
[s7b] SCAN=/mnt/game/port/s7b/layer_scan-eager-fixed.jsonl modules=396 nonfinite_modules=0  # 修复后
（pass2 记录只在出现非有限值时落盘，故修复后只剩 pass1 的 396 条）
```

**旁证：其它 fp16 `_C` 算子无辜**（`tools/dcu/s7b_kernel_sweep.py`）：

```console
[s7b-sweep] _C.rms_norm                        nonfinite=0     /65536    maxerr=0.0019393 nbad>1e-2=0
[s7b-sweep] _C.silu_and_mul (known bad)        nonfinite=493   /65536    maxerr=inf nbad>1e-2=3675
（bf16/fp32 的 silu_and_mul：nonfinite=0；fp32 maxerr=0）
```

### 3.3 最小复现（18 行，`tools/dcu/s7b_min_repro.py`）

```console
$ python tools/dcu/s7b_min_repro.py
[repro] _C.silu_and_mul  nonfinite = 417 / 24576
[repro] _C.silu_and_mul  max_abs_err = 1000000000.0     # inf 被夹到 1e9
[repro] F.silu(gate)*up  max_abs_err = 0.01857757568359375
```

输入是 `U(-5.9, 5.9)` 的 fp16（`d=3072`），**`silu` 的合法输出上界 ~5.9、与 up 相乘后 ~35**；
kernel 却给出 `inf/NaN` 与 `65376`（≈fp16 上限）——即不是精度问题，是**数值完全错**。

修复前后差异只由 §2.1 的 Python 层回退造成（kernel 本身未改）：

```console
$ python tools/dcu/s7b_moe_activation_check.py     # 直调 torch.ops._C.silu_and_mul，绕过 CustomOp
[s7b-moe] apply_moe_activation(SILU) fp16 gfx906: nonfinite=1917/98304 max_abs_err=1e+09
```

### 3.4 缺陷边界：fp16 + `__half2` 向量化 + `act_first=true`

**(a) dtype / 算子矩阵**（`s7b_activation_scope.py`，各 10 次，相对误差 >0.05 计数）：

```console
--- dtype=torch.float16 shape=(64, 6144) d=3072 (vec path) ---
[s7b-scope] torch.float16 silu_and_mul             runs_bad=10/10 worst_nonfinite=3708 worst_rel>0.05=12232 worst_rel=inf
[s7b-scope] torch.float16 mul_and_silu             runs_bad=0/10  worst_nonfinite=0    worst_rel>0.05=0     worst_rel=0.000936473
[s7b-scope] torch.float16 silu_and_mul_with_clamp  runs_bad=10/10 worst_nonfinite=871  worst_rel>0.05=3024  worst_rel=inf
[s7b-scope] torch.float16 gelu_and_mul             runs_bad=0/10  worst_nonfinite=0    worst_rel>0.05=0     worst_rel=0.000911006
--- dtype=torch.bfloat16 shape=(64, 6144) d=3072 (vec path) ---
[s7b-scope] torch.bfloat16 silu_and_mul            runs_bad=0/10  worst_rel=0.00725627
[s7b-scope] torch.bfloat16 mul_and_silu            runs_bad=0/10  worst_rel=0.00751983
[s7b-scope] torch.bfloat16 silu_and_mul_with_clamp runs_bad=0/10  worst_rel=0.00725627
[s7b-scope] torch.bfloat16 gelu_and_mul            runs_bad=0/10  worst_rel=0.00756534
[s7b-scope] fp16 scalar-branch d=3068 (d%8=4) metrics=(0, 0, 0.0009200773783959448)
```

**(b) `use_vec` 分支切换（决定性实验，`s7b_activation_vecpath.py`）**——`csrc/libtorch_stable/activation_kernels.cu:255-259`
用 `use_vec = (d % vec_size == 0)`（fp16 时 `vec_size = 16/2 = 8`）在**同一份数据的两种实现**之间切换：

```console
d     d%8  path   nonfinite  maxerr      nbad>1e-2
3064  0    vec    248        inf         1002
3068  4    scalar 0          0.0226669   158
3070  6    scalar 0          0.0198364   163
3072  0    vec    230        inf         884
3076  4    scalar 0          0.021492    157
3080  0    vec    251        inf         1000
1024  0    vec    84         inf         329
1020  4    scalar 0          0.0220146   57
512   0    vec    40         inf         154
510   6    scalar 0          0.0186691   30
```

⇒ **向量化（`__half2`）分支坏、标量分支（同一文件的 `silu_kernel<half>`）好**。
`bfloat16` 的 `__nv_bfloat162` 实例与 `gelu`（用 `erf`，无 SFU 指令）均正常。

**(c) `act_first` 维度**：把同一数学喂给 `mul_and_silu`（`act_first=false`，同样走向量化分支）
结果**精确**（`s7b_activation_localize.py`）：

```console
[s7b-loc] silu_and_mul(gate,up)              nonfinite=460/24576 maxerr=inf      nbad>1e-2=1918
[s7b-loc] mul_and_silu(up,gate) [same math]  nonfinite=0/24576   maxerr=0.0251846 nbad>1e-2=314
[s7b-loc] silu_and_mul(gate,0) all-zero: False nonfinite=442      # y=0 也出非有限 ⇒ 是 silu 侧 inf/NaN
```

**(d) 非确定性**（`s7b_activation_repeat.py`，同一输入连打 20 次）：

```console
[s7b-rep] trial= 0 nonfinite=445   ... trial= 2 nonfinite=432   ... trial=19 nonfinite=455
[s7b-rep] masks identical to trial 0: 0/19   union_bad=1953 intersection_bad=1796 trial0_bad=1934
```

⇒ 坏元素集合**不是**逐位可复现（93% 稳定 + 7% 抖动），是**按输入值 + 排布敏感**的硬件/工具链级缺陷，
不是确定性的算错公式（与 S6 归档 §3.5 描述的 gfx906 "VALU→SFU 紧邻" hazard 属同一类现象）。

**(e) gate 网格（`up=1`，应输出 `silu(gate)`）**：坏区集中在 `gate ∈ [-5.75, -1.13]`（对应 `expf` 参数
`+1.13..+5.75`），输出的隐含分母为**负数**（`1+E ≈ 0.18..0.19`，而 `E=exp(5.7)=314`）⇒ `expf`/除法
结果被"读到旧值"式的破坏，与 S6 的 SFU 现象一致：

```console
[s7b-rep] grid bad=96/3072 maxerr=86.1562
[s7b-rep] gate=   -5.7461 kernel=    -29.9219 exact=  -0.0183017
[s7b-rep] gate=   -5.6758 kernel=       -31.5 exact=  -0.0193901
```

### 3.5 真实 Kernel 身份（profiler 原文，纠正了 §3.6 的起点）

```console
$ python tools/dcu/s7b_kernel_identity.py
[s7b-id] kernel='_ZN4vllm18act_and_mul_kernelIN3c104HalfE7__half2TnPFT_RKS4_fEXadL_ZNS_11silu_kernelIS2_EES4_S6_fEETnPFT0_RKSA_fEXadL_ZNS_18packed_silu_kernelIS3_EES4_S6_fEELb1ELb1ELb0ELb0EEEvPS4_PS5_ifff'
  count=1 cuda_time_us=11.679
```

demangle 后为 `vllm::act_and_mul_kernel<c10::Half, __half2, silu_kernel<half>, packed_silu_kernel<__half2>,
true /*act_first*/, true /*use_vec*/, false /*HAS_CLAMP*/, false /*use_256b*/>`，
即 **`csrc/libtorch_stable/activation_kernels.cu` 的向量化分支**（`torch.ops._C.silu_and_mul` 由该 TU 注册）：

- `act_and_mul_kernel` 向量化实现：`csrc/libtorch_stable/activation_kernels.cu:110-159`（分支 `:140` 调 `packed_compute`）
- `packed_compute<...,act_first=true>`：同文件 `:61-90`
- `packed_silu_kernel`（`expf`，SFU）：同文件 `:171-179`
- `use_vec` 选择：同文件 `:250-259`（`const bool use_vec = (d % vec_size == 0);`）
- 注册点：`silu_and_mul` `:299-303`、`silu_and_mul_with_clamp` `:309-315`、`mul_and_silu` `:319-323`
- eager 侧选择：`vllm/model_executor/layers/activation.py:138`（`self.op = torch.ops._C.silu_and_mul`）+
  `:146-151`（`forward_cuda`），由 `CustomOp.dispatch_forward` 在 `custom_ops` 含 `all` 时选中
  （`vllm/model_executor/custom_op.py:167-200`）

> **这也解释了 S7 A/B 里"把 `ir_op_priority` 的 `rms_norm` 改成 `native` 仍然 NaN"**：`ir_op_priority`
> 只作用于编译期 IR 选择；`mode=NONE` 下 `CustomOp` 的开关由 `compilation_config.custom_ops` 决定
> （默认 `all`），所以那次改法在 eager 下是 no-op。

### 3.6 汇编级证据（`hipcc -S`，只出汇编；**未能定论**）

用 `tools/dcu/s7b_repro_act_sfu.cu` + `hipcc -x hip --cuda-device-only -S -O3 --offload-arch=gfx906`
得到 `.s`，可以看到 DTK 的 `expf` 展开为 SFU 序列且**没有等待槽**：

```asm
v_add_f32_e32 v5, v5, v6            ; VALU 写 v5
v_cvt_i32_f32_e32 v9, v10           ; VALU
v_exp_f32_e32 v5, v5                ; SFU 读 v5 —— 与 S6 的 hazard 形态一致，无 s_nop
...
v_rcp_f32_e32 v9, v4                ; SFU
v_ldexp_f32 v5, v5, v9              ; SFU
```

但 **`act_first=true` 与 `act_first=false` 两个实例的这段汇编形态相同**，且该 `.cu` 复现的是
`csrc/activation_kernels.cu`（**非** §3.5 证实的 live TU）。因此本步**没有**取得 live kernel 的反汇编，
"VALU→SFU 无等待槽"只是**与 S6 同类的合理推测**，不构成证明（见 §5.2）。它作为后续 csrc 修复的方向仍有价值。

### 3.7 修复验证（门禁 3）

**(a) eager 逐 token 与两条参考路径完全一致**（§3.1 表 + 原始日志）：

```console
[s7] S7_IDS[eager-fixed]=[12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315, 17689, 374, 24081, 13]
[s7] S7_IDS[cudagraph-none]=[12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315, 17689, 374, 24081, 13]
[s7b-hf] HF_IDS= [12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315, 17689, 374, 24081, 13]
```

**(b) 逐层扫描 0 非有限**（§3.2 统计）。

**(c) 6 个 prompt 全部恢复正常**（`s7_prompts.sh` eager + 修复；S7 修复前为 `locklock...`）：

```console
[s7] PROMPT='The capital of France is' -> ' Paris. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of China is Beijing. The'
[s7] PROMPT='2 + 2 =' -> ' 4, so the sum is 4.\n\nBut wait, the problem says "the sum of the numbers" which'
[s7] PROMPT='Once upon a time,' -> ' there were 3000 people in a town. The number of people who are in the town is 3'
[s7] PROMPT='def fibonacci(n):' -> '\n    if n == 0:\n        return 0\n    elif n == 1:\n        return 1'
[s7] PROMPT='The largest planet in the solar system is' -> '...? A. Mercury B. Venus C. Earth D. Mars\nAnswer:\nThe largest planet in the solar system'
[s7] PROMPT='Water boils at' -> ' 100°C at 1 atm pressure. What is the temperature in Fahrenheit?\n\nTo solve this, we use'
[s7] SECONDS=0.99
```

（与 S7 归档 §3.7.2 的 HF 参考逐条对得上，例如 `2 + 2 =` 的 `' 4, so the sum is 4.\n\nBut wait, the problem says'`。）

**(d) 守卫的边界行为**（`_broken_gfx906_fp16_packed_silu` 直接调用）：

```console
[s7b-guard] cpu fp16 -> False      # 非 GPU
[s7b-guard] cuda fp32 -> False     # 非 fp16
[s7b-guard] cuda fp16 -> True      # 命中
[s7b-guard] cuda bf16 -> False     # bf16 实测正常
```

**(e) 副作用/代价**：eager decode 34.6 → **37.4 ms/token（+8%）**（`act_fn` 从 1 个 kernel 变成
`F.silu` + mul 的 eager 组合）；`VLLM_COMPILE` 路径不受影响（其 `custom_ops=none`，本来就禁用该 op），
编译路径本步实测仍 32.7 ms/token（`s7b-matrix-cudagraph-none.log`，与 S7 的 33.5 同量级）。IR op priority 日志仍为 `rms_norm=['vllm_c','native'] ...`，说明其余
eager 自定义算子未被波及：

```console
INFO [kernel.py:408] Final IR op priority after setting platform defaults:
    IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native'], gelu_and_mul_sparse=['native'])
```

### 3.8 修复的覆盖缺口（同一根因的第二个入口）

`vllm/model_executor/layers/fused_moe/activation.py:356` 在**普通函数** `apply_moe_activation` 里
直调 `torch.ops._C.silu_and_mul`（`:311`/`:387` 调 `silu_and_mul_with_clamp`，`:1720` 的 GGUF 路径同样），
不经过 `CustomOp`，所以 §2.1 的 Python 回退**拦不到**：

```console
$ python tools/dcu/s7b_moe_activation_check.py
[s7b-moe] apply_moe_activation(SILU) fp16 gfx906: nonfinite=1917/98304 max_abs_err=1e+09
```

### 3.9 资源与硬约束自检

```console
$ （收工后，四卡）
devcount 4
0 Z100L 31.73 31.98    1 Z100L 31.73 31.98    2 Z100 31.73 31.98    3 Z100 31.73 31.98
cgroup_mem_current_GiB 3.54            # 上限 10 GiB
overlay / ... 648G avail ; /mnt/game 291G used/648G avail ; /dev/shm 16G 1%

$ docker ps -a
z100-eager Up 21 minutes   z100-s8 Up   z100-serve Up   z100-triton Up   z100-build-211 Up   z100-build Up
qwen38-vllm Exited (137) 14 hours ago     # 生产容器：全程只 docker ps/inspect，未 start/kill/exec
```

- **未执行任何自编译 HIP 独立可执行程序**：只做了 `hipcc -x hip --cuda-device-only -S`（纯汇编）、
  `python`（torch/vLLM/triton JIT）与 `docker exec`；GPU 全程无挂死、无 SIGABRT、无遗留进程。
- 构建树 `z100-build`/`z100-build-211`（含 `/mnt/game/port/build-211`）**未触碰**；DTK 库/RCCL/
  rocBLAS/hipBLASLt/MIOpen/triton 未替换；产物与缓存只在 `/mnt/game/{port/s7b,triton-gfx906/cache}`。
- 未改 `docs/design/*`、`docs/z100-todolist.md`、`docs/result/README.md`。
- 代码风格：`ast.parse` 通过；新增行均 ≤88 列（`ruff` 在镜像内不存在，未跑 linter）。

---

## 4. 门禁判定

| # | 门禁（原文） | 判定 | 证据指向 |
|---|---|---|---|
| 1 | 稳定复现（`mode=NONE`→NaN / `VLLM_COMPILE`→与 HF 逐 token 一致）的最小命令 | ✅ **通过** | §3.1：固定 prompt/greedy/`max_tokens=16` 一条命令拿到 `lock×16` vs 正确 16 token；§3.7(a) 三路 ids 逐 token 相同 |
| 2 | **NaN 是否定位到具体算子** | ✅ **通过** | §3.2 逐层扫描：第一个非有限值 = `model.layers.0.mlp.act_fn`（`SiluAndMul`，6/15360，`absmax` 5320 > 输入 5.922）；§3.3 离线最小复现同一算子；§3.4 划定边界（fp16 + `__half2` 向量化 + `act_first=true`）；§3.5 profiler 给出确切 kernel 实例与 `file:line` |
| 2a | 是否可能是 S6 的 SFU hazard 另一表现 | ⚠️ **同类但未证明** | §3.4(d) 非确定性、§3.4(e) 隐含分母为负、§3.6 无等待槽的 SFU 序列 ⇒ 与 S6 §3.5 同类；但 §3.6 的静态对比不构成证明（§5.2） |
| 2b | 是否涉及 gfx906 特有分支（`on_gfx906` 门控 / gfx906 专用 kernel） | ✅ **通过（否定式结论）** | `rms_norm`/`rotary`/`attention`/`k_norm`/`q_norm` 逐层输出全有限（§3.2）；`_C.rms_norm`/`_C.fused_add_rms_norm` 实测正确，S5 的 gfx906 专用 kernel（`rms_norm_gated_gfx906` 等）与 `on_gfx906()` 门控分支均**不是**病因；缺陷在**上游同步来的通用 fp16 向量化激活 kernel** |
| 3 | 结论与处置（最小修复或精确规避 + 是否改默认值） | ✅ **通过** | §2.1 落地 32 行最小修复；§3.7 端到端验证；§6.1/§6.3 给出 S9 参数与"是否改默认值"的明确建议 |
| 4 | `file:line` 级证据 | ✅ **通过** | §3.5（csrc/python 双侧 9 个锚点）、§2.1（修复行号）、§3.8（缺口行号） |
| 5 | 6–20 行最小复现脚本放 `tools/dcu/` | ✅ **通过** | `tools/dcu/s7b_min_repro.py`（18 行，§3.3 原文输出） |
| 6 | 单卡 `HIP_VISIBLE_DEVICES=2`、新建 `z100-eager`、不碰生产容器/构建树/DTK | ✅ **遵守** | §2.3、§3.9 |
| 7 | 禁止自编译 HIP 可执行程序 / 出现挂死立即如实记录 | ✅ **遵守 + 无挂死** | §3.9（只 `hipcc -S`）；全程 0 挂死、0 僵尸进程 |

**S7b 总体：✅ 门禁全过**——NaN 定位到**具体算子（`SiluAndMul`）与其内部的向量化 kernel 实例**，
根因边界收敛到 `csrc/libtorch_stable/activation_kernels.cu` 的 fp16 `__half2` + `act_first=true` 分支，
最小修复落地并端到端验证（eager == 编译 == HF）。未完成项只有"live kernel 汇编级定性"（§5.2）。

---

## 5. 未完成项 / 阻塞与原因

### 5.1 ⚠️ 修复只覆盖 `CustomOp` 路径；`fused_moe` 等直调点仍会踩雷

- 证据：§3.8（`apply_moe_activation(SILU)` 仍有 1917/98304 非有限）。
- 影响面（代码级排查，未逐一复现）：`vllm/model_executor/layers/fused_moe/activation.py:311,356,387`、
  `vllm/model_executor/layers/fused_moe/utils.py:590`、
  `vllm/model_executor/layers/quantization/gguf.py:1720`、
  `vllm/models/minimax_m3/amd/ops/swiglu_oai.py`（注释表明它**依赖**该向量化算子）。
- **推荐（彻底、且是"正确位置"的修复）**：在 csrc 侧让 gfx906 的 fp16 不走向量化分支，即可覆盖所有调用者。
  落点是 `csrc/libtorch_stable/activation_kernels.cu:259` 的 `use_vec`：

  ```cpp
  // gfx906 (DTK 26.04) miscompiles the fp16 __half2 packed path: the act_first
  // instantiation of packed_silu_kernel returns inf/NaN for a few percent of the
  // elements (docs/result/z100-07b-eager-nan-result.md).  Keep the scalar path.
  const bool use_vec = (d % vec_size == 0) && !(input.element_size() == 2 && is_gfx906());
  ```

  其中 `is_gfx906()` 只是个**未编译验证的占位说明**：该 launch 宏的 host 侧已经拿到
  `get_device_prop()`（`csrc/libtorch_stable/torch_utils.h:56`），本步**没有**验证该 DTK 构建的
  `cudaDeviceProp` 是否可用 `gcnArchName` 做前缀匹配（S6 曾因 DTK 的 `hipDeviceProp_t` 结构差异改过
  triton 的 `driver.py`），因此这段 C++ **在落地前必须先在容器里 `hipcc -S/-c` 编译确认**。
  等价的、已被本步验证过的替代路径是把 `on_gfx906()` 判定复用到直调点
  （`vllm/model_executor/layers/fused_moe/activation.py:356/311/387` 等），代价是改动点更多。
  **未落地原因**：需要重建 `_C`（S4d 口径全量 ninja ≈16 min）并替换 S4c 已"逐字节可复现"验证的四扩展产物，
  属于 Lead 决策范围；且本步不允许触碰 `z100-build*` 的构建树，新建构建树 + 复核 S4 门禁超出本步预算。

### 5.2 ISA 级机理未定性（"为什么"只到可疑级别）

- 未取得 live kernel（`csrc/libtorch_stable/activation_kernels.cu`）的反汇编：该 TU 依赖
  `torch/csrc/stable/tensor.h` 等头，本步的 `hipcc -S` 复现用错了 TU（§3.6），因此**没有**证明
  "VALU→SFU 紧邻且缺等待槽"就是 live kernel 的病因。
- 现有反证式的疑点：live kernel 的两个实例（`act_first` true/false）是本步**通过运行时行为**区分的
  （一个坏一个好），静态代码在 `packed_compute` 里是对称的，差异只可能来自寄存器分配/调度 —— 与
  "hazard 类" 相符，但不足以唯一定位。
- 若要闭合：用 `clang-offload-bundler`/`llvm-objdump` 从 `vllm/_C.abi3.so` 抽出 gfx906 code object 反汇编，
  或把 `libtorch_stable` 的 device 片段抽成独立 TU 用 `hipcc -S` 复现（两者都只编译/反汇编，不违反硬约束）。

### 5.3 明确"未执行"的项

- 任何 csrc 改动与 `_C` 重建（原因见 §5.1）；
- 任何 MoE / GGUF / MiniMax 端到端复现（本步只到 `apply_moe_activation` 的单元级证据，§3.8）；
- 4 卡 / TP>1 / 生产容器相关的任何操作；
- `ruff`/`pre-commit`（镜像内无 `ruff`；已用 `ast.parse` + 88 列人工核对替代）。

---

## 6. 对后续步骤的影响

### 6.1 S9 的验收参数

1. **`--enforce-eager` 现在可以用于 fp16 稠密模型**：Qwen3-0.6B 的 eager 输出已与 `VLLM_COMPILE` 和
   HF **逐 token 一致**（§3.7(a)），逐层无 NaN（§3.7(b)），6 prompt 全正常（§3.7(c)）。
2. **性能口径**：eager decode **37.4 ms/token**（修复后，含 +8% 代价）/ 编译不捕获 graph **32.7** /
   编译 + `FULL_AND_PIECEWISE` **11.6**（S7 §3.4）。S9 对拍基线仍应取编译路径。
3. **KV cache / 显存口径不变**（本次修复不触碰内存路径）：eager `Available KV cache memory` 仍 27.18 GiB。
4. **S9 若跑 MoE / GGUF / MiniMax 的 eager 冒烟，必须先看 §5.1**：这些路径仍会命中坏 kernel
   （`_C.silu_and_mul` 直调），表现为"logits NaN / 输出退化"，与本步修好的 `SiluAndMul` 无关。

### 6.2 是否应该改默认值？

- **不建议把 gfx906 的默认编译模式改成 `enforce_eager`**：数值上现在可行，但
  (a) §5.1 的直调点仍有雷；(b) eager 比编译 graph 慢 3.2×（37.4 vs 11.6 ms/token）；
  (c) 本修复是**绕过**而非修好 kernel。
- **建议保持默认**（`mode=VLLM_COMPILE`，`cudagraph_mode` 默认 `FULL_AND_PIECEWISE`），并把
  "csrc 侧 `use_vec` 门控"（§5.1）列为 S9/S10 固化项；在该 csrc 修复落地并复验前，任务书全局硬约束 13
  （gfx906 禁用 `--enforce-eager`）应**继续保留**——但可补充一句"fp16 稠密模型例外，见 S7b"。

### 6.3 对其它步骤的直接影响

- **S8**：attention / 后端核对与本缺陷无关（§3.2 显示 attention 链完全正常）；S8 若用 eager 做对照实验，
  请确认用的是修复后的 `activation.py`（fork 副本 `/mnt/game/port/gfx906-vllm-211` **已同步**）。
- **S10（MoE / GGUF / 多模态）**：这些模型大量走 `fused_moe/activation.py` 与 GGUF 的直调点，
  **在 gfx906 上 eager 会数值崩坏**；S10 开工前应先做 §5.1 的 csrc 修复，或全程保持编译模式。
- **环境复用**：容器 `z100-eager`（含补装 + shim + 修复后的 fork 副本）保留可用；
  `docker start z100-eager` 即可继续。
- **被本步推翻的前提**：S7 §5.1 推测的"`reshape_and_cache_kernel_flash` / `unified_attention` /
  LBNHC layout 都可能"——**全部排除**（§3.2：attention 全链有限；缺陷在 MLP 激活）。

---

## 7. 复现方式

```bash
# ---------- 0) 容器（已存在则跳过；未加 --rm，保留） ----------
docker ps -a --format '{{.Names}}\t{{.Status}}' | grep z100-eager     # 期望 Up
# 若被删：见 §2.3 的 docker run 命令（或 docker start z100-eager）

# ---------- 1) 环境前缀（每条命令都要）：卡 2 + 自编 triton + fork 副本 ----------
docker exec -it z100-eager bash
source /opt/dtk/env.sh
export S7_HIP_VISIBLE_DEVICES=2
source /mnt/game/port/gfx906-vllm-211/tools/dcu/s7_env.sh
export PYTHONPATH=/mnt/game/port/s7b:$PYTHONPATH
cd /mnt/game/port/gfx906-vllm-211

# ---------- 2) 最小复现（门禁 5；不需要模型，~1 分钟） ----------
python /mnt/game/port/s7b/s7b_min_repro.py
# 期望（坏）：nonfinite = 400±20 / 24576，max_abs_err = 1e9；F.silu 版 max_abs_err ≈ 0.019

# ---------- 3) 缺陷边界：fp16 vec 分支 vs 标量分支 vs bf16（~2 分钟） ----------
python /mnt/game/port/s7b/s7b_activation_vecpath.py     # d%8==0 坏、d%8!=0 好
python /mnt/game/port/s7b/s7b_activation_scope.py       # fp16 silu/clamp 坏；mul_and_silu/gelu/bf16 好
python /mnt/game/port/s7b/s7b_kernel_identity.py        # profiler 打出 packed_silu_kernel 实例名
python /mnt/game/port/s7b/s7b_moe_activation_check.py   # 证明 fused_moe 直调点仍坏

# ---------- 4) 逐层定位（门禁 2；~25 s 引擎启动） ----------
rm -f /mnt/game/port/s7b/layer_scan-eager.jsonl
python /mnt/game/port/s7b/s7b_scan.py /mnt/game/models/Qwen3-0.6B eager "enforce_eager=True"
grep -a NONFINITE /mnt/game/port/s7b/scan-eager.log | head
# 期望（未修复）：第一条即 model.layers.0.mlp.act_fn (SiluAndMul) out bad=6/15360
# 期望（已修复）：nonfinite_modules=0

# ---------- 5) 修复前后端到端 A/B（门禁 1/3） ----------
bash tools/dcu/s7_compare.sh /mnt/game/models/Qwen3-0.6B eager "enforce_eager=True" \
     /mnt/game/port/s7b/s7b-eager.log
grep -aE "S7_IDS|S7_TEXT" /mnt/game/port/s7b/s7b-eager.log
# 已修复期望：' Paris. The capital of Italy is Rome. The capital of Spain is Madrid.'
bash tools/dcu/s7_compile_matrix.sh /mnt/game/models/Qwen3-0.6B cudagraph-none   # 参考路径
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
p='/mnt/game/models/Qwen3-0.6B'; tok=AutoTokenizer.from_pretrained(p)
m=AutoModelForCausalLM.from_pretrained(p, dtype=torch.float16, device_map='cuda:0').eval()
i=tok('The capital of France is', return_tensors='pt').to('cuda:0')
print(m.generate(**i, max_new_tokens=16, do_sample=False)[0][i['input_ids'].shape[1]:].tolist())"
# 参考期望：[12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315, 17689, 374, 24081, 13]

# ---------- 6) 回滚修复（验证"NaN 确实是这一处"） ----------
git -C /home/lyh/z100-vllm stash push vllm/model_executor/layers/activation.py
cp /home/lyh/z100-vllm/vllm/model_executor/layers/activation.py \
   /mnt/game/port/gfx906-vllm-211/vllm/model_executor/layers/activation.py   # 需在宿主执行
# 然后重跑第 5 步：应回到 'locklock...'；恢复：git stash pop + 重新 cp

# ---------- 7) 6 prompt 回归（可选） ----------
bash tools/dcu/s7_prompts.sh /mnt/game/models/Qwen3-0.6B eager-fixed "enforce_eager=True" \
     /mnt/game/port/s7b/s7b-prompts-eager-fixed.log
grep -a PROMPT= /mnt/game/port/s7b/s7b-prompts-eager-fixed.log

# ---------- 8) （可选）只编汇编的复现模板，注意对应非 live TU ----------
hipcc -x hip --cuda-device-only -S -O3 --offload-arch=gfx906 \
      -o /mnt/game/port/s7b/repro_act_gfx906.s /mnt/game/port/s7b/s7b_repro_act_sfu.cu
grep -nE "v_(exp|rcp|ldexp)_f32" /mnt/game/port/s7b/repro_act_gfx906.s | head
```

> 宿主对 `/mnt/game` 只读：第 6 步的 `cp` 与所有产物落盘都必须在容器内（`docker exec`）执行。

---

## 8. 参考

- 前一档：[z100-07-memory-kvcache-graph-result.md](z100-07-memory-kvcache-graph-result.md)（S7；§3.7 现象、§5.1 遗留）
- S6 的同类缺陷：[z100-06-triton-gfx906-result.md](z100-06-triton-gfx906-result.md) §3.5（gfx906 SFU/VALU hazard：
  "只在 SFU 指令的操作数由紧邻的前一条 VALU 指令写入时发生"，修复是插等待槽 `s_nop 2`；该修复只在**自编 triton**里）
- 关键源码位置（本步证据）：
  - `csrc/libtorch_stable/activation_kernels.cu:110-159`（`act_and_mul_kernel`，向量化分支 `:140`）
  - `csrc/libtorch_stable/activation_kernels.cu:61-90`（`packed_compute<...,act_first=true>`）
  - `csrc/libtorch_stable/activation_kernels.cu:171-179`（`packed_silu_kernel`，`expf`）
  - `csrc/libtorch_stable/activation_kernels.cu:250-259`（`use_vec = (d % vec_size == 0)`；fp16 `vec_size=8`）
  - `csrc/libtorch_stable/activation_kernels.cu:299-303 / :309-315 / :319-323`（三个算子的注册与 launch）
  - `vllm/model_executor/layers/activation.py:114-137 / :178-181 / :282-289`（**本步修复**）
  - `vllm/model_executor/layers/activation.py:138 / :146-151 / :237`（原 `_C` 选择点）
  - `vllm/model_executor/custom_op.py:167-200`（`dispatch_forward`：启用的 CustomOp 在 ROCm 上走 `forward_hip`）
  - `vllm/model_executor/layers/fused_moe/activation.py:311 / :356 / :387`（直调 `_C`，修复未覆盖）
  - `vllm/model_executor/layers/fused_moe/utils.py:590`、`vllm/model_executor/layers/quantization/gguf.py:1720`
  - `vllm/platforms/rocm.py:270-272`（`on_gfx906()`）
- 本步脚本（仓库）：[`tools/dcu/s7b_min_repro.py`](../../tools/dcu/s7b_min_repro.py)、
  [`tools/dcu/s7b_hooks.py`](../../tools/dcu/s7b_hooks.py)、[`tools/dcu/s7b_scan.py`](../../tools/dcu/s7b_scan.py)、
  [`tools/dcu/s7b_activation_scope.py`](../../tools/dcu/s7b_activation_scope.py)、
  [`tools/dcu/s7b_activation_vecpath.py`](../../tools/dcu/s7b_activation_vecpath.py)、
  [`tools/dcu/s7b_moe_activation_check.py`](../../tools/dcu/s7b_moe_activation_check.py)、
  [`tools/dcu/s7b_repro_act_sfu.cu`](../../tools/dcu/s7b_repro_act_sfu.cu)（其余 `s7b_*` 见 §2.2）
- 原始日志/数据（全部在 `/mnt/game/port/s7b/`）：`layer_scan-eager.jsonl`、`layer_scan-eager-fixed.jsonl`、
  `scan-eager.log`、`scan-eager-fixed.log`、`s7b-eager-fixed.log`、`s7b-matrix-cudagraph-none.log`、
  `s7b-prompts-eager-fixed.log`、`repro_act_gfx906.s`
