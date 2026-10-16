# S3 bf16 门控与真机判定 — 执行结果

> 归档时间：2026-10-07（宿主机 UTC）/ 2026-10-08 00:15–00:35（Asia/Shanghai，容器日志时间戳）
> 执行者：子 agent（S3 剩余项：bf16 真机判定）
> 对应设计文档：[docs/design/z100-03-bf16-verification.md](../design/z100-03-bf16-verification.md)
> 前一档：无（`docs/result/` 内尚无 S3 前档；S3 的代码部分记在 [z100-todolist.md](../z100-todolist.md)「执行记录」2026-10-07 行）
> 状态：✅ 完成（真机判定与结论已给出；另发现并修复 `tools/dcu/env.sh` 一处会让所有 GEMM 崩溃的真实 bug）

一句话结论：**bf16 在真机上数值可用（探针 PASS，退出码 0），应保留在 gfx906 的 `supported_dtypes` 里，
`VLLM_DCU_BF16` 默认保持 `auto`（=探针通过才启用），fp16 仍排第一，`--dtype auto` 行为不变；
但 bf16 的 GEMM 普遍比 fp16 慢，不应作为默认 dtype。**

---

## 1. 目标与验收标准

### 1.1 本步门禁（[z100-todolist.md](../z100-todolist.md) 门禁表 S3 行原文）

> `tools/dcu/check_bf16.py` 给出明确 PASS/FAIL；据此决定 bf16 是否保留在 `supported_dtypes` 并记录依据

### 1.2 设计文档的四条验收（[z100-03](../design/z100-03-bf16-verification.md) §验收）

1. `check_bf16.py` 给出明确 PASS/FAIL 与数值；
2. `_rocm_supported_dtypes()` 的真机输出与判定结论一致；
3. 结论写入 `docs/hygon_z100.md` 与适配方案 §2.3/§5（A9）；
4. 现有 GPU-free 单测（`test_gfx906_bf16_enabled`、`test_gfx906_supported_dtypes_keep_fp16_first`）保持通过。

### 1.3 本次执行的标准（任务书）

- 在**真实 Z100（gfx906）**上跑 torch 层探针，拿真实命令 + 真实输出原文；
- 补两条判定：`torch.set_default_dtype(torch.bfloat16)` 到底改变了什么；
  `torch.cuda.is_bf16_supported()` 与真实数值结果是否矛盾；
- 给出 bf16 去留结论与 `auto` 门控语义；
- 归档 + 清理；**不碰生产容器 `qwen38-vllm`**；**不执行任何自编译 HIP 二进制**。

---

## 2. 实际改动

| 文件 | 位置 | 改动摘要 |
|---|---|---|
| `tools/dcu/env.sh` | 原第 88–91 行（step 5）→ 现第 88–97 行 | **删除** `export ROCBLAS_TENSILE_LIBPATH="${DTK_ROOT}/lib/rocblas/library"`，替换为一段说明该变量会让 rocBLAS Tensile host 初始化失败（`std::bad_alloc` → SIGABRT/SIGSEGV）的注释。真实 bug 修复：修复前 `source tools/dcu/env.sh` 后**任何** GEMM（含 64×64 fp16）都会让进程崩溃，`check_bf16.py` 根本无法执行（见 §3.3）。 |
| `tools/dcu/README.md` | 第 27–28 行 + 新增第 33–45 行 | 从 `env.sh` 行为清单里移除 `ROCBLAS_TENSILE_LIBPATH`，并新增「不要设置 `ROCBLAS_TENSILE_LIBPATH`」一节，附真实报错。 |
| 其余 | — | **无代码改动**：`vllm/platforms/rocm.py`、`vllm/platforms/hygon_dtk.py`、`tools/dcu/check_bf16.py` 均未改动（真机判定结果与现有实现一致，无需改门控逻辑）。 |

`git status --short` 复核：结束时工作区与开工时同形（`tools/dcu/` 本就是未跟踪目录），未新增/修改其它文件。

---

## 3. 执行过程与原始证据

所有容器操作**只读**访问生产容器（`docker inspect`），未 exec / 未重启 / 未改配置 / 未 commit。
因子 agent 会话本身无 GPU（`/dev/kfd` 打开被沙箱拒绝），全部探针在**新建的一次性容器**内执行。

### 3.1 环境与生产容器（只读 inspect 摘录）

```console
$ docker ps -a --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
NAMES             IMAGE                                                                                      STATUS
qwen38-vllm       harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413   Up 21 minutes
awesome_jemison   harbor.sourcefind.cn:5443/hcu/admin/base/pytorch:2.7.1-ubuntu22.04-dtk26.04-py3.10        Exited (137)

$ docker images | head -3
harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413   88.5GB
harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10-sf_b075  44.5GB
harbor.sourcefind.cn:5443/hcu/admin/base/pytorch:2.7.1-ubuntu22.04-dtk26.04-py3.10         34.4GB

$ docker inspect qwen38-vllm        # 仅只读；下列为关键字段摘录
Image       : harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413
              (sha256:d1c08bf30382f6d8f49e0ad7130ceb4e3a31484ce4bdd3b1aeb7787b5d2fb14e)
Devices     : [{"PathOnHost":"/dev/kfd","PathInContainer":"/dev/kfd","CgroupPermissions":"rwm"},
               {"PathOnHost":"/dev/mkfd","PathInContainer":"/dev/mkfd","CgroupPermissions":"rwm"},
               {"PathOnHost":"/dev/dri","PathInContainer":"/dev/dri","CgroupPermissions":"rwm"}]
GroupAdd    : ["video"]        IpcMode: host      ShmSize: 214748364800
Privileged  : true             SecurityOpt: ["label=disable"]   CapAdd: ["CAP_SYS_PTRACE"]
Binds       : ["/opt/hyhal:/opt/hyhal:ro", "/mnt/game/models:/models", "/mnt/game/triton-cache:/triton-cache"]
Env（节选）  : DTKROOT=/opt/dtk, ROCM_PATH=/opt/dtk, HIP_PATH=/opt/dtk/hip, MIOPEN_FIND_MODE=3,
               AMDGPU_TARGETS=gfx906;gfx926;gfx928;gfx936;gfx938, TZ=Asia/Shanghai, PYTHONUNBUFFERED=1,
               LD_LIBRARY_PATH=/opt/ucx/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:
               /opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:
               /opt/dtk/opencl/lib:/opt/mpi/lib:/opt/hwloc/lib:
               ★ 注意：生产容器**没有** ROCBLAS_TENSILE_LIBPATH（这是 env.sh step 5 引入的差异，见 §3.3）
```

宿主机侧的 GPU 现状（探针期间不变）：

```console
$ hy-smi --showmeminfo vram
DCU[0..3] : vram Total Memory (MiB): 32752
DCU[0..3] : vram Total Used  Memory (MiB): 28329      # 生产容器占 4 卡各 ~28.3 GB，剩 ~4.4 GB/卡
```

### 3.2 探针容器（照抄生产容器的 GPU 设备与 group 参数；**未加 `--privileged`**）

```console
$ mkdir -p /home/lyh/z100-vllm/.tmp-bf16
$ docker run -d --name z100-bf16-probe \
    --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
    --group-add video --ipc=host --security-opt label=disable \
    --shm-size=8g \
    -v /opt/hyhal:/opt/hyhal:ro \
    -v /home/lyh/z100-vllm:/work:ro \
    -v /home/lyh/z100-vllm/.tmp-bf16:/out:rw \
    -e HIP_VISIBLE_DEVICES=3 -e PYTHONUNBUFFERED=1 \
    --entrypoint bash \
    harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 \
    -c 'sleep infinity'
fee6e08950dcc647fc3bb93915357689a363aafdf29f1edc02688d4d6698cfb3
```

容器内 GPU 可见性（`source tools/dcu/env.sh` 后）：

```console
$ timeout 240 docker exec z100-bf16-probe bash -lc 'source /work/tools/dcu/env.sh >/dev/null 2>&1;
  python3 -u -c "
import torch, os
print(\"torch\", torch.__version__, \"hip\", torch.version.hip)
print(\"device_count\", torch.cuda.device_count())
print(\"name\", torch.cuda.get_device_name(0))
print(\"props.gcnArchName\", torch.cuda.get_device_properties(0).gcnArchName)
print(\"is_bf16_supported\", torch.cuda.is_bf16_supported())
"'
torch 2.10.0 hip 6.3.26093
device_count 1
name Z100
props.gcnArchName gfx906:sramecc+:xnack-
is_bf16_supported True
```

`env.sh` 自身输出（可见它把 `VLLM_DCU_GCN_ARCH` 归一为 `gfx906`）：

```console
[dcu] DTK_ROOT=/opt/dtk (26.04)
[dcu] rocm_smi=/opt/dtk/.hyhal/rocm_smi/lib VLLM_DCU_GCN_ARCH=gfx906
[dcu] HIP_VISIBLE_DEVICES=3 VLLM_DCU_BF16=auto
[dcu] verify with: python3 tools/dcu/check_bf16.py
```

### 3.3 第一次尝试：所有 GEMM 崩溃（发现 `env.sh` 真实 bug）

第一次执行 `check_bf16.py`（**修复前**的 `env.sh`，即仍 export `ROCBLAS_TENSILE_LIBPATH`）时，输出停在
环境报告之后，进程在 D 状态（13 个线程不可中断睡眠、CPU 时间不再增长）持续 >10 分钟，只能
`docker rm -f` 终止（该次日志原文，pytorch 2.7.1 镜像）：

```text
== environment ==
torch                 : 2.7.1
torch.version.hip     : 6.3.26045
torch.version.cuda    : None
device count          : 1
WARNING 10-08 00:15:05 [rocm.py:64] Failed to import from vllm._C with ModuleNotFoundError("No module named 'vllm._C'")
WARNING 10-08 00:15:05 [rocm.py:70] Failed to import from vllm._C_stable_libtorch with ModuleNotFoundError("No module named 'vllm._C_stable_libtorch'")
WARNING 10-08 00:15:05 [rocm.py:75] Failed to import from vllm._rocm_C with ModuleNotFoundError("No module named 'vllm._rocm_C'")

rocblaslt warning: No paths matched /opt/dtk/lib/hipblaslt/library/*gfx906*co. Make sure that HIPBLASLT_TENSILE_LIBPATH is set correctly.

rocBLAS error: Could not initialize Tensile host:
std::bad_alloc
```

改用生产镜像定位，最小化到「逐元素算子 vs 矩阵乘」：

```console
$ timeout 300 docker exec z100-bf16-probe bash -lc 'source /work/tools/dcu/env.sh >/dev/null 2>&1
python3 -u -c "
import torch
x=torch.randn(64,64,device=\"cuda\"); y=(x*2+1); torch.cuda.synchronize(); print(\"elementwise ok\", float(y.sum()))
print(\"mem_get_info\", [round(v/2**20) for v in torch.cuda.mem_get_info()])
"; echo "rc_elementwise=$?"
python3 -u -c "
import torch
a=torch.randn(64,64,device=\"cuda\",dtype=torch.float16); b=torch.randn(64,64,device=\"cuda\",dtype=torch.float16)
c=a@b; torch.cuda.synchronize(); print(\"fp16 64x64 mm ok\", float(c.float().abs().max()))
"; echo "rc_fp16mm=$?"'
elementwise ok 3875.59912109375
mem_get_info [4144, 32752]            # 卡 3 仍剩 4144 MiB，排除显存不足
rc_elementwise=0

rocblaslt warning: No paths matched /opt/dtk/lib/hipblaslt/library/*gfx906*co. ...
rocBLAS error: Could not initialize Tensile host:
std::bad_alloc
bash: line 11:  1022 Aborted                 (core dumped) python3 -u -c "..."
rc_fp16mm=134                          # SIGABRT
```

A/B/A 判定 `ROCBLAS_TENSILE_LIBPATH` 的因果（宿主可用内存同时打印，排除内存不足）：

```console
$ timeout 400 docker exec z100-bf16-probe bash -lc 'source /work/tools/dcu/env.sh >/dev/null 2>&1
mm() { python3 -u -c "
import torch
a=torch.randn(64,64,device=\"cuda\",dtype=torch.float16); b=torch.randn(64,64,device=\"cuda\",dtype=torch.float16)
print(\"  matmul ->\", float((a@b).float().abs().max()))
" 2>&1 | grep -vE "^$|rocblaslt warning"; echo "  rc=${PIPESTATUS[0]}"; }
echo "[A] Tensile libpath SET (from env.sh)"; grep MemAvailable /proc/meminfo; export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library; mm
echo "[B] Tensile libpath UNSET"; grep MemAvailable /proc/meminfo; unset ROCBLAS_TENSILE_LIBPATH; mm
echo "[C] Tensile libpath SET again"; grep MemAvailable /proc/meminfo; export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library; mm
echo "[D] env var pointing at the hipblaslt dir"; export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/hipblaslt/library; mm'
[A] Tensile libpath SET (from env.sh)
MemAvailable:    8901920 kB
rocBLAS error: Could not initialize Tensile host:
std::bad_alloc
  rc=134
[B] Tensile libpath UNSET
MemAvailable:    8270400 kB
  matmul -> 31.578125
  rc=0
[C] Tensile libpath SET again
MemAvailable:    8417916 kB
  rc=139                                   # SIGSEGV，与 [A] 同样是该变量的锅
[D] env var pointing at the hipblaslt dir
rocBLAS error: Cannot read /opt/dtk/lib/hipblaslt/library/TensileLibrary.dat: Illegal seek
  rc=134
```

根因：DTK 26.04 的 gfx906 Tensile 目录**只有 lazy 库**，没有 `TensileLibrary.dat`：

```console
$ docker exec z100-bf16-probe bash -c 'ls -la /opt/dtk/lib/rocblas/library/TensileLibrary*.dat; du -sh /opt/dtk/lib/rocblas/library'
-rw-r--r-- 1 root root   35222 /opt/dtk/lib/rocblas/library/TensileLibrary_lazy_gfx906.dat
-rw-r--r-- 1 root root  148323 /opt/dtk/lib/rocblas/library/TensileLibrary_Type_2xi16I_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx906.dat
...（其余为 gfx906 的 `TensileLibrary_Type_*.dat` / `.co`）
344M	/opt/dtk/lib/rocblas/library
# 注意：目录内不存在 TensileLibrary.dat（非 lazy 版本）
```

即：**导出 `ROCBLAS_TENSILE_LIBPATH` 会让 rocBLAS 去读 legacy 非 lazy 的 `TensileLibrary.dat`，
该文件在 DTK 26.04 / gfx906 上不存在 → Tensile host 初始化失败 → 进程 SIGABRT/SIGSEGV**。
生产容器不设这个变量，所以正常。修复见 §2（`env.sh` 不再导出该变量）。

### 3.4 修复后：`check_bf16.py` 真机 PASS（本步主证据）

命令（完整输出原文，未截断）：

```console
$ timeout 900 docker exec z100-bf16-probe bash -lc 'source /work/tools/dcu/env.sh >/dev/null 2>&1;
  export PYTHONPATH=/work; cd /work; python3 -u tools/dcu/check_bf16.py; echo "exit=$?"'
== environment ==
torch                 : 2.10.0
torch.version.hip     : 6.3.26093
torch.version.cuda    : None
device count          : 1
/work/vllm/__init__.py:7: RuntimeWarning: Failed to read commit hash:
No module named 'vllm._version'
  from .version import __version__, __version_tuple__  # isort:skip
WARNING 10-08 00:28:16 [rocm.py:64] Failed to import from vllm._C with ModuleNotFoundError("No module named 'vllm._C'")
WARNING 10-08 00:28:16 [rocm.py:70] Failed to import from vllm._C_stable_libtorch with ModuleNotFoundError("No module named 'vllm._C_stable_libtorch'")
WARNING 10-08 00:28:16 [rocm.py:75] Failed to import from vllm._rocm_C with ModuleNotFoundError("No module named 'vllm._rocm_C'")
hygon_dtk detected    : True
DTK version (sysfs)   : 26.04
gfx arch (KFD sysfs)  : gfx906
  device[0]: 'gfx906:sramecc+:xnack-' sm_count=60
is_bf16_supported()   : True
note                  : is_bf16_supported() is True on every ROCm build and does not reflect gfx906's missing bf16 ISA.

== bf16 correctness ==
[PASS] storage round-trip        max_abs_err=0.011368

rocblaslt warning: No paths matched /opt/dtk/lib/hipblaslt/library/*gfx906*co. Make sure that HIPBLASLT_TENSILE_LIBPATH is set correctly.
[PASS] bf16 GEMM vs fp32 ref     max_abs_err=0.249939 rel=0.229%
[PASS] linear(bf16 x, bf16 w)      rel=0.173%
[PASS] softmax(bf16)              max_abs_err=0.000025

== performance reference (same shapes) ==
  matmul 4096^3    torch.float16:     7.84 ms  ( 17.52 TFLOP/s)
  matmul 4096^3   torch.bfloat16:    21.54 ms  (  6.38 TFLOP/s)

== verdict ==
bf16 numerics : usable
bf16 vs fp16  : 0.36x (matmul)
recommendation: keep fp16 for throughput; enable bf16 only when a model requires it:
    export VLLM_DCU_BF16=1   # force-enable on gfx906
    vllm serve <model> --dtype bfloat16
note          : vLLM's own probe is the same check; with VLLM_DCU_BF16=auto, bf16 is offered automatically when this script passes.
exit=0
```

> 与设计文档的对照（**不改设计文档**，仅备忘）：设计文档 §目的 引用的「生产 profiler 显示 bf16 rocBLAS GEMM ≈ 8.2 ms」
> 与本机 4096³ 实测不符（bf16 21.54 ms、fp16 7.84 ms）。我们的 7.84 ms 反而接近 8.2 ms，故该数字很可能不是
> 「4096³ bf16 matmul」（或为不同 shape / 实为 fp16）。建议后续 A/B 时用 `--profiler` 复核该基线数字。

**可复现性复核（第二次运行，全新容器 `z100-bf16-probe` 重建后、相同命令）：逐字一致，仍 `exit=0`：**

```text
[PASS] storage round-trip        max_abs_err=0.011368
[PASS] bf16 GEMM vs fp32 ref     max_abs_err=0.249939 rel=0.229%
[PASS] linear(bf16 x, bf16 w)      rel=0.173%
[PASS] softmax(bf16)              max_abs_err=0.000025
  matmul 4096^3    torch.float16:     7.83 ms  ( 17.54 TFLOP/s)
  matmul 4096^3   torch.bfloat16:    21.53 ms  (  6.38 TFLOP/s)
bf16 numerics : usable / bf16 vs fp16 : 0.36x (matmul)
```

### 3.5 判定一：`torch.set_default_dtype(torch.bfloat16)` 不是能力开关

（脚本体为一次性 python 片段：每个 dtype 打印 `torch.get_default_dtype()`/`tensor/randn/zeros/ones/eye/empty/arange`
的 dtype、`nn.Linear` 权重 dtype、`is_bf16_supported()` 与 `allow_bf16_reduced_precision_reduction`，然后
`torch.set_default_dtype(torch.bfloat16)` 再打一遍；精简可粘贴版见 §7 步骤 7。）

```console
$ timeout 400 docker exec z100-bf16-probe bash -lc 'source /work/tools/dcu/env.sh >/dev/null 2>&1
python3 -u -c "<snapshot before/after set_default_dtype>"'
== torch.set_default_dtype(torch.bfloat16) semantics ==
--- before ---
  torch.get_default_dtype()      : torch.float32
  torch.tensor([1.0]).dtype      : torch.float32
  torch.randn(2,2).dtype         : torch.float32
  torch.zeros(2).dtype           : torch.float32
  torch.ones(2).dtype            : torch.float32
  torch.eye(2).dtype             : torch.float32
  torch.empty(2).dtype           : torch.float32
  torch.arange(3).dtype          : torch.int64
  nn.Linear(4,4).weight.dtype    : torch.float32
  is_bf16_supported()            : True
  allow_bf16_reduced_precision_reduction: True
--- after  ---
  torch.get_default_dtype()      : torch.bfloat16
  torch.tensor([1.0]).dtype      : torch.bfloat16
  torch.randn(2,2).dtype         : torch.bfloat16
  torch.zeros(2).dtype           : torch.bfloat16
  torch.ones(2).dtype            : torch.bfloat16
  torch.eye(2).dtype             : torch.bfloat16
  torch.empty(2).dtype           : torch.bfloat16
  torch.arange(3).dtype          : torch.int64          # 整型默认不变
  nn.Linear(4,4).weight.dtype    : torch.bfloat16
  is_bf16_supported()            : True                 # 不变
  allow_bf16_reduced_precision_reduction: True          # 不变
  cuda matmul still works, out dtype : torch.bfloat16 33.25
  explicit fp16 tensor still possible: torch.float16
```

结论：它只改「新建 **浮点** 张量的默认 dtype」（含 `nn.Linear` 权重），
整型默认不变、`is_bf16_supported()` 不变、后端开关标志不变、也不启用任何新 kernel
（bf16 matmul 在调用它之前就已可用）。**它不是 bf16 能力开关**；vLLM 的开关是 `--dtype bfloat16`
配合 `supported_dtypes` 门控。

### 3.6 判定二：`torch.cuda.is_bf16_supported()` 不可信（含 `including_emulation=False` 反证）

```console
$ timeout 500 docker exec z100-bf16-probe bash -lc 'source /work/tools/dcu/env.sh >/dev/null 2>&1;
  python3 -u -c "<is_bf16_supported / gcnArchName / decode-shaped GEMM 基准>"'
== is_bf16_supported() vs measured reality ==
gcnArchName                     : gfx906:sramecc+:xnack-
is_bf16_supported()             : True
is_bf16_supported(including_emulation=False): True      # ← 连"仅硬件"查询也返回 True
torch.cuda.get_device_capability(): (9, 0)              # torch 把 gfx906 当成 CC 9.0（Hopper 级 bf16）
```

**如实说明（不编造矛盾）**：本机上 `is_bf16_supported()` 返回 `True`，而 bf16 数值探针也 PASS，
两者在「数值可用」这一点上**并不矛盾**；它「不可信」体现在**它无法区分硬件支持与模拟**：
在汇编器已证明没有 `v_cvt_f32_bf16` / `v_dot2_f32_bf16` 的 gfx906 上，连
`is_bf16_supported(including_emulation=False)` 仍然返回 `True`（该参数在 ROCm 构建下被忽略），
且它也不反映性能（bf16 4096³ 只有 fp16 的 0.36x）。
因此它只作为对照打印，**不得作为判据**——与设计文档 §事实 一致。

### 3.7 性能补充：不同 shape / 布局的 bf16 vs fp16（真实数据，两轮复现）

`F.linear(x[M,4096], w[4096,4096])`（vLLM linear 的实际调用形态），50 iters，两轮交替顺序：

```text
== F.linear(x[M,4096], w[4096,4096]) two rounds, alternating order ==
  round1 M=    1: fp16    1.014 ms (  0.03 TFLOP/s) | bf16    0.249 ms (  0.13 TFLOP/s) | bf16/fp16  0.25x
  round1 M=    8: fp16    0.189 ms (  1.42 TFLOP/s) | bf16    0.179 ms (  1.50 TFLOP/s) | bf16/fp16  0.95x
  round1 M=  128: fp16    0.480 ms (  8.94 TFLOP/s) | bf16    0.716 ms (  6.00 TFLOP/s) | bf16/fp16  1.49x
  round1 M=  512: fp16    3.683 ms (  4.67 TFLOP/s) | bf16    2.591 ms (  6.63 TFLOP/s) | bf16/fp16  0.70x
  round1 M= 2048: fp16    4.524 ms ( 15.19 TFLOP/s) | bf16   10.315 ms (  6.66 TFLOP/s) | bf16/fp16  2.28x
  round2 M=    1: fp16    1.001 ms (  0.03 TFLOP/s) | bf16    0.174 ms (  0.19 TFLOP/s) | bf16/fp16  0.17x
  round2 M=    8: fp16    0.189 ms (  1.42 TFLOP/s) | bf16    0.180 ms (  1.49 TFLOP/s) | bf16/fp16  0.95x
  round2 M=  128: fp16    0.491 ms (  8.74 TFLOP/s) | bf16    0.716 ms (  6.00 TFLOP/s) | bf16/fp16  1.46x
  round2 M=  512: fp16    3.685 ms (  4.66 TFLOP/s) | bf16    2.592 ms (  6.63 TFLOP/s) | bf16/fp16  0.70x
  round2 M= 2048: fp16    4.528 ms ( 15.18 TFLOP/s) | bf16   10.317 ms (  6.66 TFLOP/s) | bf16/fp16  2.28x

== correctness of the same call ==
  M=   8 bf16 linear rel_err=0.200%   | M=   8 fp16 linear rel_err=0.025%
  M= 512 bf16 linear rel_err=0.311%   | M= 512 fp16 linear rel_err=0.039%
```

布局对照（NN = `x@w`，NT = `x@w.T` = `F.linear` 的底层形态）：

```text
x[M,4096] @ w[4096,4096] (NN) vs x @ w.T (NT, = F.linear layout)
  M=    1  fp16: NN  0.108 ms / NT  1.000 ms  |  bf16: NN  0.137 ms / NT  0.249 ms
  M=    8  fp16: NN  0.080 ms / NT  0.195 ms  |  bf16: NN  0.138 ms / NT  0.180 ms
  M=  128  fp16: NN  0.337 ms / NT  0.479 ms  |  bf16: NN  0.798 ms / NT  0.717 ms
  M=  512  fp16: NN  1.329 ms / NT  3.745 ms  |  bf16: NN  2.452 ms / NT  2.593 ms
```

解读（供 S9 A/B 参考，**不是结论性断言**）：
- bf16 在 plain matmul（NN）上**一致更慢**：M=8 1.7x、M=128 2.4x、M=512 1.8x；4096³ 为 0.36x（2.75x 慢）；
- `F.linear`（NT）上 bf16 反而在 M=1 / M=512 更快、M=128 / M=2048 更慢 —— 原因是每个 shape 选中的 Tensile 解不同；
- 异常点：**fp16 在 NT + M=1 上慢得离谱（1.000 ms ≈ 0.03 TFLOP/s，两轮复现）**，bf16 同 shape 仅 0.249 ms。
  这可能是 gfx906 上 fp16 某个 Tensile 解的病态表现，建议 S9 用 profiler 复核 decode 路径（M=1）；
- 结论不受影响：bf16 数值可用但总体不更快，**默认仍应 fp16**。

### 3.8 判定三：`_rocm_supported_dtypes()` 真机输出（三种 `VLLM_DCU_BF16` 取值）

```console
$ for pol in auto 0 1; do echo "=== VLLM_DCU_BF16=$pol ==="; \
    VLLM_DCU_BF16=$pol timeout 600 docker exec z100-bf16-probe bash -lc 'source /work/tools/dcu/env.sh >/dev/null 2>&1; export PYTHONPATH=/work; python3 -u -c "...";'; done
=== VLLM_DCU_BF16=auto ===
  _GCN_ARCH                      : gfx906
  on_gfx906()                    : True
  bf16_policy()                  : auto
  dcu_bf16_probe()               : True
  gfx906_bf16_enabled()          : True
  _rocm_supported_dtypes()       : ['torch.float16', 'torch.bfloat16', 'torch.float32']
=== VLLM_DCU_BF16=0 ===
  _GCN_ARCH                      : gfx906
  on_gfx906()                    : True
  bf16_policy()                  : off
  dcu_bf16_probe()               : True          # 探针仍然 True，只是策略强制关
  gfx906_bf16_enabled()          : False
  _rocm_supported_dtypes()       : ['torch.float16', 'torch.float32']
=== VLLM_DCU_BF16=1 ===
  _GCN_ARCH                      : gfx906
  on_gfx906()                    : True
  bf16_policy()                  : on
  dcu_bf16_probe()               : True
  gfx906_bf16_enabled()          : True
  _rocm_supported_dtypes()       : ['torch.float16', 'torch.bfloat16', 'torch.float32']
```

（三种取值下 `torch.float16` 都在第 0 位 → `--dtype auto` 仍选 fp16；日志中
`Failed to import from vllm._C/_rocm_C` 属预期，本 fork 未编译扩展，不影响平台层判定。）

**复现时的坑（实测，务必先 `import torch`）**：在全新进程里如果**先** `from vllm.platforms import rocm`
（即让 torch 在 vllm 的深层 import 链里才被导入），本机会稳定 abort；先 `import torch` 再导入则正常。
A/B 对照（同一容器、同一 `env.sh`、各跑 3/3 与 2/2）：

```console
$ python3 -u -c "from vllm.platforms import rocm; print('ok')"
terminate called after throwing an instance of 'std::bad_alloc'
  what():  std::bad_alloc
Aborted (core dumped)          # rc=134，3/3 次

$ python3 -u -c "import torch; from vllm.platforms import rocm; print('ok', rocm._GCN_ARCH, [str(d) for d in rocm._rocm_supported_dtypes()])"
ok gfx906 ['torch.float16', 'torch.bfloat16', 'torch.float32']   # rc=0，2/2 次
```

§3.8 上表的证据即用后者（`import torch` 在前）取得；`check_bf16.py` 模块顶部本来就 `import torch`，不受影响。
真正运行 vLLM 时 `vllm/__init__.py` 也先导入 torch，故这是「手写一行式探针」才需要注意的顺序问题。

按 §7 步骤 6 原样重跑（`import torch` 在前）的完整形态，同时给出 **vLLM 内部探针自己的数值**：

```console
$ for pol in auto 0 1; do ... python3 -u /tmp/gate.py 2>/dev/null; done
=== VLLM_DCU_BF16=auto ===
INFO 10-08 00:48:22 [hygon_dtk.py:398] bf16 probe passed on this device (round-trip 0.00780, gemm 0.12494, softmax 0.00024).
gfx906 auto True True ['torch.float16', 'torch.bfloat16', 'torch.float32']
=== VLLM_DCU_BF16=0 ===
INFO 10-08 00:48:43 [hygon_dtk.py:398] bf16 probe passed on this device (round-trip 0.00780, gemm 0.12494, softmax 0.00024).
gfx906 off True False ['torch.float16', 'torch.float32']
=== VLLM_DCU_BF16=1 ===
INFO 10-08 00:48:49 [hygon_dtk.py:398] bf16 probe passed on this device (round-trip 0.00780, gemm 0.12494, softmax 0.00024).
gfx906 on True True ['torch.float16', 'torch.bfloat16', 'torch.float32']
```

即 `dcu_bf16_probe()`（128×256 形状）本身也通过，数值与 `check_bf16.py` 同量级。

### 3.9 GPU-free 单测（门禁 4）在容器内无法运行 + 等效真机验证

容器内跑 `pytest`（生产镜像，pytest 9.0.2）失败，原因与本次改动无关，是 DTK `amdsmi` 让进程 abort：

```console
$ docker exec z100-bf16-probe bash -lc 'cd /work && python3 -m pytest tests/platforms/test_hygon_dtk.py -q'
ImportError while loading conftest '/work/tests/conftest.py'.
tests/conftest.py:7: in <module>
    from tblib import pickling_support
E   ModuleNotFoundError: No module named 'tblib'

$ # 绕过仓库 conftest，把测试文件复制到 /tmp 再跑：
bash: line 2: 12050 Aborted (core dumped) python3 -m pytest test_hygon_dtk.py -q -p no:cacheprovider
Fatal Python error: Aborted
Current thread ... (most recent call first):
  File "/usr/local/lib/python3.10/dist-packages/amdsmi/amdsmi_interface.py", line 1593 in amdsmi_init
  File "/usr/local/lib/python3.10/dist-packages/torch/cuda/__init__.py", line 834 in _raw_device_count_amdsmi
  ... 
  File "/usr/local/lib/python3.10/dist-packages/triton/utils/hcutuner.py", line 435 in get_gpu_label
  File "/usr/local/lib/python3.10/dist-packages/triton/utils/__init__.py", line 41 in <module>
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_triton.py", line 9 in has_triton_package
  ...
# 用 -p no:assertion / --assert=plain 同样 abort（torch._dynamo 已被导入）
```

等效真机验证（比单测更强，直接跑被断言的代码路径）：

| 单测断言 | 真机对应证据 |
|---|---|
| `test_gfx906_bf16_enabled`（`1/on/force`→True；`0/off`→False；`auto`→探针值） | §3.8：`auto`→probe True→True；`0`→False；`1`→True |
| `test_gfx906_supported_dtypes_keep_fp16_first`（关=`[fp16,fp32]`；开=`[fp16,bf16,fp32]`） | §3.8 三种策略的真机输出逐字一致 |
| 本机无 torch / 无 `.venv`（宿主 `python3 -c "import torch"` → `ModuleNotFoundError`） | 该测试需 `pytest`+`torch`，只能在实际 dev venv 里跑 → 见 §5 |

### 3.10 清理与生产容器复核（只读）

```console
$ docker rm -f z100-bf16-probe
$ docker ps -a --filter name=z100-bf16-probe --format '{{.Names}}'
（空）

$ # 生产容器前后状态对比（StartedAt/PID/RestartCount 完全一致，说明从未被重启/重建）
BEFORE: {"StartedAt": "2026-10-07T15:52:36.204216458Z", "Status": "running", "Pid": 534013, "Running": true, "Restarting": false, "ExitCode": 0}
AFTER : 2026-10-07T15:52:36.204216458Z running 534013 true false 0
RestartPolicy=no   RestartCount=0

$ curl -s -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:8000/v1/models
HTTP 200
```

对 `qwen38-vllm` 的全部接触 = `docker inspect`（只读）+ 上面的 `docker ps`/`docker images`/状态对比；
**未 exec、未重启、未改配置、未 commit**。探针期间没有执行任何自编译 HIP 二进制（只有 torch 层
python 探针），未在根分区落任何大文件（临时日志在仓库内 `.tmp-bf16/`，结束时已删除）。

> 说明：本步结束时 `git status --short` 里还会看到 `setup.py`、`requirements/dcu*.txt` 等改动——
> 那是**并行进行的 S4 子 agent** 的产物（mtime 16:19–16:24），与本步无关。本步只改动
> `tools/dcu/env.sh`、`tools/dcu/README.md`，并新增本归档文件。

---

## 4. 门禁判定

### 4.1 S3 门禁（todolist）

| 门禁 | 判定 | 证据指向 |
|---|---|---|
| `tools/dcu/check_bf16.py` 给出明确 PASS/FAIL | **PASS**（退出码 0，四项全 PASS） | §3.4 完整输出：storage `0.011368`、bf16 GEMM `rel=0.229%`、linear `rel=0.173%`、softmax `2.5e-05`、`exit=0` |
| 据此决定 bf16 是否保留在 `supported_dtypes` 并记录依据 | **保留**（见 §4.3 结论） | §3.4 PASS + §3.8 真机 `_rocm_supported_dtypes()`；依据写入本档 |

### 4.2 设计文档四条验收

| 验收 | 判定 | 证据指向 |
|---|---|---|
| 1. `check_bf16.py` 明确 PASS/FAIL 与数值 | **PASS** | §3.4 |
| 2. `_rocm_supported_dtypes()` 真机输出与结论一致 | **PASS** | §3.8（auto→含 bf16 且 fp16 第一；0→不含 bf16） |
| 3. 结论写入 `docs/hygon_z100.md` 与适配方案 §2.3/§5（A9） | **未执行（非本 agent 写作用域）** | 任务书与 `docs/result/README.md` §7 限定子 agent 只写自己的归档；结论已在本档 §4.3 备好，交 Lead 回写 |
| 4. 现有 GPU-free 单测保持通过 | **未在本环境执行（原因见 §5），但有等效真机验证** | §3.9：容器 pytest 因 DTK amdsmi abort；被断言的三条路径已在真机上逐条验证一致 |

### 4.3 最终结论（bf16 去留 + 默认门控语义）

**结论：保留 bfloat16，`VLLM_DCU_BF16` 默认维持 `auto`，不改 `rocm.py` / `hygon_dtk.py`。**

- `check_bf16.py` 退出码 0 → 按设计文档 §做法 判定规则，走「保留」分支；
  且实测 bf16 明显更慢（4096³ 0.36x；NN 形状 1.7–2.4x 慢）→ 走「**保留可选（`auto`），并给出
  『默认仍用 fp16』建议**」这一分支，而不是把 bf16 设为默认。
- `auto` 的语义（实测确认）：`gfx906_bf16_enabled()` = 探针 `dcu_bf16_probe()` 的结果；
  探针通过 → bf16 **插入到 fp16 之后**（`[fp16, bf16, fp32]`），探针失败 → `[fp16, fp32]`。
  由于 fp16 恒排第一，**`--dtype auto` 的行为不因本判定而改变**（仍解析为 fp16）；
  变化只在于「显式 `--dtype bfloat16` / bf16-only 权重」在 gfx906 上现在开箱可用。
- 不建议把默认改为 `0`（那会关掉 bf16-only 模型的可用性，而数值实测是可靠的）；
  也不建议改成默认 `1`（会绕过探针，且 bf16 更慢）。
- `VLLM_DCU_BF16=0` 仍是可用的保守逃生门（§3.8 验证 `[fp16, fp32]`）。

---

## 5. 未完成项 / 阻塞与原因

| 项 | 状态 | 原因 / 说明 |
|---|---|---|
| 门禁 3：回写 `docs/hygon_z100.md` 与适配方案 §2.3/§5（A9） | **未执行** | 任务书 + `docs/result/README.md` §7 规定子 agent 不改设计文档/主文档；结论与数字已在 §4.3、§3.4 备好，交 Lead 回写 |
| 门禁 4：在 dev venv 跑 `python -m pytest tests/platforms/test_hygon_dtk.py -v` | **未执行** | 本 agent 宿主无 torch、无 `.venv`；容器内 pytest 因生产镜像的 DTK `amdsmi_init` abort（§3.9）而无法运行。等效真机验证见 §3.9 |
| 与生产基线的完整 A/B 对拍（同模型/同 prompt/同并发，decode ms/token、TTFT、吞吐、质量抽查） | **未执行** | 任务书写明**不得触碰生产容器 `qwen38-vllm`**（不许 exec / 不许重启），故无法在其上做 dtype A/B；另起服务需要显存（4 卡各仅剩 ~4.4 GB）与长时间占用，超出本步范围。本档给出 GEMM/linear 层实测作为代理指标，完整 A/B 归 S9 |
| bf16 的 KV cache / 量化 / MTP 路径 | **未验证** | 不在 S3 范围（设计文档 §失败处置已说明 bf16+MTP 会以 `not implemented for BFloat16` 明确失败） |
| 设计文档引用的「生产 bf16 GEMM ≈ 8.2 ms」基线数字 | **待复核** | 与本次 4096³ 实测不符（bf16 21.54 ms / fp16 7.84 ms），可能是不同 shape 或实为 fp16；建议 S9 用 profiler 复核（不改设计文档） |
| 一次不成功的早期尝试 | 已记录 | pytorch 2.7.1 镜像的首次运行在 Tensile 失败后进程卡在 D 状态 >10 min，已 `docker rm -f` 终止；根因同 §3.3 |

无环境阻塞留给后续：本步在真机上**成功拿到了判定所需全部数据**。

---

## 6. 对后续步骤的影响

1. **S4/S5 可以继续**：bf16 门控无需改动，`vllm/platforms/*` 保持现状。
2. **S9（验收/对拍）需知的坑**：
   - 复现环境**必须先修掉 `env.sh` 的 `ROCBLAS_TENSILE_LIBPATH`**（否则一切 GEMM 崩溃）；
     其它脚本/CI/文档若复制了旧 env.sh 的导出行为，也会崩。
   - decode 路径（NT + M=1）上 fp16 的 Tensile 解异常慢（1.000 ms vs bf16 0.249 ms），
     A/B 时不要把这个当成 bf16 的优势，需用 profiler 定位（可能是 fp16 解选择问题）。
   - 生产基线数字 8.2 ms 需要重新确认口径。
   - **手写一行式探针要先 `import torch`**：先 `from vllm.platforms import rocm` 会在本机稳定
     `std::bad_alloc` abort（§3.8）。`check_bf16.py` 与实际 vLLM 启动不受影响（它们先导入 torch）。
3. **文档回写（Lead）**：把 §4.3 结论与 §3.4/§3.6/§3.7 数字写入 `docs/hygon_z100.md` §2 与适配方案 §2.3/§5（A9）；
   同时把 `env.sh` 的 `ROCBLAS_TENSILE_LIBPATH` 坑记入 `docs/hygon_z100.md`（属高危环境坑）。
4. **对 S3 状态的建议**：本步真机判定可置 ✅（门禁 3 的文档回写由 Lead 完成后收尾），
   `docs/result/README.md` 索引表 S3 行由 Lead 更新。

---

## 7. 复现方式

> 前置：当前用户在 `docker` 组；**只需只读 inspect 生产容器**；不要 exec/重启 `qwen38-vllm`。
> 全程只用 torch 层 python 探针，**不要执行任何自编译 HIP 二进制**。

```bash
# 0) 只读确认生产容器的设备/group/挂载（不要 exec、不要重启）
docker inspect qwen38-vllm | head -60
docker ps -a | grep qwen38-vllm

# 1) 准备可写挂载点（日志落这里，用完删除；不要往根分区写大文件）
mkdir -p /home/lyh/z100-vllm/.tmp-bf16

# 2) 起一次性探针容器（GPU 设备与 group 照抄生产容器；单卡 = 3 号 DCU）
docker run -d --name z100-bf16-probe \
  --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
  --group-add video --ipc=host --security-opt label=disable \
  --shm-size=8g \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v /home/lyh/z100-vllm:/work:ro \
  -v /home/lyh/z100-vllm/.tmp-bf16:/out:rw \
  -e HIP_VISIBLE_DEVICES=3 -e PYTHONUNBUFFERED=1 \
  --entrypoint bash \
  harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 \
  -c 'sleep infinity'

# 3) 确认 GPU 可见
timeout 240 docker exec z100-bf16-probe bash -lc 'source /work/tools/dcu/env.sh >/dev/null 2>&1;
  python3 -u -c "import torch;print(torch.__version__,torch.version.hip,torch.cuda.device_count(),torch.cuda.get_device_name(0),torch.cuda.get_device_properties(0).gcnArchName)"'
# 期望：2.10.0 6.3.26093 1 Z100 gfx906:sramecc+:xnack-

# 4) 主探针（PASS 期望 exit=0；FAIL=1；无 GPU=2）
timeout 900 docker exec z100-bf16-probe bash -lc 'source /work/tools/dcu/env.sh >/dev/null 2>&1;
  export PYTHONPATH=/work; cd /work; python3 -u tools/dcu/check_bf16.py; echo "exit=$?"'

# 5) 复现 §3.3 的 env.sh 崩溃（可选，验证 bug 已修）
timeout 300 docker exec z100-bf16-probe bash -lc 'source /work/tools/dcu/env.sh >/dev/null 2>&1;
  export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library;
  python3 -u -c "import torch;a=torch.randn(64,64,device=\"cuda\",dtype=torch.float16);print((a@a).sum().item())"; echo "rc=$?"'
# 期望：rocBLAS error: Could not initialize Tensile host: std::bad_alloc / rc=134（或 139）

# 6) 真机 dtype 门控（三种策略；注意必须先 import torch，见 §3.8 的坑）
docker exec z100-bf16-probe bash -lc 'cat > /tmp/gate.py <<"EOF"
import torch                      # 必须先导入 torch：否则本机 import 链会 abort (std::bad_alloc)
from vllm.platforms import rocm
from vllm.platforms.hygon_dtk import bf16_policy, dcu_bf16_probe, gfx906_bf16_enabled
print(rocm._GCN_ARCH, bf16_policy(), dcu_bf16_probe(), gfx906_bf16_enabled(),
      [str(d) for d in rocm._rocm_supported_dtypes()])
EOF'
for pol in auto 0 1; do
  echo "=== VLLM_DCU_BF16=$pol ==="
  timeout 600 docker exec z100-bf16-probe bash -lc "source /work/tools/dcu/env.sh >/dev/null 2>&1; export PYTHONPATH=/work VLLM_DCU_BF16=$pol; python3 -u /tmp/gate.py 2>/dev/null"
done
# 本档 §3.8 的真实输出形态（已验证）：
#   gfx906 auto True True ['torch.float16', 'torch.bfloat16', 'torch.float32']
#   gfx906 off  True False ['torch.float16', 'torch.float32']
#   gfx906 on   True True ['torch.float16', 'torch.bfloat16', 'torch.float32']
# 期望：gfx906 auto True True [fp16, bf16, fp32] / gfx906 off True False [fp16, fp32] / gfx906 on True True [fp16, bf16, fp32]

# 7) set_default_dtype 语义与 is_bf16_supported 反证（一次性片段，不落仓库）
timeout 400 docker exec z100-bf16-probe bash -lc 'source /work/tools/dcu/env.sh >/dev/null 2>&1;
  python3 -u -c "
import torch
print(\"before\", torch.get_default_dtype(), torch.randn(2).dtype, torch.arange(2).dtype)
torch.set_default_dtype(torch.bfloat16)
print(\"after \", torch.get_default_dtype(), torch.randn(2).dtype, torch.arange(2).dtype)
print(\"is_bf16_supported\", torch.cuda.is_bf16_supported(), torch.cuda.is_bf16_supported(including_emulation=False))
print(\"gcnArchName\", torch.cuda.get_device_properties(0).gcnArchName)"'

# 8) 清理（必须）
docker rm -f z100-bf16-probe
docker ps -a | grep z100-bf16-probe || echo "probe container removed"
rm -rf /home/lyh/z100-vllm/.tmp-bf16
git status --short
```

---

## 8. 参考

- [Todolist S3 门禁与全局硬约束](../z100-todolist.md)
- [S3 设计文档](../design/z100-03-bf16-verification.md)
- [适配记录与实测证据 §2 bf16 专项](../hygon_z100.md)
- 代码：[`vllm/platforms/hygon_dtk.py`](../../vllm/platforms/hygon_dtk.py)（`bf16_policy` / `dcu_bf16_probe` / `gfx906_bf16_enabled`）、
  [`vllm/platforms/rocm.py`](../../vllm/platforms/rocm.py)（`_rocm_supported_dtypes`）、
  [`tools/dcu/check_bf16.py`](../../tools/dcu/check_bf16.py)、
  [`tools/dcu/env.sh`](../../tools/dcu/env.sh)、[`tools/dcu/README.md`](../../tools/dcu/README.md)
- 映像：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413`
  （sha256:d1c08bf30382f6d8f49e0ad7130ceb4e3a31484ce4bdd3b1aeb7787b5d2fb14e）、
  `harbor.sourcefind.cn:5443/hcu/admin/base/pytorch:2.7.1-ubuntu22.04-dtk26.04-py3.10`（仅首次尝试使用）
- ROCm/pytorch `is_bf16_supported()` 在 ROCm 上恒为 True 的上游依据：<https://github.com/ROCm/pytorch/commit/363320f6b31645bf26f890ebc64e6ce11993a8cd>
