# S11a A7（容器内「非 DTK 行为不变」对照测试）— 执行结果

> 归档时间：2026-10-09
> 执行者：子 agent（S11a / A7）
> 对应设计文档：`docs/design/z100-09-acceptance-and-bench.md`（A7 条目）+ `docs/海光Z100-性能优化方案.md` §9
> 前一档：[z100-09-acceptance-and-bench-result.md](z100-09-acceptance-and-bench-result.md)（A7 只做了 GPU-free 的 `test_hygon_dtk.py`）
> 状态：✅ 完成（任务书列出的 1–6 步全部执行；`tests/kernels/` 抽样不在本步任务书范围，见 §5）

## 1. 目标与验收标准

**A7 原文（设计 §A7）**：「ROCm 回归为零：无 DTK 环境下抽样跑 `tests/platforms/`、`tests/kernels/`，行为与改动前一致。」

**方案 §9 给出的可执行判据**（本步任务书）：宿主没有社区 ROCm/CUDA 环境，因此在 **DTK 26.04 容器内造「非 DTK 条件」**，然后：

1. DTK 存在时 `is_hygon_dtk() is True`，记录 `gcn_arch()` / `rocm_platform_plugin()` / `_rocm_supported_dtypes()`（`VLLM_DCU_BF16=auto/0/1`）；
2. 造非 DTK：`DTKROOT=`/`HYHAL_PATH=` 置空、`ROCM_PATH=/opt/rocm`（不存在）、隐藏 `/opt/dtk/.info/rocm_version`，断言 `is_hygon_dtk() is False`；
3. 真实 pytest 跑 `tests/platforms/test_hygon_dtk.py`（DTK 存在 / 非 DTK 模拟两种环境）；
4. **与 HEAD 逐条对照**：`rocm_platform_plugin()`、`_get_gcn_arch()`、`RocmPlatform.supported_dtypes`、`get_device_name/uuid/total_memory`、`is_fully_connected()`、非 gfx906 下的 `_rocm_supported_dtypes()`；
5. `s4-harness/check_setup_dtk.py` 对 canonical 与 HEAD 各跑一次（期望 16/16）；
6. 任何在非 DTK 下被改变的行为都要写入归档（不掩盖）。

**本步验收标准**：非 DTK 模拟成立（`is_hygon_dtk() is False`）＋ pytest 两种环境结果可解释 ＋ HEAD 逐条对照给出「一致 / 差异及原因」＋ setup.py harness 结果；判据 = 与 HEAD 逐条一致（差异必须能归因）。

## 2. 实际改动

**`vllm/**`、`setup.py`、`csrc/**` 零改动**（本步只做对照测试）。新增 4 个 `tools/dcu/**` 工具与 1 份归档：

| 文件 | 位置 | 改动摘要 |
|---|---|---|
| `tools/dcu/a7_non_dtk_probe.py` | 新增（317 行） | 探针：`env` / `plugin` / `rocm` 三个子命令；`--torch-arch`（stub `torch.cuda.get_device_properties`）、`--stub-device-name`（只 stub `get_device_name`，用于无 GPU 容器里让 DTK triton 能 import）、`--block-amdsmi`、`--fake-amdsmi-arch`/`--fake-amdsmi-fail`、`--stub-arch`；每条结果包成 `{"ok":…}`/`{"exc":…}`，异常本身就是数据 |
| `tools/dcu/a7_non_dtk_gate.sh` | 新增（120 行） | 容器内取证脚本：`a7_non_dtk_gate.sh <dtk\|sim> <tree> <tag>`，跑 14 个探针配置并把 JSON 落到 `/mnt/game/port/a7/<tag>.<section>.jsonl` |
| `tools/dcu/a7_compare_jsonl.py` | 新增（48 行） | 两个 jsonl 的逐键对照（忽略 `tree` 绝对路径），rc=0 表示一致 |
| `tools/dcu/a7_setup_head_probe.py` | 新增（86 行） | 用 `ast` 从任意 tree 的 `setup.py` 抽 `_is_hip()`，在**强制非 DTK**（`_is_hygon_dtk=False`）下跑 12 组组合与 HEAD 原公式比对（S4 harness 依赖 fork 独有符号，HEAD 上跑不了，见 §3.6） |

产物（不落仓库）：`/mnt/game/port/a7/*.jsonl|*.log`（28 个探针配置 ×2 tree + 3 份 pytest/harness 日志）。

## 3. 执行过程与原始证据

### 3.1 环境

```
镜像  harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413
z100-a7        id=83c339a4c9a1  --memory=8g  devices=/dev/kfd,/dev/mkfd,/dev/dri  binds=/mnt/game,/home/lyh/z100-vllm:/src:ro
z100-a7-nodtk  id=7bbfed2585be  --memory=8g  同上 + /dev/null:/opt/dtk/.info/rocm_version
GPU：HIP_VISIBLE_DEVICES=3（单卡；hy-smi 4 卡均 0% VRAM）。未执行任何自编译 HIP 可执行程序。
torch 2.10.0+das.opt1.dtk2604.20260325.g6b060a（镜像自带；canonical 的 4 个 .so 编于 torch 2.11，
      本容器里 import 失败 `undefined symbol: …CUDACachingAllocator9allocatorE` → 只影响扩展，平台层是纯 python，不影响本步）
```

HEAD 对照树（不动主工作区，只在 `/mnt/game` 写）：

```console
$ docker exec z100-a7 bash -lc 'git config --global --add safe.directory "*"; \
    git -C /mnt/game/port/gfx906-vllm-211 worktree add --detach /mnt/game/port/a7-head HEAD'
Preparing worktree (detached HEAD 6ba236043)
HEAD is now at 6ba236043 Merge pull request #13 from ttdxq/sync/prefill-perf-20260928
$ git -C /mnt/game/port/gfx906-vllm-211 worktree list
/mnt/game/port/gfx906-vllm-211  6ba236043 [gfx906/main]
/mnt/game/port/a7-head          6ba236043 (detached HEAD)
```

- 主工作区 HEAD = canonical HEAD = `6ba236043`；HEAD worktree `git status --short` = 0 行（干净）。
- `a7-head` 里**没有** `vllm/platforms/hygon_dtk.py`、`tests/platforms/` 只有 `test_cpu.py`（`test_hygon_dtk.py` 是本 fork 新增且被 `.gitignore:301 test_*.py` 藏住 → HEAD 上不存在）。
- 指纹：canonical `vllm/platforms/rocm.py` = `f26d9d36c322b44572ae66cf7871e2a1`、`setup.py` = `4349a1cbb4a768d35332e65947cf2d76`；HEAD = `0fccd1fe1b39ae3e9082760c8b6cb681` / `302ed791fe5e38938b899eec79c6a1db`。

### 3.2 步骤 1：DTK 存在时的基线（canonical，真机 gfx906，卡 3）

命令：

```bash
docker exec z100-a7 bash -lc 'source /opt/dtk/env.sh; unset VLLM_DCU_GCN_ARCH VLLM_DCU_BF16; \
  cd /mnt/game/port/gfx906-vllm-211; export HIP_VISIBLE_DEVICES=3; \
  PYTHONPATH=$PWD python3 /src/tools/dcu/a7_non_dtk_probe.py env'
```

原始输出（`/mnt/game/port/a7/canonical.env.jsonl`，节选字段）：

```json
{"is_hygon_dtk": true, "dtk_version": "26.04", "dtk_marker_is_file": true,
 "gcn_arch": {"ok": true, "value": "gfx906"}, "gfx906_bf16_enabled": {"ok": true, "value": true},
 "hy_smi_glob": [], "rocminfo_candidates": ["/opt/rocm/bin/rocminfo", "/opt/dtk/bin/rocminfo",
 "/opt/dtk-26.04-DCC2602-0317/bin/rocminfo"],
 "env": {"DTKROOT": "/opt/dtk", "HYHAL_PATH": "/opt/hyhal", "ROCM_PATH": "/opt/dtk",
         "HIP_VISIBLE_DEVICES": "3", "VLLM_DCU_GCN_ARCH": null, "VLLM_DCU_BF16": null}}
```

`rocm_platform_plugin()`（`canonical.plugin-detect.jsonl` / `canonical.plugin-vis.jsonl`，两个变体）：

```json
{"rocm_platform_plugin": {"ok": true, "value": "vllm.platforms.rocm.RocmPlatform"},
 "current_platform": "RocmPlatform"}
```

`_rocm_supported_dtypes()` 三种 `VLLM_DCU_BF16`（`canonical.rocm-gfx906-bf16-*.jsonl`；本机是真 gfx906，
`stub_supported_dtypes` 与真值相同）：

| `VLLM_DCU_BF16` | `_rocm_supported_dtypes()` | `gfx906_bf16_enabled()` |
|---|---|---|
| `auto`（默认） | `[float16, bfloat16, float32]` | True（真机探针通过，与 S3 一致） |
| `0` | `[float16, float32]` | False |
| `1` | `[float16, bfloat16, float32]` | True |

同轮 `RocmPlatform` 其它字段（`canonical.rocm-real.jsonl`）：

```json
{"gcn_arch_module_level": "gfx906", "device_capability": "DeviceCapability(major=9, minor=0)",
 "device_name": "Z100", "device_uuid": "6fff55b7-0000-1000-8009-645be0141861",
 "device_total_memory": 34342961152, "is_fully_connected": false,
 "use_rocm_custom_paged_attention": false, "has_amdsmi": true}
```

### 3.3 步骤 2：造「非 DTK」环境

容器级隐藏（`z100-a7-nodtk` 的 bind mount，可写层未改镜像）：

```bash
docker run -d --name z100-a7-nodtk --memory=8g --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
  --group-add video --ipc=host --shm-size=16g \
  -v /mnt/game:/mnt/game -v /home/lyh/z100-vllm:/src:ro \
  -v /dev/null:/opt/dtk/.info/rocm_version \
  harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 sleep infinity
```

进程内环境（`tools/dcu/a7_non_dtk_gate.sh` 的 `sim` 分支，逐字）：

```bash
source /opt/dtk/env.sh >/dev/null 2>&1          # 只为保留 LD_LIBRARY_PATH 等运行时库
unset DTKROOT HYHAL_PATH VLLM_DCU_GCN_ARCH VLLM_DCU_BF16
export ROCM_PATH=/opt/rocm                       # 该目录不存在（已确认）
export HIP_VISIBLE_DEVICES=3
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
```

⇒ `is_hygon_dtk() is False` **成立**。原始输出（`canonical-sim.env.jsonl`）：

```json
{"is_hygon_dtk": false, "dtk_version": null, "dtk_marker_is_file": false,
 "dtk_marker_exists": true, "gcn_arch": {"ok": true, "value": "gfx906"},
 "hy_smi_glob": [],
 "env": {"DTKROOT": null, "HYHAL_PATH": null, "ROCM_PATH": "/opt/rocm",
         "HIP_VISIBLE_DEVICES": "3", "VLLM_DCU_GCN_ARCH": null, "VLLM_DCU_BF16": null}}
```

说明：`dtk_marker_exists=true` 而 `dtk_marker_is_file=false` —— bind mount 把 `rocm_version` 换成了字符设备，
`Path.is_file()` 为假（判据只看 `is_file()`）；`/opt/dtk/bin` 在磁盘上仍在（见 §5 的模拟边界）。
`VLLM_DCU_GCN_ARCH` 必须置空：它是本 fork 新增变量，HEAD 不认，若保留则对照不公平（真实非 DTK 机器不会设它）。

`rocm_platform_plugin()`（`canonical-sim.plugin-detect.jsonl` / `plugin-vis`）：

```json
{"rocm_platform_plugin": {"ok": true, "value": "vllm.platforms.rocm.RocmPlatform"},
 "current_platform": "RocmPlatform"}
```

### 3.4 步骤 3：真实 pytest（两种环境）

命令（逐字，先在 `-c` 里 `import torch`，满足硬约束 8）：

```bash
python3 -c "import torch, pytest, sys; sys.exit(pytest.main(['-v','--noconftest','tests/platforms/test_hygon_dtk.py']))"
```

| 环境 | 结果 | 日志 |
|---|---|---|
| DTK 存在（canonical，卡 3） | **52 passed / 0 failed / 0 skipped**（14.54 s，rc=0） | `/mnt/game/port/a7/pytest-dtk.log` |
| 非 DTK 模拟（canonical，卡 3） | **52 passed / 0 failed / 0 skipped**（14.95 s，rc=0） | `/mnt/game/port/a7/pytest-sim.log` |
| 非 DTK 模拟（HEAD worktree） | `collected 0 items`，`no tests ran`，rc=4 | `/mnt/game/port/a7/pytest-head-sim.log` |

- **skip 差异 = 0 条**：两种环境都没有 skip，因此没有需要逐条解释的 skip。原因是本容器里 amdsmi 存在且
  有真卡，`vllm.platforms.rocm` 在两种环境下都能完成模块级 `_GCN_ARCH` 解析（DTK：`is_hygon_dtk()` 分支；
  模拟：amdsmi→`gfx916`→我们归一化成 `gfx906`），文件末尾 3 个 `pytest.importorskip("vllm.platforms.rocm")`
  的用例都不会被 skip。
- HEAD 上该文件不存在（fork 新增 + `.gitignore`），因此 HEAD 侧无 pytest 基线可对；HEAD 的对照用 §3.5 的探针。
- 与 S9 记录的 52 passed 一致（S9 是 GPU-free stub harness）。
- 边界：如果容器**没有** GPU（本步第一版 `z100-a7-nodtk` 就是如此），DTK 镜像自带的 triton 在 import 期
  调 `torch.cuda.get_device_name()`（`triton/utils/hcutuner.py:435`），会让**任何** `import vllm` 直接
  `RuntimeError: No HIP GPUs are available`（HEAD 同样如此，与本 fork 无关）⇒ 这是环境限制，不是 A7 结论；
  最终取证容器给了 GPU。

### 3.5 步骤 4：与 HEAD 逐条对照（核心判据）

采集方式：同一容器、同一环境变量下对 canonical 与 HEAD 各跑一次 14 个探针配置，逐键 diff
（`tools/dcu/a7_compare_jsonl.py`，忽略 `tree` 字段）。逐节结果（完整 `*.jsonl` 在 `/mnt/game/port/a7/`）：

| 探针配置（sim） | 相同键 | 差异项 |
|---|---|---|
| `plugin-detect`（`rocm_platform_plugin()`，先丢弃可见设备变量） | **4/4** | 无 |
| `plugin-vis`（保留 `HIP_VISIBLE_DEVICES`） | **4/4** | 无 |
| `rocm-real`（真 amdsmi + 真卡 gfx906） | 8/14 | `gcn_arch`、`device_capability`、`on_gfx906`、`use_rocm_custom_paged_attention` + `has_amdsmi`（HEAD 无此属性） |
| `rocm-fakeamdsmi`（假 amdsmi `gfx942:sramecc+:xnack-`） | 11/14 | 仅 `_GCN_ARCH` 后缀 + `has_amdsmi` |
| `rocm-fakeamdsmi-errors`（假 amdsmi，uuid/mem/topo 查询抛错） | 12/14 | `device_uuid` + `has_amdsmi` |
| `rocm-noamdsmi`（屏蔽 amdsmi + torch stub `gfx906:sramecc+`） | 5/14 | `device_name/uuid`、`is_fully_connected`（HEAD 抛 NameError）、`device_total_memory`、`_GCN_ARCH`、dtypes、custom-paged-attn |
| `rocm-noamdsmi-nogpu`（屏蔽 amdsmi + 无可见 GPU，不 stub arch） | **3/3** | 无（两边都 `RuntimeError: No HIP GPUs are available`） |
| `rocm-gfx906-bf16-auto` / `-0` / `-1`（stub arch） | 12/20 | `supported_dtypes`（auto/1 差异、0 一致）、`use_rocm_custom_paged_attention`、capability + 模块级 gfx916 |
| `rocm-gfx942`（stub arch） | 14/20 | `stub_*` 键全一致；其余差异全部来自 HEAD 模块级 `gfx916` |
| `rocm-gfx90a`（stub arch） | 14/20 | 同 `gfx942`：`stub_*` 键全一致 |

**逐项「HEAD 输出 vs 我们输出」**（任务书要求的 5 组 + 非 gfx906）：

| # | 探针 | 我们（canonical） | HEAD | 判定 |
|---|---|---|---|---|
| 1 | `rocm_platform_plugin()`，无可见设备变量 | `"vllm.platforms.rocm.RocmPlatform"` | 同 | **完全一致** |
| 1' | `rocm_platform_plugin()`，`HIP_VISIBLE_DEVICES=3` | `"vllm.platforms.rocm.RocmPlatform"` | 同 | **完全一致** |
| 2a | `_get_gcn_arch()`，无 amdsmi + 无 GPU（真 torch 兜底） | `RuntimeError: No HIP GPUs are available`（import 期） | 同 | **完全一致**（`_lazy_init()` 先于设备号解析抛错） |
| 2b | `_get_gcn_arch()`，无 amdsmi + torch stub `gfx906:sramecc+:xnack-` | `"gfx906"` | `"gfx906:sramecc+:xnack-"` | 差异 **D3**（归一化） |
| 2c | `_get_gcn_arch()`，假 amdsmi 报 `gfx942:sramecc+:xnack-` | `"gfx942"` | `"gfx942:sramecc+:xnack-"` | 差异 **D3** |
| 2d | `_get_gcn_arch()`，真 amdsmi（Hygon 硬件） | `"gfx906"` | `"gfx916"` | 差异 **D1**（由错变对） |
| 3 | `_rocm_supported_dtypes()`，stub `gfx942` / `gfx90a` | `[f16, bf16, f32]` | `[f16, bf16, f32]` | **完全一致** |
| 3' | 同上，stub `gfx906` + `VLLM_DCU_BF16=0` | `[f16, f32]` | `[f16, f32]` | **完全一致** |
| 3'' | 同上，stub `gfx906` + `auto`（真机探针 True）/ `=1` | `[f16, bf16, f32]` | `[f16, f32]` | 差异 **D2** |
| 4a | `get_device_name/uuid/total_memory`、`is_fully_connected()`，**无 amdsmi** | `"Fake GPU"`(torch)/`"6f09645be0141861"`(sysfs)/`34342961152`/`False` | `NameError: name 'amdsmi_init' is not defined` ×3、`206158430208`(stub torch) | 差异 **D4** |
| 4b | `get_device_uuid()`，amdsmi 可用但 uuid 查询抛错 | `""` | `UnboundLocalError: device_uuid referenced before assignment` | 差异 **D5** |
| 4c | `get_device_total_memory()`/`is_fully_connected()`，amdsmi 可用但查询抛错 | `34342961152`（sysfs）/`False` | `34342961152`（torch）/`False` | **一致**（数值相同） |
| 5 | `RocmPlatform.supported_dtypes` 底层函数 = #3/#3' | 见上 | 见上 | 见上 |
| 6 | 非 gfx906 的 capability / custom-paged-attn | `(9,4)` / `True`（gfx942）、`(9,0)` / `True`（gfx90a） | 同 | **完全一致** |

**结论**：`rocm_platform_plugin()`、非 gfx906 的 dtype/capability/attention 选择、无 GPU 时的异常 ——
**逐条完全一致**。有差异的只有 4 类，均可归因：

- **D1** `_get_gcn_arch()` 把 amdsmi 报的 `gfx916` 归一化成 `gfx906`（配套 capability (9,1)→(9,0)、
  `on_gfx906()` False→True、custom-paged-attn True→False）。触发条件 = **amdsmi 返回 Hygon 专有别名**，
  只在海光硬件上出现；真实社区 ROCm 机的 amdsmi 返回规范名，不走这条。
- **D2** gfx906 上 `supported_dtypes` 可能多出 bf16（`VLLM_DCU_BF16=auto` 且探针通过，或 `=1`）。
  触发条件 = **设备真是 gfx906**（社区 ROCm 的 MI50/Radeon VII 也是 gfx906），**不经 `is_hygon_dtk()`** ⇒ 见 §4/F1。
- **D3** torch / amdsmi 返回值带 `:sramecc+:xnack-` 等后缀时，我们返回归一化短名。所有使用点都是
  `in`/`startswith` 子串判定（`_GCN_ARCH` 全仓只出现在 `vllm/platforms/rocm.py`）＋
  `_capability_from_gcn_arch()` 的 `re.match(r"gfx(\d+)")` ⇒ 语义等价（`_GCN_ARCH` 外部不可见）。
- **D4/D5** 无 amdsmi / amdsmi 查询失败时，HEAD 直接 `NameError`/`UnboundLocalError`（该路径本来就不可用，
  社区 ROCm 一定带 amdsmi），我们给出定义值 ⇒ S2 门禁明确要求的行为（"无 amdsmi 环境下不抛异常"）。

原始差异片段（`summary`，完整见各 jsonl）：

```
=== rocm-real: 8/14 keys identical
    DIFF gcn_arch_module_level: ours="gfx906"  HEAD="gfx916"
    DIFF device_capability: ours=(9,0)  HEAD=(9,1)
    DIFF use_rocm_custom_paged_attention: ours=false  HEAD=true
=== rocm-noamdsmi: 5/14 keys identical
    DIFF device_name: ours="Fake GPU"  HEAD=NameError: name 'amdsmi_init' is not defined
    DIFF device_uuid: ours="6f09645be0141861"  HEAD=NameError: name 'amdsmi_init' is not defined
    DIFF is_fully_connected: ours=false  HEAD=NameError: name 'amdsmi_init' is not defined
    DIFF device_total_memory: ours=34342961152  HEAD=206158430208
    DIFF gcn_arch_module_level: ours="gfx906"  HEAD="gfx906:sramecc+:xnack-"
=== rocm-fakeamdsmi-errors: 12/14 keys identical
    DIFF device_uuid: ours=""  HEAD=UnboundLocalError: local variable 'device_uuid' referenced before assignment
```

### 3.6 步骤 5：setup.py 非 DTK 行为

S4 留下的 harness（原样，不改）：

```console
$ python3 /mnt/game/port/s4-harness/check_setup_dtk.py /mnt/game/port/gfx906-vllm-211
...
16/16 checks passed        rc=0            # 日志：/mnt/game/port/a7/setup-harness-canonical.log

$ python3 /mnt/game/port/s4-harness/check_setup_dtk.py /mnt/game/port/a7-head
  File "/mnt/game/port/a7-head/setup.py", line 1, in <module>
NameError: name '_hygon_dtk_prefix' is not defined        rc=1
```

**HEAD 上 16/16 不可能复现**：harness 会往抽出来的 AST 里注入 `_DTK_PREFIX = _hygon_dtk_prefix()`，而
`_hygon_dtk_prefix`/`_dtk_version`/`_is_hygon_dtk` 是本 fork 新增符号，HEAD 的 `setup.py` 里根本不存在
（`git diff HEAD -- setup.py` 可见全部 6 处改动）。因此 HEAD 侧改用 `tools/dcu/a7_setup_head_probe.py`：
从任意 tree 抽 `_is_hip()`，**强制 `_is_hygon_dtk()=False`**（即 A7 关心的非 DTK 条件），与 HEAD 原公式
`(target in ("cuda","rocm")) and torch.version.hip is not None` 比 12 组组合：

```console
$ python3 tools/dcu/a7_setup_head_probe.py /home/lyh/z100-vllm
12/12 combos match HEAD's formula (/home/lyh/z100-vllm)      rc=0
$ python3 tools/dcu/a7_setup_head_probe.py /mnt/game/port/a7-head
12/12 combos match HEAD's formula (/mnt/game/port/a7-head)    rc=0
```

即：canonical 16/16（含 off-DTK `_is_hip` 12 组合、DTK 标记判定、`_dtk_version()`、warn-once）；
非 DTK 条件下 canonical 与 HEAD 的 `_is_hip()` 逐组合相同。setup.py 的其余 DTK 分支
（`PYTORCH_ROCM_ARCH` 默认、`requirements/dcu.txt`、`+dtkXX.XX` 版本号、`ROCM_HOME or _DTK_PREFIX`）
都包在 `_DTK_PREFIX is not None` / `_is_hygon_dtk()` 里，非 DTK 时不进入。

## 4. 门禁判定

| 门禁项 | 结论 | 证据 |
|---|---|---|
| A7-1 非 DTK 模拟成立 | ✅ | `canonical-sim.env.jsonl`：`is_hygon_dtk=false`、`dtk_version=null`、`is_file=false`、`ROCM_PATH=/opt/rocm` |
| A7-2 DTK 基线 | ✅ | §3.2：`is_hygon_dtk=true`、`gcn_arch()=gfx906`、`rocm_platform_plugin()=RocmPlatform`、dtypes auto/0/1 |
| A7-3 `tests/platforms/` 抽样两种环境行为一致 | ✅ | 52 passed / 0 skipped ×2（§3.4）；无 skip 需要解释 |
| A7-4 `rocm_platform_plugin()` 一致 | ✅ | plugin-detect / plugin-vis 各 4/4 键一致 |
| A7-5 `_get_gcn_arch()` 一致 | 🔶 | 无 GPU 时异常完全一致；torch/amdsmi 返回带后缀时差 D3（后缀归一化，语义等价）；Hygon 硬件上差 D1（由错变对） |
| A7-6 非 gfx906 的 `_rocm_supported_dtypes()` 一致 | ✅ | stub `gfx942`/`gfx90a`：`[f16,bf16,f32]` 双方相同 |
| A7-7 `get_device_name/uuid/total_memory`、`is_fully_connected()` 一致 | 🔶 | amdsmi 正常时一致；无 amdsmi 时 HEAD 抛异常、我们给值（D4）；amdsmi uuid 查询失败时 HEAD `UnboundLocalError`、我们 `""`（D5） |
| A7-8 setup.py 非 DTK 行为不变 | ✅ | canonical 16/16；非 DTK 下 `_is_hip()` 12/12 与 HEAD 公式相同（HEAD harness 结构性不可跑，已说明） |
| A7-9 「与 HEAD 逐条一致」 | 🔶 部分 | 多数项逐条一致；4 类差异全部归因（D1–D5），无未解释差异 |

**判定：A7 判定为「非 DTK 行为基本不变，但有两处按 arch（而非按 DTK）生效的行为改动需要 Lead 决策」**
（F1/F2），其余为「HEAD 原本就坏的路径被修好」（D4/D5）与「海光硬件专有别名被修正」（D1/D3）。

### 核心结论：非 DTK 下被我们改变的行为（不掩盖）

**F1（建议修）`_rocm_supported_dtypes()` 的 bf16 分支不按 `is_hygon_dtk()` 限定**
`vllm/platforms/rocm.py:286-290` 现在是 `if _ON_GFX906: … if gfx906_bf16_enabled(): insert(bf16)`。
在**社区 ROCm 的 gfx906 机器**（MI50 / Radeon VII）上，只要 `VLLM_DCU_BF16=auto` 且探针通过（或用户 `=1`），
`supported_dtypes` 就从 HEAD 的 `[f16, f32]` 变成 `[f16, bf16, f32]`（硬约束 6 要求新增分支被
`is_hygon_dtk()` 或 amdsmi-fallback 包裹）。
最小修复（一行，本步不落地）：

```python
if _ON_GFX906:
    dtypes = [torch.float16, torch.float32]
    if is_hygon_dtk() and gfx906_bf16_enabled():   # <-- 加 is_hygon_dtk()
        dtypes.insert(1, torch.bfloat16)
    return dtypes
```
影响面：DTK 上 `is_hygon_dtk()` 恒为 True，行为不变；只需回归 S3 的 bf16 判定与 `--dtype auto` 选择。
（若 Lead 认为「同一 ISA 同一套探针」更合理，也可保留现状，但必须在文档里明确"按 arch 生效"。）

**F2（建议保留，需记录）`use_rocm_custom_paged_attention()` 的 `on_gfx906() → False`**
同一个门控也不按 DTK 限定：社区 gfx906 上 HEAD 返回 True、我们返回 False。但 gfx906（Vega20）**没有 MFMA**，
`csrc/rocm/attention.cu` 的 CDNA 分支在 gfx906 上退化成 `UNREACHABLE_CODE`（S5 已在真机复现 device assert），
所以 HEAD 的 True 在社区 gfx906 上同样是错的（会撞 assert）。建议保留本改动，仅在文档/禁用清单里注明
"按 arch 而非按 DTK 生效"。若 Lead 要求严格 DTK 限定，改动是加上 `and is_hygon_dtk()`（不推荐）。

**F3（无需修）`_get_gcn_arch()` 归一化**
在 Hygon 硬件上是 `gfx916`→`gfx906`（这正是 S1 的目的，HEAD 在该状态下 capability 会错成 (9,1) 并选到
MFMA-only 的 ROCm attention 桩）；在真实社区 ROCm 上只剥掉 `:sramecc+:xnack-` 后缀，13 处使用点全是子串判定，
`_capability_from_gcn_arch` 用 `gfx(\d+)` 正则 ⇒ 语义等价。副作用是"amdsmi 返回非法值时改用 torch 兜底"，
HEAD 会把非法值当真。

**F4（已由 S2 授权）无 amdsmi 时的回退**
`_HAS_AMDSMI=False` 时 HEAD 在 `get_device_name/uuid/total_memory/is_fully_connected` 直接
`NameError`（这两个方法被 `with_amdsmi_context` 包着，`amdsmi_init` 未定义）；`get_device_uuid` 在
amdsmi uuid 查询失败时 HEAD 还会 `UnboundLocalError`。我们改成定义值。触发条件是"装了 ROCm 但没有 amdsmi
python 模块"，社区 ROCm 不会出现；这是 S2 门禁"无 amdsmi 环境下不抛异常"的直接结果。

## 5. 未完成项 / 阻塞与原因

1. **`tests/kernels/` 抽样未跑**（设计 §A7 提到，本步任务书的 1–6 步未列）。
   阻塞原因：镜像自带 torch 2.10，而 canonical 副本的 4 个扩展编于 torch 2.11 → 本容器里
   `import vllm._C/_rocm_C/_C_stable_libtorch` 全部 `undefined symbol`，kernel 测试无法 import；
   且 HEAD worktree 只有源码（无 `.so`），无法做"同二进制、不同源码"的对照。
   可行的解除路径（留给 Lead）：容器里已有 `/mnt/game/port/wheels-211/torch-2.11.0+das.opt1.dtk2604…whl`，
   可离线装进一个新影子容器后再抽样（属 GPU kernel 工作，本步按任务书要求只做 python 侧判定）。
2. **真机（社区 ROCm / CUDA 硬件）仍不可得**：本步是"DTK 镜像内隐藏 DTK 判据"的模拟，
   amdsmi 由 DTK 镜像提供并**报 Hygon 的 `gfx916`**，`/opt/dtk` 也仍在磁盘上。因此模拟 ≠ 真社区机；
   所有差异的归属都按"触发条件"逐条给了判断，但"真社区 ROCm 机上 0 差异"仍是推断（依据：社区 amdsmi
   不报 `gfx916`、社区 ROCm 一定带 amdsmi、`_GCN_ARCH` 只用于子串判定）。
3. **模拟不够彻底**：`rocminfo_candidates()` 在模拟环境里仍会列出 `/opt/dtk/bin/rocminfo`（磁盘上有），
   而 HEAD 只试 `/opt/rocm/bin/rocminfo`；两次 `plugin` 对照仍然 4/4 一致（第一个候选不存在即抛
   `FileNotFoundError`，两边都在同一处结束），但这点在"真无 DTK"的机器上不成立也不影响结果。
4. 未做回滚演练（本步无代码改动，无回滚对象）。

## 6. 对后续步骤的影响

- **A7 可结案为「非 DTK 行为基本不变」**，但 F1/F2 是两处 **arch 门控**（非 DTK 门控）的行为改动，
  建议 Lead 在 S9 验收/文档里明确表态（收窄到 `is_hygon_dtk()` 或声明"按 arch 生效"）。收窄 F1 的回归面很小
  （S3 的 `check_bf16.py` + 27B int8/W4A16 的 dtype 选择），不影响 S6/S8 的 attention 结论（DTK 上
  `is_hygon_dtk()` 恒 True）。
- S11b/S11c（性能剖析、并行度/内存）不受本步影响；本步没有再动 `vllm/**`。
- 若后续步骤还要做非 DTK 对照，可直接复用 `z100-a7` / `z100-a7-nodtk` + `/mnt/game/port/a7-head`
  + `tools/dcu/a7_non_dtk_gate.sh`（HEAD worktree 已建好，`git worktree remove` 即可回收）。
- `tests/kernels/` 的 A7 抽样仍是欠账（§5.1），建议与 S10 的真机矩阵一起补。

## 7. 复现方式

```bash
IMG=harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413
REPO=/mnt/game/port/gfx906-vllm-211

# 0. 对照树（HEAD，干净；不动主工作区）
docker exec z100-a7 bash -lc 'git config --global --add safe.directory "*";
  git -C /mnt/game/port/gfx906-vllm-211 worktree add --detach /mnt/game/port/a7-head HEAD'

# 1. 容器（DTK 存在）与容器（非 DTK 模拟：bind mount 盖掉 /opt/dtk/.info/rocm_version）
docker run -d --name z100-a7 --memory=8g --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
  --group-add video --ipc=host --shm-size=16g -v /mnt/game:/mnt/game -v /home/lyh/z100-vllm:/src:ro \
  --entrypoint sleep "$IMG" infinity
docker run -d --name z100-a7-nodtk --memory=8g --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
  --group-add video --ipc=host --shm-size=16g -v /mnt/game:/mnt/game -v /home/lyh/z100-vllm:/src:ro \
  -v /dev/null:/opt/dtk/.info/rocm_version --entrypoint sleep "$IMG" infinity

# 2. 探针矩阵（每条 14 个配置；把 tag 换成 head-sim 并把 tree 换成 /mnt/game/port/a7-head 即 HEAD 侧）
docker exec z100-a7         bash -lc 'bash /src/tools/dcu/a7_non_dtk_gate.sh dtk $REPO canonical'
docker exec z100-a7-nodtk   bash -lc 'bash /src/tools/dcu/a7_non_dtk_gate.sh sim $REPO canonical-sim'
docker exec z100-a7-nodtk   bash -lc 'bash /src/tools/dcu/a7_non_dtk_gate.sh sim /mnt/game/port/a7-head head-sim'

# 3. 逐键 diff（rc=0 表示一致；日志与 jsonl 在 /mnt/game/port/a7/）
python3 tools/dcu/a7_compare_jsonl.py /mnt/game/port/a7/canonical-sim.rocm-real.jsonl \
                                     /mnt/game/port/a7/head-sim.rocm-real.jsonl

# 4. pytest（DTK 存在 / 非 DTK 模拟；**必须先 import torch**）
docker exec z100-a7 bash -lc 'source /opt/dtk/env.sh >/dev/null; export HIP_VISIBLE_DEVICES=3;
  export PYTHONPATH=/mnt/game/port/gfx906-vllm-211; cd $PYTHONPATH;
  python3 -c "import torch, pytest, sys; sys.exit(pytest.main([\"-v\",\"--noconftest\",\"tests/platforms/test_hygon_dtk.py\"]))"'
docker exec z100-a7-nodtk bash -lc 'source /opt/dtk/env.sh >/dev/null;
  unset DTKROOT HYHAL_PATH VLLM_DCU_GCN_ARCH VLLM_DCU_BF16; export ROCM_PATH=/opt/rocm HIP_VISIBLE_DEVICES=3;
  unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES; export PYTHONPATH=/mnt/game/port/gfx906-vllm-211;
  cd $PYTHONPATH;
  python3 -c "import torch, pytest, sys; sys.exit(pytest.main([\"-v\",\"--noconftest\",\"tests/platforms/test_hygon_dtk.py\"]))"'

# 5. setup.py
docker exec z100-a7 bash -lc 'python3 /mnt/game/port/s4-harness/check_setup_dtk.py /mnt/game/port/gfx906-vllm-211'
python3 tools/dcu/a7_setup_head_probe.py /home/lyh/z100-vllm
python3 tools/dcu/a7_setup_head_probe.py /mnt/game/port/a7-head
```

## 8. 参考

- [docs/海光Z100-性能优化方案.md](../海光Z100-性能优化方案.md) §9（A7 容器内非 DTK 对照的做法与判据）
- [docs/design/z100-09-acceptance-and-bench.md](../design/z100-09-acceptance-and-bench.md) A7 条目
- [docs/result/z100-09-acceptance-and-bench-result.md](z100-09-acceptance-and-bench-result.md) §3.7（A7 只做了 GPU-free 部分）
- [docs/result/z100-01-arch-normalization-result.md](z100-01-arch-normalization-result.md)（`gfx916→gfx906` 归一化）
- [docs/result/z100-05-csrc-kernels-result.md](z100-05-csrc-kernels-result.md)（`paged_attention` 在 gfx906 是 MFMA-only 桩 → 这里 D1/F2 的硬件依据）
- [docs/result/z100-03-bf16-verification-result.md](z100-03-bf16-verification-result.md)（bf16 保留但默认 `auto` → F1 的来源）
- [docs/result/z100-04-build-system-dtk-result.md](z100-04-build-system-dtk-result.md)（`s4-harness/check_setup_dtk.py` 16/16 的出处）
- 原始产物：`/mnt/game/port/a7/`（28 个探针配置的 `.jsonl`+`.log`、`pytest-dtk.log`、`pytest-sim.log`、`pytest-head-sim.log`、`setup-harness-*.log`）
