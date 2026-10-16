# S1 架构归一化 `gfx916 → gfx906` — 执行结果

> 归档时间：2026-10-07
> 执行者：子 agent（z100-s1-executor）
> 对应设计文档：[docs/design/z100-01-arch-normalization.md](../design/z100-01-arch-normalization.md)
> 前一档：无（S1 首次归档；Todolist 执行记录里 "架构来源改为 env→torch→KFD 多级校验；**gfx916 归一化待补**" 即本步的前置状态）
> 状态：✅ 完成（GPU-free 部分全部通过；真机 `_GCN_ARCH` 复核见 §5）

## 1. 目标与验收标准

本步门禁原文（Todolist「门禁（Gate）汇总」S1 行）：

> `_GCN_ARCH` 为 `gfx906`；`gfx916` 与 `gfx18446744073709551615` 均被归一化/拒绝；单测覆盖两者

设计文档 `z100-01-arch-normalization.md` 的验收条目：

| # | 验收项 | 本机（无 GPU / 无 `/dev/kfd`）可验证性 |
|---|---|---|
| A1 | `normalize_gcn_arch("gfx916") == "gfx906"` | ✅ 纯函数，可直接验证 |
| A2 | `normalize_gcn_arch("gfx916:sramecc+") == "gfx906"` | ✅ 纯函数 |
| A3 | `normalize_gcn_arch("gfx18446744073709551615") is None` | ✅ 纯函数 |
| A4 | `normalize_gcn_arch("gfx90a") == "gfx90a"`（不影响其它架构） | ✅ 纯函数 |
| A5 | torch 报 `gfx916` 时 `gcn_arch()` 返回 `gfx906` | ✅ stub torch，无需真机 |
| A6 | `VLLM_DCU_GCN_ARCH=gfx916` 时 `gcn_arch()` 返回 `gfx906` | ✅ 纯 env + stub |
| A7 | 真机 `_GCN_ARCH == "gfx906"`、`_capability_from_gcn_arch(_GCN_ARCH) == (9, 0)` | ⛔ 需要 `/dev/kfd`；用 **stub 加载真实 `rocm.py`** 覆盖同一条代码路径（见 §3.3） |
| A8 | 未装 DTK / 社区 ROCm 机器行为不变（`gfx942`/`gfx90a` 原样） | ✅ stub amdsmi 路径验证 |
| A9 | 既有 29 项 GPU-free 单测不回归 | ✅ 同一 stub harness 复跑 |

**GPU-free 可验证项全部通过**；A7 的"真机"部分在本机无法执行，用等价代码路径的 stub 证据替代（见 §4 与 §5）。

## 2. 实际改动

所有行号为**当前工作区**（`git status` 时点）的行号。

| 文件 | 位置 | 改动摘要 |
|---|---|---|
| `vllm/platforms/hygon_dtk.py` | `:52-60` | 新增模块级私有常量 `_HYGON_DCU_ARCH_ALIAS = {"gfx916": "gfx906"}`，注释说明**为什么必须是 alias 而不是 allow-list**（allow-list 会留下 capability `(9,1)`，使 `capability.minor == 0` 的门控失效） |
| `vllm/platforms/hygon_dtk.py` | `:121-142` | 新增 `normalize_gcn_arch(arch) -> str \| None`：`strip` → 去 `:` 后缀 → 小写化 → 查 alias → `is_valid_gcn_arch()` 复核；空/`None`/非法返回 `None`；`gfx90a`/`gfx942`/`gfx1100` 原样返回（Google 风格 docstring，含 `Args:`/`Returns:`） |
| `vllm/platforms/hygon_dtk.py` | `:145-147` | `arch_from_gfx_target_version()` 返回值过 `normalize_gcn_arch()`（KFD 表本身给 `gfx906`，此改动保证三条来源走同一条归一化路径） |
| `vllm/platforms/hygon_dtk.py` | `:262-284` | `gcn_arch_from_torch()` 改为 `normalize_gcn_arch(prop.gcnArchName)`；返回值非 `None` 即接受，非法值仍走原 `warning_once` 分支 |
| `vllm/platforms/hygon_dtk.py` | `:290-304` | `gcn_arch()` 的 `VLLM_DCU_GCN_ARCH` 覆盖分支改为 `normalize_gcn_arch(override)`，`None` 时抛原有 `ValueError`；torch/KFD 分支不变（两条来源已在各自入口归一化） |
| `vllm/platforms/rocm.py` | `:20-27` | 导入列表：`is_valid_gcn_arch` 改为 `normalize_gcn_arch`（`is_valid_gcn_arch` 在本文件已无用例；函数本身未删除、未改名，向后兼容） |
| `vllm/platforms/rocm.py` | `:228-243` | `_get_gcn_arch()` 的 amdsmi 分支：`raw_arch = _query_gcn_arch_from_amdsmi()` → `arch = normalize_gcn_arch(raw_arch)`；`None` 时警告里打 `raw_arch`（原实现打了被覆盖后的变量）。torch 兜底分支同样包一层 `normalize_gcn_arch(...)`。未改动其它任何逻辑 |
| `tests/platforms/test_hygon_dtk.py` | `:9-12, 29-41` | 新增 `import sys` / `import types`；新增 `_fake_torch_props(gcn_arch_name)` helper（stub `torch.cuda.get_device_properties`） |
| `tests/platforms/test_hygon_dtk.py` | `:69-114` | 新增 6 个测试：`test_normalize_gcn_arch`（15 组 parametrize）、`test_is_valid_gcn_arch_accepts_gfx916_before_alias`、`test_gcn_arch_from_kfd_still_reports_gfx906`、`test_gcn_arch_from_torch_normalizes_gfx916`、`test_gcn_arch_override_reports_canonical_arch` |
| `tests/platforms/test_hygon_dtk.py` | `:290-315` | 新增 2 个 `rocm.py` 测试：`test_get_gcn_arch_normalizes_amdsmi_hygon_alias`、`test_get_gcn_arch_rejects_bogus_amdsmi_and_uses_torch`（`pytest.importorskip("vllm.platforms.rocm")`，与文件内既有风格一致） |
| `docs/result/z100-01-arch-normalization-result.md` | 本文件 | S1 归档 |

**未改动**（明确记录）：`vllm/model_executor/...` 的 `on_gfx9()`/`_ON_GFX9`、`_GCN_ARCH` 的 `_ON_*` 派生常量、`_capability_from_gcn_arch()` 解析规则、`docs/design/*`、`docs/z100-todolist.md`、`docs/result/README.md`、`vllm/envs.py`、`vllm/platforms/__init__.py`。

`git status --short` 里还有 S0/S2/S3 的既有改动与本步无关的未跟踪项：`vllm/envs.py`、`vllm/platforms/__init__.py`（S2 落地，非本步所为）、`docs/design/*`、`docs/hygon_z100.md`、`docs/z100-todolist.md`、`tools/dcu/`、`docs/result/`（含本文件），以及**并发会话产生的 `.tmp-bf16/`**（内含 `check_bf16.out`、`qwen38.json`，mtime 与本步文件不同、属 bf16 真机检查，非本步产物，未触碰）。

## 3. 执行过程与原始证据

所有命令在 `/home/lyh/z100-vllm` 下执行。本机约束：**无 torch / pytest / regex，`/dev/kfd` 不可访问，未编译或运行任何 HIP 代码**。测试通过 `/tmp` 下的 stub harness 隔离加载被测模块（harness 不进仓库）。

### 3.1 归一化单测（真实输出，节选新用例；全量 49 行 PASS 见 §3.2）

```console
$ python3 /tmp/run_hygon_tests.py
PASS test_gcn_arch_falls_back_from_torch_to_kfd
PASS test_gcn_arch_from_kfd_still_reports_gfx906
PASS test_gcn_arch_from_torch_normalizes_gfx916
PASS test_gcn_arch_override_reports_canonical_arch
SKIP test_get_gcn_arch_normalizes_amdsmi_hygon_alias: vllm.platforms.rocm unavailable: No module named 'regex'
SKIP test_get_gcn_arch_rejects_bogus_amdsmi_and_uses_torch: vllm.platforms.rocm unavailable: No module named 'regex'
PASS test_is_valid_gcn_arch_accepts_gfx916_before_alias
PASS test_normalize_gcn_arch[('gfx916', 'gfx906')]
PASS test_normalize_gcn_arch[('gfx916:sramecc+:xnack-', 'gfx906')]
PASS test_normalize_gcn_arch[(' gfx916 ', 'gfx906')]
PASS test_normalize_gcn_arch[('GFX916', 'gfx906')]
PASS test_normalize_gcn_arch[('GFX916:sramecc+', 'gfx906')]
PASS test_normalize_gcn_arch[('gfx906:sramecc+:xnack-', 'gfx906')]
PASS test_normalize_gcn_arch[('gfx90a', 'gfx90a')]
PASS test_normalize_gcn_arch[('GFX90A', 'gfx90a')]
PASS test_normalize_gcn_arch[('gfx942', 'gfx942')]
PASS test_normalize_gcn_arch[('gfx1100', 'gfx1100')]
PASS test_normalize_gcn_arch[('gfx18446744073709551615', None)]
PASS test_normalize_gcn_arch[('gfx', None)]
PASS test_normalize_gcn_arch[('', None)]
PASS test_normalize_gcn_arch[('   ', None)]
PASS test_normalize_gcn_arch[(None, None)]
PASS test_normalize_gcn_arch[('not-an-arch', None)]
PASS test_read_kfd_topology_skips_cpu_node
PASS test_rocminfo_candidates_include_dtk

49 passed, 0 failed, 3 skipped
```

```console
$ python3 /tmp/run_hygon_tests.py 2>&1 | grep -c "^PASS"; python3 /tmp/run_hygon_tests.py 2>&1 | grep -c "^SKIP"; python3 /tmp/run_hygon_tests.py 2>&1 | grep -c "^FAIL"
49
3
0
```
（最后一条 `grep -c "^FAIL"` 返回 0，`grep` 无匹配时退出码为 1——因此该命令整体 exit code 为 1，这是 grep 语义，不是测试失败。）

对比基线（改动前同一 harness）：`29 passed, 0 failed, 1 skipped`。现在 `49 passed, 3 skipped`：**已有 29 项无回归**，新增 20 项通过；新增的 2 个 `rocm.py` 用例在本机 skip（本机无 `regex`），但在 stub 加载真实 `rocm.py` 的环境下实跑通过（§3.3）。

### 3.2 测试项数分解

| 来源 | 数量 | 通过 |
|---|---|---|
| 原 29 项 | 29 | 29（无回归） |
| `test_normalize_gcn_arch` | 15（parametrize） | 15 |
| `test_is_valid_gcn_arch_accepts_gfx916_before_alias` | 1 | 1 |
| `test_gcn_arch_from_kfd_still_reports_gfx906` | 1 | 1 |
| `test_gcn_arch_from_torch_normalizes_gfx916` | 1 | 1 |
| `test_gcn_arch_override_reports_canonical_arch` | 1 | 1 |
| 合计（本机 harness 内） | 49 | 49 |

### 3.3 真实 `rocm.py` 加载探针（替代真机 A7；原始输出）

`/tmp/run_rocm_arch_probe.py` 用 stub `torch`/`regex`/`amdsmi`/`vllm` 包**从源码文件加载真实的 `vllm/platforms/rocm.py`**，然后读模块级 `_GCN_ARCH`、`_capability_from_gcn_arch` 与 `on_gfx906()/on_cdna()/on_gfx9()`。脚本把本机的 DTK 分支强制关掉（本机确实存在 `/opt/dtk/.info/rocm_version`），以便走**社区 ROCm 的 amdsmi→torch 路径**——这正是 §2 中 `rocm.py` 改动的路径。

```console
$ python3 /tmp/run_rocm_arch_probe.py
[A amdsmi reports gfx916] gcnArchName(amdsmi)='gfx916' gcnArchName(torch)='gfx916'
[A amdsmi reports gfx916] _get_gcn_arch() = 'gfx906'
[A amdsmi reports gfx916] _GCN_ARCH = 'gfx906'
[A amdsmi reports gfx916] _capability_from_gcn_arch(_GCN_ARCH) = (9, 0)
[A amdsmi reports gfx916] on_gfx906=True on_cdna=True on_gfx9=False _ON_GFX906=True
[B amdsmi bogus, torch reports gfx916:sramecc+:xnack-] gcnArchName(amdsmi)='gfx18446744073709551615' gcnArchName(torch)='gfx916:sramecc+:xnack-'
[B amdsmi bogus, torch reports gfx916:sramecc+:xnack-] _get_gcn_arch() = 'gfx906'
[B amdsmi bogus, torch reports gfx916:sramecc+:xnack-] _GCN_ARCH = 'gfx906'
[B amdsmi bogus, torch reports gfx916:sramecc+:xnack-] _capability_from_gcn_arch(_GCN_ARCH) = (9, 0)
[B amdsmi bogus, torch reports gfx916:sramecc+:xnack-] on_gfx906=True on_cdna=True on_gfx9=False _ON_GFX906=True
[C community ROCm gfx942 unchanged] gcnArchName(amdsmi)='gfx942' gcnArchName(torch)='gfx942'
[C community ROCm gfx942 unchanged] _get_gcn_arch() = 'gfx942'
[C community ROCm gfx942 unchanged] _GCN_ARCH = 'gfx942'
[C community ROCm gfx942 unchanged] _capability_from_gcn_arch(_GCN_ARCH) = (9, 4)
[C community ROCm gfx942 unchanged] on_gfx906=False on_cdna=True on_gfx9=True _ON_GFX906=False
[D community ROCm gfx90a unchanged] gcnArchName(amdsmi)='gfx90a' gcnArchName(torch)='gfx90a'
[D community ROCm gfx90a unchanged] _get_gcn_arch() = 'gfx90a'
[D community ROCm gfx90a unchanged] _GCN_ARCH = 'gfx90a'
[D community ROCm gfx90a unchanged] _capability_from_gcn_arch(_GCN_ARCH) = (9, 0)
[D community ROCm gfx90a unchanged] on_gfx906=False on_cdna=True on_gfx9=True _ON_GFX906=False

RESULT: ALL CASES OK
```

用例 D 的 `on_gfx9=True` 是因为 `on_gfx9()` 的实现是 `_ON_GFX9 = any(arch in _GCN_ARCH for arch in ["gfx90a","gfx942","gfx950"])`，"gfx90a" 命中自身，与归一化无关；本步未改 `_ON_GFX9`（硬约束 5 / 设计文档 §4），且用例 B 显示 Z100 上 `on_gfx9()=False` 仍保持 fork 的既有语义。

### 3.4 两个 `rocm.py` 单测的真实执行（stub 加载真实 rocm.py）

因为 `/tmp/run_hygon_tests.py` 本机缺 `regex` 只能 skip，用 `/tmp/run_rocm_gcn_arch_tests.py` 加载真实 `rocm.py` 后**执行同一份测试体**：

```console
$ python3 /tmp/run_rocm_gcn_arch_tests.py
loaded real rocm.py: rocm.py, _GCN_ARCH=gfx906
PASS test_get_gcn_arch_normalizes_amdsmi_hygon_alias
PASS test_get_gcn_arch_rejects_bogus_amdsmi_and_uses_torch

2 passed, 0 failed
```

### 3.5 `is_valid_gcn_arch()` 正则链路核对（任务书第 5 点）

链路：`gfx916` → `_GCN_ARCH_RE = ^gfx[0-9a-f]{3,4}$` → `"916"` 全为合法十六进制数字 → **正则放行（True）** → `normalize_gcn_arch()` 查 `_HYGON_DCU_ARCH_ALIAS` → `gfx906` → 再由 `is_valid_gcn_arch("gfx906")` 复核通过 → 返回 `gfx906`。即**正则保持纯语法职责、alias 负责纠偏**，两者职责分离；`gfx18446744073709551615` 因 20 位数字超长被正则**直接拒绝**（`None`），永远进不到 alias 表。该链路有专门断言：

```python
def test_is_valid_gcn_arch_accepts_gfx916_before_alias():
    """The regex is purely syntactic; remapping is normalize_gcn_arch's job."""
    assert hygon_dtk.is_valid_gcn_arch("gfx916") is True
    assert hygon_dtk.normalize_gcn_arch("gfx916") == "gfx906"
```
结论：**`gfx916` 预期会被正则放行，然后被 alias 归一化**——符合设计文档 §1「要点」的两条要求。

### 3.6 lint / 语法检查

```console
$ python3 -m py_compile vllm/platforms/hygon_dtk.py vllm/platforms/rocm.py tests/platforms/test_hygon_dtk.py
py_compile OK

$ for f in vllm/platforms/hygon_dtk.py vllm/platforms/rocm.py tests/platforms/test_hygon_dtk.py; do awk -v F="$f" 'length($0)>88 {printf "%s:%d:%d\n", F, NR, length($0)}' "$f"; done
vllm/platforms/rocm.py:1200:94
```

- `ruff` **未跑**：本机无 `ruff`（`command -v ruff` 无输出），仓库无 `.venv`，且任务约束禁止为 lint 安装环境。
- 行宽用 `awk` 代替：唯一 >88 的行 `rocm.py:1200` 是**改动前就存在**的（`git show HEAD:vllm/platforms/rocm.py` 对应 `:1124:94`），本步新增/修改行全部 ≤88。
- `py_compile` 通过，三个文件语法有效。

## 4. 门禁判定

| 门禁条目（Todolist S1） | 判定 | 证据指向 |
|---|---|---|
| `_GCN_ARCH` 为 `gfx906` | ✅ 通过（等价路径） | §3.3 用例 A/B：真实 `rocm.py` 模块级 `_GCN_ARCH = 'gfx906'`。**真机复核仍待做**（§5） |
| `gfx916` 被归一化 | ✅ 通过 | §3.1 `test_normalize_gcn_arch`（`gfx916`/`gfx916:sramecc+:xnack-`/` gfx916 `/`GFX916` → `gfx906`）、§3.3 A/B、§3.4 |
| `gfx18446744073709551615` 被拒绝 | ✅ 通过 | §3.1 `test_normalize_gcn_arch[(..., None)]`、§3.2 既有 `test_gcn_arch_rejects_invalid_override`（无回归）、§3.3 用例 B（amdsmi 报该值 → 丢弃 → torch） |
| 单测覆盖两者 | ✅ 通过 | §3.1/§3.2：15 组 `normalize_gcn_arch` 参数化 + 2 个 `rocm.py` 用例；`49 passed / 0 failed / 3 skipped` |

设计文档验收条目：

| 条目 | 判定 | 证据 |
|---|---|---|
| A1 `normalize_gcn_arch("gfx916") == "gfx906"` | ✅ | §3.1 |
| A2 `normalize_gcn_arch("gfx916:sramecc+") == "gfx906"` | ✅ | §3.1 |
| A3 `normalize_gcn_arch("gfx18446744073709551615") is None` | ✅ | §3.1 |
| A4 `normalize_gcn_arch("gfx90a") == "gfx90a"` | ✅ | §3.1（`GFX90A` → `gfx90a`，小写 `a` 未被破坏） |
| A5 torch 报 `gfx916` → `gcn_arch()` 得 `gfx906` | ✅ | §3.1 `test_gcn_arch_from_torch_normalizes_gfx916` |
| A6 `VLLM_DCU_GCN_ARCH=gfx916` → `gcn_arch()` 得 `gfx906` | ✅ | §3.1 `test_gcn_arch_override_reports_canonical_arch` |
| A7 真机 `_GCN_ARCH == "gfx906"` 且 capability `(9, 0)` | 🔶 部分 | §3.3 用真实 `rocm.py` + stub 设备报 `gfx916` 得到 `'gfx906'` / `(9, 0)`；**`/dev/kfd` 上的复核未做** |
| A8 未装 DTK 行为不变 | ✅ | §3.3 用例 C/D：`gfx942`→`(9,4)`、`gfx90a`→`(9,0)` 原样；`_ON_GFX906=False` |
| A9 既有 29 项无回归 | ✅ | §3.1 基线 29 passed → 现 49 passed，无 FAIL |

**S1 在 GPU-free 范围内判定：通过。** 唯一未闭环的是真机上"驱动/amdsmi 实际怎么报"这一环境事实（§5）。

## 5. 未完成项 / 阻塞与原因

| 项 | 状态 | 原因 |
|---|---|---|
| 真机 `python -c "from vllm.platforms.rocm import _GCN_ARCH, _capability_from_gcn_arch; print(_GCN_ARCH, _capability_from_gcn_arch(_GCN_ARCH))"` → 期望 `gfx906 (9, 0)` | **未执行** | 本机 `/dev/kfd` 不可访问，且未安装 torch / vllm，无任何 GPU 运行时；硬约束同时禁止运行自编译 HIP 二进制。已用「真实 `rocm.py` + stub 设备」覆盖同一代码路径（§3.3） |
| 真机 `on_gfx906()/on_cdna()/on_gfx9()` 期望 `True True False` | **未执行** | 同上；§3.3 在 stub 设备上已得 `True True False` |
| `pytest tests/platforms/test_hygon_dtk.py -v` 原样执行 | **未执行** | 本机无 pytest/torch/regex；按任务书改用 `/tmp` stub harness（§3.1），并单独用 stub 加载真实 `rocm.py` 跑 `rocm` 两个用例（§3.4） |
| `ruff check` | **未执行** | 本机无 `ruff`、无 `.venv`；不为 lint 安装环境。已用 `awk` 行宽检查 + `py_compile` 替代（§3.6） |

无阻塞性技术问题：本步代码改动与 GPU-free 验证已闭环，剩余项都是**环境条件缺失**而非实现缺陷。

## 6. 对后续步骤的影响

- **S4（DTK 构建系统）可以开始**：架构字符串从所有入口（env / torch / KFD / amdsmi 非 DTK 路径）出来都已是 `gfx906`，`PYTORCH_ROCM_ARCH=gfx906`、`CMakeLists` 的 gfx906 排除集不再需要为 `gfx916` 另做判断。
- **S3 的 bf16 门控受益**：`_ON_GFX906` 现在能真正为 True（§3.3 用例 A/B），`_rocm_supported_dtypes()` 的 gfx906 分支才会按探针在 fp16/bf16/fp32 间选择；否则 bf16 门控会被 `gfx916` 绕过。
- `capability` 从 `(9,1)` 修正为 `(9,0)`，所有 `capability.minor == 0` 的判定（如 attention 后端与量化路径的选择）恢复预期；S8 的路径核对以此为前提。
- **未推翻任何前提**：`_ON_GFX9`（本 fork 有意排除 gfx906/gfx908）与 `_capability_from_gcn_arch()` 解析规则均未改动，`on_gfx9()` 在 Z100 上仍为 `False`（设计文档 §4 明确要求）。
- 遗留的真机确认不阻塞 S4 开始；建议在 S9 真机验收时把 §7 的 A7/A8 两条命令一并跑掉。

## 7. 复现方式

```bash
cd /home/lyh/z100-vllm

# 0. 语法检查（无需任何依赖）
python3 -m py_compile vllm/platforms/hygon_dtk.py vllm/platforms/rocm.py \
    tests/platforms/test_hygon_dtk.py

# 1. GPU-free 单测（自带 stub，无需 torch/pytest；harness 不进仓库）
#    期望：49 passed, 0 failed, 3 skipped
python3 /tmp/run_hygon_tests.py

# 2. 真实 rocm.py 的架构解析探针（stub 设备报 gfx916）
#    期望：4 个用例全部 ALL CASES OK，A/B 的 _GCN_ARCH=gfx906 且 capability=(9, 0)
python3 /tmp/run_rocm_arch_probe.py

# 3. 单独跑两个依赖 rocm.py 的单测体（stub 加载真实 rocm.py）
#    期望：2 passed, 0 failed
python3 /tmp/run_rocm_gcn_arch_tests.py

# 4. 行宽检查（替代 ruff；期望只打印改动前就存在的 rocm.py:1200）
for f in vllm/platforms/hygon_dtk.py vllm/platforms/rocm.py \
         tests/platforms/test_hygon_dtk.py; do
  awk -v F="$f" 'length($0)>88 {printf "%s:%d:%d\n", F, NR, length($0)}' "$f"
done

# 5. 真机（需 /dev/kfd；本机未执行）
python3 -m pytest tests/platforms/test_hygon_dtk.py -v
python3 -c "from vllm.platforms.rocm import _GCN_ARCH, _capability_from_gcn_arch; \
            print(_GCN_ARCH, _capability_from_gcn_arch(_GCN_ARCH))"   # 期望 gfx906 (9, 0)
python3 -c "from vllm.platforms.rocm import on_gfx906, on_cdna, on_gfx9; \
            print(on_gfx906(), on_cdna(), on_gfx9())"                 # 期望 True True False

# 6. 临时兜底（若真机某条来源漏归一化）
VLLM_DCU_GCN_ARCH=gfx906 python3 -c "from vllm.platforms.rocm import _GCN_ARCH; print(_GCN_ARCH)"
```

若 `/tmp` 下的 harness 丢失，重建要点：stub `vllm.envs`（提供 `VLLM_DCU_GCN_ARCH`/`VLLM_DCU_BF16`）、`vllm.logger`、`pytest`（`importorskip`/`mark.parametrize`/`fixture`/`raises`）与 `monkeypatch`（`setenv`/`delenv`/`setattr`/`setitem`），然后按文件路径加载 `vllm/platforms/hygon_dtk.py`；进阶部分用 stub `torch`/`regex`/`amdsmi` 并从文件路径加载 `vllm/platforms/rocm.py`（`vllm.v1*` 需先注册为带 `__path__` 的包）。

## 8. 参考

- [docs/design/z100-01-arch-normalization.md](../design/z100-01-arch-normalization.md) —— 本步任务书
- [docs/z100-todolist.md](../z100-todolist.md) —— 全局硬约束与 S1 门禁
- [docs/hygon_z100.md](../hygon_z100.md) —— §2.2 误判来源、§3 本轮代码改动、§4 已完成验证（29 passed / 1 skipped 基线）
- [docs/result/README.md](README.md) —— 归档模板与规则
- `vllm/platforms/hygon_dtk.py:60,121-142,145-147,262-284,290-304` —— 归一化实现
- `vllm/platforms/rocm.py:20-27,210-243` —— `_get_gcn_arch()` 接入
- `tests/platforms/test_hygon_dtk.py:69-114,290-315` —— 新增单测
- `/tmp/run_hygon_tests.py`、`/tmp/run_rocm_arch_probe.py`、`/tmp/run_rocm_gcn_arch_tests.py` —— 本机 GPU-free 验证 harness（不在仓库内，不提交）
