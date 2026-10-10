# S1 架构归一化：`gfx916 → gfx906`

> 上层文档：[适配方案](../海光Z100-DTK26.04-gfx906-vLLM适配方案.md) · [Todolist](../z100-todolist.md) · [实测证据](../hygon_z100.md)
> 前置：[S0](z100-00-baseline-and-prereq.md) · 状态：🔶 代码已部分落地，归一化待补

## 目的

DTK 的 `amdsmi` 把这四张卡报成 **`gfx916`**（海光 SMI 命名），而**驱动/torch/所有自编译内核都是 `gfx906`**。若 `_GCN_ARCH='gfx916'`：

- `_ON_GFX906 = "gfx906" in _GCN_ARCH` → `False` ⇒ 分支里 382 处 gfx906 门控全部失效；
- `_capability_from_gcn_arch('gfx916')` → `(9, 1)` ⇒ `capability.minor == 0` 的判定全部失效；
- 若只把 `gfx916` 加进白名单，`on_gfx9()` 能开，但 capability 仍是 `(9,1)`，**第二个问题依然存在**。

因此必须**归一化**，而不是加白名单。

## 前置条件

- S0 完成，确认实际 `gcnArchName`（应为 `gfx906`）。
- 已读 [hygon_z100.md](../hygon_z100.md) §2.2。

## 做法

### 1. 在 `vllm/platforms/hygon_dtk.py` 增加归一化（唯一实现点）

```python
# 海光 SMI 把 gfx906 家族报成 gfx916；驱动/torch/内核都是 gfx906。
# 值必须是纯 ISA 名，供 rocm.py 的子串门控使用。
_HYGON_DCU_ARCH_ALIAS = {"gfx916": "gfx906"}


def normalize_gcn_arch(arch: str | None) -> str | None:
    """归一化海光 DCU 的架构别名，返回纯 gfx 名（无 target-feature 后缀）。"""
    if not arch:
        return None
    base = arch.strip().split(":", 1)[0].lower()
    normalized = _HYGON_DCU_ARCH_ALIAS.get(base, base)
    return normalized if is_valid_gcn_arch(normalized) else None
```

要点：
- `is_valid_gcn_arch()` 保持**纯语法**校验（`^gfx[0-9a-f]{3,4}$`），`gfx916` 语法合法、会被接受，**由 `normalize_gcn_arch()` 负责纠偏**，两者职责分离；
- 归一化后的值必须是纯 ISA 名（去掉 `:sramecc+:xnack-` 后缀），因为 `rocm.py` 大量使用子串判定。

### 2. 让所有架构来源都过一遍归一化

修改 `hygon_dtk` 内三处：

| 函数 | 要求 |
|---|---|
| `gcn_arch_from_torch()` | 返回值改为 `normalize_gcn_arch(...)` 的结果 |
| `gcn_arch_from_kfd()` | `arch_from_gfx_target_version()` 的映射表已给 `gfx906`，保持；表里**只保留实测过的键** |
| `gcn_arch()` | 对 `VLLM_DCU_GCN_ARCH` 覆盖值也做归一化（允许用户写 `gfx916`，但对外仍是 `gfx906`） |

### 3. 修 `vllm/platforms/rocm.py` 的非 DTK 路径

`_get_gcn_arch()` 里 amdsmi 分支目前是"校验通过就用"，需要改成"校验 + 归一化"：

```python
from .hygon_dtk import normalize_gcn_arch

    arch = normalize_gcn_arch(_query_gcn_arch_from_amdsmi())
    if arch is not None:
        return arch
```

同一条路径的 torch 兜底也包一层 `normalize_gcn_arch(...)`，最后再 `split(":")[0]`。

### 4. 明确**不做**的事

- ❌ 不把 `gfx906` 加入 `_ON_GFX9`（本 fork 的 `on_gfx9()` 有意排除 gfx906/gfx908，见 `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py:53`）。
- ❌ 不改 `on_gfx906()` / `on_cdna()` / `on_gfx9()` 的调用点。
- ❌ 不改 `_capability_from_gcn_arch()` 的解析规则（归一化后它自然得到 `(9,0)`）。

## 验收（门禁）

```bash
# 1. 归一化单元测试（GPU-free，可在开发机跑）
python -m pytest tests/platforms/test_hygon_dtk.py -v

# 2. 真机：必须得到 gfx906 与 (9,0)
python -c "from vllm.platforms.rocm import _GCN_ARCH, _capability_from_gcn_arch; \
           print(_GCN_ARCH, _capability_from_gcn_arch(_GCN_ARCH))"
# 期望: gfx906 (9, 0)

# 3. 真机：gfx906 门控确实生效
python -c "from vllm.platforms.rocm import on_gfx906, on_cdna, on_gfx9; \
           print(on_gfx906(), on_cdna(), on_gfx9())"
# 期望: True True False
```

必须新增/保留的单测（`tests/platforms/test_hygon_dtk.py`）：
- `normalize_gcn_arch("gfx916") == "gfx906"`
- `normalize_gcn_arch("gfx916:sramecc+") == "gfx906"`
- `normalize_gcn_arch("gfx18446744073709551615") is None`
- `normalize_gcn_arch("gfx90a") == "gfx90a"`（不影响其它架构）
- `gcn_arch()` 在 torch 报 `gfx916` 时返回 `gfx906`

## 失败处置

| 现象 | 处置 |
|---|---|
| 真机 `_GCN_ARCH` 仍是 `gfx916` | 说明某条来源漏了归一化；用 `VLLM_DCU_GCN_ARCH=gfx906` 临时兜底，再定位漏点 |
| `on_gfx9()` 期望为 True | **不要改**；先确认调用方到底需要哪个能力（MFMA？通用 gfx9 内核？），本 fork 的语义是有意的 |
| torch 报 `gfx906:sramecc+:xnack-` | 归一化会剥掉后缀，属于预期 |

## 参考

- 前序计划 §5.2（归一化补丁原型）：`/home/lyh/server/gfx906_port/适配计划-gfx906-vLLM-DTK.md`
- 生产版 `rocm.py` 对照：`/home/lyh/server/gfx906_port/artifacts/rocm.py.production`
