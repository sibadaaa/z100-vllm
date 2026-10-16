# S11m 修 `=1` 的 AOT 缓存重载 + 跨重启复现性 — 执行结果

> 归档时间：2026-10-09
> 执行者：子 agent（S11m）
> 对应设计文档：无（本步来自 `docs/result/z100-11l-sdot4-reacceptance-result.md` §5.2/§6.1/§6.3 的待办项 ②）
> 前一档：[`z100-11l-sdot4-reacceptance-result.md`](z100-11l-sdot4-reacceptance-result.md)
> 状态：✅ **完成**（门禁 M1–M3/M5–M8 通过；M4 为**有条件通过**，口径见 §4）

## 0. 承接与本步要点

S11l 已经把 P2（`VLLM_DCU_USE_INT8_SDOT4=1`）的性能与数值都验完（decode 3.82×、比生产基线快 1.458×、
线性层与 Triton 逐位相同、7 文件 129/129），**唯一挡住上线的是复现性**：

| S11l 结论 | 位置 |
|---|---|
| `=1` 每次启动 AOT 产物都加载失败 → 重编（两个 rank 各一条 `Source code has changed`） | S11l §3.2.5 |
| 叠加 inductor `benchmark_combo_kernel=True` ⇒ 同开关两次启动文本不同（arm1 vs arm3：341/490 位置不同、max Δ 1.401 nat） | S11l §3.2.5/§3.3.3 |
| 同引擎内两遍逐 token + logprob 完全一致（flips=0、Δ=0.0000） | S11l §3.2.4 |
| 下一步线索：`VLLM_LOGS=DEBUG`、`VLLM_FORCE_AOT_LOAD` | S11l §5.2 |

本步的门禁（任务书原文）：**① 同开关 ≥3 次启动 greedy token ids 完全一致；② 测试矩阵补齐到 S9 的
7 文件 146 passed；③ 三相位（uncached/cache-fill/cache-hit）重启耗时实测**。

### 0.1 完成情况一览（边做边写：每完成一段就追加）

| # | 事项 | 状态 |
|---|---|---|
| 1 | 建骨架并持续追加 | ✅ 本节 |
| 2 | 定位 AOT 重载失败的**真机制** | ✅ §3.1（字节级定位到"**缺尾换行**"） |
| 3 | 最小复现（秒级、不起 27B） | ✅ §3.2（`tools/dcu/s11m_min_repro.py`，< 30 s） |
| 4 | 落地修复 | ✅ §2（`decorators.py`+`caching.py`，11 行有效改动） |
| 4b | 用 S11l 的**既有** =1 产物验证修复 | ✅ §3.3（P0 探针：同一份 `2b4e7331…` 产物这次**加载成功**） |
| 5 | 三相位重启耗时实测（`=1`/`=0`） | ✅ §3.4（`=1` 1663→1553–1573 s；`=0` 1804→1633 s） |
| 6 | ≥3 次启动 token ids 一致 | 🔶 §3.5（**`=1` 热启动 3/3 逐位一致 S2=S3=S4**、`=0` 热启动 2/2 Z2=Z3；但冷编译那次不同 ⇒ 有条件通过；另有重要修正：变体跟产物 vintage 走） |
| 7 | 测试矩阵 146 passed（`=1`） | ✅ §3.6（**146/146**，含 `test_qwen3_5_gdn.py` 20 passed） |
| 8 | workaround 验证（`benchmark_combo_kernel` / `VLLM_FORCE_AOT_LOAD`） | ✅ §3.7（`benchmark_combo_kernel=false` 生效但换 AOT key；`VLLM_FORCE_AOT_LOAD=1` 的 F0b **通过**，并暴露 DEBUG 会换 AOT key 的坑） |
| 9 | 证据归置到 `/mnt/game` | ✅ §3.8（`.tmp-11l`/`.tmp-11k`/`.tmp-11m` 全部 md5 校验后搬走，工作区只留脚本） |

**一句话结论（详见 §3.1/§3.5）**：`=1` 每次启动必重编的**真机制**是
`_verify_source_unchanged` 拿"产物内嵌的 `inspect.getsource` 快照（缺尾换行会被 `linecache` 补一个 `\n`）"
去比"磁盘裸读"，`gfx906_sdot4.py` 恰好缺尾换行 ⇒ 永远判 `Source code has changed`；**已修**（11 行、两个文件），
修后既有产物直接加载（`torch.compile` 54.15 s → 19.7 s），`=1` 下 **146 passed**。

**跨重启复现性的实测结构（比 S11l 的描述更细，也更关键）**：
* **AOT 加载路径是确定的**：`=1` 三次独立热启动 S2=S3=S4 **0/512 位置不同**（§3.5.4）；
  `=0` 两次热启动 Z2=Z3 **0/512**（§3.5.6）——这就是生产预热后的状态。
* **"填充缓存的那次冷编译"跑的是另一个变体**：S1 vs S2/S3/S4 差 318/490（§3.5.3）⇒ 冷启动必须当**预热步骤**。
* **变体主要由"产物/cache 的 vintage"决定，不由开关决定**：同开关、不同 vintage 的两次热启动差
  **422/512**（Z2 vs arm2）；同会话内 `=0` vs `=1`（不同产物）只差 **2/512**（§3.5.5/§3.5.6）。
  ⇒ S11l §3.3/L2/L3 里"`=0` vs `=1` 的 428/512 翻转"**主要是产物 vintage 差异**，不是"换内核"差异；
  S11l 真正站得住的是同进程内的那些结论（内核 294/294 逐位相同、同引擎两遍 0 差异）。

## 1. 目标与验收标准

任务书门禁原文（S11m）：

1. **AOT 重载**：定位 `=1` 的 AOT 产物每次加载失败的真机制（不是"缓存没命中"这种笼统说法），
   给出**可执行的最小复现**；按代价从低到高验证 workaround（`-cc.inductor_compile_config.
   benchmark_combo_kernel=False`、`VLLM_FORCE_AOT_LOAD`）；根因清楚且修法小就**落地修复**。
2. **跨重启复现性（P2 上线的最后一道门）**：同一开关 **≥3 次启动的 greedy token ids 完全一致**。
3. **三相位重启成本**：`uncached / cache-fill / cache-hit` 各自耗时，"修好后生产重启能到多少秒"
   要有实测数字（`=0` 与 `=1` 都给）。
4. **测试矩阵**：`=1` 下跑 `tests/model_executor/test_qwen3_5_gdn.py`（S9 基线 20 passed），
   把 S9 的 7 文件合计补到 **146 passed**。
5. **收尾**：`.tmp-11m/` 等证据目录整理到 `/mnt/game`，工作区只留脚本，归档里记路径。

硬约束（本步遵守）：27B 用卡 0、1（TP=2）、`--dtype float16`、**不用** `--enforce-eager`；
`VLLM_CACHE_ROOT=/mnt/game` 下**用独立子目录**；不跑自编译 HIP 可执行程序；单条阻塞命令加 `timeout`；
不 start/kill/exec/commit 生产容器 `qwen38-vllm`；不动 DTK 库；**不改 `VLLM_DCU_USE_INT8_SDOT4` 默认值（保持 OFF）**。

## 2. 实际改动

| 文件 | 位置 | 改动摘要 |
|---|---|---|
| `vllm/compilation/caching.py` | `:608-631`（新增 `_read_traced_file`）、`:6`（`import tokenize`） | 新增"按 `inspect.getsource` 的方式读文件"的读取器：`tokenize.open` 解码 + **最后一行没有换行时补一个 `\n`**（与 CPython `linecache.updatecache:140` 对齐）；读不到的文件返回 `""`（失败闭合，走重编）。`_compute_code_hash`（缓存 key 路径）**不动**，避免历史缓存失效 |
| `vllm/compilation/decorators.py` | `:266-289`（`_verify_source_unchanged`） | 校验从"**产物快照 vs 磁盘裸读**"改成"**产物快照 vs 同一读取器读出的磁盘内容**"；顺带把 S11l §5.2 想要的 `Traced files (to be considered for compilation cache)` DEBUG 日志搬到这里（原来在 `_compute_code_hash` 里，本路径不再调用它） |

`md5sum`（改动后）：

```
# 被改动的产品代码（md5 为最终值）
cef517974532d5ac5a9fd8928dc5f9fd  vllm/compilation/decorators.py
6555d89be203cadcd3752a11c76ae4c9  vllm/compilation/caching.py
# 本步新增的工具（工作区只留这些脚本 + 本归档）
baf4d567c8ce57bc076ded1dc9f74885  tools/dcu/s11m_aot_probe.py
2c74a2656c7d73476daa561a185fd1df  tools/dcu/s11m_min_repro.py
6baeef47c1b84e7753722c13fbaaa82d  tools/dcu/s11m_variant_matrix.py
186e30a00bfb86f1d55841ac60b5e819  tools/dcu/s11m_flip_matrix.py
1780b493c825a8295302a793ac540a78  tools/dcu/s11m_kernel_fingerprint.sh
fcbf383a61bef94f88c8c847ca573361  tools/dcu/s11m_tidy_evidence.sh
6103351739152fd878e95cd47f66bdd0  tools/dcu/s11m_tests.sh
b4c547c10254e1560c6eb9bdf0669222  tools/dcu/s11m_probe_fresh.sh
dddb32ea2f350e815f8d2404dd62aac5  tools/dcu/s11m_run_one.sh
36e9fbfeb40163669f889c8b1c91e9d3  tools/dcu/s11m_arm_env.sh
35e4a945971e97e862e0a744ad795735  tools/dcu/s11m_matrix.sh
079d182e0ac4f3c82316d41d3a257800  tools/dcu/s11m_batch2.sh
47550b14c2fff03c7dd330638e2aa74c  tools/dcu/s11m_batch3.sh
b42483c40ff2821361e20393bcf73ac4  tools/dcu/s11m_batch4.sh
61e9e87126a220720b209a7abf440772  tools/dcu/s11m_probes_stage.sh
```

有效 diff（`git diff vllm/compilation/`）：

```diff
--- a/vllm/compilation/caching.py
+++ b/vllm/compilation/caching.py
@@ -6,6 +6,7 @@
 import pickle
+import tokenize
@@ -604,6 +605,31 @@ def _compute_code_hash_with_content(...)
+def _read_traced_file(filepath: str) -> str:
+    """Read a traced file exactly the way ``inspect.getsource`` reads it. ..."""
+    if not os.path.isfile(filepath):
+        return ""
+    try:
+        with tokenize.open(filepath) as f:
+            file_contents = f.read()
+    except (OSError, UnicodeDecodeError, SyntaxError, LookupError):
+        return ""
+    if file_contents and not file_contents.endswith("\n"):
+        file_contents += "\n"
+    return file_contents
+
--- a/vllm/compilation/decorators.py
+++ b/vllm/compilation/decorators.py
@@ -266,7 +266,7 @@ def _verify_source_unchanged(...)
-    from .caching import _compute_code_hash, _compute_code_hash_with_content
+    from .caching import _compute_code_hash_with_content, _read_traced_file
@@ -274,8 +274,17 @@
         file_contents[file] = source.content
+    logger.debug(
+        "Traced files (to be considered for compilation cache):\n%s",
+        "\n".join(sorted(file_contents)),
+    )
     expected_checksum = _compute_code_hash_with_content(file_contents)
-    actual_checksum = _compute_code_hash(set(file_contents.keys()))
+    # `source.content` comes from `inspect.getsource`, so the on-disk side must
+    # be read the same way (see `_read_traced_file`); a raw read reports every
+    # file without a trailing newline as modified.
+    actual_checksum = _compute_code_hash_with_content(
+        {file: _read_traced_file(file) for file in file_contents}
+    )
```

新增工具（都在 `tools/dcu/`，本档的复现全部靠它们）：

| 工具 | 作用 |
|---|---|
| `s11m_aot_probe.py` | **不导入 torch** 就能解开 AOT 产物（pickle 流，用 stub `find_class`），列出全部 `inlined_sources` 并**逐文件与磁盘比对**（`--dump-module` 可 dump 出产物里那份源码） |
| `s11m_min_repro.py` | **< 30 s** 复现/回归本缺陷（合成"缺尾换行"模块 + 可选对真产物复算），并验证"真改代码仍报错" |
| `s11m_kernel_fingerprint.sh` | 两份产物的**生成代码指纹**（`triton_*`/`*_fused_*` 名集合）对比——产物字节不可比（内嵌随机 `CompileId`，见 §3.1） |
| `s11m_arm_env.sh` / `s11m_run_one.sh` / `s11m_matrix.sh` | 单次 27B 启动（显式 `VLLM_CACHE_ROOT`、可选 `startup` 快速探针）+ 批次编排 |
| `s11m_probe_fresh.sh` | 只等到"AOT 决策行"就杀（~2–4 min），用于冷编译/加载路径的快速判定 |
| `s11m_batch2.sh` / `s11m_batch3.sh` / `s11m_batch4.sh` | F0/F2/S4/Z1/Z2、F0b/W1/W2、Z3（都带"等上一批结束"的守卫） |
| `s11m_tests.sh` | S9 七文件（146）+ S11l 替身（3）的 `=1`/`=0` 测试矩阵 |
| `s11m_flip_matrix.py` | N 路 token 电池对比（逐位置 token id + 匹配位置 logprob 差） |
| `s11m_variant_matrix.py` | 全量两两"token 变体矩阵"（`=` 表示 0 差异）+ 逐 prompt 首个分叉位；§3.5.5 的表由它生成 |
| `s11m_tidy_evidence.sh` | 证据目录搬到 `/mnt/game` 并做 md5 全量校验（§3.8） |

**未改动**：`gfx906_sdot4.py` 一个字节都没动（S11l §6.3 的 md5 `59e66964…` 仍是当前值），
`VLLM_DCU_USE_INT8_SDOT4` 默认值不动（仍 `False`，`vllm/envs.py:169`/`:1414-1415`）。

## 3. 执行过程与原始证据

### 3.0 运行环境

```
# 容器：z100-s9（root）；/src = /home/lyh/z100-vllm（只读挂载，被验证的 Python 树）
# env：. /opt/dtk/env.sh; . /src/tools/dcu/env.sh
#      PYTHONPATH=/src:/mnt/game/triton-gfx906/install
#      TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache
# 27B：HIP_VISIBLE_DEVICES=0,1（TP=2）、--dtype float16、无 --enforce-eager、--max-model-len 4096
# 本步新增缓存根：/mnt/game/vllm-cache-11m/{one,zero}（=1/=0 各一个独立子目录，互不污染）
# 宿主时钟 = 容器时钟 - 8h（S11l 的 12:05/13:16 对应宿主 04:05/05:16）
```

### 3.1 真机制：**产物里的源码快照带了尾换行，磁盘文件没有** —— 逐字节定位

S11l §5.2 只到了"`=1` 的产物把 `gfx906_sdot4.py` 当 traced file、哈希对不上"这一步。本步不再起引擎，
而是**直接把 S11l 留下的 =1 产物解开看**（产物是 pickle 流，不是 zip；`torch.compiler.load_compiled_function`
的对应物），逐条重算 `_verify_source_unchanged` 的两个哈希。

工具：`tools/dcu/s11m_aot_probe.py`（宿主 `python3` 即可跑，不需要 torch：用 stub `find_class` 反序列化）。
目标产物：`/mnt/game/vllm-cache-27b/torch_compile_cache/torch_aot_compile/2b4e7331…/rank_0_0/model`
（= S11l arm1/arm3 的 =1 产物，md5 `85998013bb90f12a884425fcaa68a83c`）：

```
# python3 tools/dcu/s11m_aot_probe.py <artifact> --diff-root /home/lyh/z100-vllm --map /src=/home/lyh/z100-vllm
# artifact: .../2b4e7331b5917485697b9af412969a813712449805d4e9a957ccf454ca6bfc15/rank_0_0/model
# source_info entries found: 28
  09365678f3cc975c     18962B    634L  torch._dynamo.polyfills
  ...(略 24 条，见 .tmp-11m/probe-art2b4e.json)...
  9917f72436c69f98     10185B    257L  vllm.model_executor.kernels.linear.scaled_mm.gfx906_sdot4
  d39f811f9163d9e5     85008B   2148L  vllm.model_executor.models.qwen3_5
  ...
# on-disk comparison
  MATCH        vllm._custom_ops  artifact=8cf8d7f6a279 disk=8cf8d7f6a279 first_diff_line=-1
  ...(26 条 MATCH)...
  DIFF         vllm.model_executor.kernels.linear.scaled_mm.gfx906_sdot4  artifact=9917f72436c6 disk=ac4f9a69e0b3 first_diff_line=-1
```

**28 条里 27 条逐字节相同，只有 `gfx906_sdot4.py` 一条 DIFF，而且 `first_diff_line=-1`
（按行比没有一行不同）** —— 这直接把嫌疑锁定到"行尾/行结构"而不是代码内容。把产物里那份内容 dump 出来
与磁盘文件 `cmp`：

```
# python3 tools/dcu/s11m_aot_probe.py <artifact> --dump-module \
#     vllm.model_executor.kernels.linear.scaled_mm.gfx906_sdot4 --dump-out .tmp-11m/art-gfx906_sdot4.py
# dumped ... (10185B) -> .tmp-11m/art-gfx906_sdot4.py
$ cmp .tmp-11m/art-gfx906_sdot4.py vllm/model_executor/kernels/linear/scaled_mm/gfx906_sdot4.py
cmp: EOF on vllm/model_executor/kernels/linear/scaled_mm/gfx906_sdot4.py after byte 10184, in line 257
$ tail -c 40 .tmp-11m/art-gfx906_sdot4.py | xxd | tail -2
00000000: 7970 6529 0a20 ... 7572 6e20 6f75 742e 7669 6577 286f 7574
00000020: 5f73 6861 7065 290a                        _shape).
$ tail -c 40 vllm/model_executor/kernels/linear/scaled_mm/gfx906_sdot4.py | xxd | tail -2
00000000: 7479 7065 290a 2020 ... 7265 7475 726e 206f 7574 2e76 6965
00000020: 742e 7669 6577 286f 7574 5f73 6861 7065 29   t_shape)
```

⇒ **产物内嵌的源码 = 磁盘源码 + 一个尾换行**（10185 vs 10184 字节，257 vs 256 个 `\n`）。

为什么产物会多一个换行？三处代码一对就清楚了：

| 环节 | 代码 | 行为 |
|---|---|---|
| torch 建快照 | `torch/_dynamo/package.py:162` `_get_module_content()` = **`inspect.getsource(module)`**；`:184` `InlinedSource(..., content=_get_module_content(module))` | 快照 = `inspect.getsource` 的输出 |
| CPython 读文件 | `inspect.getsource`(module) → `linecache.getlines` → `linecache.updatecache`，**`/usr/lib/python3.10/linecache.py:140` `if lines and not lines[-1].endswith('\n'): lines[-1] += '\n'`** | **最后一行没换行就补一个** |
| vLLM 校验 | `vllm/compilation/decorators.py:266` 旧代码：`expected=哈希(快照)` vs `actual=_compute_code_hash(files)` → `caching.py:634` **`with open(filepath) as f: f.read()`**（裸读，不补） | 两边口径不一致 |

于是：**任何"最后一行没有尾换行"的被 traced 文件，都会在每次加载时被判成"源码变了"**。
`=1` 之所以 100% 命中这个坑，是因为它把 `gfx906_sdot4.py`（S11l 诊断改动后**最后一个字节不是 `\n`**）
拉进了 traced files；`=0` 的 traced files 里没有它，所以从来都是干净的。

这不是"缓存没命中"，而是**校验函数自己引入了不一致**：`source.content` 是 linecache 规范化过的，
磁盘侧却是裸字节。同一份文件、同一个进程内，torch 自己的 `dynamo.source_info` 校验（`package.py:622-628`
`_hash_sourcelines`）反而不会误报，因为它两边都用 `inspect.getsourcelines`（同样规范化）。

**旁证：全部 10 份历史产物的扫描**（`tools/dcu/s11m_aot_probe.py` 遍历，原始表见 §3.4）：

```
mtime               key            #ent traces_gfx906 content_len  diffs
10-08 14:25:10  b548de046c23     33 False                None  ['_custom_ops']
10-08 14:57:58  89ba96b5b5a3     33 False                None  ['_custom_ops']
10-08 16:15:24  430798d90652     33 False                None  ['_custom_ops']
10-08 17:02:32  10152eba2ef4     33 False                None  ['_custom_ops']
10-08 23:51:57  ce081eee30f6     33 False                None  []
10-09 02:17:50  e7d6503b1847     33 False                None  []
10-09 03:17:02  bfbdfaa3fb4f     28 True                 9120  ['gfx906_sdot4']
10-09 03:22:40  bca2a2932f12     33 False                None  []
10-09 05:16:48  2b4e7331b591     28 True                10185  ['gfx906_sdot4']
10-09 05:51:05  cdf688ef4aa6     28 True                10185  ['gfx906_sdot4']
```

* **`=0` 的产物（33 条、`traces_gfx906=False`）0 个 DIFF** ⇒ 每次都能吃缓存（S11l 观察一致）。
* `=1` 的产物（28 条）全部命中 `gfx906_sdot4`：`bfbdfaa3`（10-09 03:17）内嵌的是 **9120 字节**的
  **S11k 旧版**文件（真有代码差异，`first_diff_line=24`，属"合法过期"）；`2b4e7331`/`cdf688ef`
  内嵌的是 **10185 字节**（= 当前文件 + 尾换行）⇒ **唯一差异就是尾换行**。
* Oct 8 那批 `_custom_ops` DIFF 同样是"合法过期"（之后 `_custom_ops.py` 改过）。
* `2b4e7331` 的 key 被 arm1(12:05) 与 arm3(13:16) **共用**（AOT key = `sha256(str(aot_compile_hash_factors
  + [_model_hash_key(forward)]))`，`decorators.py:548-557`，**与源码内容无关**）⇒ S11l 说的"arm1 与 arm3 用同一份
  产物路径、各自 fresh compile 一次、后一次覆盖前一次"得到确认。`cdf688ef` 是 arm4（`maxlen=8192`，
  config_hash 不同 ⇒ 另一个 key）。
* **方法学警示（本步实测）**：**AOT 产物的字节不能当"编译是否可复现"的判据**。
  产物里内嵌 `CompileId.id_ = str(uuid.uuid4())`（`torch/_dynamo/utils.py:1895-1897`），
  所以同一份产物两次编译的 `__compiled_fn_1_<uuid>` 一定不同、文件大小/字节一定不同。
  本步实测：S1（本步 `=1` 冷编译产物，md5 `911769244fccaac536a3cf0ff699d736`，16,690,336 B）
  vs S11l arm3 的产物（md5 `85998013bb90f12a884425fcaa68a83c`，16,678,379 B）字节不同，但把
  **129 个 `triton_*`/`*_fused_*` kernel 名（即生成代码的指纹）取集合做 diff，结果为 0 行差异**
  （`diff kA.txt kB.txt` → 0）。⇒ 想验证"两次 fresh compile 是否等价"必须看 **token ids**
  或生成代码指纹，不能看产物 md5。§3.5 因此只认 token ids。

### 3.2 最小复现（秒级，不起 27B）

`tools/dcu/s11m_min_repro.py`：构造一个"最后一行没有换行"的临时模块，按 torch 的方式
（`inspect.getsource`）造 `SourceInfo`，分别跑**旧口径**（裸读）与**当前 `_verify_source_unchanged`**，
再验证"真的改了源码仍能被查出"（不放宽校验）。同时可 `--artifact` 直接对着真产物复算。

```
$ docker exec z100-s9 bash -lc '... cd /src && python tools/dcu/s11m_min_repro.py --artifact <2b4e7331/rank_0_0/model>'
[case1] no trailing newline: legacy(old) check equal=False
        expected(artifact)=4e49d01aebf0fa29 actual(raw disk)=3bea9f2cb25cfbfa
[case1] new _verify_source_unchanged: PASS (no exception)
[case2] trailing newline  : PASS (both old and new code pass)
[case3] real change       : detected (RuntimeError) OK

[artifact] vllm.model_executor.kernels.linear.scaled_mm.gfx906_sdot4
           artifact bytes=10185 disk bytes=10184
           legacy(old) equal=False  expected=aeae96755e6cb10e actual=8e36ee98eddf62eb
           new(normalized) equal=True

# cases failed: 0
```

价值：① 30 s 内复现/回归这个缺陷，不需要 27B；② **`case3` 证明修复没有削弱校验**
（真改代码照样 `RuntimeError` → 重编）。诚实地说明放宽了什么：**只放宽"最后一个字节的换行差异"**
（linecache 视角下两份内容完全相同，Python 解析结果也相同），其余任何字节差异仍会报警。

顺带把 S11l §5.2 想要的 DEBUG 日志补上并验证（新代码 `decorators.py:277`）：

```
$ VLLM_LOGGING_LEVEL=DEBUG python -c '…调 _verify_source_unchanged(合成 SourceInfo)…'
DEBUG 10-09 15:18:18 [compilation/decorators.py:277] Traced files (to be considered for compilation cache):
DEBUG 10-09 15:18:18 [compilation/decorators.py:277] /tmp/tmpz28habb_/dbgmod.py
OK
```

（原来这行在 `_compute_code_hash` 里；新的校验路径不再调用它，所以搬到了 `_verify_source_unchanged`，
`VLLM_LOGGING_LEVEL=DEBUG` 下仍然能看到"这次到底把哪些文件算进了校验"。）

### 3.3 用 S11l 的**既有** =1 产物验证修复（P0 探针，100 s）

修复后第一件事：拿 **S11l 留下的那份** `2b4e7331…` 产物（一个字节都没重新编译），**原样**再起一次 `=1`，
看到 AOT 决策行就停（`s11m_run_one.sh <name> 1 <cache> <port> 4096 startup`）。

```
$ tools/dcu/s11m_run_one.sh P0aotload 1 /mnt/game/vllm-cache-27b 8031 4096 startup
[orch] probe hit after 100s

# 引擎日志（.tmp-11m/P0aotload.log）：
125: (Worker_TP0) INFO [default_loader.py:484] Loading weights took 23.96 seconds
149: (Worker_TP1) INFO [decorators.py:321] Directly load AOT compilation from path /mnt/game/vllm-cache-27b/…/2b4e7331…/rank_1_0/model
222: (Worker_TP0) INFO [backends.py:290] Directly load the compiled graph(s) for compile range (1, 2048) from the cache, took 12.701 s
223: (Worker_TP0) INFO [decorators.py:321] Directly load AOT compilation from path /mnt/game/vllm-cache-27b/…/2b4e7331…/rank_0_0/model
224: (Worker_TP0) INFO [monitor.py:53] torch.compile took 19.54 s in total
```

对照 S11l arm1（**同一份产物、同一套参数**，修复前）：

```
141: (Worker_TP0) WARNING [decorators.py:322] Compiling model again due to a load failure from …/2b4e7331…/rank_0_0/model,
     reason: Source code has changed since the last compilation. Recompiling the model.
142: (Worker_TP1) WARNING [decorators.py:322] … rank_1_0/model, reason: Source code has changed …
152: (Worker_TP0) INFO [decorators.py:717] saved AOT compiled function to …/2b4e7331…/rank_0_0/model
153: (Worker_TP0) INFO [monitor.py:53] torch.compile took 54.15 s in total
```

⇒ **`Source code has changed` 消失，两个 rank 都从既有产物直接加载**；而且连 inductor 的
`Directly load the compiled graph(s) … from the cache`（12.70 s）也命中了 ⇒ **编译产物完全复用**。
`torch.compile` 19.54 s vs 修复前 54.15 s（**-34.6 s**）。

### 3.4 三相位重启耗时（`uncached` / `cache-fill` / `cache-hit`）

口径（照 `docs/海光Z100-性能优化方案.md` §7）：每个开关值用**一个独立的 `VLLM_CACHE_ROOT` 子目录**
（`/mnt/game/vllm-cache-11m/{one,zero}`），先空目录冷启动一次（= `uncached`，该次同时把产物写下去 = `cache-fill`），
再在同一个目录上热启动（= `cache-hit`）。每个 run 都跑同一套 8 prompt × ≤64 token 的 greedy 电池。

* `uncached`：`VLLM_CACHE_ROOT` 子目录为空 → 需要完整冷编译（inductor codegen + AOT 产物落盘）。
* `cache-fill`：与 `uncached` 是**同一次启动**（冷启动必然填缓存）；这里把它的"填缓存"代价单独记账：
  `saved AOT compiled function` + `torch.compile took` + `init engine (… compilation: X s)`。
* `cache-hit`：产物已存在且校验通过 → `Directly load AOT compilation` + `Directly load the compiled graph(s)`。

> 注意（诚实标注）：三次相位只清空 **vLLM/torch.compile 的缓存根**；`TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache`
> 是**共享且热**的（生产也这么配），所以"uncached"不含 Triton kernel 的首次编译。

| run | 开关 | 缓存相位 | `Loading weights` | `torch.compile took` | AOT 决策 | inductor graph cache | 两轮 graph capture | `init engine …(compilation:` | 到 ready 墙钟 |
|---|---|---|---|---|---|---|---|---|---|
| S1 `S1sw1fill` | `=1` | uncached（空目录，同次 fill） | 23.97 s | **93.65 s**（冷编译 + `saved AOT compiled function`） | 编译（无产物可载） | 0 命中 / 1 落盘 | 693 s + 700 s | **1525.19 s (compilation: 95.05 s)** | **1663 s** |
| S2 `S2sw1hit` | `=1` | cache-hit | 24.16 s | **19.70 s**（`Directly load AOT compilation` ×2 rank） | **加载成功**（`ind_hit=1`） | 1 命中 / 0 落盘 | 687 s + 708 s | **1447.35 s (compilation: 19.70 s)** | **1573 s** |
| S3 `S3sw1hit` | `=1` | cache-hit | 24.31 s | **20.00 s** | **加载成功** | 1 命中 / 0 落盘 | 687 s + 708 s | **1441.09 s (compilation: 20.00 s)** | 1563 s |
| S4 `S4sw1hit` | `=1` | cache-hit | 24.64 s | **19.60 s** | **加载成功** | 1 命中 / 0 落盘 | 681 s + 702 s | **1435.82 s (compilation: 19.92 s)** | 1553 s |
| Z1 `Z1sw0fill` | `=0` | uncached（空目录，同次 fill） | 20.70 s | **156.48 s**（`Compiling a graph … takes 107.64 s` + `saved AOT compiled function`） | 编译（无产物可载） | 0 命中 / 1 落盘 | 706 s + 718 s | **1657.32 s (compilation: 156.48 s)** | **1804 s** |
| Z2 `Z2sw0hit` | `=0` | cache-hit | 22.82 s | **23.15 s** | **加载成功**（`…/e7d6503b…` ×2 rank） | 1 命中 / 0 落盘 | 698 s + 720 s | **1515.02 s (compilation: 23.23 s)** | **1633 s** |
| Z3 `Z3sw0hit` | `=0` | cache-hit（第二样本） | 19.15 s | **23.84 s** | **加载成功** | 1 命中 / 0 落盘 | 702 s + 720 s | **1519.99 s (compilation: 23.84 s)** | **1633 s** |

S1（`=1` 冷启动）原始行：

```
(Worker_TP0) INFO [default_loader.py:484] Loading weights took 23.97 seconds
(Worker_TP0) INFO [decorators.py:726] saved AOT compiled function to /mnt/game/vllm-cache-11m/one/torch_compile_cache/torch_aot_compile/2b4e7331…/rank_0_0/model
(Worker_TP0) INFO [monitor.py:53] torch.compile took 93.65 s in total
(Worker_TP0/TP1) INFO [model_runner.py:1060] Graph capturing finished in 693 secs, took 1.67 GiB
(Worker_TP0/TP1) INFO [model_runner.py:1060] Graph capturing finished in 700 secs, took 1.48 GiB
(APIServer) INFO [core.py:381] init engine (profile, create kv cache, warmup model) took 1525.19 s (compilation: 95.05 s)
(APIServer) INFO:     Application startup complete.
```

**一个必须先说清的结论**：27B 的重启耗时**几乎不由编译决定**——两轮 CUDA/HIP graph capture
（`=1`：693+700 s / 681+702 s；`=0`：706+718 s / 698+720 s，即 **1383–1424 s ≈ 23–24 min**）
占了 `init engine` 的 **92–95%**。修 AOT 省下的编译时间与它相比是小头：

| 量 | `=1` uncached → cache-hit | `=0` uncached → cache-hit |
|---|---|---|
| `torch.compile`（含 AOT 加载/落盘） | 93.65 s → **19.6–20.0 s**（**−74 s**） | 156.48 s → **23.15 s**（**−133 s**） |
| `init engine (… compilation: X)` | 1525.19 s (95.05) → **1435.8–1447.4 s (19.7–20.0)**（**−78…−90 s**） | 1657.32 s (156.48) → **1515.02 s (23.23)**（**−142 s**） |
| **到 ready 墙钟** | 1663 s → **1553–1573 s**（**−90…−110 s**） | 1804 s → **1633 s**（**−171 s**） |
| 两轮 capture | 1393 / 1383 / 1395 s（≈91% of init） | 1424 / 1418 s（≈86–94% of init） |

⇒ **"修好后生产重启能到多少秒"的实测答案：`=1` 热重启 ≈ 1553–1573 s（26 min 上下）、
`=0` 热重启 ≈ 1633 s（27 min）**；其中权重加载 ~24 s、编译/加载 ~20 s，
**其余 ~23 min 全是两轮 CUDA/HIP graph capture**（51 段 PIECEWISE + FULL，每个 rank 各一轮）。
P5 里"cache-hit 仍要重新加载权重并重新捕获 graph"这句在 27B 上被实测坐实。
**修 AOT 的价值主要在"确定性"（§3.5）而不是"启动速度"**。

**三相位口径的总结**（`cache-fill` 与 `uncached` 是同一次启动，这里把"填缓存"的代价单独记账）：

| 相位 | `=1` | `=0` |
|---|---|---|
| `uncached`（= 冷启动，空缓存根） | 1663 s（其中冷编译 93.65 s、落盘 1 份产物） | 1804 s（冷编译 156.48 s） |
| `cache-fill`（同一次启动写下的东西） | AOT 产物 16.69 MB ×2 rank + `inductor_cache`（`one/…/2b4e7331…/inductor_cache`） | AOT 产物 ×2 rank + `inductor_cache`（`zero/…/e7d6503b…/inductor_cache`，401 MB） |
| `cache-hit`（热启动） | **1553–1573 s**（编译只花 19.6–20.0 s；3 个样本） | **1633 s**（23.15 / 23.84 s；2 个样本） |

### 3.5 ≥3 次启动的 greedy token ids 对比（**本步最重要的结果，且不是好消息**）

口径：每次启动都跑 `tools/dcu/s11l_flip_stats.py` 的 8 prompt × ≤64 token greedy 电池
（`temperature=0`、`top_p=1.0`、`logprobs=true`、`return_token_ids=true`），
比对工具 `tools/dcu/s11m_flip_matrix.py`（逐位置 token id + 匹配位置 logprob 差值）。

#### 3.5.1 先复现 S11l 的两个基准数

| 对比 | 位置差 | max_matched_delta | 说明 |
|---|---|---|---|
| arm3 vs arm3b（**同一引擎**两遍，S11l） | **0/490** | 0.000000 | ✅ 同引擎内逐位稳定（与本步工具一致复现 S11l 的 flips=0） |
| arm1 vs arm3（**两次启动**，S11l） | **341/490** | 1.401050 | ✅ 与 S11l §3.2.5 的"341/490、max Δ 1.401"逐位一致（数量与数值都对上） |

#### 3.5.2 本步新增的跨启动对比（`=1`）

| 对比 | 位置差 | max Δ | 备注 |
|---|---|---|---|
| **S1（冷编译/fill）vs arm3（S11l 的 =1 引擎）** | **0/490** | **0.000000** | ‼️ **两次"各自 fresh compile"的引擎给出逐位完全相同的 token 与 logprob**（不同天、不同缓存根、arm3 还是 `INNER=triton`） |
| S1 vs arm3b（arm3 引擎第二遍） | 0/490 | 0.000000 | 同上 |
| **S1 vs S2（同一个缓存根，S2 = AOT 加载）** | **318/490** | 1.400876 | ‼️ 用 **S1 自己写下的产物**加载的 S2，token 与 S1 不同 |
| S2 vs arm3 | 318/490 | 1.400876 | S2 与"S1==arm3"这一族不同 |
| S2 vs arm1 | 363/512 | 1.543604 | |
| S1 vs arm1 | 341/490 | 1.401050 | = S11l 的 arm1-vs-arm3 数（因为 S1==arm3） |

**由此可确定的结论（本步已足够，加上 §3.5.3 的 S3）**：

1. **`=1` 的 greedy token 串不是开关的稳定属性** —— S11l §3.3.0(b) 已看到过一例（S11k 冷编译臂 vs arm1），
   本步再给出两例（S1 vs S2/S3；S2 vs arm1/arm3）。**至少观察到 3 种不同的 token 串**。
2. **"两次 fresh compile 一定不同"也不成立**：S1 与 arm3 是两次独立的 fresh compile，
   却**逐位（含 logprob）完全相同**（0/490、Δ=0.000000）。⇒ 变体数有限（不是连续噪声），
   像"若干个离散的编译变体"。
3. **修好 AOT 重载不等于拿到复现性**：S2 加载的正是 S1 写下的产物
   （日志 `Directly load AOT compilation from path …/2b4e7331…` + `Directly load the compiled graph(s) … took 12.836 s`，
   `ind_hit=1`、`ind_save=0`、无任何 `Compiling a graph …` 行），**却与 S1 不是同一串 token**。
   ⇒ "同一份编译产物 ⇒ 同一串 token"这条前提在本树上**不成立**，扰动源不止"重编/不重编"。

#### 3.5.3 S3（第三次 `=1` 启动，同样走 AOT 加载）：**热启动之间是逐位一致的**

```
$ python tools/dcu/s11m_flip_matrix.py /mnt/game/s11m/flip-S2sw1hit.jsonl /mnt/game/s11m/flip-S3sw1hit.jsonl
[PAIR] S2sw1hit vs S3sw1hit: 0/512 positions differ, max_matched_delta=0.000048
# ALL_IDENTICAL=True
$ python tools/dcu/s11m_flip_matrix.py /mnt/game/s11m/flip-S1sw1fill.jsonl /mnt/game/s11m/flip-S3sw1hit.jsonl
[PAIR] S1sw1fill vs S3sw1hit: 318/490 positions differ, max_matched_delta=1.400876
# ALL_IDENTICAL=False
```

⇒ **本步的核心判读**：这套栈上的 token 串分成两类稳定的变体，

| 变体 | 谁属于它 | 说明 |
|---|---|---|
| **冷编译变体（A）** | S1（它写下产物）、arm3、arm3b | "在自己进程里刚编译出来的那份代码"跑出的结果；S1 与 arm3（另一次独立冷编译、另一天、另一个缓存根、`INNER=triton`）**0/490 逐位相同** |
| **AOT 加载变体（B）** | **S2、S3**（都加载 S1 写下的同一份产物） | 两次独立启动 **0/512 逐位相同、logprob 最大差 4.8e-5** ⇒ **可复现** |
| 第三变体（C） | arm1（S11l 的另一次冷编译） | 与 A、B 都不同（341/490、363/512） |

也就是说：**"编译这个动作本身"会在若干离散变体里挑一个（对得上 S11l 的 timing-based
`benchmark_combo_kernel` 解释），而"加载 AOT 产物"这条路是确定的**。同一份产物、
同一份 inductor cache，两次启动（S2/S3）跑出完全相同的 token 与几乎相同的 logprob；
而**编译进程自己执行的那份代码**与它**存下来给后面加载的那份**不是同一个变体（S1 属于 A，S2/S3 属于 B）。
这对生产的意义非常直接：

* 修好 AOT 重载后，**生产（缓存已预热）的每次重启都逐 token 可复现**（S2==S3==S4，见 §3.5.4）；
* **但"填充缓存的那一次冷启动"与之后的重启不是同一串文本** ⇒ 部署流程必须把冷启动当作
  **预热步骤**（先跑一次、再重启对外服务），否则"重启换输出"这件事仍然存在一次。

#### 3.5.4 S4（第三次热启动）与同引擎两遍：**门禁①在本口径下通过**

S4 = `=1` + 缓存根 `one`（已预热），启动时同样 `Directly load AOT compilation from path …/2b4e7331…` ×2 rank、
`torch.compile took 19.60 s`。它跑了两遍电池（`flip` + `flipb`，第二遍命中前缀缓存）：

```
$ python tools/dcu/s11m_flip_matrix.py .../flip-S2sw1hit.jsonl .../flip-S4sw1hit.jsonl
[PAIR] S2sw1hit vs S4sw1hit: 0/512 positions differ, max_matched_delta=0.000000   # 逐位 + logprob 完全相同
$ ... flip-S3sw1hit.jsonl flip-S4sw1hit.jsonl
[PAIR] S3sw1hit vs S4sw1hit: 0/512 positions differ, max_matched_delta=0.000048
$ ... flip-S4sw1hit.jsonl flip-S4sw1hitb.jsonl      # 同引擎第二遍
[PAIR] S4sw1hit vs S4sw1hitb: 0/512 positions differ, max_matched_delta=0.000000
$ ... flip-S1sw1fill.jsonl flip-S4sw1hit.jsonl
[PAIR] S1sw1fill vs S4sw1hit: 318/490 positions differ, max_matched_delta=1.400876
```

⇒ **S2、S3、S4 是三次独立的进程启动，都在 AOT 加载路径上，greedy token ids 三次完全一致**
（512/512 相同；logprob 最大差 ≤ 4.8e-5，即"token 完全一致、logprob 有 5e-5 级浮点抖动"）。
**门禁①（同开关 ≥3 次启动 token ids 一致）在"缓存已预热后的重启"这个口径下 ✅ 通过。**

**产物没有被重写**（S2/S3/S4 都是真加载，不是"又编了一遍恰好一样"）：

```
$ md5sum one/…/2b4e7331…/rank_0_0/model      # S1 写完时、S4 之后各查一次，值不变
911769244fccaac536a3cf0ff699d736
```

#### 3.5.5 全部可用 run 的 token 变体矩阵（含 S11l 的四个臂）—— **变体数 > 2，且与开关取值不对应**

把 S11l 与本步所有能拿到的 greedy 电池（8 prompt × ≤64 token）两两对比（`= 表示 0 处不同`）：

```
             arm1     arm3     arm3b    S1sw1    S2sw1    S3sw1    S4sw1    S4sw1b   arm2     Z1sw0    arm4
arm1(=1冷)    =       363/490  363/490  363/490  363/512  363/512  363/512  363/512  428/512  385/512  399/512
arm3(=1冷)  363/490     =        =        =       340/490  340/490  340/490  340/490  402/490  337/490  427/490
arm3b       363/490     =        =        =       340/490  340/490  340/490  340/490  402/490  337/490  427/490
S1sw1fill   363/490     =        =        =       340/490  340/490  340/490  340/490  402/490  337/490  427/490
S2sw1hit    363/512  340/490  340/490  340/490     =        =        =        =       423/512  128/512  425/512
S3sw1hit    363/512  340/490  340/490  340/490     =        =        =        =       423/512  128/512  425/512
S4sw1hit    363/512  340/490  340/490  340/490     =        =        =        =       423/512  128/512  425/512
S4sw1hitb   363/512  340/490  340/490  340/490     =        =        =        =       423/512  128/512  425/512
arm2(=0温)  428/512  402/490  402/490  402/490  423/512  423/512  423/512  423/512     =      421/512  433/512
Z1sw0fill   385/512  337/490  337/490  337/490  128/512  128/512  128/512  128/512  421/512     =      428/512
arm4(=1冷)  399/512  427/490  427/490  427/490  425/512  425/512  425/512  425/512  433/512  428/512     =
```

等价的"簇"（互相 0 差异）：`{arm3, arm3b, S1}`、`{S2, S3, S4, S4b}`、`{arm1}`、`{arm2}`、`{Z1}`、`{arm4}`
⇒ **10 次独立启动里至少 6 个不同的 token 变体，而且它们与"开关 =0/=1"并不对应**
（两个 =0 引擎 arm2 与 Z1 之间差 421/512，两个 =1 引擎 arm1 与 S2 之间差 363/512）。

这一条有两个必须写下来的后果（都很重要）：

1. **"`=1` 改变输出"这件事，用"跨引擎对整个 prompt 的翻转率"是证明不了的**：
   同开关跨引擎的差异（arm2 vs Z1，都是 `=0`：**421/512**）与跨开关的差异
   （arm1 vs arm2：**428/512**）**同样大**。S11l §3.3 的 L2/L3（翻转点重合、翻转率、max Δ）
   因此只能支持"**这是一族贴边翻转**"，不能单独支持"翻转是 `=1` 造成的"。
   另外，**所有变体对的首个分叉点都落在同一批位置**（`zh_hz@9`、`zh_ml@7`、`en_fr@13`、
   `en_sea@10`、`en_math@2`）——这正是 S11l §3.3 里"5 个早期分叉点重合"的来源，
   但它**对区分"开关效应"和"运行间抖动"没有判别力**（任何两串都会在这些并列点分叉）。
   S11l 真正站得住的证据是**同一进程内**的那几条：内核 294/294 逐位相同、
   同引擎两遍 0 差异（本步 S4 vs S4b 再次复现（§3.5.4））。
2. **本步的 AOT 修复因此更关键**：在修复前，`=1` **永远走"冷编译"这条路**（而 `=0` 一直走"加载"），
   所以修复前 `=1` 的每次启动都在"每次重新抽一个变体"；修复后 `=1` 才第一次具备 `=0` 本来就有的
   "**加载路径可复现**"这一属性（S2/S3/S4 = 3/3 逐位一致）。

#### 3.5.6 Z2（`=0` 热启动）：**变体由"产物/缓存 vintage"决定，不由开关决定** ← 本步第二个重要发现

Z2 的日志确认走加载路径（`Directly load AOT compilation from path …/e7d6503b…` ×2 rank、
`torch.compile took 23.15 s`、`ind_hit=1`、`ind_save=0`），且**确实是 `=0` 路径**：
`TritonInt8ScaledMMLinearKernel`（对照 S1/S2 是 `Gfx906Sdot4Int8ScaledMMLinearKernel`）。

```
$ python tools/dcu/s11m_flip_matrix.py …/flip-Z2sw0hit.jsonl …/flip-arm2.jsonl
[PAIR] Z2sw0hit vs arm2: 422/512 positions differ, max_matched_delta=0.809530     # 都是 =0、都走加载路径
$ … flip-Z2sw0hit.jsonl flip-Z1sw0fill.jsonl
[PAIR] Z2sw0hit vs Z1sw0fill: 127/512 positions differ, max_matched_delta=0.770719  # 同一份产物：加载 vs 编译
$ … flip-Z2sw0hit.jsonl flip-S2sw1hit.jsonl
[PAIR] Z2sw0hit vs S2sw1hit: 2/512 positions differ, max_matched_delta=0.117957     # =0 加载 vs =1 加载
$ … flip-Z2sw0hit.jsonl flip-S4sw1hit.jsonl
[PAIR] Z2sw0hit vs S4sw1hit: 2/512 positions differ, max_matched_delta=0.117957
```

对照上一节的变体矩阵，可读出这样一张图（"变体"= 互不相同的 token 串）：

| 观察 | 数字 | 含义 |
|---|---|---|
| `=1` 的**热启动**彼此 | S2=S3=S4=S4b，**0/512** | 加载路径**可复现** |
| `=0` 热启动 vs `=0` 热启动（**不同 vintage 的产物**：Z1 新编的 `e7d6503b…` vs S11l 02:17 那份同 key 产物） | **422/512** | 变体跟着**产物/cache 的 vintage**走 |
| `=0` 热启动 vs `=1` 热启动（**不同 key、不同内核**，但都是本步新编的产物） | **2/512**（max Δ 0.118 nat） | **开关不是主要变量** |
| 同一份产物：**编译进程自己跑** vs **后续加载跑** | 127/512（Z1 vs Z2）、318/490（S1 vs S2/S3/S4） | "编译-执行"与"加载-执行"仍是两个变体（§5.1） |

完整的 15 路两两矩阵（含 Z3/W1/W2/arm4）已落盘：
`/mnt/game/s11m/evidence-11m/variant-matrix.txt` + `variant-matrix.json`
（生成工具 `tools/dcu/s11m_variant_matrix.py`）。其中的"精确同族"（0 差异）是：
`{arm3, arm3b, S1sw1fill}`、`{S2sw1hit, S3sw1hit, S4sw1hit, S4sw1hitb}`、`{Z2sw0hit, Z3sw0hit}`，
其余（`arm1`、`arm2`、`Z1sw0fill`、`W1`、`W2`、`arm4`）各自一族。
注意 `=0 热` 与 `=1 热` 两族之间是 **2/512**（不是 0）——这一对**同时**换了产物与开关，
所以这 2 个位置**不能干净地归给开关**（只能作为"同一会话内两种 config 的最小可观测差"）。

⇒ **两个必须写下来的修正**：

1. **`VLLM_DCU_USE_INT8_SDOT4` 不是 token 串的主要决定因素**。本步实测：
   * 同开关、不同 vintage 产物：**422/512** 不同（Z2 vs arm2，都是 `=0`）；
   * 不同开关、同 vintage（都本步新编）：**2/512** 不同（Z2 vs S2/S4）。
   因此 S11l §3.3/L2/L3 里"`=0` vs `=1` 的 428/512 翻转"**无法与"产物 vintage 不同"区分开**
   ——它测到的绝大部分是"两次不同编译产物"的差异，不是"换内核"的差异。
   （S11l 真正站得住的是**同进程内**的结论：内核逐位相同 294/294；本步的 §3.5.5 也复现了
   "同引擎两遍 0 差异"。）
2. **对生产的意义反而更强**：只要缓存固定（同一个 `VLLM_CACHE_ROOT`、不让它重编），
   输出就是稳定的；而"修好 AOT 重载"正是让 `=1` 也能固定在加载路径上的前提。
   **严禁**在同一套服务里换 `VLLM_CACHE_ROOT` / 删缓存 / 改 `compilation_config`（会换 key ⇒ 换变体）。

**Z3（`=0` 热启动第二样本，加载同一份 `e7d6503b…` 产物）**：

```
$ python tools/dcu/s11m_flip_matrix.py …/flip-Z2sw0hit.jsonl …/flip-Z3sw0hit.jsonl
[PAIR] Z2sw0hit vs Z3sw0hit: 0/512 positions differ, max_matched_delta=0.000015
# ALL_IDENTICAL=True
$ … flip-Z3sw0hit.jsonl flip-S2sw1hit.jsonl
[PAIR] Z3sw0hit vs S2sw1hit: 2/512 positions differ, max_matched_delta=0.117957   # en_fr@62
```

⇒ **`=0` 的加载路径同样可复现（Z2 == Z3，2 个样本，0/512）**，且与 `=1` 的加载变体只差 2/512。
**"加载路径可复现"这件事对两个开关取值都成立**；`=1` 的独特点只在于"修复前它根本走不到加载路径"（§3.5.5 结论 2）。

#### 3.5.7 W1/W2：`benchmark_combo_kernel=false` 的冷/热对拍

W1 = `=1` + `-cc.inductor_compile_config.benchmark_combo_kernel=false` + **空缓存根 `combo`**
（冷编译：`init engine … took 1523.50 s (compilation: 82.53 s)`，比默认冷编译的 93.65 s 略快，
因为没有 combo kernel 的计时对比）。它的 token：

```
$ … flip-W1sw1nocombo.jsonl flip-S2sw1hit.jsonl
[PAIR] W1sw1nocombo vs S2sw1hit: 398/506 positions differ, max_matched_delta=0.874296
$ … flip-W1sw1nocombo.jsonl flip-S1sw1fill.jsonl
[PAIR] W1sw1nocombo vs S1sw1fill: 378/490 positions differ, max_matched_delta=1.116405
$ … flip-W1sw1nocombo.jsonl flip-arm3.jsonl
[PAIR] W1sw1nocombo vs arm3: 378/490 positions differ, max_matched_delta=1.116405
```

⇒ **W1 是又一个独立变体**：关掉 `benchmark_combo_kernel` **并不能**把冷编译变体变成
"默认配置的热启动变体"（398/506、378/490 都很大）。也就是说这个开关**不是**"让冷启动等于热启动"的钥匙
（至少不是对**别的** config 的热变体而言）。

W2 = 同 config（flag）、同缓存根 `combo` 的热加载（`ready=1573 s`、`init engine … took 1461.73 s
(compilation: 16.85 s)`、`aot_load=2 / ind_hit=1 / ind_save=0`），回答**唯一还有意义的问题**：
**"关掉这个开关后，冷编译 == 它自己的热加载" 是否成立**。

```
$ … flip-W1sw1nocombo.jsonl flip-W2sw1nocombo.jsonl
[DIFF] zh_ml   : 39/64 positions, first=25, max_matched_delta=0.173088
[DIFF] zh_sort : 19/64 positions, first=45, max_matched_delta=0.143229
[PAIR] W1sw1nocombo vs W2sw1nocombo: 58/506 positions differ, max_matched_delta=0.508883
$ … flip-W2sw1nocombo.jsonl flip-S2sw1hit.jsonl
[PAIR] W2sw1nocombo vs S2sw1hit: 399/506 positions differ, max_matched_delta=0.874296
```

**判定（这是本步对"候选 workaround"的最终结论）**：

| 配置 | 冷启动 vs 它自己的热启动 | 与默认配置的热变体 |
|---|---|---|
| 默认（`benchmark_combo_kernel=True`） | S1 vs S2/S3/S4：**8/8 prompt 分叉、318/490** | — |
| `benchmark_combo_kernel=false` | W1 vs W2：**2/8 prompt 分叉、58/506**（6/8 prompt 逐位含 logprob 相同） | W2 vs S2：**399/506**（换 config ⇒ 换 AOT key ⇒ 换变体） |

⇒ **这个开关能"减小"但不能"消除"冷/热差异**（8/8 → 2/8，318/490 → 58/506），
而且它自己也要换一份 AOT key（多一次冷编译预热）。因此：
* **它不能替代"部署预热"**；
* 若某个部署确实无法预热（每次都是冷启动），开它可以把"频繁换文本"降到"少数 prompt 偶尔换"，
  但**不能承诺逐 token 一致**；
* **本步的推荐是：默认配置 + 预热**（§6.1），不把它写进生产。

#### 3.5.8 环境干扰记录（诚实标注）

S3 启动期间（07:38 起）宿主上有**另一个 agent 的 GPU 作业**在同一个容器 `z100-s9` 内运行：
`python /src/tools/dcu/p2p_verify.py`（`unset HIP_VISIBLE_DEVICES`，即用全部 4 张卡，900 s 超时）。
它抢占了卡 0/1 的算力，S3 的 graph capture 进度条瞬时速率从 S1/S2 的 ~13.6 s/图 变成 ~21 s/图。
**实测总量影响很小**：S3 两轮 capture = 687+708 s、S4 = 681+702 s，S1 = 693+700 s、S2 = 687+708 s，
四次都在 1383–1401 s 区间（±1.5%）⇒ 墙钟仍然可比（进度条前段的速率只是估计不准）。
**它不影响 token 的正确性**（前向是确定的，只是变慢）。
另外，Lead 已把这次事故写成硬约束 25，并另记硬约束 24（PCIe ACS 关闭发生在本次电池期间）——
本步的 token 结论是数值性的、不受 ACS 影响；但 §3.4 的墙钟数字跨了 ACS 前后，
按约束 24 的口径，正式对外的"启动耗时"应在 ACS 关闭后整体重测。


### 3.6 测试矩阵：`=1` 下 S9 的 7 文件 **146 passed**（补齐 S11l 的 129）

S11l 用 `tests/kernels/test_gfx906_packed_silu_fp16.py`(3) 替掉了 S9 的
`tests/model_executor/test_qwen3_5_gdn.py`(20)，所以只有 129。本步把两个集合**都**跑（8 个文件），
一次得到 "S9 七文件 = 146" 与 "并集 = 149" 两个口径。

容器用 **`z100-serve`**（不是 `z100-s9`）：两者挂的是同一份只读 `/src`，但 `z100-serve` 是独立
cgroup（8 GiB），不会和正在跑的 27B（`z100-s9`，32 GiB 上限）抢内存；GPU 用**卡 3**（27B 占 0/1）。
先核对容器内确实看到修复后的树：

```
$ docker exec z100-serve bash -lc '… cd /src && python -c "import torch,triton,vllm; …"'
vllm /src/vllm/__init__.py
_read_traced_file True
$ … s11m_tests.sh 1 /mnt/game/s11m/tests-sw1
[s11m-tests] sw=1 devices=3 out=/mnt/game/s11m/tests-sw1 host=b4e70c991884
[s11m-tests] vllm=/src/vllm/__init__.py sdot4_env=True
S11M_TEST sw=1 file=tests/kernels/attention/test_triton_unified_attention_gfx906.py   rc=0 passed=39
S11M_TEST sw=1 file=tests/kernels/attention/test_triton_unified_attn_gfx906_v1.py    rc=0 passed=12
S11M_TEST sw=1 file=tests/kernels/mamba/test_fused_recurrent_packed_decode.py        rc=0 passed=5
S11M_TEST sw=1 file=tests/kernels/mamba/test_fused_sigmoid_gating.py                 rc=0 passed=5
S11M_TEST sw=1 file=tests/kernels/quantization/test_gguf_fixed_cols_gfx906.py        rc=0 passed=13
S11M_TEST sw=1 file=tests/model_executor/test_qwen3_5_gdn.py                         rc=0 passed=20   ← S11m 补齐
S11M_TEST sw=1 file=tests/platforms/test_hygon_dtk.py                                rc=0 passed=52
S11M_TEST sw=1 file=tests/kernels/test_gfx906_packed_silu_fp16.py                    rc=0 passed=3    ← S11l 的替身
```

逐文件日志与汇总（`/mnt/game/s11m/tests-sw1/`，`SUMMARY.txt` 由各 log 末行生成）：

```
sw1-test_triton_unified_attention_gfx906.log        39 passed, 1 warning in 18.67s
sw1-test_triton_unified_attn_gfx906_v1.log          12 passed, 1 warning in 18.57s
sw1-test_fused_recurrent_packed_decode.log           5 passed, 1 warning in 16.45s
sw1-test_fused_sigmoid_gating.log                    5 passed, 1 warning in 16.44s
sw1-test_gguf_fixed_cols_gfx906.log                 13 passed, 3 warnings in 142.11s (0:02:22)
sw1-test_qwen3_5_gdn.log                            20 passed, 17 warnings in 41.70s
sw1-test_hygon_dtk.log                              52 passed in 8.80s
sw1-test_gfx906_packed_silu_fp16.log                 3 passed, 17 warnings in 43.39s
```

* **S9 七文件合计 = 39+12+5+5+13+20+52 = 146 passed / 0 failed**（与 S9 §3.6 逐文件计数完全一致），
* 并集（+packed_silu 3）= **149 passed / 0 failed**（= S11l 的 129 + 20，两个口径互相印证）。
* `=1` 时 `envs.VLLM_DCU_USE_INT8_SDOT4=True` 由脚本首行断言打印，避免"跑了但开关没生效"。

### 3.7 workaround 验证（`benchmark_combo_kernel=False` / `VLLM_FORCE_AOT_LOAD=1`）

#### 3.7.1 `-cc.inductor_compile_config.benchmark_combo_kernel=False`

任务书要求"先试这个候选"。**开关本身能生效**（启动日志的 effective config 原文）：

```
'inductor_compile_config': {'benchmark_combo_kernel': False, 'enable_auto_functionalized_v2': False,
                            'size_asserts': False, 'alignment_asserts': False, 'scalar_asserts': False}
'saved AOT compiled function to /mnt/game/vllm-cache-11m/freshB/torch_compile_cache/torch_aot_compile/
 ab6111e9916394d7f46d421b97a5f86a81fb739b7df82e1e311505cb5b25bea4/rank_0_0/model'
'torch.compile took 91.14 s in total'      # 空缓存 → 冷编译
```
（命令用**小写 `false`**：`-cc` 的 dotted 解析走 `json.loads(value_str)`，
写 `False` 会被当成字符串 `"False"`（`vllm/utils/argparse_utils.py:428-445`）——一个真实存在的坑。）

**关键副作用**：`benchmark_combo_kernel` 属于 `compilation_config` ⇒ 进 `config_hash`
⇒ **AOT key 从 `2b4e7331…` 变成 `ab6111e99163…`**，即"改这个开关 = 换一份编译缓存，
必须重新冷编译一次预热"。所以它不是"零成本 workaround"，而是"换一条编译路径"。
它对**冷编译变体是否与热启动一致**的验证 = W1/W2 两臂（batch3，见 §3.5.7）。

#### 3.7.2 `VLLM_FORCE_AOT_LOAD=1`：暴露了第二个坑（DEBUG 会换 AOT key）

第一次试（`F0forceload`，`VLLM_FORCE_AOT_LOAD=1 VLLM_LOGGING_LEVEL=DEBUG`，缓存根=`one`）**直接崩**：

```
(Worker_TP0) ERROR Traceback (most recent call last):
  … v1/worker/gpu/model_runner.py:949 profile_run → v1/worker/gpu/model_runner.py:836 _dummy_run …
(Worker_TP0) ERROR FileNotFoundError: [Errno 2] No such file or directory:
  '/mnt/game/vllm-cache-11m/one/torch_compile_cache/torch_aot_compile/8fb515037802b667be608e8bdd78a7ddd19aa7c13df9d04f96ce3acc17afc8ee/rank_0_0/model'
```

注意 key 是 **`8fb51503…`**，不是 §3.3 里那个 `2b4e7331…` —— 也就是说这次算出来的 AOT key 不一样。
逐项排查后定位到（S1 vs F0 的 effective config 逐字段 diff，只有这 3 项不同）：

```
- 'size_asserts': False            + 'size_asserts': True
- 'alignment_asserts': False       + 'alignment_asserts': True
- 'scalar_asserts': False          + 'scalar_asserts': True
```

根源在 `vllm/config/compilation.py:941-948`：

```python
if not is_torch_equal_or_newer("2.12.0.dev"):
    enable_asserts = envs.VLLM_LOGGING_LEVEL == "DEBUG"
    for key in ("size_asserts", "alignment_asserts", "scalar_asserts"):
        self.inductor_compile_config.setdefault(key, enable_asserts)
```

⇒ **本树（torch < 2.12）上 `VLLM_LOGGING_LEVEL=DEBUG` 会打开这三个 assert ⇒
`inductor_compile_config` 变 ⇒ `config_hash` 变 ⇒ AOT key 变 ⇒ 复用不了生产缓存**。
后果两条：
① S11l §5.2 建议的"用 `VLLM_LOGS=DEBUG` 起一次看 traced files"**在带缓存的机器上会顺带把 AOT key 换掉**（本步因此改用 §3.2 的直接调用方式拿到 DEBUG 的 `Traced files` 输出）；
② `VLLM_FORCE_AOT_LOAD=1` 的语义是"连**产物不存在**都要抛"（`decorators.py:327-334`：`if envs.VLLM_FORCE_AOT_LOAD: raise e`，
不区分"文件损坏/源码变了"与"这次算出的 key 根本没有产物"），所以一旦 key 变了就是硬崩，
**不能用它做"换个 log level 顺手验证一下"**。
正确做法：`VLLM_FORCE_AOT_LOAD=1` + **INFO**（它本身在 `envs.compile_factors()` 的 `ignored_factors` 里，
`vllm/envs.py:2361`，不会换 key）。

**F0b（`FORCE_AOT_LOAD=1` + INFO，缓存根 `one`）通过** ——

```
$ S11M_FORCE_AOT_LOAD=1 tools/dcu/s11m_probe_fresh.sh F0bforceload /mnt/game/vllm-cache-11m/one
[probe] decision after 105s
125: (Worker_TP0) INFO [default_loader.py:484] Loading weights took 25.31 seconds
149: (Worker_TP1) INFO [decorators.py:321] Directly load AOT compilation from path …/2b4e7331…/rank_1_0/model
151: (Worker_TP0) INFO [decorators.py:321] Directly load AOT compilation from path …/2b4e7331…/rank_0_0/model
152: (Worker_TP0) INFO [monitor.py:53] torch.compile took 19.89 s in total
```

⇒ **加载路径里没有任何被吞掉的异常**（`VLLM_FORCE_AOT_LOAD=1` 会把任何异常直接抛出），
两个 rank 都是干净加载。**"静默回落重编"这条路已被彻底关掉。**

### 3.8 证据归置（全部 md5 校验后搬入 `/mnt/game`，工作区只留脚本）

用 `tools/dcu/s11m_tidy_evidence.sh`（容器内跑：`cp -a` + 全量 `md5sum` 对比 + 打印 `MD5_ALL_MATCH`）
把三个证据目录搬到 `/mnt/game/s11m/`，**校验通过后**才删除工作区副本：

```
$ docker exec z100-s9 bash /src/tools/dcu/s11m_tidy_evidence.sh \
      /src/.tmp-11m /mnt/game/s11m/evidence-11m \
      /src/.tmp-11l /mnt/game/s11m/evidence-11l \
      /src/.tmp-11k /mnt/game/s11m/evidence-11k
[tidy] MD5_ALL_MATCH /src/.tmp-11l files=34 -> /mnt/game/s11m/evidence-11l files=34
[tidy] MD5_ALL_MATCH /src/.tmp-11k files=4  -> /mnt/game/s11m/evidence-11k files=4
[tidy] done rc=0
```

| 内容 | 路径 | 体量 |
|---|---|---|
| 本步全部门禁/引擎日志、批次编排日志、token 比对 JSON、`summary.tsv`、产物 dump、15 路变体矩阵 | `/mnt/game/s11m/evidence-11m/` | 2.3 MB / 35 文件（33 份日志+JSON，另加 §3.5.5 的 `variant-matrix.txt/.json`） |
| S11l 的证据目录（原 `.tmp-11l/`，34 文件：臂日志、flip jsonl 副本、cmp JSON） | `/mnt/game/s11m/evidence-11l/` | 1.4 MB |
| S11k 遗留证据（原 `.tmp-11k/`，4 文件） | `/mnt/game/s11m/evidence-11k/` | 20 KB |
| 容器侧 token 电池原始 jsonl（`flip-*.jsonl`，15 份） | `/mnt/game/s11m/` | — |
| `=1` 测试矩阵逐文件日志（8 个）+ `SUMMARY.txt` | `/mnt/game/s11m/tests-sw1/` | 40 KB |
| 关键 AOT 产物备份（S11l 的 `2b4e7331…`、本步 S1 写的同一 key） | `/mnt/game/s11m/backup-s11l-artifact/`、`/mnt/game/s11m/artifacts/` | 17 MB×4 |
| 本步使用的 4 份独立缓存根（`=1` 热、`=0` 冷/热、flag 冷/热、freshB 探针） | `/mnt/game/vllm-cache-11m/{one,zero,combo,freshA,freshB}` | 各 ~1 GB |

工作区内**只剩脚本**（`tools/dcu/s11m_*.py|*.sh`）+ 本归档；
`.tmp-11m/`、`.tmp-11l/`、`.tmp-11k/` 已删除。`.tmp-p2p/`（12 KB，PCIe ACS apply 现场快照）
是 **Lead 的证据**（硬约束 24/25），本步**未动**。

## 4. 门禁判定

| # | 门禁（任务书原文） | 判定 | 证据 |
|---|---|---|---|
| M1 | 定位 AOT 重载失败的**真机制**（不是"缓存没命中"） | ✅ **通过** | §3.1：字节级定位到"**产物快照（`inspect.getsource`/`linecache` 规范化）vs 磁盘裸读**"的口径不一致；`=1` 的产物 28 条快照里**只有 `gfx906_sdot4.py` 一条 DIFF，且 0 行内容差异**（10185 vs 10184 字节，差的正是尾换行） |
| M2 | 给出**可执行的最小复现** | ✅ **通过** | §3.2：`tools/dcu/s11m_min_repro.py`，**< 30 s、不起 27B**；旧口径 False / 新口径 PASS / 真改代码仍报错 |
| M3 | 按代价从低到高验证 workaround + 根因清楚就落地修复 | ✅ **通过** | §2/§3.3：修复 11 行（`caching.py:608`、`decorators.py:266-291`），S11l 既有产物原样加载成功（`torch.compile` 54.15 s → 19.54 s）；§3.7：`benchmark_combo_kernel=false` 会换 AOT key，且只能把冷/热差异从 **8/8 prompt（318/490）降到 2/8（58/506）**、不能消除 ⇒ **不推荐**；`VLLM_FORCE_AOT_LOAD=1`+INFO 的 F0b **通过**（无静默回落），并暴露"DEBUG 换 AOT key"的坑 |
| **M4** | **同开关 ≥3 次启动的 greedy token ids 完全一致** | 🔶 **有条件通过（必须按新口径读）** | §3.5.3/§3.5.4：**三次热启动（AOT 加载路径）S2=S3=S4 完全一致（0/512 位置不同）** ✅；但 **"填充缓存的那次冷编译"与热启动不同（318/490）** ⇒ 生产必须把冷启动当**预热步骤**（§6.1）；**若把冷启动和热启动混在一起数，则 3 次不一致**。附加发现（§3.5.5/§3.5.6）：变体主要由**产物/cache 的 vintage** 决定，不由开关决定（同开关不同 vintage 422/512；不同开关同 vintage 2/512） |
| M5 | 三相位（uncached/cache-fill/cache-hit）重启耗时实测（`=0`/`=1` 都给） | ✅ **通过** | §3.4：`=1` uncached **1663 s** → cache-hit **1553–1573 s**（`compilation` 95.05 → 19.6–20.0 s）；`=0` uncached **1804 s** → cache-hit **1633 s**（156.48 → 23.15 s）；**重启耗时由两轮 graph capture（23–24 min）决定，修 AOT 省的是 78–90 s（`=1`）/ 142 s（`=0`）** |
| M6 | `=1` 下跑 `test_qwen3_5_gdn.py`（20 passed）并把 7 文件补到 **146** | ✅ **通过** | §3.6：S9 七文件 **146 passed / 0 failed**（39/12/5/5/13/**20**/52），并集 8 文件 149 passed |
| M7 | 边做边写 | ✅ **通过** | 本档从骨架开始，§0.1 随执行更新，§3.1–§3.8 均带原始日志与文件路径 |
| M8 | 证据归置 `/mnt/game`，工作区只留脚本 | ✅ **通过** | §3.8：三个证据目录（35+34+4 文件）`cp -a` + 全量 md5 校验通过后删除工作区副本 |

**总体判读**：
1. **AOT 重载缺陷已根治**（M1/M2/M3）：真机制是校验函数的读法不一致（`inspect.getsource` 的快照 vs
   磁盘裸读），不是缓存策略；修复小、可辩护、**不改变缓存 key**（不会让历史缓存失效）；
   `=1` 从此不再每次重编（`torch.compile` 54.15 → 19.7 s，且不再每次抽一个新编译变体）。
2. **复现性拿到"生产可用"的那一半**（M4）：缓存预热后，`=1` **连续三次独立重启逐 token 完全一致**（3/3），
   `=0` 两次热启动同样一致（2/2）。剩下的那一半是"**填充缓存的那次冷编译 ≠ 它自己的加载**"
   （`=1` 318/490、`=0` 127/512）——这不是"每次启动抽签"，而是"**每个产物 vintage 一个稳定变体**"，
   因此用**部署时预热**即可规避；但必须把它写成硬性步骤。
3. **重启成本的真瓶颈是 CUDA/HIP graph capture（23–24 min）**，不是编译（M5）。修 AOT 的收益是
   **确定性 + 每次省 78–90 s（`=1`）/ 142 s（`=0`）**，不是"把 27 min 变成 3 min"。
4. **必须转达给 Lead 的一条修正**（§3.5.5/§3.5.6）：本树上 token 变体主要由**编译产物/cache 的 vintage**
   决定（同开关不同 vintage 差 422/512），**不由 `VLLM_DCU_USE_INT8_SDOT4` 决定**
   （同会话 `=0` vs `=1` 只差 2/512）。S11l 的"`=1` 改变输出（428/512）"因此**主要是 vintage 差异**；
   引用 S11l 那两条上线注意时必须改口径。

## 5. 未完成项 / 阻塞与原因

### 5.1 未定位：**为什么"编译进程执行的代码"与"它存下来给后续加载的代码"不是一个变体**

现象（§3.5.3）：S1 空缓存冷编译，自己跑出变体 A；同一进程把产物写到 `…/2b4e7331…/rank_{0,1}_0/model`
并把 inductor cache 写到 `…/2b4e7331…/inductor_cache`；
随后 S2/S3/S4 三次独立启动**都** `Directly load AOT compilation from path …/2b4e7331…` +
`Directly load the compiled graph(s) … from the cache`（`ind_hit=1`、`ind_save=0`、无任何 `Compiling a graph …`），
且**三次彼此完全一致（变体 B）**，但 **B ≠ A**。

已经排除的：
* 不是"缓存没命中/又编了一遍"（日志与产物 md5 `911769244fccaac536a3cf0ff699d736` 全程不变）；
* 不是 sdot4 内核数学（S11e/S11k/S11l 共 294/294 逐位相同；且 S1（`INNER=sdot4`）与 arm3（`INNER=triton`）
  逐位相同）；
* 不是采样随机性/前缀缓存（同引擎两遍 S4 vs S4b = 0/512）。

**未定位的是**：加载路径与编译路径之间到底哪一处让"生成的代码/运行配置"不同。可疑点（下一步按序排查）：
1. `InductorAdaptor.load` 会打补丁 `FxGraphCache._get_shape_env → AlwaysHitShapeEnv`
   （`vllm/compilation/compiler_interface.py:661-690`）——加载路径与编译路径的形状/guard 行为不同；
2. piecewise 子图（`graph_index`）在"反序列化后的 `GraphModule`"上重新切分时是否与原切分**逐个**对上
   （`vllm/compilation/backends.py:221-259` 的 `self.cache[(compile_range, graph_index, compiler)]` 映射）；
3. `benchmark_combo_kernel=True` 在**冷编译**时按实测计时挑 combo kernel，而加载路径不会重挑
   ——若第 2 点里有任何一块子图没有命中缓存、走了重新 codegen，就会重新计时挑一次。
   这也解释了 §3.7.1 里那个开关为什么"会换一份缓存"。
**建议**：把它作为独立立项（与 S11l §5.2 的 AOT 项合并），用 `VLLM_FORCE_AOT_LOAD=1`（INFO）
+ 逐 `graph_index` 打印命中/未命中的方式定位；本步不做（每次实验 ~30 min×N）。

### 5.2 已做：`=0` 的对照（Z1 冷 / Z2、Z3 热）——两个结论都拿到了

① `=0` 的三相位耗时见 §3.4（uncached 1804 s → cache-hit 1633 s，编译 156.48 → 23.15/23.84 s）；
② **"冷/热变体差"不是 `=1` 独有的**：`=0` 的 Z1（冷）与其产物加载（Z2/Z3）差 127/512，
而 Z2=Z3=0/512 ⇒ 该现象是这套栈的**通用属性**（compile-execute ≠ load-execute），与 sdot4 开关无关；
③ 顺带得到 §3.5.6 的关键修正：变体跟**产物 vintage** 走，不跟开关走。

### 5.3 已做：`benchmark_combo_kernel=false` 的冷/热对拍（W1/W2，结论见 §3.5.7）

结论：**能减小、不能消除**冷/热差异（8/8 prompt → 2/8；318/490 → 58/506），且会换 AOT key；
**不推荐进生产**。若部署无法预热，可作为"降级"手段，但不能承诺逐 token 一致。

### 5.4 环境/口径限制（诚实登记）

1. **PCIe ACS 在本步电池期间被关闭**（Lead 硬约束 24）：§3.4 的墙钟数字跨了这次改动，
   对外引用前应在 ACS 关闭后重测（token 结论是数值性的，不受影响）。
2. **本步电池期间有另一个 agent 的 `p2p_verify.py`**（Lead 硬约束 25，已由 Lead 记录）：
   实测两轮 capture 总量仍在 1383–1401 s 的正常波动内（±1.5%），未污染 token 结论。
3. **TP=4 未对拍**（生产基线是 TP=4；本步按硬约束用 TP=2 / 卡 0,1）。
4. **8K（`max-model-len 8192`）未重跑**：本步全部用 4096（与 S11l 的 flip 电池同口径）。
5. **logprob 有 ≤5e-5 级抖动**（S2 vs S3 4.8e-5、S3 vs S4 4.8e-5、S2 vs S4 0）：
   token 完全一致，抖动源未定位（不排除 sampler/amax 归约顺序）。

## 6. 对后续步骤的影响

### 6.1 生产上线配置（最终定稿，`27B` W8A8 INT8 / TP=2 / 卡 0,1）

```bash
# ---- 一、预热（部署/升级时做一次，不可省）----
export VLLM_DCU_USE_INT8_SDOT4=1
export VLLM_CACHE_ROOT=/mnt/game/vllm-cache-27b     # 硬约束 17：必须显式、必须落 /mnt/game
export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache
export PYTHONPATH=/mnt/game/triton-gfx906/install:/mnt/game/port/gfx906-vllm-211
# 起一次、等 "Application startup complete"、然后停掉。这一次会编译并写下 AOT 产物（~95 s），
# 它自己那一次服务的文本属于"冷编译变体"，不要把它当生产输出。
# 日志验收：出现 "saved AOT compiled function to …"。

# ---- 二、正式启动（此后每次重启都逐 token 可复现）----
python -c 'import torch; from vllm.entrypoints.cli.main import main; main()' serve \
  /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 \
  --dtype float16 --tensor-parallel-size 2 --gpu-memory-utilization 0.9 \
  --disable-custom-all-reduce --skip-mm-profiling --enable-prefix-caching \
  --mamba-cache-mode align --max-model-len 8192 --max-num-batched-tokens 2048 \
  --served-model-name qwen3.8-27b --host 127.0.0.1 --port <PORT>
  # 不要加 --enforce-eager（硬约束 13）；不要为了"看日志"加 VLLM_LOGGING_LEVEL=DEBUG（§3.7.2）
# 日志验收：两个 rank 各一条 "Directly load AOT compilation from path …"，且
# 无 "Compiling model again due to a load failure"、无 "saved AOT compiled function"。
```

| 项 | 定稿值 | 依据 |
|---|---|---|
| 仓库默认 `VLLM_DCU_USE_INT8_SDOT4` | **保持 `False`（OFF）** | 与 S11l §6.1 一致：`=1` 会改变输出（贴边翻转） |
| 生产 | **显式 `=1`** | S11l 的 3.82× / 快 1.458× |
| 冷启动 | **必须当预热步骤** | 本步 §3.5.3：冷编译变体 ≠ 热启动变体 |
| `benchmark_combo_kernel` | **不改**（保持默认 `True`） | §3.5.7：实测关掉它只能把冷/热差异从 8/8 prompt（318/490）降到 2/8（58/506）、**不能消除**；且它会换 AOT key（`ab6111e9…`）⇒ 多一次冷编译预热，不划算 |
| `VLLM_FORCE_AOT_LOAD` | **不要**在生产开 | §3.7.2：它把"产物不存在"也变成硬崩 |
| `VLLM_LOGGING_LEVEL` | **生产保持 INFO**；DEBUG 仅用于离线诊断（且会换 AOT key） | §3.7.2 |
| `VLLM_CACHE_ROOT` | **固定不变**；严禁在服务生命周期内换目录/删缓存/改 `compilation_config` | §3.5.5/§3.5.6：换 key = 换产物 vintage = **换 token 变体** |
| 文本对比口径 | 只在**同一份缓存**内比，不要跨 vintage 比（否则会把"产物差异"误读成"开关差异"） | §3.5.5/§3.5.6 |
| 启动耗时预期（热缓存） | **~26 min**（其中两轮 graph capture ≈ 23.2 min，权重 ≈ 24 s，编译 ≈ 20 s） | §3.4 |
| 逐 token 一致性口径 | "**同一份预热缓存的多次重启之间**逐 token 一致"；"跨冷编译 / 与 `=0` 路径"不一致 | §3.5 |

### 6.2 代码固化（给 Lead 的两个决定）

1. **本步的两个 Python 改动要进 canonical 副本**（`/mnt/game/port/gfx906-vllm-211/vllm/compilation/{caching,decorators}.py`）：
   纯 Python、无 ABI 影响；沙箱限制下本步只改了工作区 `/src`（= `/home/lyh/z100-vllm`），
   canonical 的同步请 Lead 决定（与 S11l §6.3 的 `gfx906_sdot4.py` 一并处理）。
2. **`gfx906_sdot4.py` 的尾换行**：本步**没有**改它（保持 S11l 的 md5 `59e66964…`）。
   修复在 vLLM 侧，所以"文件缺尾换行"这件事不再影响加载；但若将来要回退本修复，
   必须同时给该文件补尾换行（`printf '\n' >> …`），否则 `=1` 又会每次重编。

### 6.3 后续步骤

| 步骤 | 能否开工 | 说明 |
|---|---|---|
| **S11c**（P1：4B/0.6B 的 TP 缩放曲线） | ✅ **可以开工** | 与本步无依赖（不同模型、不同问题） |
| **S11f**（P6：FP16 模拟 FP8） | ✅ **可以开工** | 与本步无依赖；注意它也会引入新的自定义 op，**新 op 的实现文件若缺尾换行会重蹈本步的坑**（本步修复已从根上挡掉） |
| **S12**（ACS 关闭后的通信/并行度复测） | ✅ **可以开工**（本步交付了它的前置结论） | 硬约束 24 要求 S11m 的电池在 ACS 关闭后重跑；本步的 token 结论不受 ACS 影响，墙钟数字需在 S12 里重测 |
| S11n/S11o（若有） | — | 本步未涉及 |
| 立项（建议） | ⏳ | ① §5.1 的"编译路径 vs 加载路径变体差"；② `=0` 是否同样存在冷热变体差（Z1/Z2 会给初步答案） |

## 7. 复现方式

```bash
# 7.0 通用 env（容器 z100-s9，root；/src = /home/lyh/z100-vllm 只读挂载）
#   . /opt/dtk/env.sh; . /src/tools/dcu/env.sh
#   export PYTHONPATH=/src:/mnt/game/triton-gfx906/install
#   export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache
#   export VLLM_CACHE_ROOT=/mnt/game/vllm-cache-11m/<phase>   # 每个相位一个独立子目录

# 7.1 最小复现（<30 s，无需 27B；旧口径 False / 新口径 PASS）
docker exec z100-s9 bash -lc 'export PYTHONPATH=/src:/mnt/game/triton-gfx906/install; \
  . /opt/dtk/env.sh >/dev/null 2>&1; . /src/tools/dcu/env.sh >/dev/null 2>&1; cd /src; \
  python tools/dcu/s11m_min_repro.py \
    --artifact /mnt/game/vllm-cache-27b/torch_compile_cache/torch_aot_compile/2b4e7331…/rank_0_0/model'

# 7.2 直接解开 AOT 产物看"校验到底比什么"（宿主 python3 即可，不需要 torch）
python3 tools/dcu/s11m_aot_probe.py <artifact> \
  --diff-root /home/lyh/z100-vllm --map "/src=/home/lyh/z100-vllm" --json /tmp/probe.json
python3 tools/dcu/s11m_aot_probe.py <artifact> \
  --dump-module vllm.model_executor.kernels.linear.scaled_mm.gfx906_sdot4 --dump-out /tmp/art.py
cmp /tmp/art.py vllm/model_executor/kernels/linear/scaled_mm/gfx906_sdot4.py

# 7.3 启动/缓存探针（~3-5 min，看到 AOT 决策行就杀）
tools/dcu/s11m_probe_fresh.sh F1fresh /mnt/game/vllm-cache-11m/freshA
S11M_FORCE_AOT_LOAD=1 tools/dcu/s11m_probe_fresh.sh F0bforceload /mnt/game/vllm-cache-11m/one   # 注意：不要加 DEBUG

# 7.4 单次完整启动 + token 电池（~32 min）
tools/dcu/s11m_run_one.sh S4sw1hit 1 /mnt/game/vllm-cache-11m/one 8048 4096 flip,flipb

# 7.5 三个批次（本步实际用的编排）
tools/dcu/s11m_matrix.sh stage1      # =1：fill + 2× hit（stage1 的 3 次）
setsid bash tools/dcu/s11m_batch2.sh > .tmp-11m/batch2.log 2>&1 &   # F0/F2 + S4 + Z1/Z2
setsid bash tools/dcu/s11m_batch3.sh > .tmp-11m/batch3.log 2>&1 &   # F0b + W1/W2（combo=false）

# 7.6 token 对比（纯 stdlib）
python3 tools/dcu/s11m_flip_matrix.py \
  /mnt/game/s11m/flip-S2sw1hit.jsonl /mnt/game/s11m/flip-S3sw1hit.jsonl /mnt/game/s11m/flip-S4sw1hit.jsonl

# 7.7 产物指纹对比（产物字节不可比，见 §3.1 的方法学警示）
tools/dcu/s11m_kernel_fingerprint.sh <artifactA> <artifactB>

# 7.8 测试矩阵（=1，卡 3，容器 z100-serve：独立 cgroup，不与 27B 抢内存）
docker exec -e S11M_DEVICES=3 z100-serve bash /src/tools/dcu/s11m_tests.sh 1 /mnt/game/s11m/tests-sw1
```

## 8. 参考

1. [`z100-11l-sdot4-reacceptance-result.md`](z100-11l-sdot4-reacceptance-result.md)（S11l：§3.2.5 AOT 现象、§5.2 未定位项、§6.1/§6.3 待办）
2. [`z100-11k-sdot4-compile-integration-result.md`](z100-11k-sdot4-compile-integration-result.md)（S11k：`torch.library.custom_op` 编译集成修法）
3. [`z100-09-acceptance-and-bench-result.md`](z100-09-acceptance-and-bench-result.md)（S9：§3.6 七文件 146 passed 口径）
4. `docs/海光Z100-性能优化方案.md` §1.2/§1.3/§7（P5 三相位冷启动）；`docs/z100-todolist.md` 硬约束 1–23
   （本步电池期间新增的 24/25 见该文件：ACS 关闭、卡位纪律）
5. torch `_dynamo/package.py:153-184`（`InlinedSource` / `_get_module_content` -> `inspect.getsource`）、
   CPython `/usr/lib/python3.10/linecache.py:140`（缺尾换行补 `\n`）、
   `vllm/config/compilation.py:941-948`（DEBUG 打开 inductor asserts -> 换 AOT key）
6. 本步证据（md5 校验后落盘）：`/mnt/game/s11m/evidence-11m/`（含 `variant-matrix.txt`/`.json`、
   `summary.tsv`、各 run 引擎日志）、`/mnt/game/s11m/evidence-11l/`、`/mnt/game/s11m/tests-sw1/`、
   `/mnt/game/s11m/artifacts/`、`/mnt/game/vllm-cache-11m/`
