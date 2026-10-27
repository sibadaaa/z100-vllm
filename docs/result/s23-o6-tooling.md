# S23 O6 — 端到端 A/B / parity / 回滚工具（工具说明与自检记录）

> 归属：perf-verifier（O6 / O7）。规格来源：`docs/design/z100-fp8-emu-optimization-plan.md` §3 O6、§5、§6。
> 协议：`.s23/S23_EXEC_PROTOCOL.md`。本文只描述**工具与协议**；各窗口的实测数据在
> `.s23/w1/O6/<window>/window-summary.md`，O6/O7 结论由 lead 汇总进最终结果文档。

## 1. 为什么不是改 `s20_ab.sh`

`s20_ab.sh` 是给**旧栈**写的：docker 容器里的进程、`.so` 换装路径
`/mnt/game/port/gfx906-vllm-211/vllm/`、没有 API key、没有 python-only 臂、没有 pre-flight 门禁。
S23 的生产已迁到 bare-metal 的 **systemd user 单元**
`z100-vllm-host.service` → `tools/dcu/serve_host_dflash2.sh`，`.so` 经由
`vllm/_rocm_C.abi3.so -> /mnt/game/port/build-sync/temp/_rocm_C.abi3.so`。

因此 `s23_ab.sh` 重写了驱动层，但**复用了 S20 的测量代码**：`s20_single_probe.py` /
`s20_conc_sweep.py` 通过 `s23_probe.py` / `s23_sweep.py` 原样调用，只补一个 Authorization 头，
口径（引擎 `iteration_tokens_total` 步进、fwd t/s、accept）与 S20 报告完全一致，不存在两份实现漂移。

## 2. 组件

| 文件 | 作用 |
|---|---|
| `tools/dcu/s23_ab.sh` | 窗口驱动：抢 `gpu.lock` → 逐臂 pre-flight → 重建 unit + 臂 env → ready/smoke/身份校验 → 4 次 c=1 probe → parity → 可选 c=4/c=16 sweep → 窗口结束恢复默认配置 |
| `tools/dcu/s23_arms/<arm>.env` | 臂描述文件：`ARM_DESC` / `ARM_ENV`（全部 `VLLM_DCU_*` 开关值）/ `ARM_SO`（可选换 `.so`）/ `ARM_LAUNCHER`（可选换启动脚本，O7 用）/ `ARM_PREFLIGHT_EXPECT` |
| `tools/dcu/s23_preflight.py` | 容器内 import smoke：`gfx906_fp8`、`torch.ops.vllm.fp8_emu_mm_block`、三个开关是否存在于 `vllm.envs`、`--expect` 值是否按臂解析 |
| `tools/dcu/s23_descriptor.py` | 臂身份记录 JSON：git rev、脏文件 md5、`.so` md5、启动脚本、**live 进程 `/proc/<pid>/environ` 里的全量 `VLLM_*`**，并对齐臂请求值（不一致即判臂无效） |
| `tools/dcu/s23_probe.py` / `s23_sweep.py` | S20 probe 的鉴权包装（service `/v1/*` 需要 key） |
| `tools/dcu/s23_parity.py` | greedy 8 prompt 逐 token parity：`--mode capture` 记基线，`--mode check` 比对；分歧报首个 token 下标并可选重试一次识别臂内不确定性 |
| `tools/dcu/s23_smoke.py` | 每次重启后的真实 chat smoke（记录 usage）+ 固定 prompt 完成串（回滚判据） |
| `tools/dcu/s23_report.py` | 只在本窗口内比较：p50 / 均值 / 输出 t/s / accept / parity / sweep 表，输出 `window-summary.{json,md}` |
| `tools/dcu/s23_launch_piecewise.sh` | O7 专用启动脚本（与生产脚本的差异只有两处 graph-mode flag，见文件头） |

## 3. 一次窗口里发生了什么

1. **抢锁**：`flock /home/lyh/z100-vllm/.s23/gpu.lock`（容器内是 `/src/.s23/gpu.lock`，同一 inode），
   窗口期间一直持有，别人的 kernel probe 会被阻塞而不是污染数字。
2. **每次重启前的门禁**（任一失败：不重启，臂标 `INVALID`）
   - 最近 90 s 内有 `.py` 被改 → 等 20 s 再看一次，仍在改则中止（共享工作树下"队友正在编辑"）；
   - `git status --porcelain -uall` 里每个改动的 `.py` 跑 `py_compile`；
   - 容器内 import smoke（带本臂开关值，验证开关名在 `vllm.envs` 里解析到期望值）。
3. **同窗口一致性门禁**：本窗口第一臂记下 `python 树指纹` 与 `.so md5`；后续臂二者任一变化即中止该臂
   （`csrc/*.cu` 文本变动本身不中止——只有 rebuild 把新代码写进 `.so` 才影响被测对象）。
4. **臂启动**：unit 是 transient 的（`systemd-run --user --unit=z100-vllm-host --collect`），
   drop-in 不可用，所以每臂 `stop` + 用 `systemd-run` 按 `tools/dcu/z100-vllm-host.service` 的属性重建，
   开关经 `systemctl --user set-environment` 注入（已实测 transient 单元继承 manager env）。
5. **身份校验**：读 live 引擎进程的 `/proc/<pid>/environ`；臂请求的开关值没到进程里就判无效
   （"开关设了但没生效"这类假阴性在 S20 出过）。
6. **测量**：真实 chat smoke（记录 usage）→ 4 次 c=1 probe（512 prompt / 128 输出，引擎步进口径）
   → parity（首臂 capture，其余 check）→ 可选 `c=4,16` sweep。
7. **窗口收尾**：`unset-environment` 三个开关 → 若 `.so` 与基线 md5 不同则从
   `/mnt/game/perf1/_rocm_C.<md5>.so` 回滚 → 用生产启动脚本重建 unit → ready + 固定 prompt smoke
   与窗口 control 的 smoke 比对。**任何窗口结束都必须停在默认配置且健康**（trap 保证异常路径也走这里）。

## 4. 自检记录（2026-10-19 10:07–10:12 UTC，未重启生产）

| 项 | 结果 |
|---|---|
| 鉴权 | service `/v1/*` 需 `Authorization: Bearer <api-key>`；`/metrics`、`/health` 开放 |
| parity 决定性 | 同一进程内 capture→check，8/8 prompt `token_ids_equal=True`（首臂 capture 已记录 8/54/412/465/1678/1643/3254/16 prompt tokens，各 64 生成 token） |
| smoke | `completion=' Paris.\nThe capital of Germany is Berlin.\nThe'`，chat usage `completion_tokens=22` |
| c=1 probe | 4 次 probe：step p50 **144.2 ms**、均值 144.3、输出 **26.30 t/s**、accept 7.00、tok/step 7.53 |
| 参考口径 | 上表进程是 2026-10-18 启动的，**不是任何窗口的 control**；它说明同日同机已比 plan 里 193.8 ms 的窗口 A 快 25%，跨窗口比较确实不可用 |

原始输出：`.s23/w1/O6/_selftest/`。

## 5. 已修的一个真坑

第一版 `s23_common.install_auth()` 只 patch `urllib.request.urlopen`，而 `post()` 直接调了模块导入时
捕获的原始 `urlopen`，于是 smoke/parity 全部 401 静默失败。现在 `authed_request()` 显式加头，
`install_auth()` 仅保留给 S20 probe 复用。

## 6. 回滚方法（一句话）

```bash
docker exec z100-regress bash -lc "cp -p /mnt/game/perf1/_rocm_C.f205be8f43d3ffb1cfb3c6f093bfab4b.so \
  /mnt/game/port/build-sync/temp/_rocm_C.abi3.so && md5sum /mnt/game/port/build-sync/temp/_rocm_C.abi3.so"
bash tools/dcu/s23_ab.sh --no-restore <window> control   # 或直接 systemctl --user restart
```
判据：md5 == `f205be8f43d3ffb1cfb3c6f093bfab4b` 且 `smoke.json` 的固定 prompt 完成串与基线一致。
