# z100 vLLM 生产部署（S34 MTP k=2 臂）—— 运维手册

> 部署时间：2026-10-27。配置 = s33 全套（s30c ctrim M1-GDN + WP2 push 模式
> one-shot AR + piecewise graph [1,2,4]）**加一项原生 MTP k=2 投机解码**。
> 目标模型自带 `mtp.*` 权重（`mtp_num_hidden_layers: 1`），无需单独 draft 模型。
>
> 同口径实测（`runs/specbench/REPORT.md`，7469-token probe prompt，单流）：
>
> | 口径 | 无投机（s33） | MTP k=2（本臂） |
> |---|---|---|
> | 96 tok 贪心 | 36.9 t/s | **51.6 t/s（+40%）** |
> | 512 tok temp0.7+rp1.05 | 34.7 t/s | **40.6 t/s（+17%）** |
> | warm TTFT | 0.72 s | 2.40 s（+1.7 s） |
>
> 接收长度：贪心 ~2.1–2.6，采样 ~1.9–2.3。注意 s28d 的 accept "重启彩票"
> （±6–10%）：跨重启 t/s 有波动属正常。

## 服务

- systemd 系统单元：`z100-vllm-mtp2.service`（**开机自启**，崩溃 15 s 后自动拉起）
- 与 `z100-vllm-s33.service` **互斥**（双向 `Conflicts=`）：启动任一方自动停另一方
- 启动链：`z100-vllm-mtp2.service` → `.s34/serve_deploy.sh` → `.s34/launch_mtp2.sh`
  （launcher 洍生于 `.s33/launch_ctrim_ar.sh`，仅注入 `--speculative-config`）
- 端口/监听：**0.0.0.0:8000**；鉴权 = `/mnt/game/host-serve/api-key`（Bearer token）
- 当前开机自启：`z100-vllm-mtp2` enabled，`z100-vllm-s33` disabled

## 常用命令

```bash
bash .s34/start.sh        # 启动 + 等 /health + 校验 spec/policy 行（~6-8 min）
bash .s34/stop.sh         # 停止（端口空出，无服务）
bash .s34/rollback.sh     # 一步回滚到 s33 无投机臂并改回开机自启
systemctl status z100-vllm-mtp2
journalctl -u z100-vllm-mtp2 -f
```

## 与 s33 的差异（只此一项）

`launch_mtp2.sh` 在 exec 的 vllm serve 命令里加了：
`--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`

覆盖方式（改 spec 不用改文件）：
- `SPEC_SET='{"method":"mtp","num_speculative_tokens":1}' bash .s34/launch_mtp2.sh`
- `SPEC_OFF=1` = 无投机（对齐 s33 行为）

## 环境变量（同 s33，勿动）

| env | 值 | 作用 |
|---|---|---|
| VLLM_DCU_CUSTOM_AR | 1 | custom AR（PCIe-only 4-rank） |
| VLLM_DCU_CUSTOM_AR_ALLOC | finegrained | IPC buffer 细粒度分配（gfx906 必需） |
| VLLM_CUSTOM_ALLREDUCE_ALGO | oneshot | push 模式 one-shot AR |
| **VLLM_DCU_FP8_GEMV_STREAM_ROWS / _M1** | **4 / 1** | **S34-WP3 GEMV stack（−2.0 ms/步，已采纳）** |
| **VLLM_DCU_CUSTOM_AR_MAX_KIB** | **128** | **S34-WP2 AR size 门（c4 −4.6 ms/步，已采纳）** |
| MAX_SEQS / MBT | 4 / 8192 | 与 s33 一致 |

> 以上 S34 增量已写入 `serve_deploy.sh`（systemd 链）与 `.s34/launch_arm.py`
> （快照复现链）两处，两条启动路径配置一致。终局报告：
> `docs/result/z100-s34-final.md`；长上下文曲线与 S35 移交项：
> `docs/result/z100-s34-wp6-longctx.md`。

## S45-N2a 采纳:SiluAndMul 折进 W8A16 down_proj GEMV(2026-11-01)

| env | 值 | 作用 |
|---|---|---|
| **VLLM_DCU_W8A16_SILU_MUL** | **1** | **S45-N2a:MLP 激活折入 wholea_m1 整行 staging(M=1 decode;位同,已采纳)** |

- 证据:`.s45/n2a_parity.py` torch.equal max\|d\|=0;board first_div=None;
  `mul_silu` 内核出账(−47 发/step);`.s45/s45-adopt.md`。
- **`.so` 变更**:`_rocm_C` md5 `f72bb39d` → `6dee7fad`(smul 变体;原路径
  逐位不变)。回滚 = 注释本行 + restart + 恢复
  `.r3/so-backup/_rocm_C.s44-before.so`。
- **禁令(S45 审计)**:勿设 `VLLM_QWEN35_EMPTY_CORE_ATTN_OUT=1`——
  core_attn_out 的 zeros 承重(empty 使贪心流分叉,op 有未写元素被下游读)。
- S45 空隙归因:5.38 ms/step GPU 空闲在 piecewise 图段边界(92%),
  是下一轮(host 侧)的主池子。

policy 指纹（journalctl 可查）：`FP e9f6b1ff41e682d6` —— 与 s33 完全一致
（MTP 只经 CLI 传参，不动任何 VLLM_* 环境旋钮）。

## S44-A4 采纳：attn 出口 push-AR+rmsnorm 融合（2026-11-01）

| env | 值 | 作用 |
|---|---|---|
| **VLLM_DCU_CUSTOM_AR_FUSION** | **1** | **S44 attn 出口 AR+add-rmsnorm 融合（混合臂，已采纳）** |
| **VLLM_DCU_CUSTOM_AR_FUSION_MLP** | **0** | **MLP 出口不融合（两出口全融合 = 乱码，禁开）** |

- 证据：B/A/B/A e2e W1 41.73/40.70 vs 40.03/39.59 t/s（配对 +4.25%/+2.81%）；
  norm 类目账 1.635 → 0.392 ms/step；`runs/s44/e2e-s44-ar-*.json`。
- **禁令：`VLLM_DCU_CUSTOM_AR_FUSION_MLP=1`（全融合）在 piecewise capture 下
  输出乱码（S38 原始损坏 S44 复现，`docs/perf/data/s44-a0-debug.txt`），
  永不开启，除非先修复 S38 §2 根因。**
- 另：`VLLM_DCU_CUSTOM_AR_FUSION_DEBUG=1` 的对拍分支含逐 rank 数据相关的
  额外集体操作，warmup 期可致 AR flag 序列跨 rank 分叉 → 服务挂死
  （S44 三次复现）。**生产与 ABAB 一律 DEBUG=0**；修复见 `.s44/ar-adopt.md`。
- 回滚：注释上两行 env + `sudo systemctl restart z100-vllm-mtp2`（env 去 0
  = 逐位恢复 S43 态；`.so` 未动，`_C_stable_libtorch` md5 `54ccf2e2`）。

## S46 复役：MTP k=2 于 S44/S45 采纳栈（2026-11-01）

| 项 | 值 |
|---|---|
| **SPEC_OFF** | **（已注释）** 移除后 `launch_mtp2.sh` 默认 `--speculative-config {"method":"mtp","num_speculative_tokens":2}` |

- 证据：同日 ABAB W1 med B 51.35/52.72 vs A 38.44/39.16 t/s（配对
  +33.6%/+34.6%，门 +29%）；W2 B 池中位 42.4 ≥ 施工图绝对下限 41.3
  （严格配对 +12.4%，B2 配对 +18.1%；当日 A 臂 W2 偏强属分母效应）；
  board `first_div=None`（verify 无损先例）；accept/drafts 代理
  2.46 true tok/step（≥2.0）。`runs/s46/e2e-s46-mtp-*.json`、
  `runs/s44/board-s46-*.json`、`docs/perf/data/s46-*.txt`、`.s46/mtp-adopt.md`。
- 长 ctx（同日同口径，软门仅记录）：50k +179%（24.8 vs 8.9 tps）；
  128k −16%（15.2 vs 18.0，S37 +28% 未在本栈复现——dynk k=1 区间经济学，
  S47 候选）；128k TTFT +6%（370 vs 347 s）。
- 运营：MTP 臂跑 `/start_profile` 沿用 S38/S41 预案（stop 超时 → health
  000 → restart 恢复；本日 1/1 见 `.s46/mtp-adopt.md` §异常，干净路径不触发）。
- 回滚：取消注释上行 + `sudo systemctl restart z100-vllm-mtp2`
  （env 级，零 `.so`/代码变更）。

## 注意

- **DSH/agent 负载影响**：每请求 TTFT +~1.7 s（draft 预填充），生成段 −17% 左右。
  输出越长越划算；短回复请求净收益小。
- **qwen-image21 容器保持停止**：0.0.0.0:8000 端口映射与本服务冲突（同 s33）。
- 排障时若 GPU 有占用但无进程：`pgrep -af "VLLM[:] :Worker"` 找孤儿 worker，
  kill -9 后 hy-smi 归零再操作。
- 验收基线：`runs/specbench/e2e-B-mtp2.json`（贪心流 parity 与基线首 diff @46
  属 batch 形状浮点抖动，verify 本身无损）。

## S39-C1 容量口径(2026-10-31,4 并发验证)

- 单流(M=1)现役:贪心 ~38–39 t/s(W1 96 tok,当日带);TTFT ~0.71 s。
- 4 路并发(7469-probe,96 tok × 4 流):**聚合 ~76 t/s(×1.95)**,单流
  16–23 t/s(ITL ~2× solo),TTFT 0.8–2.7 s。
- 机理:batch M=4 下 stream-GEMV 实测 1.45–1.75× M=1 成本(内核族无 M-flat
  替代:tile GEMM cfg-35 在 M=4 慢 12×),×1.95 为物理上限而非缺陷。
- 结论:4 并发是吞吐口径(≈2 倍产能);单流时延口径仅 M=1 独占(现役路径
  不受任何影响)。数据:`runs/s39/e2e-s39-c1-*.json`;判定:`.s39/c1-close.md`。
