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
| MAX_SEQS / MBT | 4 / 8192 | 与 s33 一致 |

policy 指纹（journalctl 可查）：`FP e9f6b1ff41e682d6` —— 与 s33 完全一致
（MTP 只经 CLI 传参，不动任何 VLLM_* 环境旋钮）。

## 注意

- **DSH/agent 负载影响**：每请求 TTFT +~1.7 s（draft 预填充），生成段 −17% 左右。
  输出越长越划算；短回复请求净收益小。
- **qwen-image21 容器保持停止**：0.0.0.0:8000 端口映射与本服务冲突（同 s33）。
- 排障时若 GPU 有占用但无进程：`pgrep -af "VLLM[:] :Worker"` 找孤儿 worker，
  kill -9 后 hy-smi 归零再操作。
- 验收基线：`runs/specbench/e2e-B-mtp2.json`（贪心流 parity 与基线首 diff @46
  属 batch 形状浮点抖动，verify 本身无损）。
