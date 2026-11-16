# z100 vLLM 生产部署 —— 运维手册

> 服务：`z100-vllm-mtp2`（systemd 系统单元，开机自启，崩溃 15 s 后自动拉起）
> 地址：`0.0.0.0:8000`，鉴权 `/mnt/game/host-serve/api-key`（Bearer token）
> 模型：`/mnt/game/models/Qwen3.8-27B-FP8`，TP=4，`--dtype float16`

## 一、启动链（固化后只有一条）

```
z100-vllm-mtp2.service
  └─ ExecStart: serve_deploy.sh      # env 旋钮唯一真源
       └─ exec: launch_mtp2.sh       # CLI 参数 + DTK/HIP 运行时
            └─ vllm serve ...
```

配置分三层，各管一件事，**不要跨层抄**：

| 文件 | 管什么 | 改动方式 |
|---|---|---|
| `serve_deploy.sh` | 已采纳的 `VLLM_*`/`S59_*`/`KV_DTYPE` 旋钮 | 改默认值或单次 `X=... bash serve_deploy.sh` |
| `launch_mtp2.sh` | `vllm serve` CLI、编译缓存、HCC/DTK 环境 | 一般不动；A/B 用 `COMPILATION_CONFIG`/`SPEC_OFF` |
| systemd 单元 + drop-in | 拉起 serve_deploy + 同值 `Environment=` | 有 root 时删 drop-in（脚本已同值） |

`serve_deploy.sh` 与 drop-in 的取值已逐项对拍一致（143 个策略旋钮 + 命令行逐字相同），
两种启动路径等价。

## 二、常用命令

```bash
bash start.sh              # 启动 + 等 /health + 校验 spec/指纹（~4-8 min）
bash start.sh status       # 单元/端口/健康/.so/指纹 一览
bash start.sh restart      # 重启 + 等健康
bash start.sh stop         # 停止（端口空出，清孤儿 worker）
bash start.sh logs -f      # 跟随日志
bash start.sh config       # 干跑：打印解析后的 env + 命令行（不启动）
```

`stop.sh` 保留为兼容入口（等价 `start.sh stop`）。

## 三、当前采纳栈（2026-11-15）

| 层 | 配置 | 收益 / 说明 |
|---|---|---|
| 投机 | MTP k=2（模型自带 `mtp.*`） | W1 配对 +33.6%/+34.6%；TTFT +1.7 s |
| 图 | `FULL_DECODE_ONLY`，capture `[2,3,6,12]` | S55 采纳；decode TPOT 主收益 |
| KV | `int8_per_token_head` | KV 容量 1.03M→1.97M token；c=4 TPOT −10% |
| prefill GEMM | `VLLM_DCU_PREFILL_INT8_PC=1`，`MIN_M=1024` | S99/S100a：8k −16.9%、32k −19.4%、131k −11.8% |
| attention | HIP inner（decode S53 + prefill S58），3D 门 4096 | 131k kernel 1491→1126 ms |
| 前缀缓存 | `--prefix-match-unit 16` + mamba checkpoint + retention 1616 | 131k 共享前缀+8k 尾 332.7→44.4 s |
| GEMV/AR | S34 栈 + S41 wholea + S40 lm_head + S44 混合融合 | S44 配对 +4.25%/+2.81% |
| 调度 | Mamba partial-tail stop = 1（S100b B2 否决） | 保住 n-16 checkpoint，多轮 4.5x |

策略指纹：`FP 71c3af949b564865`（`journalctl -u z100-vllm-mtp2 | grep "policy: FP"`）。

## 四、生产 `.so` 与回滚点

`vllm/_rocm_C.abi3.so` 是指向内核变体的符号链接：

| 用途 | 路径 | md5 |
|---|---|---|
| **生产（S99 PC）** | `.s99/arms/build_pc/_rocm_C.abi3.so` | `3ccebe80` |
| 回滚 1（S97 phsr） | `.s97/arms/build_phsr/_rocm_C.abi3.so` | `adba0acb` |
| 回滚 2（S93 dot2） | `.s93/arms/_rocm_C_dot2.abi3.so` | `7d2ed2a2` |
| 回滚 3（S91 kvint8） | `.s91/arms/_rocm_C_kvint8fix.abi3.so` | `5af8b848` |

切换内核（不改代码）：

```bash
ln -sfn /home/lyh/z100-vllm/.s97/arms/build_phsr/_rocm_C.abi3.so vllm/_rocm_C.abi3.so
bash start.sh restart
```

**重编或新增内核变体**（宿主 relink、编译门禁、符号校验、换装验收）的完整步骤见
[`BUILD.md`](BUILD.md) §3–§6；本文只负责「用哪个产物」和「怎么切」。

被否决的中间变体已于 2026-11-15 清理（释放 ~2.7 GB），清单与理由见
`docs/result/z100-cleanup-20261115.md`。

## 五、禁令（违反会乱码 / 挂死 / 分叉）

1. **`VLLM_DCU_CUSTOM_AR_FUSION_MLP=1`**（两出口全融合）：piecewise capture 下输出乱码
   （S38 损坏复现），脚本已硬锁 0。
2. **`VLLM_DCU_W8A16_SILU_MUL=1`**：S49 已否决，A 流量翻倍，down GEMV 534→319 GB/s。
3. **`VLLM_QWEN35_EMPTY_CORE_ATTN_OUT=1`**：`core_attn_out` 的 zeros 承重，
   greedy 流会分叉（S45 审计）。
4. **`VLLM_USE_BREAKABLE_CUDAGRAPH=1`**：数值错误 + VM fault，已判死。
5. **生产/ABAB 一律 `FUSION_DEBUG=0`**：debug 分支含数据相关集体操作，会挂死。
6. **不要用 `logprobs` 探测本栈**：会偶发 invalid token id 打死 APIServer（2026-11-15 两次）。
7. **`/stop_profile` 后不要立刻发请求**：本栈出现过 `running=1 / 0 tok/s` 假死。

## 六、排障

- 冷启慢：首次换指纹会重建编译缓存（225 s 属正常）。
- 端口占用：`qwen-image21` 容器必须保持停止（映射 0.0.0.0:8000）。
- GPU 有占用但无进程：`pgrep -af "VLLM[:]Worker"` 找孤儿 worker，kill -9 后
  `hy-smi` 归零再操作。
- 架构未就绪：启动会以 `VLLM_DCU_GCN_ARCH is not gfx906` 拒绝启动（S89 事故加固），
  等 hymgr 驱动探测完成或显式 `VLLM_DCU_GCN_ARCH=gfx906`。
- DSH/agent 负载影响：每请求 TTFT +~1.7 s（draft 预填），长输出才划算。

## 七、变更历史（细节见对应文档）

| 日期 | 采纳项 | 证据 |
|---|---|---|
| 2026-10-19 ~ 10-27 | S23/S24/S29/S30c/S33/S34 基础栈 | `docs/result/z100-*.md` |
| 2026-11-01 | S44 AR 融合、S45 silu 折入（后回退）、S46 MTP 复役 | `.s44/`、`.s46/mtp-adopt.md` |
| 2026-11-05 | S55 FULL decode graph | drop-in `full-graph.conf` |
| 2026-11-12 | S59 前缀缓存、S89 G2 MS16 | `.s89/` |
| 2026-11-14 | S97 phsr 内核 | `.s97/deploy-adopt.md` |
| 2026-11-15 | S99 int8 prefill GEMM + S100a MIN_M=1024 | `.s99/deploy-adopt.md`、`runs/s100a-minm1024/` |
| 2026-11-15 | S100b B2 否决（tail stop 保持 1） | `runs/s100b-tailoff/RESULT.md` |
