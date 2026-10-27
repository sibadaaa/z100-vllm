# S34 施工图：按 WP 分步的实施任务书（2026-10-27）

> 设计依据：`docs/design/z100/s34/z100-s34-design.md`（先读，含 16:05 误判的纠偏）。
> 基线快照：`.s34/base_cmdline.txt` / `.s34/base_environ.txt` /
> `.s34/so_md5_baseline.txt`（05:44 生产 MTP 臂逐字固化）。
> 臂操作统一走 `.s34/restart_arm.sh <tag> [KEY=VALUE | -KEY ...]`；
> 探针统一走 `tools/dcu/s20_single_probe.py`（c=1）与 `.s34/probe_c4.py`（c=4）。
> 结果落 `.s34/` 与 `docs/result/z100-s34-*.md`；每步命令必须可复现。

## 通用纪律

1. 重启前 `restart_arm.sh` 自查在飞请求（running/waiting ≠ 0 即拒止）。
2. 孤儿处置：逐 pid `kill` → `kill -9 "VLLM[:]::Worker"` → hy-smi VRAM 归零
   才起服（S33 教训，勿 `pkill -f VLLM::Worker` 裸匹配）。
3. 起服后必查：`/health` 200 → 新主进程 environ 的 AR/knob 行 → serve 日志
   `policy:` 指纹行。
4. 探针顺序：先 7.5k 协议（7469 doc / 96 tok / 3 reps，与 S29–S33 同口径），
   再 60k 生产口径（`.s34/longctx_corpus.md` truncate 60000 / 256 tok / 3 reps），
   WP2 加 c=4。
5. 结果 json 一律落 `.s34/`，命名 `s20_s34-<wp>-<tag>_<ts>.json`。

## WP1：MTP 臂 AR A/B

```bash
# 腿 A（AR on）已测：
#   .s34/s20_s34-wp1-base-r1_*.json        7.5k，step p50 49.44 ms
#   .s34/s20_s34-wp1-base-60k_*.json       60k，后台任务 $JOB_ID
# 腿 B（AR off）：
bash .s34/restart_arm.sh s34-wp1-aroff \
  -VLLM_DCU_CUSTOM_AR -VLLM_CUSTOM_ALLREDUCE_ALGO -VLLM_DCU_CUSTOM_AR_ALLOC
# 7.5k + 60k 探针（同参数，tag=s34-wp1-aroff）
# 恢复（WP3 直接叠加，见下）
```

判据：step p50 差 ≥1.5 ms（7.5k）与 S33 方向一致；60k 同向。
流门：MTP accept-lottery，跨臂流位同不作门；in-session 3 rep 位同 + 0 错误。
文档：`docs/result/z100-s34-wp1-ar-ab.md`（含 16:05 误判纠偏说明）。

## WP3：GEMV stack（AR on + 两 env）

```bash
bash .s34/restart_arm.sh s34-wp3-ar-gemv \
  VLLM_DCU_FP8_GEMV_STREAM_ROWS=4 VLLM_DCU_FP8_GEMV_M1=1
# 7.5k + 60k 探针（tag=s34-wp3-*）
PYTHONPATH=tools/dcu VLLM_PROBE_BASE=http://127.0.0.1:8000 \
  .venv/bin/python .s33/capture.py --tag s34-wp3 --out .s34   # 与 WP1-A 臂 capture 比对 token_ids
```

判据：7.5k step p50 −1.0 ms 级；capture token_ids 与 A 臂位同
（GEMV 位同 + draft 塔 bf16 路径不受影响）；0 错误。
文档：`docs/result/z100-s34-wp3-gemv-stack.md`。

## WP2：push AR size 门 env 化（64 → 可调，默认 64 = 零变更）

1. `vllm/envs.py` DCU 段新增 `VLLM_DCU_CUSTOM_AR_MAX_KIB: int = 64`。
2. `vllm/distributed/device_communicators/custom_all_reduce.py` 的
   `if inp_size > 64 * 1024: return False` 改为读该 knob（×1024）。
   纯 Python；kernel region chunk = 2 MiB ≥ 256 KiB，无 csrc 重编译。
3. 微测（复用 `tools/dcu/car_bench.py`，torchrun 4 rank）：
   载荷 {10, 30, 64, 128, 256} KiB × {nccl, push} × 100 样本，fp64 误差 + NaN 表。
   NaN 或误差劣化 → 停，回默认 64。
4. 服务级：`VLLM_DCU_CUSTOM_AR_MAX_KIB=256` 臂上跑 `.s34/probe_c4.py`
   （verify 120 KiB 实走 push）→ c=4 step period 应低于 64 KiB 臂 ≥0.5 ms/step。
5. 30 min 长跑（复用 S33 长跑法，0 fail 门）。
6. 产物：`.s34/wp2_car_bench_*.json`、`c4_*.json`、
   `docs/result/z100-s34-wp2-size-gate.md`；git diff 入台账。

## WP6：长上下文口径（最终胜出臂上）

```bash
for CTX in 32768 65536 131072; do
  PYTHONPATH=tools/dcu .venv/bin/python -c "...s20... --prompt-file .s34/longctx_corpus.md \
    --truncate-tokens $CTX --max-tokens 256 --repeat 3 --tag s34-wp6-c1-$CTX --out .s34"
done
for CTX in 32768 131072; do
  PYTHONPATH=tools/dcu .venv/bin/python .s34/probe_c4.py s34-wp6-c4-$CTX --ctx $CTX --max-tokens 128
done
```

注意：每个新长度首跑付冷 prefill（60k ≈ 数分钟，128k 更长），探针本身
顺序执行即可；分析维度：step 周期随 ctx 的增量（attention 3D kernel、
块表 host、AR 门命中）、ms/tok 曲线、warm TTFT 2.4 s 异常定位、
KV-fp8 / lm_head-fp8 在长 ctx 的重估表。
文档：`docs/result/z100-s34-wp6-longctx.md`。

## WP4 / WP5：设计 + 施工图（本轮交付文档，实施留裁决）

- `docs/design/z100/s34/z100-s34-wp4-ar-fusion.md`：push 核 add-rmsnorm epilogue 草图、
  集成点（`cross_device_reduce_push` 出口 + `vllm::all_reduce` 调用侧）、
  位同风险（归约序不变；norm fp32 中间量）、门（per-op −20%、e2e −0.8 ms）、
  回滚（env 不设）。
- `docs/design/z100/s34/z100-s34-wp5-splitk-gemv.md`：split-K 触发（qkv 402.5 / o 361.3
  < 450）、K 分块与分片累加序的位同策略、零翻转硬门方案、预期 −1 ms 级。

## 收尾

1. 胜出臂固化为生产：无 override 或按 WP2/WP3 结果带 env，重启并更新
   `.s33/DEPLOY.md`（或写 `.s34/DEPLOY.md`）。
2. `docs/result/z100-s34-final.md`：WP 状态表 + 分账 + md5 台账 + 回滚序。
3. 约束外路线保持 S32 账本引用，不施工。
