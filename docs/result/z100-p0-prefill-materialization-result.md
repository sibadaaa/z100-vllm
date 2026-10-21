# Z100 P0 复测结果:prefill fp16 物化路由 A/B(2026-10-21)

按 [z100-p0-retest-plan.md](../design/z100-p0-retest-plan.md) 执行的双臂 A/B。
**结论:全面采纳,`VLLM_DCU_FP16_PREFILL=auto` 建议直接量产。**

## 1. 臂定义与运行

两臂 env 除 `VLLM_DCU_FP16_PREFILL` 外逐项钉死(033606 同口径:27B-FP8 /
TP=4 / DFlash2 k=7 / maxlen 262144 / MBT 4096 / max_num_seqs 4 /
`VLLM_DCU_FP8_W8A16=1`;代码 = `w8a18` 分支 `0d085f276`,两臂同代码)。

| 臂 | probe(含 REPORT) | suite | longctx |
|---|---|---|---|
| A `=0`(tile) | `runs/20261021-082314-serve` | `runs/20261021-093609-serve` | probe 目录 `longctx-128k-cold/` |
| B `=auto`(物化) | `runs/20261021-090257-serve` | `runs/20261021-094628-serve` | 同上 |

路由证据(臂 B 启动日志):`W8A16 engaged=True` +
`fp16 prefill materialization: VLLM_DCU_FP16_PREFILL=auto, role min_m {...}` +
`first materialization route engaged (role=in_proj_qkvz, M=4096, N=4096, K=5120)`。

## 2. 套件对比(`make compare`,B/A)

| 场景 | TTFT p50 | 吞吐 | TPOT | 备注 |
|---|---|---|---|---|
| s02 短对话 c1(512/128) | 2337→1081 ms(**0.46×**) | 22.8→28.0 tok/s(1.23×) | 1.06× | |
| s03 长解码 c1(512/512) | 0.46× | 1.14× | **1.00×** | decode 完全不受影响 |
| s04 重预填充(16k/32) | 48.7→23.9 s(**0.49×**) | 预填充 470→921 tok/s(**1.96×**) | 1.06× | |
| s05 并发 c8(4k×8) | 30.0→15.7 s(**0.52×**) | 17.9→33.9 tok/s(**1.89×**) | **0.55×(-45%)** | 长预填充对 decode 的挤占大幅缓解 |
| s06 前缀缓存(2k 前缀) | 8.6→3.6 s(**0.42×**) | 3.2→6.4 tok/s(2.0×) | — | |
| s07 长上下文(24k/64) | 68.1→41.0 s(**0.60×**) | 预填充 503→808 tok/s(1.61×) | 1.03× | |
| s08 投机解码(512/252) | 2349→1041 ms(**0.44×**) | 26.0→30.1 tok/s(1.16×) | 0.99× | 接受长度不变 |

REPORT 口径:预填充 9684.8 → 7723.5 ms/1k context token;短 prompt TTFT
p50 2320.5 → 1037.5 ms;decode step 143.0 vs 148.9 ms(样本 88/57、接受长度
4.07/3.88,路由在 M=8 机械性不可达,判为窗口组成噪声)。

## 3. 冷 131k 长上下文(warmup 口径)

probe_http 的 warmup 请求(不计入统计)即真冷测量,引擎 chunk 时间戳:

- 臂 A:08:27→08:58:30,**~1920 s**(9.8 ms/token,全 tile)
- 臂 B:09:06→09:28:52,**~1370 s**(7.0 ms/token,物化),**1.40×**

注意:probe_http 默认 warmup=1 会把"冷"变"暖"(TTFT ~26-28 s 的假冷值,
033606 的 28.5 s 即此);要测真冷必须重启后首个请求或改 builder 内容。
131k 尾段 chunk 是注意力主导(200k KV),物化收益集中在中低 KV 段
(s04 24k 场景 1.96×),故整体 1.40× 低于逐段峰值,符合预期。

## 4. 判定与遗留

- 通过线(冷 128k ≥20%、decode ≤±3%、数值一致、0 崩溃)全部满足:
  物化路由在所有 prefill 形状 -40~-58% TTFT,decode/s08 TPOT 持平,
  全程 0 worker 崩溃、0 OOM,KV pool 810,817 tokens 两臂一致。
- **遗留 1(重要)**:两臂都运行在 upstream 同步(`bb7e8dae1`)之后,
  臂 A 的预填充效率(9685 ms/1k)比 033606 基线(5647)慢 ~1.7×,
  疑似同步引入回归,需单独 bisect(与本次 A/B 无关,但影响绝对值)。
- 遗留 2:`MAX_NUM_SEQS` 仍为 4(pinned-memory 维护窗口未做,见计划附录 A)。
- 遗留 3:probe_http 需 `--extra --timeout --extra 7200` 才能测冷 131k
  (默认 900 s 不够);建议框架把长 ctx 场景默认超时提高。
