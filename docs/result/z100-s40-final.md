# S40 终局报告:GEMV 读上限逼近(pf[4] + fp16 lm_head stream 化 + A-slice 双缓冲)

> 2026-10-31。输入:`docs/design/z100/s40/z100-s40-design.md` /
> `z100-s40-construction.md`;基线:S39 终局(无 MTP 臂,systemd
> `z100-vllm-mtp2`,`SPEC_OFF=1`,FP `2f3793deb97e138f`,`.so` md5
> `ca05f746`/`54ccf2e2`)。**全部判定台账:`.s40/{a-close,b-adopt,
> c-close}.md` + `docs/perf/data/s40-*.txt`。**

## 0. 结果一览(全部 WP,逐门验收)

| WP | 门 | 实测 | 判定 |
|---|---|---|---|
| D0 测量基建 | (工具,无门) | 窗口口径 harness(`.s40/gemv_bw_bench.py`,30rep 单事件对)+ lm_head 形状;窗口≈trace 交叉验证(lm_head 896.5 vs 899.3 µs,0.3%) | **完成** |
| B0 数值口径裁决 | 读账 + trace 定案 | **形状账本修正**:lm_head 实为 [62080,5120] fp16(635.7 MB/发,grid 62080)@**707 GB/s**,非设计图 [38484,5120] @417;非位同序确认 | **修正 + 闭账**(`s40-b0-verdict.md`) |
| A pf[4] 预取环 | 位同 `torch.equal` + gate_up/o_out ≥+6% | 位同**全绿**(两轮);微测**全面回退 −11~−20%**(gate_up 510→409 GB/s;PF4 无关的 lm_head 线 0.0% 证明设备状态稳定) | **FAIL 关档**(`.s40/a-close.md`) |
| B fp16 lm_head stream | 窗口 ≥752(重述)+ rel≤3e-4 + S29 板 | **797.0 GB/s(+12.3%)**;rel **2.74e-4** PASS;S29 A-vs-B **0.0107 nat / 0 错位**;服务 trace **808.6 µs vs 899.3**(−0.09 ms/step) | **采纳**(`.s40/b-adopt.md`) |
| C A-slice 双缓冲 | 探针 ≥+3%,否则关 | 设计图 ×2 环**有竞态**;安全 ×3 环(两种实现)**244–275 vs 581 GB/s(−53%)** | **关档,未进内核**(`.s40/c-close.md`) |
| DEP 部署 + ABAB | 合并门 −0.8 ms(设计值) | 实际可采纳项仅 B(−0.09 ms/step);e2e ABAB 配对中位差 **+0.36 t/s**(当日带 ±2 t/s 内不可分辨,无回归) | **采纳态上线**(设计合并门按 B 单项重述) |

**净效果**:`.so` md5 `ca05f746` → **`f72bb39d`**(pf2 fp8 路径与旧库七形状
`torch.equal` 全绿);env 增量仅 `VLLM_DCU_FP16_GEMV_STREAM=1`;lm_head 内核
899.3 → 808.6 µs(服务 trace 口径 **−0.09 ms/step**,−0.36% step);W1
贪心 36.0–37.4 t/s(当日带内)。**1 采纳、2 关档、1 账本修正。**

## 1. 本轮学到了什么(证据强度)

1. **裸形 probe 的"深度"不可移植到内核**(A 轨,强证据)。ceiling-probe
   的 deep4(+20%)是**每 lane 4 条独立 load 同时在飞**(chunk 步长
   64×4);内核 pf[4] 不制造并发 —— 每 slice 每 lane 仍 1 发 refill,
   深度只是把复用距离 2→4 slice 并 +8 寄存器,实测 −11~−20%。
   **可移植的方向是"多流并发",不是"环深"**。
2. **lm_head 账本修正**(B0,trace 实证):40 发/38 步、grid [62080,1,1] ×
   block [64,1,1] → [62080,5120] fp16 = 635.7 MB/发 @707 GB/s。S39 账本的
   "417 GB/s"是形状配错的账。桶大小 0.95 ms/step 不变;stream 几何把它
   压到 808.6 µs(786 GB/s,服务级)。
3. **单 barrier/ slice 的 A 双缓冲在结构上不成立**(C 轨,三重证据):
   ×2 旋转单 barrier 有 read/write 竞态(安全结构需 ×3);修正后的 ×3 环
   两种实现(动态基址/字面量下标)都 −53% —— **2-barrier 不是可回收的税**。
4. **微测判据链的价值**:b2_delta(随机权重 max|Δ|)拦住了首版寻址 bug
   (fp16 uint4=8 元素应为 `>>3`,误用 fp8 的 `>>4` —— 性能正常、数值
   全错);窗口/trace 双口径互证(lm_head 0.3% 偏差)给了采纳信心。

## 2. 部署现状(终态)

- 在役:`.so` md5 **f72bb39d**(含休眠 pf[4]/双缓冲代码,env 默认关);
  `serve_deploy.sh` 新增 **ADOPTED** 段:`export VLLM_DCU_FP16_GEMV_STREAM=1`
  (回滚 = 注释 + restart)。旧库备份 `.r3/so-backup/_rocm_C.s40-before.so`
  (md5 `ca05f746`)。
- S29 板(真实权重):A vs B `max|d top1| = 0.0107 nat`,**token 序零错位**
  —— 贪心文本逐位一致。
- **基线漂移备忘**:今天的 A/B 两臂与 S39 的 `parity-s39-b1-post.json` 在
  probe7469(7469-token 长 prompt)上续写整体不同(两个 short 用例逐位
  一致);经旧 `.so` 位同验证(七形状 torch.equal)排除内核因素 —— 归因
  为长 prefill 路径的环境漂移(triton autotune/prefill chunking 类),
  与 S40 代码无关,已记录待后续核验。
- FP 工具缺口:`VLLM_DCU_FP16_GEMV_STREAM` 未入
  `tools/dcu/dcu_policy_fingerprint.py` 的 key 表(FP 不能反映 S40 开关);
  待补。当前工具输出 `270fbac5c3c1f953`。

## 3. 模型现状(操作者口径)

- 服务:qwen3.8-27b,TP4,无 MTP(SPEC_OFF=1),单流贪心
  **36.0–37.4 t/s**(S40 当日 ABAB 带;S39 当日带 35.4–39.1),TTFT
  0.73–0.83 s;4 并发口径未在本轮重测(S39 C1 口径 ×1.95 维持)。
- decode 步构成(S39 账本 − 本轮修正):GEMV 12.30 ms、**lm_head 0.95 →
  ~0.86 ms**(本轮)、push-AR 3.76、attn 1.42、norm/碎核 ~2.6、GDN 1.00、
  bf16 小核 0.55、AG 0.19;GPU busy ~24 ms/step。

## 4. 下轮候选(按本轮修正后的账本排序)

0. **GEMV 裸形天花板钉死**(2026-10-31 追加探针 `s40-ceiling2-probe.txt`):
   宽 lane 连续 ×4 深(K-J,即 triton-905 模式的 GEMV 化)窗口仅
   **611 GB/s**,远低于 strided-deep4(K-E)的 **783** —— 905 的 wide-load
   模式在 GEMV lane 结构下**不可移植**。任何生产 GEMV 的诚实上限:
   轮转裸形 783 × LUT 税(−7.7%)× 内核效率(~0.95)≈ **690**;
   即便做权重合并分配(拿回轮转税)也只有 828×0.923≈764。**880 不可达**
   (高于一切实测 GEMV 形状裸形,包括无轮转的 828)。

1. **fp8 GEMV 多流并发**(A 轨的正确续作):m1 内核每 lane 的 W refill
   改为 2 条独立 load 在飞(wholea 的 kPf=3 无 barrier 环已有先例,
   deep4 的多流发射是探针证明的方向),目标 GEMV 桶 510→~580(窗口)。
2. **push-AR 3.76 ms**(现第二大桶)与 norm/碎核 ~2.6:S38 的 fused
   AR+rmsnorm 因 piecewise capture 损坏关档,eager 路径可再探(需先解
   capture 兼容)。
3. **长 prompt 预填路径的确定性**(本轮备忘):probe7469 续写漂移的根因
   (autotune 缓存 / chunking)值得一次专项核验——它影响一切贪心流逐位
   对账的基线纪律。
4. FP 工具补 `VLLM_DCU_FP16_GEMV_STREAM` key(一行)。

## 5. 台账(施工即台账)

- diff:`.s40/s40-a1b1-csrc.diff`(758 行)、`s40-b1-python.diff`(116)、
  `s40-deploy.diff`(37)
- 数据:`docs/perf/data/s40-{d0-baseline,d0-lmhead,a2-pf2,a2-pf4,
  a2-parity,b2-on,b2-delta-pf0,b2-delta-pf1,s29-board,c1-probe}.txt`、
  `s40-anchor-after.txt`、`s40-b0-verdict.md`、`.s40/gemv_bw_win_*.json`、
  `.s40/a2-{pf2,pf4,pf2-oldso}.pt`、`runs/s40/e2e-s40-ab-*.json`、
  `/mnt/game/s30c-prof/s34nospec_rank*.1793418038542*.trace.json.gz`(B 臂)
- `.so`:`.r3/so-backup/s40-before.md5` + `_rocm_C.s40-before.so`(改前);
  `.s40/s40-after-a1b1.md5` / `s40-after-b1fix.md5`(改后,现役 f72bb39d)
- 判定:`.s40/{a-close,b-adopt,c-close}.md`;S29 板 json
  `runs/s38/parity-s40-{a1,b1}.json`(副本 `runs/s40/`)
- 服务在役:`.s34/serve_deploy.sh` S40 ADOPTED 段(`VLLM_DCU_FP16_GEMV_STREAM=1`)
