# S44 施工图:push-AR 出口融合收尾(W4')+ 碎核账本猎杀(Track N)

> 前置阅读:同目录 `z100-s44-design.md`(方案/门/风险)。事实基线:
> S37 W4 部分交付(微测 per-op 0.81-0.88×、residual 位同 d=0)、S38 A1
> 模型穿透已在树、S39 decode 账本、S43 关档。**部署基线**:无 MTP 臂在役
> (systemd `z100-vllm-mtp2`,`SPEC_OFF=1`)+ S40 段 + S41 段。
> **本轮 `.so` 特殊性**:custom AR 库 = `_C_stable_libtorch.abi3.so`
> (md5 `54ccf2e2`,fused 核已在内),GEMV 库 = `_rocm_C.abi3.so`
> (`f72bb39d`)——Track AR 零重编;N1 起重编 `_C_stable_libtorch`,
> **必须 build-sync2**(裸 ninja 编到陈旧 hipify 副本,S37 踩坑)。

---

## 0. 通用纪律(承 S41/S43,差异项加粗)

1. **双 `.so` 台账**:备份/恢复面 = `_C_stable_libtorch.abi3.so` +
   `_rocm_C.abi3.so` 两份;构建命令
   `docker exec -e MAX_JOBS=8 z100-regress bash -lc
   'bash /mnt/game/port/build-sync2.sh _C_stable_libtorch'`;
   **先验符号**:`llvm-readelf --dyn-symbols` 无新增 UND + `strings`
   含 `cross_device_reduce_push_fused`。
2. **窗口口径(强制)**:30 rep 背靠背单事件对;每轮前后锚点漂移 >3%
   作废本轮;**AR/碎核判读一律单发中位 + 类目账**
   (`.s44/ledger_classes.py`,楔死窗桶中位数不可用——S41 教训已固化)。
3. 数值门分两层:**位同硬门(torch.equal,适用面:AR 和/residual、
   quant 映射、GDN 链逐元素)+ S29 板(适用面:norm Σs² 序差)**;
   "0 错位"门对序差变更不设(S34 先例)。
4. profile 规程:一发 + 立即 `/stop_profile`;楔死 → restart(~270 s);
   判读一律单发中位 + 类目账。
5. 每步原始输出落 `docs/perf/data/s44-<wp>-<名>.{txt,json}`;diff 入
   `.s44/*.diff`。
6. e2e:`VLLM_PROBE_BASE=http://127.0.0.1:8000 .venv/bin/python
   runs/specbench/e2e_ts.py --tag s44-<wp>-<arm> --out runs/s44`
   (W1 贪心 96×3 + W2 采样 512×2;ABAB 必配对)。

先建目录、备份与锚点:

```bash
mkdir -p .s44 runs/s44 docs/design/z100/s44   # design 已就位
md5sum vllm/_C_stable_libtorch.abi3.so vllm/_rocm_C.abi3.so \
  | tee .r3/so-backup/s44-before.md5
cp -L vllm/_C_stable_libtorch.abi3.so .r3/so-backup/_C_stable_libtorch.s44-before.so
cp -L vllm/_rocm_C.abi3.so        .r3/so-backup/_rocm_C.s44-before.so
source .s34/base_env.sh; export HIP_VISIBLE_DEVICES=3
export VLLM_DCU_FP8_GEMV_M1=1 VLLM_DCU_FP8_GEMV_PF4=0 VLLM_DCU_FP8_GEMV_WA=1
.venv/bin/python .s40/gemv_bw_bench.py --anchor --tag s44-anchor \
  | tee docs/perf/data/s44-anchor.txt        # 对照 S43:792.6/567.9
```

---

## 1. Track AR:W4' 收尾(0.5-1d,零重编)

### A0 功能探针(0.5h,graph 形态必验)

```bash
# 现役服务在跑请求时禁止 restart(restart_arm.sh 自带在途检查)
bash .s34/restart_arm.sh s44-ar-fused \
  VLLM_DCU_CUSTOM_AR_FUSION=1 VLLM_DCU_CUSTOM_AR_FUSION_DEBUG=1
# 等 /health(~270 s)后打一发 probe 触发 warmup(eager 段出 DEBUG 对拍):
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"ping"}],"max_tokens":32}' | head -c 200
journalctl -u z100-vllm-mtp2 --since "-10 min" | grep 'S38A1 fused debug' \
  | tee docs/perf/data/s44-a0-debug.txt
```

- 独立复验(离线,含 capture 路径):`torchrun --nproc-per-node 4
  .s38/fused_capture_check.py 2>&1 | tee docs/perf/data/s44-a0-capture.txt`
- **门**:16 条对拍 `d_res = 0`(硬)、`d_out ≤ 5e-4`;journal 确认融合
  路径在 piecewise graph 捕获内生效(无回退警告刷屏)。
- **边界显式核验**(设计 §1.3):journal 检查尾层 next-norm / 首层 skip
  无异常(层计数 = 64×2 出口);异常则先 `_MLP=0` 混合臂重跑 A0。

### A1 数值门(0.5h,S29 板重建)

`.s33/capture.py` 已不在树——重建 `.s44/board.py`(贪心 96 token ×3 rep,
top5 logprobs 留档,`--tag` 区分臂,输出 `runs/s44/board-<tag>.json`):

```bash
VLLM_PROBE_BASE=http://127.0.0.1:8000 .venv/bin/python .s44/board.py \
  --tag s44-ar-fused --rep 3 | tee docs/perf/data/s44-a1-board-fused.txt
# 对照臂 = A0 前的基线(先把 FUSION=0 跑一遍留档再翻转,或用 runs/s43 记录)
```

- **门**:无乱码/无复读/长度分布正常(序差类变更,S34 push-AR 采纳
  先例);首分歧位置与 top1 logprob 偏移分布留档(参考量级:S37 微测
  \|Δout\| ≤4.9e-4 → logits 偏移 ~1e-3 nat 级;塌陷式分歧 = 关)。

### A2 微核窗复测(0.5h)

```bash
docker exec z100-regress torchrun --nproc-per-node 4 \
  /src/.s37/car_fused_bench.py 2>&1 | tee docs/perf/data/s44-a2-micro.txt
```

- **门**:融合 per-op ≤ 未融合对 ×0.90(当日锚点漂移 ±3% 内;S37 实测
  0.81-0.88×,此处为复测防漂)。

### A3 e2e ABAB(0.5d,env 翻转,同一 `.so`)

```bash
# A 臂 = FUSION=0(基线),B 臂 = FUSION=1;B/A/B/A 四臂,每臂:
bash .s34/restart_arm.sh s44-ar-<arm> VLLM_DCU_CUSTOM_AR_FUSION=<0|1>
VLLM_PROBE_BASE=http://127.0.0.1:8000 .venv/bin/python \
  runs/specbench/e2e_ts.py --tag s44-ar-<arm> --out runs/s44
# trace 桶(B 臂一次):
VLLM_PROBE_BASE=http://127.0.0.1:8000 VLLM_PROBE_OUT=/home/lyh/z100-vllm/runs/s44 \
  .venv/bin/python runs/s34nospec/profile_window.py
.venv/bin/python .s41/parse_trace_s41.py /mnt/game/s30c-prof/s34nospec_rank3.<新>.gz \
  --out-dir runs/s44 --tag s44-ar-B
.venv/bin/python .s44/ledger_classes.py /mnt/game/s30c-prof/s34nospec_rank3.<新>.gz \
  | tee docs/perf/data/s44-a3-ledger-B.txt
```

- **门**:GPU busy **−0.8 ms/step**(桶账:allreduce 桶不升 ±5%,
  norm 五类 1.50 → ≤0.2);W1 配对 **≥ +2%**;TTFT 无回归;S29 板语义门
  维持。
- 判读注意:楔死窗的 allreduce 桶含自旋污染,以单发中位(基线 14.7 µs)
  与 norm 类目账为准。

### A4 采纳(0.5h)

- `serve_deploy.sh` 注入 `VLLM_DCU_CUSTOM_AR_FUSION=1`(`_MLP` 按实际
  采纳形态);`.s34/DEPLOY.md` 增行 + 回滚行(env 去 0 = 逐位恢复);
- 判定落 `.s44/ar-adopt.md`(对齐 .s41 体例);FP 指纹更新记录。

---

## 2. Track N:碎核猎杀(2-3d)

### N0 账本归档(已完成)

```bash
.venv/bin/python .s44/ledger_classes.py \
  /mnt/game/s30c-prof/s34nospec_rank3.1793448006522872837.pt.trace.json.gz \
  --top 30 | tee docs/perf/data/s44-n0-ledger.txt
```

基线数字(本设计 §0 表):norm 五类 1.50、GDN 链 2.13(0.61+0.58+0.36+
0.34+0.24)、fp8 quant 链 0.24、poi/per 碎核 0.52+、reduce_segments 0.32;
**scope 合计 5.19 ms/step**。A3 采纳后重跑一次,更新对照列。

### N1 fp8 act-quant 折进 AR 出口(0.5-1d,csrc)

```bash
# 1) csrc:cross_device_reduce_push_fused 增 quant_out 变体
#    (epilogue 出 e4m3 + per-128 scale;S37 W4 设计 §2 草图)
git diff csrc/custom_all_reduce.cuh csrc/custom_all_reduce.cu \
  > .s44/s44-n1-csrc.diff
# 2) python:vllm/_custom_ops.py + custom_all_reduce.py 增 quant_out 参数
git diff vllm/_custom_ops.py vllm/distributed/device_communicators/custom_all_reduce.py \
  > .s44/s44-n1-python.diff
# 3) 重编(必须 build-sync2)+ 验符号:
docker exec -e MAX_JOBS=8 z100-regress bash -lc \
  'bash /mnt/game/port/build-sync2.sh _C_stable_libtorch'
llvm-readelf --dyn-symbols vllm/_C_stable_libtorch.abi3.so | grep -c UND
md5sum vllm/_C_stable_libtorch.abi3.so | tee .s44/s44-n1-so.md5
# 4) 微测+位同:quant 路径 vs 现役 quant 核 torch.equal(硬门);
torchrun --nproc-per-node 4 /src/.s38/fused_capture_check.py \
  2>&1 | tee docs/perf/data/s44-n1-check.txt
# 5) A3 式 ABAB(对照臂 = A4 采纳态),e2e + 桶账
```

- **门**:quant 位同 = 1(硬);AR+norm 桶不回归;busy 相对 A4 态再
  −0.15~−0.25 ms(quant 0.24 + 空隙);e2e 无回归。
- **止损**:quant epilogue 使融合核寄存器/占用劣化(S42 教训:点亮前
  看 ISA)→ 先编后测,ISA 有 spill 即关 N1,不硬上。

### N2 GDN 链融合(1d,逐项独立)

- **N2a** `mul_silu_slice`(0.24)→ `bf16_emu_gemv_dot2` epilogue:
  gating 输入即 in_proj_a/b 输出,双行一次算完再 silu·mul。
  位同硬门(silu 确定性,序不变);微测 → 窗口 → e2e。
- **N2b** `causal_conv1d_update`(0.34)→ `fused_recurrent` prologue:
  收益在空隙不在字节;状态布局复杂化即关档(允许 kill,不牵连 N2a)。
- 落点:`csrc/rocm/fp8_emu_gemm.cu`(bf16 dot2/GDN 核同文件);
  `git diff ... > .s44/s44-n2-csrc.diff`;重编 `_rocm_C`(build-sync2)。
- N2 合计现实预期 **−0.3~−0.5 ms/step**(含 96 处核间空隙)。

### N3 triton 碎核定性与处置(0.5d)

- 对象:`triton_per_fused_1`(0.36)、`triton_poi_fused_0`(0.29,63 发
  /step)、`_6`(0.23)——inductor 产物,先归因:
  `VLLM_COMPILE_DEBUG=1`/torch profiler stack 对齐源算子,落
  `docs/perf/data/s44-n3-attr.txt`。
- 处置三分:并邻居 / 声明最小形态关档 / python 侧冗余调度上游修;
  逐项微-窗-板-四步,账本行引用制。

### N4 reduce_segments(0.5d,可选)

- 15.6 发/step × 19.8 µs = 0.32;折进 attention 内核 epilogue 须过
  attn 位同门(S37 A1 同门);视 N1-N3 后余量决定,上限 0.3 ms。

### Track N 总门

- scope 桶(账本口径)**5.19 → ≤3.7 ms/step**(AR 已吃 1.5,N1-N4
  再吃 0.5-1.0,留 ±0.3 折损);e2e 汇总相对 S44 起点 **W1 ≥ +5%**;
- 判定落 `.s44/{n-adopt|n-close}.md`,逐项台账在案。

---

## 3. 执行顺序与并行度

```
A0 → A1 → A2 → A3(ABAB)→ A4 采纳            # Track AR,零重编,1 天内
        └─(A0 通过后)N0 归档 → N1(csrc,重编)
                     └─ N2a ∥ N3(与 N1 无依赖,可穿插编译窗)
                          └─ N2b → N4(可选)
```

- AR 全链先行:收益最确定、零重编、为 N1 的 quant 变体定形态;
- N1 与 N2a 分属两个 `.so`(`_C_stable_libtorch` / `_rocm_C`),编译窗
  天然隔离,可交替;
- 每项落地后基线快照更新(md5/桶账/W1);终局写
  `docs/result/z100-s44-final.md`(模板对齐 s41)。

## 4. 产物清单(施工即台账)

- diff:`.s44/s44-n1-{csrc,python}.diff`、`s44-n2-csrc.diff`、
  `s44-board.py`(A1 重建工具)
- 数据:`docs/perf/data/s44-{anchor,a0-debug,a0-capture,a1-board-*,
  a2-micro,a3-*,n0-ledger,n1-check,n2-*,n3-attr}.txt`、
  `runs/s44/{e2e-s44-ar-*,board-*,steps-s44-*}.json`
- `.so`:`.r3/so-backup/s44-before.md5` + 双 before 副本;N1 改后 md5
  落 `.s44/s44-n1-so.md5`
- 判定:`.s44/{ar-adopt,n-adopt|n-close}.md`
- 服务变更:`serve_deploy.sh` env 一行 + `.s34/DEPLOY.md`(回滚 = env
  去 0 + restart,逐位恢复)
