# S38 施工图:AR 复合路径收口 + FP8 GEMV 小 K 特化

> 前置阅读:同目录 `z100-s38-design.md`(方案/门/风险)。事实基线:
> `runs/s34nospec/REPORT.md`(2026-10-30 无 MTP 臂 39-step trace + GEMV 分级
> 带宽);工具直接复用 `runs/s34nospec/` 与 `.s37/`。
> **部署基线**:无 MTP 臂在役(systemd `z100-vllm-mtp2`,`SPEC_OFF=1`,
> FP `2f3793deb97e138f`);`.so`: `_rocm_C.abi3.so` md5 `ca05f746`,
> `_C_stable_libtorch.abi3.so` md5 `7dcce4f2`;备份惯例 `.r3/so-backup/`。

---

## 0. 通用纪律(沿用 S37 §0,本阶段增补)

1. **csrc 改动四步走**:备份在役 `.so` + md5 → 容器重编译 → 微测 → 数值对照
   → 服务 smoke → e2e;失败恢复备份并回 md5。容器构建后 `.so` 属主 root,
   回滚用 `docker exec cp`。
   - `_rocm_C.abi3.so`:`docker start z100-regress; docker exec z100-regress
     bash -c 'cd /mnt/game/port/build-sync/temp && ninja _rocm_C.abi3.so'`
   - `_C_stable_libtorch.abi3.so`:**必须走 build-sync2**(先刷 hipify 暂存,
     裸 ninja 编到陈旧副本,S37 已踩坑):
     `docker exec -e MAX_JOBS=8 z100-regress bash -lc
     'bash /mnt/game/port/build-sync2.sh _C_stable_libtorch'`
2. **profile 窗口规程(S37 事故教训,强制)**:一发请求 + 立即
   `/stop_profile`,不依赖 max_iterations 兜底;工具
   `runs/s34nospec/profile_window.py`(改 `profile_prefix` 即可)。每窗口
   之后 health 确认;异常楔死 → `sudo systemctl restart z100-vllm-mtp2`
   (~60 s,编译缓存热)。
3. **离线微测环境**:`DTK_ROOT=/mnt/game/host-runtime/dtk2604` +
   `PYTHONPATH=/home/lyh/z100-vllm:/mnt/game/triton-gfx906/install`
   (26.10 用户态缺 libhipblas.so.2,S38 前置实测已踩);单卡
   `HIP_VISIBLE_DEVICES=3`,分配 ≤2.5 GB(服务占 85%)。
4. 每步原始输出落 `docs/perf/data/s38-<wp>-<date>.txt`,不誊写。
5. e2e 口径:`runs/specbench/e2e_ts.py --tag s38-<wp>-<arm> --out runs/s38`
   (W1 贪心 96×3 + W2 采样 512×2);step p50 = 1000/W1_tps_med。
6. 服务启停:`sudo systemctl stop|start|restart z100-vllm-mtp2`;启动后核对
   日志三行(`SPEC_OFF=1`、`spec=` 空、`policy: FP`);`.s34/start.sh` 的
   spec 行检查在本臂下打印 `NO SPEC LINE` 属预期。

---

## 1. A1/W4 收尾:融合 AR 部署 + 模型层穿透(0.5–1d)

### 1.1 第一步:重建 `_C_stable_libtorch`(0.5h)

```bash
md5sum /mnt/game/port/_C_stable_libtorch.abi3.so          # 记录 7dcce4f2
cp /mnt/game/port/_C_stable_libtorch.abi3.so \
   .r3/so-backup/_C_stable_libtorch.s38-before.so
docker start z100-regress
docker exec -e MAX_JOBS=8 z100-regress bash -lc \
  'bash /mnt/game/port/build-sync2.sh _C_stable_libtorch'
md5sum /mnt/game/port/_C_stable_libtorch.abi3.so          # 新 md5 入台账
```

### 1.2 第二步:模型层 ar_epilogue 穿透(0.5d)

- 文件:`vllm/model_executor/models/qwen3_5.py`(decoder 层两处出口)+
  `vllm/model_executor/kernels/linear/scaled_mm/...`(或 RowParallelLinear
  所在层文件)按 S37 python diff(`.s37/s37-w4-python.diff`)续做:
  - 在 Qwen3NextDecoderLayer 的 attn 出口与 mlp 出口,当
    `custom_all_reduce_fused` 可用(形状门 + env)时以
    `vllm.all_reduce_fused_rmsnorm(hidden, residual)` 取代
    AR + `fused_add_rms_norm` 两步;不可用回退现路径(逐位一致)。
- `git diff > .s38/s38-a1-python.diff` 入台账;`pre-commit run` 过。

### 1.3 第三步:微测复测(0.5h,门:ratio ≤0.85 / Δout ≤5e-04 / Δresidual=0)

```bash
bash .s37/car_fused_bench.py   # rows 1,4;输出 docs/perf/data/s38-a1-car.txt
```
不过门 → 停,记 FAIL,回滚 `.so`(见 1.5)。

### 1.4 第四步:服务 A/B(AB 各 2 实例,~3h)

```bash
# B = 融合关(基线),A = 融合开;次序 B→A→B→A
sudo systemctl restart z100-vllm-mtp2 && sleep 400   # 等健康(带 SPEC_OFF=1)
VLLM_PROBE_BASE=http://127.0.0.1:8000 .venv/bin/python \
  runs/specbench/e2e_ts.py --tag s38-a1-B1 --out runs/s38
# A 实例:serve_deploy.sh 临时加 export VLLM_DCU_CUSTOM_AR_FUSION=1 → restart → 同上 --tag s38-a1-A1
# ……重复 B2/A2;每实例加抓一个 39-step trace(1.4)
```

- **trace 窗口(A 实例)**:`profile_window.py` prefix `s38a1`;确认
  `cross_device_reduce_push_fused` 出现、`fused_add_rms_norm*` 在图内计数
  下降、host gap(inter_cpu_gap)不涨 → `parse_trace.py` 对比。
- **门**:e2e step p50 A 中位 vs B 中位 **−0.8 ms**;W1 in-arm 一致;
  parity 板 max|d logprob| ≤0.2255 nat(8-prompt,S29 界)。

### 1.5 采纳与回滚

- 过门 → `serve_deploy.sh` 固化 `export VLLM_DCU_CUSTOM_AR_FUSION=1`
  (注释回滚 = 注释该行 + restart);写 `.s38/a1-adopt.md`(md5、指纹、门数据)。
- 回滚:env 注释 + `docker exec cp` 还原备份 `.so` + git 还原 python diff。

---

## 2. A2:nccl 回退 AR 归位(1d)

### 2.1 第一步:定位(0.5d,kill-fast)

- csrc `custom_all_reduce.cuh/cpp` 的 nccl 回退分支加一次性日志
  (env `VLLM_DCU_CUSTOM_AR_FALLBACK_LOG=1`,rank0,每 (op,bytes,capture) 首见
  打一行),容器重编 `_rocm_C`(小改,rebuild ~10 min)。
- 跑 1 个 profile 窗口(prefix `s38a2probe`)+ 96-token 请求;从日志 +
  trace(`ncclDevKernel_AllReduce_Ring` 的 args)裁决:
  - **H1 图捕获回退**(调用点在 CompiledFxGraph / capture 中)→ 走 2.2a;
  - **H2 尺寸门**(bytes >128 KiB)→ 走 2.2b。

### 2.2 第二步:按位点修(0.5d)

- **2.2a(H1)**:优先"把 AR 移出编译区"——核对 `vllm.all_reduce` custom op
  为何未在该点形成 piecewise 分裂(疑被融合/concat 包进 graph);在其前显式
  分裂或改包装为 custom op。**备选** push-AR capture-safe 化(仅当
  finegrained flag/无主机同步被证实合法;否则不做)。
- **2.2b(H2)**:该消息单独评估 push(two-shot)或拆分;超门消息维持 nccl 并
  在设计附注豁免理由。
- **微测**:`car_fused_bench.py` 同 harness 对比该消息形状 nccl vs push。

### 2.3 验收(门)

- trace:`ncclDevKernel_AllReduce_Ring` decode 计数/步 = 0;push-AR 计数
  与现役一致(129/step);冷启动 + 全捕获形状 [1,2,4] smoke 零 fail。
- e2e:step p50 **−0.5 ms**;贪心 parity 同 A1 门。
- 回滚:env `VLLM_DCU_CUSTOM_AR_FALLBACK_LOG` 默认 0;行为改动以 env
  `VLLM_DCU_CUSTOM_AR_NO_NCCL=1` 控制(默认现役),单 diff 还原。

---

## 3. A3:logits / AllGather 定位(0.5d)

1. **读路径(0.5h)**:`vllm/model_executor/layers/logits_processor.py`
   (M=1 分支)+ `qwen3_5.py` 的 lm_head 构造(ParallelLMHead 切法)+
   shard 头:`grep lm_head /mnt/game/models/Qwen3.8-27B-FP8/*.safetensors`
   (经 header 解析)确认 dtype/shape。
2. **观测(0.5h)**:`profile_window.py` prefix `s38a3` 的 trace 里取
   AllGather 事件的 `args`(bytes/stream/包含 op);对照 `all_gather`
   cpu_op 的输入张量尺寸,判定传的是什么。
3. **裁决**:
   - 若 logits GEMV 确实走了某已有快路径 → 闭账:更新 REPORT 字节账
     (§6.6 → 结论),不改码,WP 关档。
   - 若 AllGather 是小消息(≤256 KB)→ 换 custom one-shot allgather
     (对齐 push-AR 实现,≤50 µs 可达):csrc + 绑定 + env 门,微测
     (同 car harness)+ trace 门(AllGather ≤100 µs)+ e2e 门 −0.2 ms。
   - 若是 124 MB 级 → 字节账模型修正,并评估 reduce-scatter+gather 替代
     (单列 follow-up,不强制本期)。

---

## 4. Track B:FP8 GEMV 小 K 特化(1–2d)

### 4.1 B1-a:RW=2 双行(0.5h,先做)

- 改动:`stream_block_select()`(fp8_emu_gemm.cu)加一条形状规则——
  `K<=2048 && N>=4096` 时 `rows=8; nt=256;`(RW=rows/(nt>>6)=2),
  以 env `VLLM_DCU_FP8_GEMV_SMALLK=1` 启用(默认 0 = 现役逐位不变);
  kernel 本体零改(RW 模板已有,核对 `row < N` 边界与 `pf[2][RW]`)。
- 重编 `_rocm_C`(容器 ninja,~10 min)→ 微测:

```bash
DTK_ROOT=/mnt/game/host-runtime/dtk2604 \
PYTHONPATH=/home/lyh/z100-vllm:/mnt/game/triton-gfx906/install \
HIP_VISIBLE_DEVICES=3 VLLM_DCU_FP8_GEMV_M1=1 VLLM_DCU_FP8_W8A16=1 \
VLLM_DCU_FP8_GEMV_STREAM_ROWS=4 VLLM_DCU_FP8_GEMV_SMALLK=1 \
  .venv/bin/python runs/s34nospec/gemv_bw_bench.py   # 对照 rows4 基线 JSON
```

- **门**:o/out 中位 −20%(同 harness,对照 `gemv_bw_rows4.json` 的
  44.8/44.5 µs)且 gate_up/down/qkv 回归 ≤2%;数值:七形状 `torch.equal`
  (小脚本:同输入跑 SMALLK=0/1 比对 out)。
- 不过 → B1-b。

### 4.2 B1-b:slice512(1d,视 B1-a 结果)

- `kStreamSlice` 从 constexpr 提为模板参 `SLICE`,小 K 形状实例化 512
  (静态断言 `kSliceChunks == SLICE/16` 相应泛化);kb 折叠分组随绝对 k
  不变(核 `kb = ((s0>>4)+lane)/cpb` 的绝对索引注释仍成立)。
- 微测门:o/out −20% 且 `max|Δ| ≤3e-4`(非位同,累加分组变化);
  gate_up/down/qkv 回归 ≤2%。
- 不过 → B1-c(尾 slice lane 拆分,peel 宏重写;设计同 design §4.2,
  先出 1 页 micro-design 再施工)。

### 4.3 B2:部署与验收(0.5d)

- 部署态:形状门自动生效 + `VLLM_DCU_FP8_GEMV_SMALLK=1` 写入
  `serve_deploy.sh`(回滚 = 该 env 置 0,无需动 `.so`)。
- trace 复测:prefix `s38b2`;门:o/out **≥450 GB/s**(现 309)、
  fp8 GEMV 步合计 ≤10.2 ms(现 10.84)。
- e2e 门:step p50 **−0.3 ms**;parity 同 A1 门。

---

## 5. 执行顺序与并行度

```
D1  A1.1 重建 ─ A1.3 微测 ────────── A1.4 服务 ABAB(3h)── A1.5 采纳
D1  B1-a 微测(与 A1.2 并行,单卡 3 空闲)→ 过门即 B2 trace 复测排 D2
D2  A2.1 定位(0.5d)→ D3 A2.2 修复
D2  A3 定位(0.5h)→ 视结果改码或闭账
D3  A2/A3/B2 e2e(可合并为一个 ABAB 批次:基线 vs 三改动全开,过者逐项
    单开复核归因)
```

- 合并批注意:三改动无耦合(AR 出口/nccl 回退/GEMV 派发),先合并测总量,
  再逐项单开归因——省 2 次重启批次;若总量不过门则逐项独立 A/B 定位回退。
- 每项落地后基线快照更新(md5/指纹/step p50),终局写
  `docs/result/z100-s38-final.md`(模板对齐 s37-final:结果一览表 +
  逐 WP 判据 + 台账)。

## 6. 产物清单(施工即台账)

- diff:`.s38/s38-a1-python.diff`、`s38-a2-*.diff`、`s38-b1-*.diff`
- 数据:`docs/perf/data/s38-*.txt` / `runs/s38/e2e-*.json` /
  `/mnt/game/s30c-prof/s38*_rank*.pt.trace.json.gz`
- `.so`:前后 md5 对(含 `.r3/so-backup/` 备份对)
- 采纳记录:`.s38/a{1,2,3}-adopt.md`、`b2-adopt.md`(env、指纹、门数据、
  回滚一行命令)
