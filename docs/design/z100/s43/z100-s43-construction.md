# S43 施工图:INT4 w4a16 GEMV(质量先行 + 几何探针 + wholea 骨架换载荷)

> 前置阅读:同目录 `z100-s43-design.md`(方案/门/风险)。事实基线:
> `docs/result/z100-s41-final.md`(S41 终局)、`.s42/b0-verdict.md`
> (a-reg/ldcs 关档)、`docs/design/z100/s42/int4-w4a16-prework.md`
> (收益账)。**部署基线**:无 MTP 臂在役(systemd `z100-vllm-mtp2`,
> `SPEC_OFF=1`)+ S40 段(`VLLM_DCU_FP16_GEMV_STREAM=1`)+ S41 段
> (`VLLM_DCU_FP8_GEMV_WA=1`);`.so` md5 `f72bb39d`;fp8 模型目录
> `/mnt/game/models/qwen3.8-27b`(只读)。本轮动 `_rocm_C`、launch
> 模型路径,新增 INT4 checkpoint 目录。

---

## 0. 通用纪律(承 S41 §0,差异项加粗)

1. **csrc 四步走**:备份在役 `.so` + md5 → 容器重编 → 微测(窗口)→
   数值对照 → 服务 smoke → e2e;失败恢复备份回 md5。构建命令
   `docker exec -e MAX_JOBS=8 z100-regress bash -lc 'bash /mnt/game/port/
   build-sync2.sh _rocm_C.abi3.so'`(**先验符号**:`llvm-readelf
   --dyn-symbols` 无新增 UND)。**K0 动内核前先备份**:
   `cp -L vllm/_rocm_C.abi3.so .r3/so-backup/_rocm_C.s43-before.so` +
   `md5sum ... | tee .r3/so-backup/s43-before.md5`。
2. **窗口口径(强制)**:30 rep 背靠背单事件对;每轮前后锚点
   (`--anchor`)漂移 >3% 作废本轮;**INT4 一律物理 GB/s 口径
   (0.53 B/元素)**,与 fp8 账换算 ×1.94。
3. **数值门(分两层,与 S41 不同)**:
   - 内核算术:内核(量化权重)vs fp64 逐元素参考 ≤ **1e-3 相对**
     (同权重,验算术);
   - 模型质量:Q1 PPL + Q2 语义门(**torch.equal / 3e-4 / S29 板
     本轮不适用——换权重**)。
4. profile 规程:一发 + 立即 `/stop_profile`;楔死 → restart(~270 s);
   **判读一律单发中位 + 内核总账**(S41 教训:桶中位数在步界破碎时不可用)。
5. 每步原始输出落 `docs/perf/data/s43-<wp>-<名>.{txt,json}`;diff 入
   `.s43/*.diff`。
6. e2e:`runs/specbench/e2e_ts.py --tag s43-<wp>-<arm> --out runs/s43`
   (W1 贪心 96×3 + W2 采样 512×2;ABAB 必配对)。

先建目录与锚点:

```bash
mkdir -p .s43 runs/s43 docs/design/z100/s43   # design 已就位
md5sum vllm/_rocm_C.abi3.so | tee .r3/so-backup/s43-before.md5
source .s34/base_env.sh; export HIP_VISIBLE_DEVICES=3
export VLLM_DCU_FP8_GEMV_M1=1 VLLM_DCU_FP8_GEMV_PF4=0 VLLM_DCU_FP8_GEMV_WA=1
.venv/bin/python .s40/gemv_bw_bench.py --anchor --tag s43-anchor \
  | tee docs/perf/data/s43-anchor.txt        # 对照 S41:792.9/532.7
```

---

## 1. Track Q:质量先行(0.5-1d,零内核成本,kill-fast)

### Q0 量化 checkpoint(0.5d)

```bash
mkdir -p /mnt/game/models/qwen3.8-27b-gptq-int4 /mnt/game/models/qwen3.8-27b-rtn-int4
# 校准集:runs/specbench prompt_ids 池采样 256 条 x 2048 token
.venv/bin/python .s43/build_calib.py --src runs/specbench/prompt_ids.json \
  --n 256 --seq 2048 --out .s43/calib_ids.json
# GPTQ g128 对称(工具链任一:auto-gptq / llm-compressor,容器外 HF 环境跑)
#   范围:K>=4096 的 GEMV 大权重;qkv/gdn_in/gate_up/z/down
#   保留:lm_head、o/out、GDN 小参数(A_log/dt_bias/conv)、embed/norm
#   输出:qweights int32 [N, K/8] 沿K打包 + q_s fp32 [N, K/128] + dtype 标记
.venv/bin/python .s43/quantize_gptq.py --base /mnt/game/models/qwen3.8-27b \
  --calib .s43/calib_ids.json --group 128 --sym \
  --skip-shapes o,out,lm_head,gdn_small \
  --out /mnt/game/models/qwen3.8-27b-gptq-int4 2>&1 | tee docs/perf/data/s43-q0-gptq.txt
# RTN 同配置(对照 + K2 数值参考)
.venv/bin/python .s43/quantize_rtn.py --base /mnt/game/models/qwen3.8-27b \
  --group 128 --sym --skip-shapes o,out,lm_head,gdn_small \
  --out /mnt/game/models/qwen3.8-27b-rtn-int4 2>&1 | tee docs/perf/data/s43-q0-rtn.txt
```

### Q1 PPL 评估(0.5d)

```bash
# 三臂:fp8 基线 / GPTQ / RTN;HF transformers 直跑(与 vLLM 内核无关)
.venv/bin/python .s43/eval_ppl.py --model /mnt/game/models/qwen3.8-27b \
  --datasets wikitext2,calib-domain --stride 512 \
  --tag fp8base | tee docs/perf/data/s43-q1-ppl-fp8.txt
.venv/bin/python .s43/eval_ppl.py --model /mnt/game/models/qwen3.8-27b-gptq-int4 \
  --datasets wikitext2,calib-domain --stride 512 \
  --tag gptq | tee docs/perf/data/s43-q1-ppl-gptq.txt
.venv/bin/python .s43/eval_ppl.py --model /mnt/game/models/qwen3.8-27b-rtn-int4 \
  --datasets wikitext2,calib-domain --stride 512 \
  --tag rtn | tee docs/perf/data/s43-q1-ppl-rtn.txt
```

**门**:PPL(GPTQ) ≤ PPL(fp8) × **1.03**;且 GPTQ 明显优于 RTN
(否则先修量化配置,不是关档)。

### Q2 任务小集 + 语义对照(0.5h)

```bash
# fp8 基线在役时:抓 8 个用例贪心输出留档
VLLM_PROBE_BASE=http://127.0.0.1:8000 .venv/bin/python .s43/task_probe.py \
  --tag s43-q2-fp8 --out runs/s43
# INT4 checkpoint 起临时实例(或 K 阶段后):同脚本 --tag s43-q2-gptq
# 规则查:无乱码 / 无复读 / 长度分布;人工并排留档
```

**Q1 或 Q2 不过 → 整轨关档**,落 `.s43/q-close.md`(总投入 1d)。

---

## 2. Track P:几何探针(0.5d,kill-fast)

```bash
cp .s42/widen_tax.cu .s43/widen_tax.cu
# mode 20 "U int4-v1":tile-2048 直 pack,lane 每 tile 1x uint4(32 元素)
# mode 21 "V int4-v2":tile-1024,lane 每 tile 1x dword(16 元素,访存靠合并)
# 共同:wholea 骨架 + 16-entry LUT 寄存器驻留 + per-32 折算 + A 走 LDS 整行
# 权重 bin:离线对现 fp8 权重做 group-128 对称 RTN,按 V1/V2 两布局各 pack 一份
.venv/bin/python .s43/gen_int4_probe_bins.py \
  --src /mnt/game/models/qwen3.8-27b --layout v1 --out .s43/int4_v1.bin
.venv/bin/python .s43/gen_int4_probe_bins.py \
  --src /mnt/game/models/qwen3.8-27b --layout v2 --out .s43/int4_v2.bin
docker exec z100-regress bash -c '/opt/dtk/bin/hipcc -O3 --offload-arch=gfx906 \
  -I/src/csrc/rocm /src/.s43/widen_tax.cu -o /src/.s43/widen_tax'
docker exec -e HIP_VISIBLE_DEVICES=3 z100-regress bash -c '/src/.s43/widen_tax' \
  2>&1 | tee docs/perf/data/s43-p0-probe.txt
```

**门(物理口径)**:gate_up 形 **≥ 550 GB/s**(= 643 × 0.855);两变体
取优,全不过 → **P/K/DEP 关档**(Q 轨产物保留,INT4 检查点另作他用)。
锚点:同轮 mode 15(fp8 wholea)漂移 ±3% 内有效。**同时记录 K=5120
利用率差(83% vs 100%)在实测中的实际代价**,喂给 K0 的 A0 预检。

---

## 3. Track K:内核化(1.5-2d,视 P0)

### K0 内核(1-1.5d)

- `csrc/rocm/fp8_emu_gemm.cu`:新增 INT4 载荷兄弟模板(对齐
  `fp8_emu_gemv_wholea_m1_kernel` 结构):slice-2048 walk(P0 赢家
  布局)、16-entry LUT 寄存器驻留(备选位技巧,由 P0 数据定)、
  per-32 折算(`kb = (s0 + lane*32)/128`,lane 块不跨组边界按构造)。
- 路由:`M == 1 && K >= 4096 && int4_enabled()` 且层 dtype=INT4;
  **o/out 与 lm_head 不接管**。
- env:`VLLM_DCU_INT4_GEMV=1`(默认 0)。
- **A0 预检**:编译无 spill + `llvm-readelf` 无新增 UND + VGPR ≤
  wholea+8(S42 教训)+ 微测不降速。diff:
  `git diff csrc/rocm/fp8_emu_gemm.cu > .s43/s43-k0-csrc.diff`。

### K1 loader/packer(1d)

- 离线 packer:GPTQ checkpoint → P0 赢家布局 + `q_s` fp32;**混合目录
  一次性生成**(量化层 INT4 + 保留层 fp8 原样),避免运行时双源加载。
- loader:按层 dtype 标记分发到 INT4/fp8 内核路径;`vllm/_custom_ops.py`
  扩展 w4a16 op 分支(或独立 op),参数表对齐现 `fp8_emu_gemm_block_w8a16`。

### K2/K3 门(0.5h + 0.5h)

```bash
# K2 数值门:内核 vs fp64 参考(同一量化权重),GPTQ/RTN 各一遍
VLLM_DCU_INT4_GEMV=1 .venv/bin/python .s43/k2_parity.py \
  --weights .s43/int4_v1.bin --ref fp64 \
  | tee docs/perf/data/s43-k2-parity.txt      # 门:max rel <= 1e-3
# K3 窗口门(锚点前后)
VLLM_DCU_INT4_GEMV=1 .venv/bin/python .s40/gemv_bw_bench.py \
  --tag s43-k3-int4 | tee docs/perf/data/s43-k3-window.txt
# 对照:fp8 形状(o/out)与 lm_head 同轮复测,零回归门(±2%)
```

**门**:K2 ≤1e-3;K3 gate_up/qkv 物理 ≥550;o/out/lm_head 零回归。
MS 内核跟随:不做(单流场景,RW=1 覆盖)。

---

## 4. Track DEP:部署 ABAB(0.5d,checkpoint 级)

```bash
# B 臂 = INT4 混合目录:launch_mtp2.sh 模型路径切换(S43 标注段)
sudo systemctl restart z100-vllm-mtp2   # ~270 s;权重读取减半,加载略快
# smoke + trace + e2e + Q2 复跑;A 臂 = fp8 目录;B/A/B/A 四臂
VLLM_PROBE_BASE=http://127.0.0.1:8000 .venv/bin/python \
  runs/specbench/e2e_ts.py --tag s43-ab-B1 --out runs/s43
VLLM_PROBE_BASE=http://127.0.0.1:8000 VLLM_PROBE_OUT=/home/lyh/z100-vllm/runs/s43 \
  .venv/bin/python runs/s34nospec/profile_window.py
python3 .s41/parse_trace_s41.py /mnt/game/s30c-prof/s34nospec_rank3.<新>.gz \
  --out-dir runs/s43 --tag s43-ab-B1
```

**门**:trace GEMV 桶 ≤ **6.5 ms/step**(楔死则按单发中位口径:
INT4 单发中位 ≤ fp8 40.8 µs 的 ~0.62,即 ≤25.3 µs);W1 配对 ≥ **+15%**;
Q2 语义门维持;TTFT 无回归。
**回滚**:launch 模型路径改回 fp8 + restart(逐位恢复);INT4 内核
env 默认 0 不触 fp8 路径;`.so` 异常时恢复 `.r3/so-backup/`。

---

## 5. 执行顺序与并行度

```
Q0 量化(0.5d)─┬─ Q1/Q2 质量门 ─┬─ 不过 → 整轨关档(q-close)
              │                └─ 过 ↓
P0 探针(0.5d,Q0 生成权重 bin 后并行)─┬─ <550 → 关档
                                      └─ ≥550 ↓
K0 内核(1-1.5d)→ K1 loader(1d)→ K2/K3 门 → DEP ABAB(0.5d)
```

- Q 与 P 在 Q0 产出后并行;K0 编译窗穿插 P0 复测;每项落地后基线快照
  更新(md5/桶/step p50);终局写 `docs/result/z100-s43-final.md`
  (模板对齐 s41)。

## 6. 产物清单(施工即台账)

- diff:`.s43/s43-k0-csrc.diff`、`s43-k1-python.diff`
- 数据:`docs/perf/data/s43-{anchor,q0-gptq,q0-rtn,q1-ppl-*,p0-probe,
  k2-parity,k3-window,ab-trace-*.txt}`、`runs/s43/e2e-s43-ab-*.json`、
  `runs/s43/task_probe-*`
- checkpoint:`/mnt/game/models/qwen3.8-27b-{gptq,rtn}-int4`(GPTQ 上线,
  RTN 留档)、`.s43/{calib_ids,int4_v1,int4_v2}.bin`
- `.so`:`.r3/so-backup/s43-before.md5` + `_rocm_C.s43-before.so`(改前);
  改后 md5 落 `.s43/`
- 判定:`.s43/{q-close|p-close|k-adopt,dep-adopt}.md`(对齐 .s41 体例)
- 质量证据:Q1 三臂 PPL json、Q2 双臂输出对照
