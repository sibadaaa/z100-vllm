# S39 施工图:GEMV LDS 消除 + 集合通信归位 + 4 并发口径(W4 capture 探索)

> 前置阅读:同目录 `z100-s39-design.md`(方案/门/风险)。事实基线:
> `runs/s34nospec/REPORT.md`(§3 分环节、§8 口径修正)、`.s38/a1-close.md`
> (W4 收档)、`docs/perf/data/s38-bw-ceiling.txt`(纯读 905 校准)。
> **部署基线**:无 MTP 臂在役(systemd `z100-vllm-mtp2`,`SPEC_OFF=1`,
> FP `2f3793deb97e138f`);`.so`:`_rocm_C.abi3.so` md5 `ca05f746`、
> `_C_stable_libtorch.abi3.so` md5 `54ccf2e2`;备份惯例 `.r3/so-backup/`。

---

## 0. 通用纪律(承 S38 §0,本阶段增补两条)

1. **csrc 四步走**:备份在役 `.so` + md5 → 容器重编 → 微测 → 数值对照 →
   服务 smoke → e2e;失败恢复备份回 md5。容器构建后 `.so` 属主 root,回滚用
   `docker exec cp`。
   - `_rocm_C.abi3.so`(Track A:`fp8_emu_gemm.cu` / `gfx906_emu_common.cuh` /
     `fp8_dequant.cuh`):`docker start z100-regress; docker exec z100-regress
     bash -c 'cd /mnt/game/port/build-sync/temp && ninja _rocm_C.abi3.so'`
   - `_C_stable_libtorch.abi3.so`(Track B:`custom_all_reduce.*`):
     **必须走 build-sync2**:`docker exec -e MAX_JOBS=8 z100-regress bash -lc
     'bash /mnt/game/port/build-sync2.sh _C_stable_libtorch'`
2. **带宽口径统一**:一切"上限/占比"以纯读 **905 GB/s** 为分母
  (`s38-bw-ceiling.txt`);harness 一律冷读轮转法(`gemv_bw_bench.py` 现行)。
3. profile 窗口规程(强制):一发请求 + 立即 `/stop_profile`;楔死 → restart。
4. 数值门:凡声称位同者以 `torch.equal` 判定,失败 = bug;非位同改动走
   `max|Δ|≤3e-4` + logprob 板(0.2255 nat,S29)。现役路径行为差异先跑
   位同语料:`/tmp/s36_parity.py check`(`.s36/parity/` 120 张量)。
5. 每步原始输出落 `docs/perf/data/s39-<wp>-<名>.{txt,json}`(不誊写);diff 入
   `.s39/*.diff`。
6. e2e:`runs/specbench/e2e_ts.py --tag s39-<wp>-<arm> --out runs/s39`(W1 贪心
   96×3 + W2 采样 512×2;step p50 = 1000/W1_med);服务启停
   `sudo systemctl stop|start|restart z100-vllm-mtp2`,启动后核对日志三行。

先建目录与首份台账:

```bash
mkdir -p .s39 runs/s39 docs/design/z100/s39   # design 已就位
```

---

## 1. A1:widen 税微基准(0.5h,kill-fast)

写 `.s39/widen_tax.py`(单卡离线,`HIP_VISIBLE_DEVICES=3`,分配 ≤2.5 GB):
gate_up(8704×5120)与 o/out(5120×1536)两形状,43 份轮转权重 defeat L2,
三内核各 30 rep CUDA-event 计时:

- K-A 纯读:`uint4` 载入 + 宽度不变量累加(参照);
- K-B LUT widen:载入 + `fp8x4_to_f16x2x2(sl,…)×4` + 校验和寄存器(现役路径);
- K-C 位技巧 widen:1.4 公式 + 校验和(不建 LDS)。

```bash
DTK_ROOT=/mnt/game/host-runtime/dtk2604 \
PYTHONPATH=/home/lyh/z100-vllm:/mnt/game/triton-gfx906/install \
HIP_VISIBLE_DEVICES=3 .venv/bin/python .s39/widen_tax.py \
  | tee docs/perf/data/s39-a1-widentax.txt
```

**门(K=B 与 A、C 与 B 的中位比)**:(B/A ≤ 0.75) 且 (C/B ≥ 1.15) → 进 A2/A3;
否则 LDS 假说证伪,A2/A3 关档留证(**A5 照常**)。注意 K-A 应 ≈905 纯读,
若 K-A 明显偏低先查微基准本身(并行度/轮转)。

---

## 2. A2:LUT dword 化(0.5d,位同)

| 位置 | 改动 |
|---|---|
| `gfx906_fp8.py`(host 建表) | 256 表项 half → **uint32**(低 16 位 = 原 half bits;高 16 位冗余 0/同值) |
| `gfx906_emu_common.cuh::fp8_lut_u16` | 签名改 `const uint32_t*`,`return lut[code & 0xFF]`(4B 读,取低 16 位) |
| `fp8_emu_gemm.cu` 各 `__shared__ __half sl[256]` | → `__shared__ uint32_t sl[256]`(+512 B/block;MS=8 总 LDS 仍 <20 KiB) |
| env | `VLLM_DCU_FP8_LUT32=1` 门(默认 0 = 现役) |

```bash
# 备份 + 重编(Track A → _rocm_C)
md5sum vllm/_rocm_C.abi3.so | tee .r3/so-backup/s39-before.md5
cp -L vllm/_rocm_C.abi3.so .r3/so-backup/_rocm_C.s39-before.so
git diff csrc/rocm/fp8_emu_gemm.cu csrc/rocm/gfx906_emu_common.cuh \
  > .s39/s39-a2-csrc.diff
docker start z100-regress
docker exec z100-regress bash -c \
  'cd /mnt/game/port/build-sync/temp && ninja _rocm_C.abi3.so' | tail -5
# 单卡微测(门:≥2 形状 +8%)与位同(同输入 LUT32=0/1 比对 out)
HIP_VISIBLE_DEVICES=3 PYTHONPATH=/home/lyh/z100-vllm:/mnt/game/triton-gfx906/install \
DTK_ROOT=/mnt/game/host-runtime/dtk2604 .venv/bin/python \
  runs/s34nospec/gemv_bw_bench.py | tee docs/perf/data/s39-a2-harness.txt
```

位同检查:七形状同输入跑 `LUT32=0/1` 比对 out(小脚本入 `.s39/a2_parity.py`);
`torch.equal` 全绿才算过。**不过门 → 留档,跳 A3。**

---

## 3. A3:位技巧 widen(1d,位同,核心)

### 3.1 第零步:1 页推导 + 256 码全表对拍(0.5h,硬门)

`.s39/a3_derive.py`:对 256 个 e4m3fn 码逐码比较 `LUT 值` vs `trick+修正值`
(`torch.equal`,fp16 位级):

```
naive  = ((b & 0x80) << 8) | ((b & 0x7f) << 7) | 0x2000     # normal 码精确
修正   = (b & 0x78) == 0 ? ((b & 0x7) ? naive - 0x1000      # subnormal→exp6
                                     : 0)                    # 零码
         : naive                                            # 其余不变
NaN 码(0x7f/0xff):host 加载端 clamp(量化权重必有限,留痕日志)
```

全表绿才进内核;e5m2 单独一页推导(其 subnormal 落 fp16 subnormal,naive
公式不同)。**失败 = 推导错误,修推导,不开工内核。**

### 3.2 第一步:内核集成(~40 行)

| 位置 | 改动 |
|---|---|
| `gfx906_emu_common.cuh` | 新增 `fp8_bits_widen(uint32_t four, uint32_t& lo, uint32_t& hi)`(位技巧 + branchless 修正;e5m2 模板参) |
| `fp8_emu_gemm.cu` W 路径 | `fp8x4_to_f16x2x2` 调用点按 env `VLLM_DCU_FP8_WIDEN_BIT`(默认 0)二选一;`sl`/`fp8_lut_stage` 在该路径下跳过 |
| A 侧(a16=0 时) | 同开关跟随(A 侧查表已按块摊销,收益小,保持 LUT 亦可——先只切 W 路径,归因干净) |

重编 `_rocm_C`(同 A2)→ 单卡微测 harness → 位同(七形状 `LUT` vs `BIT`
同输入 `torch.equal`)→ 服务 smoke。

### 3.3 第二步:验证序

```bash
# trace 复测(prefix s39a3,一发+立即 stop):分类速率
VLLM_PROBE_BASE=http://127.0.0.1:8000 python3 runs/s34nospec/profile_window.py
#   (profile_prefix 改 s39a3)→ parse_trace.py 对比分类 GB/s
# e2e:A 组合并门(A2+A3+A4+A5 全开一臂 vs 基线,B→A→B→A)
VLLM_PROBE_BASE=http://127.0.0.1:8000 .venv/bin/python \
  runs/specbench/e2e_ts.py --tag s39-a-A1 --out runs/s39
```

**门**:harness gate_up/down ≥700、加权 ≥650;e2e A 组合并 −0.8 ms;
过门后逐项单开复核(A2 / A3 / A5 各一臂)归因。

---

## 4. A4:dequant 归并定位(0.5h + 视结果 0.5d)

```bash
# 调用点与形状
grep -rn "fp8_dequant_to_fp16\|dequant_to_fp16" csrc/rocm/fp8_dequant.cuh \
  vllm/_custom_ops.py vllm/model_executor/kernels/linear/ | head
# trace 计数核验(现有 s34/s38 trace 即可)
python3 - <<'EOF'
import gzip, json, collections
d=json.load(gzip.open("/mnt/game/s30c-prof/s34nospec_rank0.1793360594970020927.pt.trace.json.gz",'rt'))
ev=d["traceEvents"] if isinstance(d,dict) else d
c=collections.Counter(e["name"] for e in ev if e.get("ph")=="X" and e.get("cat")=="kernel" and "dequant" in e.get("name","").lower())
print(c.most_common(4))
EOF
```

裁决分支:(i) 线性层前激活宽展 → 该 linear 改 a16=0 喺 stream kernel(内 prologue
宽展,查表已摊销)或上游直接给 fp16 → 消 0.65 ms;(ii) 同步伪影(scale 的
`.item()`/grid 过小)→ 修 launch 形状;(iii) 必要独立 pass → 闭账记录。
**门**:dequant trace 计数 → 0(或 <20 µs/次);并入 A 组 e2e 门;位同。

---

## 5. A5:K=1536 跑道(0.5h + 1d,承 S38 §4)

- **B1-a RW=2**(0.5h):`stream_block_select()`(fp8_emu_gemm.cu)加形状规则
  `K<=2048 && N>=4096 → rows=8; nt=256`(RW=2),env
  `VLLM_DCU_FP8_GEMV_SMALLK=1`(默认 0);kernel 零改。
  微测门:o/out −20%(对照 `gemv_bw_rows4.json` 44.8/44.5 µs 同 harness)、
  其余形状回归 ≤2%、七形状 `torch.equal`。
- **B1-b slice512**(1d,视 B1-a):`kStreamSlice` 模板参化(SLICE),小 K 实例
  512(静态断言 `kSliceChunks == SLICE/16` 泛化);门:o/out −20%、
  `max|Δ| ≤3e-4`、其余形状 ≤2%。
- 与 A2/A3 叠加部署(A5 单独 env 门,归因分开);**B1-a 不过即 B1-b,B1-b
  不过即 B1-c 出 1 页 micro-design 再施工**(S38 原文)。

---

## 6. B1:nccl ring 回退归位(1d,承 S38 §2.1-2.3)

```bash
# 定位:回退分支一次性日志(csrc custom_all_reduce.cuh/cpp → _C_stable_libtorch)
md5sum vllm/_C_stable_libtorch.abi3.so | tee -a .r3/so-backup/s39-before.md5
git diff csrc/custom_all_reduce.cuh > .s39/s39-b1-probe.diff   # 先存 fallback-log 版
docker exec -e MAX_JOBS=8 z100-regress bash -lc \
  'bash /mnt/game/port/build-sync2.sh _C_stable_libtorch'
sudo systemctl restart z100-vllm-mtp2   # env VLLM_DCU_CUSTOM_AR_FALLBACK_LOG=1 临时注入
# 1 个 profile 窗口(prefix s39b1)+ 96-token 请求 → 日志 + trace args 裁决 H1/H2
```

- H1(图捕获回退)→ 2.2a:AR 移出编译区(核对 `vllm.all_reduce` 分裂点)或
  push-AR capture-safe 化(仅当 finegrained flag 合法且无主机同步);
- H2(尺寸门)→ 2.2b:该消息单独评估 push/拆分;超门维持 nccl + 豁免记录。
- **验收**:trace `AllReduce_Ring` decode 计数 = 0;push-AR 129/步不变;冷启动
  全形状 [1,2,4] smoke 零 fail;e2e −0.5 ms;parity ≤0.2255 nat。
- 回滚:`VLLM_DCU_CUSTOM_AR_NO_NCCL=1`(默认现役)+ 单 diff 还原;
  `.so` 恢复 `docker exec cp` 备份。

## 7. B2:AllGather 定位(0.5d,承 S38 §3)

1. 读路径:`vllm/model_executor/layers/logits_processor.py`(M=1 分支)+
   `qwen3_5.py` lm_head 构造 + `grep lm_head` safetensors header(0.5h);
2. 观测:`profile_window.py` prefix `s39b2` 取 AllGather args(bytes/stream);
3. 裁决:闭账 / custom one-shot allgather(≤50 µs,csrc+绑定+env 门+微测)/
   124 MB 级则修字节账单列。**门**:AllGather ≤100 µs;e2e −0.2 ms 或闭账。

## 8. C1:4 并发验证(0.5h,零代码)

```bash
# 4 路并发(e2e_ts 并发改造或 4 进程各 7469-probe 96tok),聚合口径:
for i in 0 1 2 3; do (VLLM_PROBE_BASE=http://127.0.0.1:8000 .venv/bin/python \
  runs/specbench/e2e_ts.py --tag s39-c1-s$i --out runs/s39 --w1-reps 1 \
  --skip-w2 > /tmp/c1-$i.log 2>&1 &) ; done; wait; sleep 300
# 随后 solo 复测(M=1 回归门 ≤2%)
```

- 记录:聚合 tps、每流 ITL(stream chunk 间隔)、TTFT;**门**:聚合 ≥×3、
  每流 ITL ≤ solo +15%、solo 回归 ≤2%。
- 产物:`runs/s39/e2e-s39-c1-*.json` + 容量口径一段写入 `.s34/DEPLOY.md` 备注。

## 9. D1:W4 capture 根因(0.5d,time-box)

按序探针,任一命中即停:

1. `EAGER=1` 全量 W1 parity + 计时(eager 惩罚定量;若 <0.5 ms → "eager 采纳"
   进入采纳评估,绕开 capture);
2. `vllm.all_reduce_fused_rmsnorm` 加入 `splitting_ops`
   (`compilation_config` 传参即可,零代码)→ 两出口 op 出图复测位同;
3. 影子校验:replay 路径一次性对照(`FUSION_DEBUG` 思路移植)。

位同(piecewise)→ 补 ABAB(门 −0.5 ms);0.5d 未破 → 关档留探针证据。

## 10. 执行顺序与并行度

```
D1  A1 证伪(0.5h)─┬─ 证伪 → A5(RW=2)先行,C1 同日
                   └─ 证实 → A2(0.5d)→ A3(1d)→ A4 → A5 → A6 部署
D1  C1(0.5h,与 A1 并行,零代码)
D2  B1 定位(0.5d)→ B1 修(0.5d);B2(0.5d)
D3  A 组 e2e ABAB(B→A→B→A,~3h)→ 采纳;D1 time-box 穿插
```

- A/B 两轨不同 `.so`,重编互不影响;同日多臂时用**合并批**(A2+A3+A4+A5 全开
  一臂)先测总量,过门再逐项单开归因(S38 §5 惯例)。
- 每项落地后基线快照更新(md5/FP/step p50);终局写
  `docs/result/z100-s39-final.md`(模板对齐 s38/s37:结果一览 + 逐 WP 判据 +
  台账)。

## 11. 产物清单(施工即台账)

- diff:`.s39/s39-a2-csrc.diff`、`s39-a3-csrc.diff`、`s39-a4-*.diff`、
  `s39-a5-*.diff`、`s39-b1-probe.diff`、`s39-b1-*.diff`、`s39-b2-*.diff`、
  `s39-d1-*.diff`
- 数据:`docs/perf/data/s39-*.txt|json`、`runs/s39/e2e-*.json`、
  `/mnt/game/s30c-prof/s39*_rank*.pt.trace.json.gz`
- `.so`:前后 md5 对(`.r3/so-backup/_rocm_C.s39-before.so`、
  `_C_stable_libtorch.s39-before.md5` 追加)
- 采纳记录:`.s39/a{2,3,4,5}-adopt.md`、`b1-adopt.md`、`b2-adopt.md`
  (env、md5、FP、门数据、回滚一行命令);C1 容量口径入 `.s34/DEPLOY.md`
