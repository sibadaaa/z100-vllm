# S41 施工图:GEMV 第二轮(wholea 采纳 + 多流发射 + o/out 固定开销)

> 前置阅读:同目录 `z100-s41-design.md`(方案/门/风险)。事实基线:
> `docs/result/z100-s40-final.md`(S40 终局)、`docs/perf/data/
> s40-{wholea-window,rows8-window,ceiling2-probe}.txt`(本轮动机数据)。
> **部署基线**:无 MTP 臂在役(systemd `z100-vllm-mtp2`,`SPEC_OFF=1`)
> + S40 ADOPTED 段(`VLLM_DCU_FP16_GEMV_STREAM=1`);`.so` md5 `f72bb39d`,
> 备份 `.r3/so-backup/`。本轮**只动 `_rocm_C`** 与 `serve_deploy.sh`。

---

## 0. 通用纪律(承 S40 §0,全量沿用)

1. **csrc 四步走**:备份在役 `.so` + md5 → 容器重编 → 微测(窗口)→
   数值对照 → 服务 smoke → e2e;失败恢复备份回 md5。构建命令
   `docker exec -e MAX_JOBS=8 z100-regress bash -lc 'bash /mnt/game/port/
   build-sync2.sh _rocm_C.abi3.so'`(**先验符号**:`llvm-readelf
   --dyn-symbols` 无新增 UND —— S40 教训:hipify 竞态会产静默缺符号)。
2. **窗口口径(强制)**:30 rep 背靠背单事件对;每轮前后 K-A 锚点
   (`--anchor`)漂移 >3% 作废本轮。
3. **数值门**:位同者 `torch.equal`;非位同者 `max|Δ|≤3e-4` + S29 板
   (`.s38/parity_probe.py --capture/--compare`,0.2255 nat)+ 贪心流记录。
4. profile 规程:一发 + 立即 `/stop_profile`;楔死 → restart(~270 s)。
5. 每步原始输出落 `docs/perf/data/s41-<wp>-<名>.{txt,json}`;diff 入
   `.s41/*.diff`。
6. e2e:`runs/specbench/e2e_ts.py --tag s41-<wp>-<arm> --out runs/s41`
   (W1 贪心 96×3 + W2 采样 512×2;ABAB 必配对)。

先建目录与基线锚点:

```bash
mkdir -p .s41 runs/s41 docs/design/z100/s41   # design 已就位
md5sum vllm/_rocm_C.abi3.so | tee .r3/so-backup/s41-before.md5
cp -L vllm/_rocm_C.abi3.so .r3/so-backup/_rocm_C.s41-before.so
source .s34/base_env.sh; export HIP_VISIBLE_DEVICES=3
.venv/bin/python .s40/gemv_bw_bench.py --anchor --tag s41-anchor \
  | tee docs/perf/data/s41-anchor.txt
```

---

## 1. D0:测量基建(0.5h,先行)

1. `cp .s40/widen_tax.cu .s41/widen_tax.cu` 并加两模式(§3 B0):
   - **mode 12 "K-M group4"**:K-E 的 strided-deep4 chunk 模式
     (`c = lane + 64d`,d=0..3,组步进 256 chunk),组首背靠背 4 发,
     组内逐 slice 插入 A-stage 模拟(2 KB LDS 写 + 2 `__syncthreads()`);
   - **mode 13 "K-N wholea4"**:整行 A 一次进 LDS、零 barrier,W 环
     kPf=4(K-E 同款 chunk 模式)。
   编译:`docker exec z100-regress bash -c '/opt/dtk/bin/hipcc -O3
   --offload-arch=gfx906 -I/src/csrc/rocm /src/.s41/widen_tax.cu -o
   /src/.s41/widen_tax'`(容器内跑,`/src/.s39/lut_e4m3.bin` 依赖)。
2. 锚点当日值落 `docs/perf/data/s41-anchor.txt`。

---

## 2. Track A:wholea 采纳(0.5d,零施工)

### A0 位同复核(0.5h,kill-fast)

```bash
# 臂1:现役 m1(WA 不设)
VLLM_DCU_FP8_GEMV_WA=0 VLLM_DCU_FP8_GEMV_M1=1 VLLM_DCU_FP8_GEMV_PF4=0 \
  .venv/bin/python .s40/a2_parity.py .s41/a-wholea-ref.pt m1
# 臂2:wholea 抢占(WA=1;M1 留 1 验证抢占路径)
VLLM_DCU_FP8_GEMV_WA=1 VLLM_DCU_FP8_GEMV_M1=1 VLLM_DCU_FP8_GEMV_PF4=0 \
  .venv/bin/python .s40/a2_parity.py .s41/a-wholea-new.pt wa
# 比对:torch.equal 全绿 → A1;失败 = 构造判据翻车,先修
```

### A1 部署 ABAB(0.5d)

```bash
# B 臂:serve_deploy.sh 增(S41 标注段)
#   export VLLM_DCU_FP8_GEMV_WA=1
sudo systemctl restart z100-vllm-mtp2   # ~270 s
# smoke + S29 板(应 0 错位)+ trace + e2e
VLLM_PROBE_BASE=http://127.0.0.1:8000 .venv/bin/python .s38/parity_probe.py \
  --capture s41-a-b1
VLLM_PROBE_OUT=/home/lyh/z100-vllm/runs/s41 .venv/bin/python \
  runs/s34nospec/profile_window.py    # 一发 + 立即 stop;楔死则 restart
python3 .s40/parse_trace_s40.py /mnt/game/s30c-prof/s34nospec_rank3.<新>.gz \
  --out-dir runs/s41 --tag s41-a-b1
VLLM_PROBE_BASE=http://127.0.0.1:8000 .venv/bin/python \
  runs/specbench/e2e_ts.py --tag s41-ab-A-b1 --out runs/s41
# A 臂:注释 WA → restart → e2e --tag s41-ab-A-a1;B/A/B/A 四臂
```

**门**:trace GEMV 桶(`fp8_emu_gemv_wholea_m1_kernel`)分类速率
**≥ +6%**;S29 板 0 错位;e2e 无回归(当日带内)。
**回滚**:WA 不设 = 现役逐位(位同已证);`.so` 不需回退。

---

## 3. Track B:多流发射(0.5d 探针 + 0.5-1d 内核)

### B0 探针(0.5d,kill-fast)

```bash
docker exec -e HIP_VISIBLE_DEVICES=3 z100-regress bash -c '/src/.s41/widen_tax' \
  | tee docs/perf/data/s41-b0-probe.txt
# 读 K-M(mode 12)/ K-N(mode 13)窗口,gate_up + o_out 两形状
```

**门**:K-M 或 K-N 窗口 **≥ 830 GB/s**(= K-E 783 × 1.06)。对照:
K-E 783(无 barrier 上限)、K-F 723(含 LUT 下限)。两模式全 <830 →
**B 整轨关档**(组发射连 barrier 都扛不住,不施工)。

### B1 内核化(赢者定形态)

| 探针赢者 | 内核改动 | env 门 |
|---|---|---|
| K-M(group4 + barrier) | `fp8_emu_gemv_stream_m1_kernel`(~:925):K 循环改组结构——组首 4 发 `wv[c]`(`c = 组基 + lane + 64d`),组内逐 slice stage/barrier/MAC 消费第 d 寄存器;lane→chunk 映射与 k 序不变 | `VLLM_DCU_FP8_GEMV_M1_G4=1`(默认 0) |
| K-N(wholea 零 barrier) | `fp8_emu_gemv_wholea_m1_kernel`(~:1227):`kPf` 3→4(环声明/预填/peel 循环同构展开) | `VLLM_DCU_FP8_GEMV_WA4=1`(默认 0,依赖 WA=1) |

- A0 预检:编译无 spill + 微测不降速(S40 惯例);MS=8 实例不开。
- 与 Track A 叠加矩阵:A ∈ {WA=0/1} × B ∈ {off/G4/WA4},合并批先测总量,
  过门后逐项单开归因。
- diff:`git diff csrc/rocm/fp8_emu_gemm.cu > .s41/s41-b1-csrc.diff`。

### B2/B3 门(0.5h + 视况)

```bash
# 窗口:七形状,off vs 赢者臂(各带锚点)
VLLM_DCU_FP8_GEMV_M1_G4=1 ... .s40/gemv_bw_bench.py --tag s41-b2-g4 \
  | tee docs/perf/data/s41-b2-g4.txt
# 位同:两进程 dump + torch.equal(MULTI/G4 按构造位同,失败 = bug)
```

**门**:gate_up、attn_qkv **≥ +6%**(对照 wholea 臂),其余形状 ≤2% 回归;
`torch.equal` 全绿。MS 内核跟随(视 B2):同构 + 预检,RW≤2 only。

---

## 4. Track C:o/out 固定开销(time-box 0.5d,旋钮先行)

```bash
# C1 零施工:rows=16 离线微测(RW=2,nt=512)
VLLM_DCU_FP8_GEMV_STREAM_ROWS=16 ... .s40/gemv_bw_bench.py --shape attn_o,gdn_out \
  --tag s41-c1-rows16 | tee docs/perf/data/s41-c1-rows16.txt
```

**门**:o/out **≥ +10%**(=298 GB/s)才立案 C2;否则关(0.5h 出局)。
C2(过门才做):`stream_block_select`(~:1649)加 K 条件化规则
`K < 2048 → rows=16, nt=512`,env `VLLM_DCU_FP8_GEMV_SMALLK16=1`(默认 0;
**只动 K<2048 类**,S39 A5 教训)。**非位同**(warp→行归属变化改变归约
划分)→ 数值门(max|Δ|≤3e-4 + S29 板)。

---

## 5. 部署与验收(合并批 → 归因)

```bash
# 合并臂(WA=1 [+ G4/WA4/SMALLK16 若过])注入 serve_deploy.sh(S41 段)
sudo systemctl restart z100-vllm-mtp2
# 服务 smoke + S29 板 + trace(prefix s41)+ e2e ABAB 四臂
VLLM_PROBE_BASE=http://127.0.0.1:8000 python3 runs/s34nospec/profile_window.py
VLLM_PROBE_BASE=http://127.0.0.1:8000 .venv/bin/python \
  runs/specbench/e2e_ts.py --tag s41-ab-B1 --out runs/s41
# 过合并门(GEMV 桶 ≥+6% 服务级)→ 逐项单开归因
```

- **合并门**:trace GEMV 桶分类速率 ≥+6%(微测已 +8~13%,服务折损预留);
  e2e 无回归;S29 板 0 错位(位同项)/ 0.2255 内(非位同项)。
- 回滚:各 env 归零 = 现役逐位;`.so` 恢复 `.r3/so-backup/
  _rocm_C.s41-before.so` + md5。
- 注意:S40 的 `VLLM_DCU_FP16_GEMV_STREAM=1` 维持在役(本轮不动 lm_head)。

## 6. 执行顺序与并行度

```
D0  测量基建(0.5h)→ 锚点 + 探针模式
A0  wholea 位同 dump(0.5h)─┬─ 过 → A1 部署 ABAB(0.5d)
B0  探针(0.5d,A1 编译窗穿插)─┬─ ≥830 → B1 内核化(0.5-1d)→ B2 门 → B3 视况
C1  rows16 旋钮微测(0.5h)─── 过 → C2(0.5d)→ 数值门
DEP 合并臂 → ABAB → 归因(~3h)
```

- A 与 B 探针并行;C 全程穿插;每项落地后基线快照更新(md5/FP/step p50);
  终局写 `docs/result/z100-s41-final.md`(模板对齐 s40)。

## 7. 产物清单(施工即台账)

- diff:`.s41/s41-b1-csrc.diff`、`s41-c2-csrc.diff`(如立案)
- 数据:`docs/perf/data/s41-{anchor,b0-probe,a0-parity,a1-trace,
  b2-*,c1-rows16}.txt`、`runs/s41/e2e-*.json`、`runs/s41/parity-*`、
  `/mnt/game/s30c-prof/s41*_rank*.pt.trace.json.gz`
- `.so`:前后 md5 对(`.r3/so-backup/s41-before.md5`、
  `_rocm_C.s41-before.so`)
- 采纳记录:`.s41/{a,b,c}-adopt.md` 或 close(对齐 `.s40/` 体例);
  探针/门不过的 WP 各一页 close
- 数值证据:S29 板 json、max|Δ| 脚本输出、`torch.equal` dump
