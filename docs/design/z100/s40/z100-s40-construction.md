# S40 施工图:GEMV 读上限逼近(pf[4] + fp16 lm_head stream 化 + A-slice 双缓冲)

> 前置阅读:同目录 `z100-s40-design.md`(方案/门/风险)。事实基线:
> `docs/result/z100-s39-final.md`(账本修正)、`.s39/ceiling-probe.md`
> (读带宽阶梯)、`.s39/{a1,b1,b2,c1}-close.md`(已证伪项)。
> **部署基线**:无 MTP 臂在役(systemd `z100-vllm-mtp2`,`SPEC_OFF=1`,
> FP `2f3793deb97e138f`);`.so`:`_rocm_C.abi3.so` md5 `ca05f746`、
> `_C_stable_libtorch.abi3.so` md5 `54ccf2e2`;备份惯例 `.r3/so-backup/`。
> 本轮**只动 `_rocm_C`**(Track B 的 `_C_stable_libtorch` 不涉及)。

---

## 0. 通用纪律(承 S39 §0,增补两条)

1. **csrc 四步走**:备份在役 `.so` + md5 → 容器重编 → 微测(窗口)→
   数值对照 → 服务 smoke → e2e;失败恢复备份回 md5。容器构建后 `.so`
   属主 root,回滚用 `docker exec cp`。
   - `_rocm_C.abi3.so`(全部 Track):`docker start z100-regress;
     docker exec z100-regress bash -c 'cd /mnt/game/port/build-sync/temp &&
     ninja _rocm_C.abi3.so'`
2. **窗口口径(新增,强制)**:微测计时 = 30 rep 背靠背发射、单事件对包
   住全部 rep(`.s39/widen_tax.cu` mode-6/7 的 window 方法);每 rep 事件
   只作参考。S39 实测每 rep 口径把 o/out 类短核压 34%。
3. **锚点复核(新增)**:每轮微测前后各跑一次 K-A 参照(裸纯读),窗口值
   漂移 >3% 则本轮数据作废(设备状态守门)。
4. profile 窗口规程(强制):一发请求 + 立即 `/stop_profile`;楔死 →
   `sudo systemctl restart z100-vllm-mtp2`。
5. 数值门:凡声称位同者以 `torch.equal` 判定,失败 = bug;非位同改动走
   `max|Δ| ≤ 3e-4` + logprob 板(0.2255 nat,S29,`.s38/parity_probe.py
   --capture/--compare`)。现役路径行为差异先跑贪心流逐位对
   (`runs/s38/parity-s38-prod-restore.json` 为基线,允许重排级差异但须记录)。
6. 每步原始输出落 `docs/perf/data/s40-<wp>-<名>.{txt,json}`(不誊写);diff
   入 `.s40/*.diff`。
7. e2e:`runs/specbench/e2e_ts.py --tag s40-<wp>-<arm> --out runs/s40`
   (W1 贪心 96×3 + W2 采样 512×2;ABAB 必配对)。

先建目录与基线锚点:

```bash
mkdir -p .s40 runs/s40 docs/design/z100/s40   # design 已就位
md5sum vllm/_rocm_C.abi3.so | tee .r3/so-backup/s40-before.md5
cp -L vllm/_rocm_C.abi3.so .r3/so-backup/_rocm_C.s40-before.so
```

---

## 1. D0:测量基建(0.5h,先行,一切判据的仪器)

1. `gemv_bw_bench.py` 加 `--window` 模式(30 rep 单事件对)与
   `--shape lm_head`(38484×5120,copies=4:394 MB × 4 = 1.6 GB,defeat L2;
   注意 ≤2.5 GB 显存约束)。放 `.s40/gemv_bw_bench.py`(副本改造,不覆盖
   runs/s34nospec 原件)。
2. 锚点:窗口口径 K-A(gate_up 纯读)当日值落
   `docs/perf/data/s40-anchor.txt`,前后对照用。
3. trace 分类(parse_trace.py)补 lm_head 桶:
   `bf16_emu_gemv_dot2_kernel` 且 avg>300 µs 的发数单列。

```bash
HIP_VISIBLE_DEVICES=3 PYTHONPATH=/home/lyh/z100-vllm:/mnt/game/triton-gfx906/install \
DTK_ROOT=/mnt/game/host-runtime/dtk2604 LD_LIBRARY_PATH=$DTKL .venv/bin/python \
  .s40/gemv_bw_bench.py --window | tee docs/perf/data/s40-d0-baseline.txt
```

---

## 2. A:pf[4] 预取环(0.5d + 0.5h)

### A0 预检(0.5h,kill-fast)

```bash
# 临时改 pf[2]→pf[4](仅 m1 内核)编译,看 VGPR/寄存器与 spill:
docker exec z100-regress bash -c 'cd /mnt/game/port/build-sync/temp && \
  ninja _rocm_C.abi3.so && \
  /opt/dtk/llvm/bin/llvm-objdump -h ../temp/_rocm_C.abi3.so 2>/dev/null | head -3'
# VGPR 判定:编译日志/hipcc -W 无 spill;或对比改前后 .so 内 kernel 的
# VGPR 段(llvm-readelf --debug-info 不便时用:跑 A2 微测不降速即视为过)
```

过 → 留改造;不过 → A 轨降级"m1 专用 pf[4]",A3 关档。

### A1 内核改造(0.5d)

| 位置 | 改动 |
|---|---|
| fp8_emu_gemm.cu:945(m1)`uint4 pf[2][RW]` | env 门下 `pf[4][RW]`;预填 4 发(:948-958 同构展开) |
| :960 `VLLM_FP8_STREAM_M1_SLICE` 宏 | 消费序不变,仅 REFILL 步长改为 +4·kStreamSlice 旋转写 buffer |
| :1014-1016 peel 循环 | peel-by-4(4 个字面量 buffer 下标)+ 尾部(<4 slice)回退现序 |
| :721(MS 内核) | 同构,**仅模板实例 RW≤2 启用 pf4**(RW≥4 编译期保持 pf[2]) |
| env | `VLLM_DCU_FP8_GEMV_PF4=1`(默认 0 = 现役 pf[2];`static const` 一次读取) |

```bash
git diff csrc/rocm/fp8_emu_gemm.cu > .s40/s40-a1-csrc.diff
docker start z100-regress; docker exec z100-regress bash -c \
  'cd /mnt/game/port/build-sync/temp && ninja _rocm_C.abi3.so' | tail -3
```

### A2 门(0.5h)

```bash
# 窗口微测:pf2 臂 / pf4 臂(同进程只能跑一臂,分两次,前后各一个 K-A 锚点)
HIP_VISIBLE_DEVICES=3 ... .venv/bin/python .s40/gemv_bw_bench.py --window \
  | tee docs/perf/data/s40-a2-pf2.txt    # VLLM_DCU_FP8_GEMV_PF4=0
# PF4=1 臂同上 → docs/perf/data/s40-a2-pf4.txt
# 位同:七形状同输入 PF4=0/1 比对(.s40/a2_parity.py,torch.equal 全绿)
```

**门**:gate_up、o_out ≥ +6%,其余形状 ≤2% 回归;`torch.equal` 全绿。
不过 → 关档留证(env 归零即回滚)。

### A3 MS 内核跟随(视 A2)

同构改造 MS 内核(RW≤2 实例),重复 A2 门;RW≥4 实例明确不动。

---

## 3. B:fp16 lm_head stream GEMV(1d,下限最硬)

### B0 数值口径裁决(0.5h)

读 `csrc/rocm/bf16_emu_gemv.cu:79-127`:确认累加序 = 线程内 i=t+j·nt +
shared 树归约;与 stream 序(lane 内 chunk 串行 + shfl 树)**不同** →
数值门定案:`max|Δ| ≤ 3e-4`(对现役输出)+ S29 板 ≤0.2255 nat;
`docs/perf/data/s40-b0-verdict.md` 一页记录。

### B1 新内核(1d,~150 行)

| 项 | 内容 |
|---|---|
| 内核 | `fp16_gemv_stream_m1`(fp8_emu_gemm.cu,m1 骨架删减):W 为 uint4 = 8 fp16 元素;slice = 2048 B(128 uint4,lane 每 slice 2 发,1 KiB/请求);A 行 fp16 直入 LDS(2 KB/slice);**无 LUT/sl、无 a_s/w_s**;pf[4] 预取(A 轨成品形);epilogue = shfl 树 + bias |
| 绑定 | `torch.ops._rocm_C.fp16_gemv_stream_m1`(torch_bindings.cpp,SIG 对齐 bf16_emu_gemv 的 T/bias 可选) |
| 路由 | `vllm/model_executor/layers/utils.py:614` 选择器:`VLLM_DCU_FP16_GEMV_STREAM=1`(默认 0)且 `M==1 && N>=2048 && K%128==0` → 新 op;否则现役 bf16 路径原样 |
| 复核 | grid = (N+rows-1)/rows @rows=4 → lm_head 9621 blocks;LDS = 2 KB slice + 无 scale 表 |

```bash
git diff csrc/rocm/fp8_emu_gemm.cu csrc/rocm/torch_bindings.cpp \
         vllm/model_executor/layers/utils.py vllm/envs.py > .s40/s40-b1.diff
docker exec z100-regress bash -c 'cd /mnt/game/port/build-sync/temp && \
  ninja _rocm_C.abi3.so' | tail -3
```

### B2 门(0.5h)

```bash
# 微测(窗口):lm_head 形状,STREAM=0(现役 bf16 路径)vs =1(新内核)
HIP_VISIBLE_DEVICES=3 ... .venv/bin/python .s40/gemv_bw_bench.py --window \
  --shape lm_head | tee docs/perf/data/s40-b2-{off,on}.txt
# max|Δ|:七形状(含 lm_head)新内核 vs 现役输出,.s40/b2_delta.py
# 服务 smoke:重启带 STREAM=1 → W1 出字 + S29 板 capture/compare
```

**门**:窗口 ≥550 GB/s(+32% = −0.22 ms/step 下限;stretch 650);
`max|Δ| ≤ 3e-4`;S29 板 ≤0.2255 nat。不过 → 关档留证。

---

## 4. C:A-slice 双缓冲(time-box 0.5d,探针先行)

1. 探针(1h,kill-fast):widen_tax 加 mode"双 A-slice + 单 barrier"
   (纯读形,隔离变量)。**<+3% 直接关档**,不进内核。
2. 探针过 → m1 内核改造:LDS A-slice ×2 旋转(M=1 +2 KB;MS=8 上限
   33.3 KB,44 KB 门内),每 slice 一次 barrier;env
   `VLLM_DCU_FP8_GEMV_DBLSICE=1`(默认 0)。
3. 门:窗口 ≥+3%;`torch.equal`(序不变);并入部署合并臂。
4. diff:`.s40/s40-c2-csrc.diff`。

## 5. 部署与验收(合并批 → 归因)

```bash
# 合并臂(PF4=1 + STREAM=1 [+ DBLSLICE=1 若 C 过])注入 serve_deploy.sh
#   (临时段,标注 S40;终态回滚 = 删段)
sudo systemctl restart z100-vllm-mtp2    # 重启后核对日志三行 + FP 变更记录
# 服务 smoke:W1 出字 + 贪心流记录(允许重排级差异,记录首 diff 位)
# trace:一发 + 立即 stop(prefix s40a)→ GEMV/lm_head 分类速率
VLLM_PROBE_BASE=http://127.0.0.1:8000 python3 runs/s34nospec/profile_window.py
# e2e ABAB(B→A→B→A 各 3rep):
VLLM_PROBE_BASE=http://127.0.0.1:8000 .venv/bin/python \
  runs/specbench/e2e_ts.py --tag s40-ab-B1 --out runs/s40
# 过合并门(−0.8 ms)→ 逐项单开归因(A 臂 PF4 单开;B 臂 STREAM 单开)
# 数值:每臂 S29 板 + max|Δ| 复核;冷启动全形状 [1,2,4] smoke 零 fail
```

- **合并门**:e2e step p50 **−0.8 ms**(ABAB 配对中位);trace:GEMV 分类
  ≥650 GB/s、lm_head ≥600 GB/s。
- 回滚:各 env 归零 = 现役逐位;`.so` 恢复 `.r3/so-backup/
  _rocm_C.s40-before.so`(`docker exec cp`)+ md5 `ca05f746`。
- 注意:S39 的休眠 SMALLK 门仍在树内(默认 0),S40 重编的 `.so` 天然
  含它——部署记录 md5 时注明;SMALLK 保持 unset。

## 6. 执行顺序与并行度

```
D0  测量基建(0.5h)→ 当日锚点
A0  pf4 预检(0.5h)─┬─ 过 → A1(0.5d)→ A2 位同/微测 → A3 视况
B0  lm_head 口径(0.5h,与 A0 并行)→ B1(1d)→ B2 门
C1  探针(1h,A2 等编译窗穿插)→ 过 → C2(0.5d)
DEP 部署合并臂 → ABAB → 归因臂(~3h)
```

- A/B 同一 `.so`、同批重编,合并批先测总量,过门后逐项单开归因。
- 每项落地后基线快照更新(md5/FP/step p50);终局写
  `docs/result/z100-s40-final.md`(模板对齐 s39)。

## 7. 产物清单(施工即台账)

- diff:`.s40/s40-a1-csrc.diff`、`s40-b1.diff`、`s40-c2-csrc.diff`
- 数据:`docs/perf/data/s40-{anchor,d0-baseline,a2-pf2,a2-pf4,
  b2-off,b2-on,c1-probe}.txt`、`runs/s40/e2e-*.json`、
  `/mnt/game/s30c-prof/s40*_rank*.pt.trace.json.gz`
- `.so`:前后 md5 对(`.r3/so-backup/s40-before.md5`、
  `_rocm_C.s40-before.so`)
- 采纳记录:`.s40/{a,b,c}-adopt.md`(env、md5、FP、门数据、回滚一行命令);
  探针/门不过的 WP 各出一页 close(对齐 `.s39/*-close.md` 体例)
- 数值证据:S29 板 json(runs/s40/parity-*)、max|Δ| 脚本输出
