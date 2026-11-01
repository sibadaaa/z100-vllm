# S44 终局报告:push-AR 出口融合收尾(混合臂采纳)+ 碎核账本猎杀(归因收账)

> 2026-11-01。输入:`docs/design/z100/s44/{z100-s44-design,
> z100-s44-construction}.md`;基线:S43 终局(无 MTP 臂,systemd
> `z100-vllm-mtp2`,`SPEC_OFF=1`,S40/S41 段,双 `.so` md5
> `54ccf2e2`/`f72bb39d`)。**本轮双 `.so` 零改动**(md5 全程不变,备份
> `.r3/so-backup/*s44-before*`),唯一采纳项 = 两个 env 开关。
> 判定台账:`.s44/{ar-adopt,n-close}.md` + `docs/perf/data/s44-*.txt` +
> `runs/s44/*`。

## 0. 结果一览(逐门验收)

| 门 | 判据 | 实测 | 判定 |
|---|---|---|---|
| 锚点 | 漂移 ≤3% | read 792.9 GB/s(S43 792.6,+0.04%);gate_up 563.2(−0.8%) | PASS |
| A0 全融合(FUSION=1) | d_res=0 硬 + 输出连贯 | op 级对拍全绿(d_res=0×64,d_out≤9.8e-4)但**输出乱码**——S38 capture 损坏原样复现 | **FAIL,维持禁用** |
| A0 混合臂(_MLP=0) | 输出连贯 | 推理+正文双段正常 | PASS |
| A1 S29 板 | 无乱码/无复读/长度正常 | 96×3 正常;首分歧 @2,\|Δlp\|=2.88e-3 nat(S34 @46/0.0107 同档),分叉后两臂各自连贯 | PASS |
| A2 微核窗 | fused ≤ pair×0.90 | rows1 0.855-0.861 / rows4 0.868-0.871 / rows8 0.860-0.864;rows12 0.908 边缘(pair 当日异常快,fused 39.9 µs 与 S37/S38 同档);decode 实际命中 rows1-4 | PASS |
| A3 e2e ABAB | W1 ≥+2%;norm 桶降;TTFT 无回归 | W1:B1 41.73 / A1 40.03 / B2 40.70 / A2 39.59 → 配对 **+4.25% / +2.81%**;norm 类目账 **1.635→0.392 ms/step(−1.24)**;TTFT +17 ms ≈ A 臂自身组内漂移(A1→A2 +18 ms),判噪声 | **PASS** |
| A4 采纳 | env 入 serve_deploy + DEPLOY | `VLLM_DCU_CUSTOM_AR_FUSION=1` + `_MLP=0` 注释块 + 禁令;systemd 重启生效,health 200 | done |
| N0-N4 | 见 `.s44/n-close.md` | scope 5.189→3.907;N1 关档(宿主消失)、N2a 归因修正后顺延 S45、N2b/N4 关档、N3 三核源码级归因 | 收账 |

**净效果**:`.so` 不变;env 增量 = `VLLM_DCU_CUSTOM_AR_FUSION=1` +
`VLLM_DCU_CUSTOM_AR_FUSION_MLP=0`(回滚 = 注释两行 + restart,逐位恢复)。
**当前速度/耗时(采纳态,systemd 臂)**:W1 贪心 96 tok **38.6-40.7 t/s**
(单臂跨轮带,本箱 ±7% 时钟漂移内;ABAB 配对口径净收益 **+2.8~+4.3%**,
折 −1.02 ms/token)、W2 采样 512 tok **38.2 t/s**(37.6-39.3,稳定)、
**TTFT 0.73 s**(持平)。decode GPU busy scope 账 5.189→**3.907 ms/step**。

## 1. 本轮学到了什么(证据强度)

1. **S38 全融合 capture 损坏第三次确认且未随时间自愈**(强证据):op 级
   DEBUG 对拍全绿 + 服务输出乱码,与 S38 §2 根因链逐点吻合(eager 干净 /
   piecewise 坏,嫌疑 = 第二 auto_functionalized op 的残差 clone/weight
   绑定)。S44 设计把修复希望押在"A0 重验"上——重验的结果是否定的。
   **两出口全融合从此为禁用项**(DEPLOY.md 载明),除非先修 S38 §2。
2. **混合臂(+2.8~+4.3%)是本轮全部净收益**,且恰好吃到高频半:
   消除的 attn 出口 norm 是账本里最大的类(22.9 µs 级),MLP 出口保留的
   是 5.9 µs 级——"×0.6 收益"的设计折算实际兑现为 ×0.83(1.24/1.5)。
3. **A0 DEBUG 自检分支本身是生产级 bug(新发现,三次复现)**:
   `parallel_state.py:855` 的对拍条件含逐 rank 数据相关的
   `tensor.float().abs().max().item() > 0`,满足者多打一发
   `_all_reduce_out_place`(集体)→ AR flag 序列跨 rank 失步 → 服务
   挂死(py-spy 四 rank 同卡 conv1d fallback 首个同步点,GPU 自旋,
   无 page fault)。**修法(S45)**:条件改 rank 无关或对拍移出
   capture-warmup 段;修复前生产/ABAB 一律 DEBUG=0。
4. **平台教训:裸 `torch.cuda.graph` 捕获 fused-AR op 触发 hycu XID:81
   page fault 且楔死驱动队列**(独立 capture 复验脚本两次复现;楔死后
   后续 boot 在不同阶段挂死,`hy-smi --hcureset -d N` ×4 恢复)。
   该脚本的捕获方法未复现 vLLM piecewise 前置条件,结论仅作旁证;
   运行须在服务下线窗且跑完即 reset。
5. **碎核归因的正确姿势是 inductor artifact 源码反查**(`strings
   artifact_* | grep "def triton_..."`),比 profiler stack 关联可靠
   (后者在本构建上 External id 断裂)。三项碎核全部解密:
   per_fused_1 = GDN 门控 rmsnorm(已三合一)、poi_fused_0 = **纯零填充
   (每步每层,疑似可省)**、poi_fused_5 = conv 输出 chunk 拆分。
6. **N2a 宿主修正(纠正设计图)**:mul_silu_slice = GDN 层 dense MLP 的
   SiluAndMul,宿主 GEMV = `fp8_emu_mm_block_w8a16_auto`(W8A16 族,
   gate_up [8704,5120]),**不是**设计所写的 bf16_emu_gemv_dot2(时序
   11.7 µs 读不动 89 MB,亦证伪)。折入 = W8A16 核心块映射改造,留 S45
   并附位同复现配方(`.s44/n-close.md` §1)。

## 2. 数字表(全部当日,证据文件)

| 指标 | 基线(FUSION=0) | 采纳(混合臂) | 证据 |
|---|---|---|---|
| W1 贪心 96(配对) | 40.03 / 39.59 | 41.73 / 40.70 | runs/s44/e2e-s44-ar-{B1,A1,B2,A2}.json |
| W2 采样 512 | 39.03 / 38.92 | 39.22 / 38.93 | 同上 |
| TTFT med | 0.710 / 0.728 | 0.733 / 0.739 | 同上 |
| norm 类目账 | 1.635 ms/step | 0.392 | s44-a3-buckets-abab.txt |
| scope 账(Track N 类) | 5.189 ms/step | 3.907 | s44-n0-ledger.txt / s44-a3-ledger-B2.txt |
| AR 形态(发/step) | 126.6 plain | 58.9 fused + 59.3 plain | 同上 |
| 微核 per-op 比 | 1.0 | 0.855-0.871(rows1-8) | s44-a2-micro.txt |
| 锚点 GB/s | 792.6 | 792.9 | s44-anchor.txt |

楔死窗判读纪律执行记录:两次 profile 均楔死(stop_profile 超时),AR 桶
含自旋污染不可比(fused med 47 µs vs 微测 38-39),busy interval-union
21.6→22.9 ms/step 不作数;门判读一律 e2e 配对 + 类目账(施工图纪律 2/4)。

## 3. 移交项(S45)

1. **N2a:W8A16 gate_up 折入 SiluAndMul**(0.24 ms + 48 发空隙,位同
   配方在 `.s44/n-close.md` §1;落点 csrc/rocm/fp8_emu_gemm.cu + 重编
   `_rocm_C`,build-sync2);
2. **poi_fused_0 零化审计**(0.29 ms,63 发/step;先语义审计再动手);
3. **A0 DEBUG 分支修复**(rank 无关条件化),修后全融合二分可重启
   (S38 §2 下轮入口:splitting_ops / 影子缓冲);
4. MTP 复役(用户既定后置)在采纳态基线上叠加 S37 GO 口径。

## 4. 产物清单(施工即台账)

- 判定:`.s44/{ar-adopt,n-close}.md`(本轮两个关档/采纳判定档)
- 工具:`.s44/board.py`(S29 板重建)、`.s44/launch_arm.py` 恢复 +
  `base_cmdline/base_environ` 快照重建(pid 944 活体导出)
- 数据:`docs/perf/data/s44-{anchor,a0-debug,a0-capture,a1-board-*,
  a2-micro,a3-ledger-B2,a3-buckets-abab,a3-busy,n0-ledger,n3-attr}.txt`
- e2e/板:`runs/s44/{e2e-s44-ar-*,e2e-s44-final*,board-*,parity-*}.json`
- `.so`:md5 全程不变(`.r3/so-backup/s44-before.md5` + 双 before 副本)
- 服务变更:`serve_deploy.sh` S44-A4 注释块(两行 env + 禁令)+
  `.s34/DEPLOY.md` S44 节(回滚 = 注释 + restart)
