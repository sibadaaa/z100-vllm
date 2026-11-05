# S41 终局报告:GEMV 第二轮(wholea 采纳 + 多流发射证伪 + o/out 几何证伪)

> 2026-10-31。输入:`docs/design/z100/s41/{z100-s41-design,
> z100-s41-construction}.md`;基线:S40 终局(无 MTP 臂,systemd
> `z100-vllm-mtp2`,`SPEC_OFF=1`,S40 ADOPTED 段 `VLLM_DCU_FP16_GEMV_STREAM=1`,
> `.so` md5 `f72bb39d`)。**本轮 `.so` 零改动**(md5 全程不变),唯一采纳项
> 为 env 单开关。判定台账:`.s41/{a-adopt,b-close,c-close}.md` +
> `docs/perf/data/s41-*.txt`。

## 0. 结果一览(全部 WP,逐门验收)

| WP | 门 | 实测 | 判定 |
|---|---|---|---|
| D0 测量基建 | (工具,无门) | `.s41/widen_tax.cu`(+mode 12/13/14 + 逐字生产 wholea 15/16 + mode8 越界修复);锚点 read-only 792.9 GB/s(S40 793.2,漂移 0.04%);当日基线 gate_up 532.7(在役 m1 窗口) | **完成** |
| A0 wholea 位同 | 七形状 `torch.equal` | 7/7 绿 ×2(默认 env + 完整服务 env) | **PASS**(`s41-a0-parity.txt`) |
| A1 部署 ABAB | trace GEMV 桶 ≥+6%;S29 板 0 错位;e2e 无回归 | 单发中位 **+6.27%**(43.4→40.8 µs);S29 **0 错位/0.0107 nat**;W1 配对 **+1.21 t/s**、W2 +0.82,TTFT 不变;GEMV 桶 **−0.67 ms/step**(中位口径) | **采纳上线**(`.s41/a-adopt.md`) |
| B0 多流探针 | K-M 或 K-N 窗口 ≥830 | **K-M 562.3(< K-H 基线 581.7,−3.3%);K-N 633-636,kPf 3→4 一致 −1.2~−1.6%**;K-E 782.0/K-F 723.6/K-J 610.5 与 S40 逐点吻合 | **整轨关档**(`.s41/b-close.md`) |
| B1/B2 内核化 | (视 B0) | 不施工;`M1_G4`/`WA4` 两 env 不立项 | 关档 |
| C1 rows16 | o/out ≥ +10%(≥298) | **attn_o −9.9%、gdn_out −9.3%**(226.6/252.0 vs 251.4/277.8);rows8 ≈ rows4 | **关档,不立案 C2**(`.s41/c-close.md`) |
| DEP 合并部署 | 合并臂 = A 唯一采纳项 | WA=1 在役 + S40 段维持;5 次 restart 全部按预案执行 | **完成** |

**净效果**:`.so` 不变(`f72bb39d`);env 增量仅 `VLLM_DCU_FP8_GEMV_WA=1`
(回滚 = 注释 + restart,位同路径恢复)。fp8 GEMV 桶 11.11 → 10.44 ms/step
(中位口径,−0.67);W1 贪心当日带 A 37.95-38.31 → B 38.73-39.96 t/s
(配对 +3.2%)。**1 采纳、2 关档、0 代码改动。**

## 1. 本轮学到了什么(证据强度)

1. **组发射 + barrier 结构第三次证伪**(B0,强证据):忠实复刻探针发射
   时序的内核化真实形态(K-M:组首 4 背靠背 + 逐 slice A-stage 2 barrier)
   窗口 562.3,连现役 pf[2] 环(581.7)都不如。S40 结论"可移植方向是多流
   并发而非环深"需要再加半句:**在每 slice 有 A-stage barrier 的内核里,
   多流并发本身也不可移植** —— barrier 税把 4 发在飞的收益全部吃掉。
2. **wholea 环深已到头**(B0,双实现互证):零 barrier 骨架下 kPf 3→4 在
   骨架(636.4→636.4)和逐字生产文本(643.0→633.1)两种实现里一致回退。
   wholea 的 W 预取在无 barrier 循环里已完全隐藏,加深只加寄存器。
3. **wholea 采纳的机理注脚**(A 轨旁证):零 barrier 形(643)vs m1 形
   (582)窗口 +10.5%,与 S33(+13.3%)、S40(+11.6%)三measure同向;
   服务级折损到 +6.27%(单发中位)——设计图预留的折损余量恰好够用。
4. **探针工程教训(方法论,两次翻车两次修复)**:
   - `pf[]` 数组任何**动态下标**(tail 的 `b % kPf`)都会让 HIP 编译器
     生成灾难代码——wholea 形全族(含逐字生产文本)虚低 2.3×(249 vs
     586 GB/s),改字面量下标(生产 peel 形)后与 .so 实测对齐。
     **以后内核探针一律禁用运行时 ring 下标。**
   - S40 遗留的 mode 8(1 GiB 顺序模式)对 `out[row]` 越界写(行数
     ≫ N)在 o_out 段撞崩进程;本轮修复(输出缓冲按行数分配)。
5. **服务级按臂确定分叉异常**(A 轨,机制未明,不阻塞采纳):op 级位同
   三重验证全绿,但服务 greedy 流按 WA env 确定性分叉——实为一次近平手
   argmax 翻转(数字合并边界),两流 ~6 token 后重新收敛;S29 板 32 token
   内 0 错位。已排除 split-K 抢占、WA 其他消费方、full-env dump、内核
   总账差异。**该异常同时重新解释 S40 备忘的"长 prefill 环境漂移"**——
   S42 优先核验。
6. **profile 规程升级需求**:S38 教训的"一发 + 立即 stop"预案在本轮
   两臂全部楔死(n=0/320 s,health 000,需 restart);trace 本身完整,
   但 execute_context 步界破碎使桶中位数不可比——**单发中位/内核总账
   才是楔死窗口下的可采纳口径**。根因待查(可能与 today 的引擎状态有关)。

## 2. 部署现状(终态)

- 在役:`.so` md5 **f72bb39d**(不变);`serve_deploy.sh` 新增 **S41 段**:
  `export VLLM_DCU_FP8_GEMV_WA=1`(回滚 = 注释 + restart)。
  S40 段(`VLLM_DCU_FP16_GEMV_STREAM=1`)维持。
- S29 板:A vs B **0 token mismatch**,max|Δtop1| = 0.0107 nat(≤0.2255)。
- FP 工具缺口延续:`VLLM_DCU_FP16_GEMV_STREAM` 与 `VLLM_DCU_FP8_GEMV_WA`
  均未入 `tools/dcu/dcu_policy_fingerprint.py` key 表(FP 不反映
  S40/S41 开关),待补。

## 3. 模型现状(操作者口径)

- 服务:qwen3.8-27b,TP4,无 MTP(SPEC_OFF=1),wholea 臂在役;当日 ABAB
  W1 贪心 A 37.95-38.31 / B 38.73-39.96 t/s,TTFT 0.706-0.75 s。
- decode 步构成(本轮 trace 修正):fp8 GEMV 桶 11.11 → **10.44 ms/step**
  (单发中位 × 256 发;均值口径 11.57→11.14);lm_head 808.5 µs(两臂同,
  S40 采纳维持);其余桶未重测。

## 4. 下轮候选(按本轮证据排序)

0. **GEMV 桶的诚实上限再次下修**:wholea 后 643 GB/s(窗口)vs 裸形
   782;剩余差距 = A 读 LDS/MAC 循环本身(4-way bank 冲突是结构性的,
   S33 §5 已证不可消除)。**GEMV 内核结构空间已穷尽**(组发射、环深、
   双缓冲、宽 lane、位技巧全部证伪),桶内剩余 ~0.6 ms(o/out 271 级)
   的小形状只剩 epilogue/固定开销类假设,风险收益比差。
1. **服务级按臂分叉根因**(本轮异常,优先级最高的一笔"欠账"):它影响
   一切贪心流逐位对账纪律(S29 板的 0.0107 nat 就是它的影子),也可能
   是 S40"环境漂移"的真身。建议:同一进程内翻转 WA 前后各抓一次
   decode logprob 对账 + prefill-only 对账,把扰动定位到 prefill/decode。
2. **push-AR 3.76 ms**(现第二大桶)与 norm/碎核 ~2.6:S38 fused 关档的
   eager 路线可再探(需先解 piecewise capture 兼容)。
3. **profile 楔死根因**:两臂 100% 复现(n=0/320 s),影响一切 trace
   采集;建议抓 start_profile 前后的 engine 日志与迭代计数。
4. FP 工具补 `VLLM_DCU_FP16_GEMV_STREAM` + `VLLM_DCU_FP8_GEMV_WA` key
   (两行)。

## 5. 台账(施工即台账)

- diff:无(`.so` 零改动);`.s41/widen_tax.cu` 为探针源(S40 版 +
  mode 12-16 + 两处修复)
- 数据:`docs/perf/data/s41-{anchor,b0-probe,c1-rows16,a0-parity,
  a1-trace,a1-e2e}.txt`;`.s40/gemv_bw_win_s41-*.json`(锚点/C1 三臂);
  `.s41/a-wholea-{ref,new}.pt`、`.s41/a-fullenv-{ref,new}.pt`(位同 dump)
- 板:`runs/s38/parity-s41-a-{a1,b1}.json`
- trace:`/mnt/game/s30c-prof/s34nospec_rank3.{1793447156597201903(A),
  1793448006522872837(B)}.pt.trace.json.gz`;`runs/s41/steps-s41-a-{a1,b1}.json`
- e2e:`runs/s41/e2e-s41-ab-{B1,A1,B2,A2}.json` + `parity-s41-ab-*.json`
- `.so`:改前备份 `.r3/so-backup/_rocm_C.s41-before.so` +
  `s41-before.md5`(f72bb39d,与现役一致——本轮无换库)
- 判定:`.s41/{a-adopt,b-close,c-close}.md`
- 服务在役:`.s34/serve_deploy.sh` S41 段(`VLLM_DCU_FP8_GEMV_WA=1`)
