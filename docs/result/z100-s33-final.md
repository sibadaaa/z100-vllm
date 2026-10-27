# S33 终局报告：fp8 约束内对齐天花板 —— WP 状态与裁决清单（2026-10-26）

> 施工依据：`docs/design/z100-s33-fp8-scope-construction.md`（WP 状态表已同步
> 更新）；设计依据：`docs/design/z100-s33-fp8-scope-design.md`。
> 基线（勿重推导）：s30c-full 现役臂 step p50 = 29.48 ms（N=3：29.24/29.48/30.05）。

## 0. 结论速览

| 项 | 结果 |
|---|---|
| 采纳进现役的变更 | **0 项**（现役 = s30c ctrim 臂，行为与开工前位同） |
| WP0 空隙分账 | 完成：真实核间空隙 ≈**3.8 ms/round**（~1000 处 × 5 µs），分账 100% 闭合 |
| WP1 whole-A GEMV | 位同 PASS、+9~17%/形状带宽，判据 A 未达 → 按图停，**待裁决** |
| WP2 one-shot AR | 分配修复 ✓、e2e **−3.40 ms** 超验收门，但贪心流两次落世界外 → 按图停，**待裁决** |
| WP3 融合 | 不执行（依赖项全停 / <0.3 ms 规则） |
| WP4 attention 分段 | 维持现状（分段已产线激活，扫档无过门增益） |
| WP5 lm_head/KV fp8 | 按图不默认执行，备料见 §5 |
| 今日现役复测（终局 N=3） | **29.331 / 29.228 / 29.201**（中位 29.228，落基线带内）✓ 无漂移 |
| GPU 现场变更 | qwen-image21 容器已按用户裁决停止（`docker start qwen-image21` 可恢复）；其 :8000 端口与 4 卡 HBM 归 s33 使用 |

## 1. WP 状态表（与施工图同步）

| WP | 判据 | 状态 | 证据 |
|---|---|---|---|
| WP0 空隙分账 | 覆盖 ≥90%；只测量 | **完成** | `docs/result/z100-s33-gap-accounts.md`、`.s33/gap_accounts.json` |
| WP1 whole-A | 位同；带宽门；e2e −1.5 ms | **位同 PASS；判据 A 未达 → 按图停** | `docs/result/z100-s33-wholea-gemv.md` |
| WP2 one-shot AR | e2e 门 −2 ms；贪心流零翻转；长跑 | **速度超门；贪心流两次落世界外 → 按图停** | `docs/result/z100-s33-oneshot-ar.md` |
| WP3 融合 | per-op −20%；e2e −0.8 ms | **不执行**（依赖项全停；GDN CFG 按"单项 <0.3 ms 不采纳"出局） | §2.3/§3 |
| WP4 split-KV | per-op −30%@7.5k；e2e −0.3 ms | **维持现状**（分段已激活；seg32/64 扫档无过门增益） | `.s33/s20_s33wp4-seg*.json` |
| WP5 lm_head/KV fp8 | eval + 用户裁决 | **按图不默认执行**（§5 备料） | — |
| 终局 | s33-full N≥3 | 完成（无采纳项 = 现役复测） | 本档 §0 |

## 2. WP0 分账（要点）

- 口径：torch profiler CPU decode 窗口（20 rounds），busy = 窗口内 GPU 活动
  **区间并集**（kdiff (name,grid) 求和会把跨窗核重复计入，30.74 > wall，弃用）。
- 插桩 wall 30.49 = busy 24.86（81.5%）+ head 0.54 + 核间隙 5.10（1023 处 ×
  5.1 µs）→ 分账 100%。
- ITER_DETAILS（未插桩）引擎 round p50 28.63 → **profiler 工件 ≈1.86 ms**，
  真实核间空隙 ≈3.8 ms/round；host 步间 ≈0.7 ms。
- nccl 行跨 trace 方差大（s30cm1b 8.56 ms vs s33gap 4.95 ms p50）→ 勘误①的
  窗口工件实证；WP2/未来窗口必须用本窗 min 去偏。
- 对 WP3 的含义（空隙大 → 融合升值）：方向正确，但融合的两个落点分别随
  WP2（AR 核 norm）与 WP1（whole-A epilogue）停。

## 3. WP1：whole-A M=1 GEMV（停点 + 裁决材料）

- 位同（判据 B）**PASS**：六形状 M=1 max|Δ| 与现路径逐值相同（qkv 4.77e-07 /
  o 6.10e-05 / gateup 1.53e-05 / gdn_in 3.05e-05 / gdn_out 6.10e-05），M=8 不受影响。
- 带宽（判据 A）**未达标**（rows=4 最优档、干净机器复核 ±0.5%）：
  gateup 586.9（门 600）/ qkv 402.5（450）/ o 361.3（400）/ gdn_in 440.8（450）/
  gdn_out 363.6（450）。
- 折算 e2e ≈ −0.9~−1.1 ms/round < −1.5 ms 采纳门。
- WP-f3：A 读 4-way bank 冲突确认，且为位同序约束下的**结构性**下限
  （b128 + 16B 对齐不可消除；pad 1×uint4 无效）；与生产核同款，非回归。
- **裁决项 A**：(a) 弃（保持默认关）；(b) 条件采纳（rows=4 + WA=1 跑一次
  服务 A/B 实测，−1 ms 级收益换 TTFT/流风险复验）。内核默认关，二进制含核
  （md5 见 §6），回滚 = env 不设。

## 4. WP2：gfx906 one-shot AR（停点 + 裁决材料）

- 分配修复 ✓（finegrained，历史 OOM 阻塞点根治）；size 门 ≤64 KiB ✓；
  capture 无 hang ✓；e2e **−3.40 ms**（25.929 vs 29.331）超 −2 ms 验收门。
- **贪心流硬门两次落世界外**（`.s29/cmp_capture` 判定）：
  - graph 臂（部署形态）：`8283,1023,1023,…` 确定性劣化（3 rep 位同地错）；
  - eager 臂（隔离）：3 rep 三种流，非确定性。
  根因指向 one-shot 核的 flag/data 先后序在 gfx906 fine-grain PCIe 上不被兑现
  （graph replay 路径另有 registered-buffer 陈旧读问题）。
- 微测判据口径注记："10 KiB ≤25 µs"为服务级口径，bench 上下文（冷时钟独立
  进程）连 NCCL 都是 155 µs；相对改善 oneshot vs nccl = 23~41%（无扰动）、
  13~38%（skew20）。
- **裁决项 B**：(a) 弃（维持 RCCL，接受 0 收益）；(b) 立项修复（最小面：
  producer 侧 `__threadfence_system()` gfx906 条件编译 + replay 的
  `register_graph_buffers` 路径核查 → 重走零翻转门 + 长跑 30 min）。潜在收益
  −2.5~−4 ms/round，是本图最大单项。

## 5. WP5 备料（按图不默认执行；随裁决 A/B 一并定优先级）

- lm_head fp8（W8A16 路线现成）：907→~460 µs，e2e −0.45 ms；改 top-1 logprob
  → 需贪心 diff 板 + top5 偏移报表 + eval，单独裁决。
- KV fp8 LUT：先微测证伪"ITL +41% = 反量化未融合"假设（设计 D6），未开工。
- 优先级建议：裁决项 B（AR 修复，−2.5~−4 ms）≫ 裁决项 A（−1 ms）> WP5
  （−0.45 ms，改输出分布）。理由：AR 是唯一没被带宽墙限死的杠杆。

## 6. md5 / 指纹 / 回滚台账

| 件 | 值 |
|---|---|
| `_rocm_C.abi3.so`（含默认关的 whole-A 核） | `a9325bffadb7b487d53b2641e6babf25` |
| `_rocm_C.abi3.so.bak-s33`（= s30c 现役原版） | `f5d279c5b5a47dcfaa443d0bc3eea4a0` |
| `.bak-s30c`（更早备份） | `f42d006481efa8e12504a1b73bbabdb5` |
| `_C.abi3.so`（GDN） | 未动（s30c：`34d8b074775fa547a6098864f40eff23`） |
| 新增 env（全部默认关/空） | `VLLM_DCU_FP8_GEMV_WA`、`VLLM_CUSTOM_ALLREDUCE_ALGO`（KNOBS 已入） |
| 现役 policy 指纹 | `.s30c/serve-s33-final-r3.log`（canonical，与 s30c 相同） |
| 回滚 | 二进制 cp bak-s33；臂脚本只用 `.s30c/`；WP2/WP4 env 不设 |

## 7. 产物归档

- `.s33/`：gap_accounts.json、wp1_bench_*.json（4）、wp1_so_md5.log、
  wp2_car_oneshot{,_skew20}.json、capture-s33wp2-{ar2,ear}.json、
  s20_s33-*.json（e2e 8 份）、gap_accounts.py、restart.sh、probe.sh、
  capture.py、launch_ctrim_ar.sh、launch_ctrim_eager_ar.sh、
  launch_ctrim_eager_ar.sh、wp0_iterdetails_summary.txt。
- traces：`/mnt/game/s30c-prof/s33gap_rank*.gz`（4 rank）。
- 文档：`docs/result/z100-s33-gap-accounts.md`、
  `docs/result/z100-s33-wholea-gemv.md`、`docs/result/z100-s33-oneshot-ar.md`、
  本档。

---

## 8. 修复后更新（2026-10-27，用户裁决"先修 WP2，再试 WP1"之后）

### WP2 修复 = 采纳 ✓（详见 z100-s33-oneshot-ar-fix.md）

- 根因：gfx906 PCIe **出站读不观察目标 L2 脏行**（pull 核读陈旧）；
  **入站写对目标 L2 一致**（旗标握手一直工作的原因）。
- 修复：push 模式单核 `cross_device_reduce_push`（本地激活直读 → plain 16B
  入站推写每个 peer 的 fine-grain region → volatile mid 屏障 → 本地归约）；
  staging 拷贝删除；python 门限 oneshot/1stage；捕获不走 graph-pool P2P。
- 验收：贪心流 = **基线 r1 世界逐位一致**（零翻转门 PASS）；e2e
  **27.26/27.26/26.70 ms**（N=3 跨两次重启，基线 29.23/29.48 → **−2.0~−2.6 ms**）；
  TTFT 728–743 ✓；**长跑 30 min 104 探测（312 请求）0 失败 0 hang** ✓。
- `_C_stable_libtorch.abi3.so` md5：`7dcce4f23fc2f939028648255a0b6f02`
  （修复前 bak-s33 = `0855b2a3…`，cp 回即回滚；env 三变量不设 = RCCL）。

### WP1 修复尝试 = 不达门，两变体留档（详见 z100-s33-wholea-gemv.md §7）

- LDS 变体（bank 冲突结构性 4-way）：+9~17%/形状，门未达。
- **L2 直读变体**（免冲突，同值同序位同）：+5~10%/形状，门仍未达
  （gateup 564/600、qkv 387/450）。
- 结论：门需换算法类（split-K GEMV，改累加序 → 零翻转硬门 + 裁决）；
  施工图 WP4 的 split-K 触发条件（qkv/o < 450）已成立，留待裁决。

### 当前现役态

- 服务 = ctrim 臂 + push AR（三 env）+ `.s33/launch_ctrim_ar.sh`，
  step p50 = **26.70 ms**（基线 29.48 → **−2.8 ms，−9.4%**），TTFT 正常。
- 回滚优先级：env 三变量不设（一步回 RCCL）→ 二进制 cp bak-s33。
- qwen-image21 容器保持停止（用户裁决）；`docker start qwen-image21` 可恢复，
  但其 :8000 端口映射会与现役 AR 臂冲突（服务须换回 .s30c 端口或停容器）。
