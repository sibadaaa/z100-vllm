# S14d 任务书(Lead 预设;S14c 结束后立即派发)

## 目标
回答 **"graph 模式的 token 流到底可不可依赖"**,给出**可执行的预热/部署配方**;若不可依赖,明确说"graph 不能上生产"。

## 背景(全部已有实测,不要重复验证)
* S14b(`docs/result/z100-14b-graph-vs-eager-result.md`,md5 `abf25e19…`)判定:**以"可解释数值"为主** —— 真值锚(未量化 4B、TP=1、CPU HF,首 token 4/4 一致)显示 **eager 差真值 0.0025 nat、graph 0.0023 nat、graph↔eager 仅 0.054 nat**(编译模式本身只值 0.05 nat);27B 上的 1–4 nat(尾部 10–12 nat)是**动态 per-token INT8 激活量化 × 64 层 GDN 递推**对这点差异的放大。**质量/精度参照 = eager**(逐位可复现:1024 token / 8/8 flip / 同引擎重复 Δ=0.000)。
* **但存在一个 A 形状缺陷**:graph **运行间不可复现** —— S14 的 P2(graph)给出 `'The'`/4.750 + 英文重复循环,用**逐字同配置、同 cache、同 AOT hash、同冷首请求**重跑得到 `'用户'`/gap 1.047(与 eager 仅差 0.06–0.20 nat)⇒ P2 是一次**坏 draw**。
* **H-AOT 只部分成立**:A18 重载 A5 的 cache 后 3/4 prompt 逐位一致,**第 4 个 prompt 从第 59 位起差到 9.5 nat**(`p2_zh_slip`)。
* **S14m/S11m 的既有口径**:`=1` 热启动 3/3 逐位一致(0/512);**冷编译产物 ≠ 它自己的磁盘重载**(314–318/490);生产配方因此要求"先空缓存预热一次,再以重载方式对外服务"。**S14d 要检验这条配方在 graph 下是否真的够。**
* **S14c(`docs/result/z100-14c-tp4-graph-memory-result.md`)已证**:① 峰值是**稳态**而非启动尖峰(peak−steady 仅 1469/4/215 MiB);② **AOT 重载把启动时间减半(351→230 s)但内存不降(+1.8%)**;③ graph 比 eager 每 worker 多 ≈1.4 GiB(TP 对齐后净 ≈4 GB);④ `MALLOC_ARENA_MAX=1` 无效。

## 必做实验
1. **E1 H-AOT 2×2**(现场编译 / 磁盘重载 × `benchmark_combo_kernel`/`combo_kernels` 开/关):
   每个格子至少 2 次独立启动,给出**逐 prompt 逐位对比**与首个分歧位置 + 间距(nat)。
   **注意**:这两个开关会**换 AOT key**,必须每次用独立 cache 根并标注 vintage。
2. **E2 `p2_zh_slip` 最小复现**:同 cache 根连跑两次(相同请求序),给出"第 59 位起差到 9.5 nat"是否稳定复现;若复现,继续二分(同一进程内两遍 vs 两次进程)。
3. **E3 conc=2 一致性**:S14b 记过一次 13.7 nat 的并发观测(未消解)。用 conc=2 同上下文对照,判定是调度效应还是数值效应。
4. **E4 预热配方的有效性检验**:按生产配方走一遍(空 cache 起 → 写产物 → 重启载入 → 服务),对 n≥8 prompt 比较"预热那次"与"服务那次"的输出是否逐位一致;给出**不一致率**与最坏间距。
5. **E5 收口"C2 vs P4e 的 12%"**(Lead 记在 `docs/result/z100-14c-lead-crosscheck.md`):同一 TP=4 **eager** 配置,分别在 **① 挂 16 GiB swapfile** 与 **② 只挂 `/swap.img` 4 GiB** 两种内存形态下各测一次 decode(n≥4)。若两次一致 ⇒ 该 12% 是抖动/vintage,结论里写明;**若①明显快于② ⇒ "生产机需要内存/swap 余量"这条对 eager 也成立,必须写进生产配方**。

## 达标线(与 S14c 一起构成"2.44× 能否上生产")
* 若 E4 的**不一致率 = 0**(或只有 <1e-3 nat 的舍入级差异)且 E1/E2/E3 不能复现坏 draw ⇒ 判"预热配方足够,graph 可上生产",并给出**配方 + 残余风险**;
* 若坏 draw 可复现且概率不可忽略 ⇒ 判"**graph 不能上生产**",并给出替代(TP=2 graph 84.22 是否同样有风险?必须一并给结论)。

## 纪律
* 不动生产容器 `qwen38-vllm`;不跑独立自编译 HIP 可执行文件;显式 `HIP_VISIBLE_DEVICES`(不用 `unset`);单一 GPU owner;杀进程 `pkill VLLM`。
* ACS 保持部署态(不切);swap 若用须记录 `/proc/swaps` 前后并**收尾还原**。
* 写作用域:`tools/dcu/s14d_*`、`docs/result/z100-14d-determinism-result.md`、`/mnt/game/s14d/**`;**不改** `docs/z100-todolist.md`/`docs/hygon_z100.md`/`docs/海光Z100-性能优化方案.md`(Lead 维护),不改 s14/s14b/s14c 脚本。
* **边做边写归档**,预算 ≤4 h。
