# Z100 S68 注意力专项总图 — 结果

日期：2026-11-08。执行依据 `docs/design/z100/s68-attn-final/`
（设计图 + 施工图）。台账 `.s68/`，原始数据 `runs/prefill-lctx/s68/`。
环境与 s63 完全一致（GPU 实测经 `with_serve_env.py`，服务 :8000）。
**生产态已复原并验证**：`vllm/_rocm_C.abi3.so` → `.s58/`（md5 8d0bb215），
服务 argv == s63 收尾臂，`/health` 200，全程 journalctl AER/DPC = 0。

## 三 Track 判定表

| Track | 内容 | 判定 | 关键数字 |
|---|---|---|---|
| B1 | 2D 路径 TILE=16 sweep（108 配置 ×3 形状） | **闭案，不采纳** | 8192: 97.3→92.5 ms (+5.1%)；4096: +3.1%；2048: +0.3%——收益随 ctx 缩短衰减，短段不成立 |
| B2 | 16k 门槛拆账 | **闭案，不立案** | 16k 头 chunk（2D）8279 ms vs 8192 全程 2D 7846 ms，差 ~0.4 s < 1.5 s 立案门 |
| C | hycu `noretry` 前置门 | **闭案暂缓** | `/sys/module/hycu/parameters/noretry` 存在（当前 -1）；262k 冒烟需重启窗口，已留执行清单 |
| A | 6 q-head/token 合并 CTA（寄存器 Q 驻留） | **止损（-72% 实测证伪）** | 131k/7488: 基线 1184 ms vs merge 4208 ms；max\|d\|=0 但 ISA 确认 qrow 整行溢出 scratch（528 B/线程 + 128 spill store + 每 tile 16 dword 重载） |

## 各 WP 结果（详细见 `.s68/` 台账）

- **WP-B1**（`.s68/wp-b1-sweep.md`）：TILE=16 只在配 BM≥64 时有效且
  仅限大 ctx；反哺 Track A 的 launch 查表档位 (BM64,T16,W4,S1) 已记录；
- **WP-B2**（`.s68/wp-b2-verdict.md`）：`MIN_CTX 16384→8192` 维持现状；
- **WP-C**（`.s68/wp-c-verdict.md`）：hycu 认 `noretry` 参数——设计图
  "hycu 不认即闭案"的预期**被推翻**，实验值得在下一个维护窗口执行
  （GRUB + `noretry=0` + 重启 + 262k ×2 reps，判定矩阵照施工图）；
- **WP-A0/A1**（`.s68/wp-a0-verdict.md`、`wp-a1-verdict.md`）：
  - A2 静态审**修正了设计图**：原 lane=(token,dim-slice) 的 softmax
    归约轴跨 warp（bits 6/7），不可行；改为 CTA=16 q-token × 1 q-head
    （grid.y=nq），总 CTA 数不变、归约轴保持连续；
  - 数值门全绿（max|d|=0，fp32 参考 ≤1.5e-5）——排列/索引正确；
  - 性能门惨败：**-72%**。ISA 复核定位根因：跨 2048-trip 循环持有
    128 VGPR 的 Q 行数组被编译器整体溢出 scratch，每 tile 重载 +
    串行等待——**"Q 寄存器驻留"假设在本工具链上不成立**；
  - 两轮止损条款触发，回退完成。

## 过程事故与教训（已修复，建议入纪律）

1. **op 签名不可动**：给 `gfx906_attnd_decode_inner` 加参数导致 stale
   `torch_bindings.cpp.o` 引用旧符号 → 整库加载失败 → Python 静默回落
   Triton 路径，**测出过一版假数据**（两臂同为 2511 ms）。修复：改用
   C++ dispatch 内 live getenv（decode 旋钮同款），零 ABI 变化。
   **微测前必须先 load-test .so**；
2. sweep 工具改造要先确认网格真的变了（第一版跑了未含 tile=16 的网格）；
3. spill 检测流程（fatbin→code object→`llvm-readelf -n` 读 vgpr/scratch
   →`llvm-objdump --triple=amdgcn--amdhsa -d`）已在本专项打通，可复用。

## 对设计图 §0 表的回填

Track A 天花板终局：**不可行**（-72% 实测 + ISA 根因）。S63 遗留的
四条线（S64/S65/TILE=16/iacopPBK 对读）至此全部收口：
- KV 量化 / 稀疏 / pk_fma / Q-in-LDS：此前已证伪（设计图 §2 排除表）；
- **Q 寄存器驻留（S65 合并头）：本专项证伪**——排除表新增一行
  "跨长循环的大数组寄存器驻留（本工具链会溢出）"；
- 2D TILE=16：不采纳（增益限于大 ctx 且 <5%）；
- 16k 门槛：维持现状（差值 < 立案门）；
- noretry：待维护窗口的判定性实验（唯一开口项）。

## 工件清单

| 文件 | 内容 |
|---|---|
| `.s68/wp-b1-sweep.md` / `wp-b2-verdict.md` / `wp-c-verdict.md` / `wp-a0-verdict.md` / `wp-a1-verdict.md` | 各 WP 台账 |
| `.s68/attn2d_sweep16.py` / `a1_merge_ab.py` | sweep 与 A/B 微测工具 |
| `.s68/attnd_inner.a0.hip` / `attnd_inner.mergeqtok.v1.hip` / `v2.hip` | 基线留档 + merge 内核两版 |
| `.s68/rocm_c_rebuild/` | s68 构建产物（已与生产 symlink 断开） |
| `runs/prefill-lctx/s68/attn2d_tile16*.json` / `a131k_wp_a1*.json` / `stages_*.json` | 全部原始数据 |
| `docs/result/z100-s68-prefill-speed.md` | prefill 各阶段速度测量 |
| 设计图/施工图 | 状态行待改"已结案"（见下） |
