# S82-b 结果：V 侧 LDS 压缩两臂 —— X4 合成臂（V 单缓冲 + X-REG + 3 CTA）内核级 −12%

日期：2026-11-11。施工图：`docs/design/z100/s82b-v-lds-shrink/z100-s82b-v-lds-shrink-construction.md`
台账：`.s82b/`；原始数据：`runs/prefill-lctx/s82b/`。

---

## 0. 三句话

1. **WP0 判决：4 KiB 粒度风险不存在，真实粒度是 512 B**，3 CTA 的 LDS 上限是
   **21504 B**（不是 21845）。X1（20608→20992）与 X2（20736→20992）都放行。
2. **WP4 合成臂（X1 单缓冲 + X-REG + `launch_bounds(256,3)`）在内核级把四个形状
   全部压掉 11–13%，且 `out_md5` 与基线逐位相同**：131k 1003.0→881.3 ms、
   8k 63.7→55.5 ms、decode M=1 0.4918→0.4339 ms、verify M=12 1.9374→1.6899 ms。
   这是本专项唯一过门的臂，也是历史上第一次真正把 int8 内核推上 3 CTA/CU。
3. **WP5 服务级配对完成，X4 过门**：131k TTFT **232.7 → 216.3 s（−7.1%）**、
   32k −2.0%、TPOT c=1 −1.1%、c=4 0.0%，**accept_len 逐位一致**（8k 在单发
   噪声带内）。中途踩到一次与本专项无关的全局编译缓存污染（服务 decode 55× 慢，
   用生产 `.so` 原样复现），已定位并绕开，详见 §5。

---

## 1. 静态账（开工复核）

| 项 | 值 |
|---|---|
| 生产 `.so` | `.s82/rocm_c_release/_rocm_C.abi3.so` md5 `5fd07f21` |
| `attnd_inner_kvint8.hip` md5 | `a430bdd7`（S82 采纳态，PV unroll 4） |
| 内核实例 | `ILb1ELi4E` vgpr 120 / spill 0 / LDS 28800（bucket 29184 → 2 CTA） |
| LDS 组成 | K 8192 + V 16384 + Q 4096 + ks 128 = 28800 |
| 基线计时（复现 S82，9 iters 中位） | 131k 1003.0 / 8k 63.7 / dec 0.4918 / ver 1.9374 ms |

`out_md5`：prefill 131k `278905b1`、8k `7dd8029d`、decode `f244f342`、verify `e715a0bb`。

---

## 2. WP0：LDS 分配粒度（判决：512 B，不是 4 KiB）

方法：`__launch_bounds__(256,3)` + `__shared__ char pad[PAD]` 的 spin 内核，
一次发 3×60=180 个 block；3 CTA/CU 时 1 波（≈0.198 ms），2 CTA/CU 时 1.5 波
（≈0.39 ms）。源码与输出：`.s82b/wp0/`，完整结论：`.s82b/wp0-lds-granularity.md`。

| PAD | API occ | 实测驻留 | ceil512 |
|---:|---:|---:|---:|
| 20608 (X1) | 3 | **3** | 20992 |
| 20736 (X2) | 3 | **3** | 20992 |
| 21503 / 21504 | 3 | **3** | 21504 |
| **21505** | 3 | **2** | 22016 |
| 21845 / 21760 | 3 | **2** | 22016 |
| 28800（生产） | 2 | 2 | 29184 |

- 11 个点与 `ceil512` 模型逐点吻合，0 反例；粒度排除 2 KiB/4 KiB
  （21504 不是它们的整数倍却实测 3 CTA）。
- **施工图 §2 的"20864→24576 会被 4 KiB 桶吃掉"没有发生**；但"21845 是关口"
  也不成立：`(21504, 21845]` 全部落到 22016 桶 → 2 CTA。**X1/X2 的 20992 桶安全**。
- 文档口径：`rocminfo` 只报 `GROUP segment 64 KB`、`Alloc Granule 0KB`；
  DTK 头文件里唯一的 4 KB 粒度是 `hipMemAllocation*` 的线性内存粒度，与 LDS 无关。

---

## 3. WP1：X-REG（删 `pr[16]`，PV 内重算 shuffle）

- 改动：`.s82b/arms/xreg_u4.hip`（生产内核基础上，删 16 条预取 shuffle 与
  `float pr[16]`，PV 循环内 `__shfl(p, gbase | j2, 16)`）。
- 静态：**vgpr 120/121 → 90/92，spill 0，LDS 28800**。
- `out_md5` 四形状与基线逐位相同。
- 计时（oc2 bound，9 iters 中位）：131k **991.97（−1.1%）**、8k 62.60（−1.7%）、
  decode 0.4902（−0.3%）、verify 1.901（−1.9%）。
- 裁决：**单臂不达标**（vgpr 90 > 85 的 WP4 关口），但 −1% 门通过，且它是
  X1/X2 转 3 CTA 的寄存器基础。备刀（acc 8+8 接力）未启用——由 WP4 的
  `launch_bounds(256,3)` 直接把 vgpr 压到 69/71。

---

## 4. WP2/WP3/WP4

### WP2 X1 = V 单缓冲（`.s82b/arms/x1_singlebuf.hip`）

- LDS 28800 → **20608**（bucket 20992 → 3 CTA）；vgpr 121 / spill 0；
  双屏障按 S79 结论**完整保留**。
- 计时（仍是 oc2 bound，即 2 CTA）：131k 1017.4（+1.4%）、8k 63.65（+0.0%）、
  decode 0.4971（+1.1%）、verify 1.9419（+0.2%）。`out_md5` 逐位相同。
- 裁决：**单臂不过收益门**（staging/PV 重叠消失，+1.4% prefill），但它把
  LDS 让到 20992 桶，是 WP4 的另一半。

### WP3 X2 = V 存 int8（`.s82b/arms/vint8.hip`）

- LDS 20736（bucket 20992）；vgpr 69/71，spill 0（V 的 16 个反量化 cvt 从
  staging 移到 PV 后，寄存器反而更低）。
- 静态 ISA（`ILb1ELi4E`）：`ds_read_b128` 16→13、`ds_write_b128` 7→5、
  总指令 1015→978（**−3.6%**，未超施工图 +5% 红线）。
- 数值（施工图放宽门）：`numgate` 48 例 rel ≤1.7e-3；`kl_gate` KL 3.4e-5–6.7e-5；
  `wrapper_gate1616` 6/6 rel ≤4.7e-4。**`out_md5` 与基线不同（符合施工图预期）**。
- 计时（oc2 bound，2 CTA）：131k 1007.3（+0.4%）、8k 63.20（−0.8%）、
  decode 0.4848（−1.4%）、verify 1.9254（−0.6%）。
- 裁决：**单臂不过收益门**（PV 每元素 1 条 cvt 把 3 CTA 的账吃掉）；
  3 CTA 版本（X2+oc3）与 oc2 完全同速，证明它已经吃到 3 CTA 的收益但被 cvt 抵消。

### WP4 X4 = X1 + X-REG + `launch_bounds(256,3)`（`.s82b/arms/x4_x1_xreg_oc3.hip`）

- 静态：**LDS 20608（bucket 20992 → 3 CTA）/ vgpr 69–71 / spill 0 / priv 0**。
- 计时（9 iters 中位，`out_md5` 四形状与基线**逐位相同**）：

| 形状 | 基线 | X4 | Δ |
|---|---:|---:|---:|
| prefill 131k | 1003.0 | **881.3** | **−12.1%** |
| prefill 8k | 63.70 | **55.46** | **−12.9%** |
| decode M=1 @131k | 0.4918 | **0.4339** | **−11.8%** |
| verify M=12 @131k | 1.9374 | **1.6899** | **−12.8%** |

- **3 次交替 A/B（12 次测量，同进程外交替跑）**：8k −12.97%、131k −12.13%、
  decode −11.42%、verify −12.86%，离散 <0.1%，`out_md5` 全部相同。
- **归因对照（关键）**：同一份 SASS、只把动态 LDS 申请 +4096 B（→ 25088 桶 →
  2 CTA/CU）：131k **1042.2 ms**、decode 0.5020、verify 1.9712——即 **还原成
  基线慢速**。同一`.so`、同一代码，唯一变量是驻留 CTA 数 ⇒ **−12% 全部来自
  3 CTA/CU，不是代码微调**。
- 另一对照 `launch_bounds(256,2)` 版本（`x4_ctrl_oc2`）：vgpr 仍 69、LDS 仍
  20992 → **自动也是 3 CTA**，计时与 X4 完全一致（881.6/55.5）——说明占用率由
  资源决定，`minBlocksPerMultiprocessor` 只是给编译器的预算提示。

**结论：WP4 过全部内核门（decode ≥ −3% 的"路线活着"判据以 −11.4% 远超）**，
且 prefill 也 −12%，因此不需要 decode 专用实例（施工图 §6 的 tps_hint 分发
后续项可以不做——本臂对 prefill 同样是净收益）。

---

## 5. WP5：服务级配对（X4 vs 生产基线，同语料同口径）

### 前置事故与修复

本机在 07:22 出现一次**全局编译缓存污染**，与本专项内核无关：

- 我用当时路径写错的 `.s82b/run_serve.sh`（指向不存在的 `.s82/arms/build_x4`）
  启动过一次服务 → `vllm/_rocm_C.abi3.so` 悬空 → `import vllm._rocm_C` 失败 5 次 →
  vLLM 把 block-scaled FP8 GEMM 编成 **Triton 软件内核**并写进新缓存 key
  `4b4f462333`；之后每次重启只要读该 key，服务 decode 就从 **37.7 → 2088 ms/step
  （55×）**，`fp8_utils.py:1070` 警告 20 条，编译图里 `fp8_emu_mm_block` 0 次
  （正常 832 次）。
- 该现象**用未改动的生产 `.so`（md5 `5fd07f21`）就能复现**，内核级计时反而完全
  正常（131k 1003.0 ms），因此与 S82-b 的任何改动无关。
- 修复：删掉被污染的 key，并按 `VLLM_DISABLE_COMPILE_CACHE=1` 启动（放弃跨重启
  编译缓存，换回正确内核选择）。修复后生产 `smoke` 37.5 ms/step、8k TTFT 2.15 s，
  与 S82 收工态一致。

> **口径说明**：下面两侧都在 `VLLM_DISABLE_COMPILE_CACHE=1` 下测得（同一
> `.s82/corpus.md`、同一 `s20_single_probe`、`max_tokens=8`、`temperature=0`），
> 所以配对有效；但它不等于 S82 §4 的旧口径（那次带编译缓存），跨报告不可直接比。

### 结果

| 指标 | 基线（`5fd07f21`） | X4（`x4_x1_xreg_oc3`） | Δ |
|---|---:|---:|---:|
| TTFT 8k | 8130 ms | 8272 ms | **+1.7%**（单发噪声带内；复测基线 8365 ms） |
| TTFT 32k | 33.08 s | 32.44 s | **−2.0%** |
| TTFT 131k | **232.7 s** | **216.3 s** | **−7.1%** |
| TPOT c=1 | 36.98 ms | 36.59 ms | **−1.1%** |
| TPOT c=2 | 85.69 ms | 86.62 ms | +1.1%（S82 已记录的双峰噪声带） |
| TPOT c=4 | 174.05 ms | 174.08 ms | 0.0% |
| accept_len | 1.415 / 1.153 | 1.415 / 1.153 | **逐位一致** |

- **131k TTFT 与 32k TTFT 是清晰的净收益**（−7.1% / −2.0%），与内核级 131k
  prefill −12% 的方向一致（服务级只摊到 attention 那一段，其余是 GEMM/通信）。
- decode 段服务级只有 −1.1%，因为 4097-token 上下文里 attention 占比远小于
  prefill；不是 kernel 收益消失（内核 decode M=1 是 −11.8%）。
- 8k 在单发噪声带内（同一基线两次测得 8130 / 8365 ms，跨度 2.9%）。
- **accept_len 与基线逐位一致**（服务级红线）；`out_md5` 内核级四形状逐位一致。
- 结论：**X4 过服务级门**（TTFT 不劣、131k 明显改善、accept_len 不变）。

### 生产态处置

- `.so` 已指回 `.s82/rocm_c_release/_rocm_C.abi3.so`（md5 `5fd07f21`，未改）；
- 服务用 `KV_DTYPE=int8_per_token_head` + `MIN_CTX=4096` 重启，`int8-KV decode
  inner loop enabled`（head=256, block_size=1616, segments=256）日志在位；
- 因编译缓存事故，当前生产以 `VLLM_DISABLE_COMPILE_CACHE=1` 运行（代价是重启
  多约 8–10 min 现场编译）。**下次维护窗口应重建一次干净缓存**，恢复带缓存启动。

## 6. 产物清单

| 文件 | 内容 |
|---|---|
| `.s82b/wp0-lds-granularity.md` + `.s82b/wp0/` | LDS 粒度探针（源码、二进制、逐点输出） |
| `.s82b/arms/xreg_u4.hip` / `xreg_oc3.hip` / `xreg_u2.hip` | WP1 |
| `.s82b/arms/x1_singlebuf.hip` | WP2 |
| `.s82b/arms/vint8.hip` / `x5_vint8_oc3.hip` | WP3 |
| `.s82b/arms/x4_x1_xreg_oc3.hip` | **WP4 冠军** |
| `.s82b/arms/x4_ctrl_oc2.hip` / `x4_dynpad.hip` | 归因对照（2 CTA） |
| `.s82b/isa/census.py` | 逐实例 vgpr/spill/LDS/bucket/CTA/opcode 清点 |
| `.s82b/run_kernel_shapes.sh`、`ab_kernel.sh`、`ab_summary.py` | 内核计时与配对 A/B |
| `runs/prefill-lctx/s82b/` | 全部计时 JSON 与 A/B 日志 |

## 7. 建议的下一步（未做）

1. **采纳 X4**：`x4_x1_xreg_oc3.hip` 覆盖 `csrc/rocm/attnd_inner_kvint8.hip`，
   重建 `.so`（md5 会变，需登记 `serve_deploy.sh` 与回滚），先跑
   `.s82b/run_kernel_shapes.sh` 复验四形状 md5，再进服务级。
2. WP5 前先修环境：把 `torch_compile_cache` 里所有 `code_hash != 7ab06af5`
   的 key 清掉，并确认服务冷启动后 `fp8_utils:1070` 警告数为 **0**；
   否则服务级数字不可用。
3. 服务级口径照抄 S82 §4（`.s82/corpus.md`、TTFT 8k/32k/131k、TPOT c=1/2/4、
   accept_len 逐位）。
