# S35 施工图：MoE 架构适配 —— 实施任务书（2026-10-27）

> **交给实施 agent 的完整任务书**。设计依据：`docs/design/z100/s35/z100-s35-moe-design.md`
> （**先读**；风险编号 R1–R11 / N1–N7 均出自它与其引用的 MoE 调查
> `docs/result/z100-moe-feasibility-investigation.md`）。
> 范围：正确性 + 基线 + 内核性能优化。卡分配执行时定（WP0 门禁）。
> 产物目录 `.s35/`；结果文档 `docs/result/z100-s35-*.md`（每 WP 一篇，
> 结构：目标/改动/原始证据/门禁判定/未完成项/对后续影响/复现方式）。
> **写作用域**：新增 `.s35/**`、`tests/kernels/test_gfx906_moe.py`、
> `tools/dcu/s35_*.py`、`docs/result/z100-s35-*.md`；WP7 起方可动
> `csrc/quantization/gguf/**`（+CMake 注册）与 `vllm/envs.py`（新 env）。
> **只读参考**：`.s30c/**`、`.s29/**`、`tools/dcu/**`（可复制改造，不原地改）。
> **红线**：不 kill 生产服务（:8000）；不动 DTK 库 / `$DATA_ROOT/host-runtime`；
> 不动 `.s30c` 现役臂。

---

## 0. 环境须知（先读，全是实测踩过的坑）

1. **栈确认 + GPU 自检（每次开工先跑）**：
   ```bash
   cd $WORKDIR
   bash tools/dcu/use_stack.sh status          # 必须在 DTK 树（本工作目录）
   export DTK_ROOT=$DATA_ROOT/host-runtime/dtk2604
   set +u; source tools/dcu/env.sh; set -u
   .venv/bin/python -c "import torch; x=torch.randn(8,8,device='cuda'); print(torch.cuda.is_available(), torch.cuda.device_count())"
   hy-smi                                        # 4 卡 VRAM%（生产在役时 ~91%）
   ```
   torch 报 `libgalaxyhip.so.5` 缺失 = env 没加载，不是机器坏了。
2. **生产服务红线**：`Qwen3.8-27B-FP8 TP=4`（:8000）占满 4 卡。
   **任何情况下不 kill、不重启、不降它的 util**。MoE 服务一律用
   `HIP_VISIBLE_DEVICES` 指定空闲卡 + 端口 8101/8102。无空闲卡 → 按 WP0
   判据停并报告，**不得**为腾卡动生产。
3. **pkill 禁忌**：不要在内联命令里手写 `pkill -f "VLLM::Worker"`——模式会
   匹配自己的命令行把自己杀掉（S33 实测）。清理自己起的 MoE 服务：
   `pgrep -f "entrypoints[.]api_server.*810[12]"` 拿 pid 后 `kill -9 <pid>`。
4. **csrc 构建换装（仅 WP7 用到，C7 纪律）**：GGUF MoE 内核在
   **`_C_stable_libtorch`**：
   ```bash
   docker exec z100-regress bash -c 'cp $DATA_ROOT/port/build-sync/temp/_C_stable_libtorch.abi3.so \
     $DATA_ROOT/port/build-sync/temp/_C_stable_libtorch.abi3.so.bak-s35 && md5sum $DATA_ROOT/port/build-sync/temp/_C_stable_libtorch.abi3.so*'
   docker exec -e MAX_JOBS=5 z100-regress bash -lc 'bash $DATA_ROOT/port/build-sync.sh _C_stable_libtorch'
   ```
   竞态坑：build log 只有 `[1/6] Running hipify` 且 md5 未变 = hipify/ninja
   竞态，**原样重跑即恢复**；禁手改 temp 产物或删 stamp。改后 `nm -DC` 确认
   新符号。
5. **profiler collect 会 wedge 服务**（S33 须知 4）：`s27_five_phase_profile
   --phase collect` 后立刻重启服务再继续。
6. **指纹工具**：新增任何 `VLLM_*` knob（WP7 的 `VLLM_DCU_MOE_GEMV`）必须
   同步加进 `tools/dcu/dcu_policy_fingerprint.py` 的 KNOBS。
7. **度量工具全套现成（勿重写）**：
   - trace/kdiff：`tools/dcu/s27_five_phase_profile.py --phase collect` +
     `.s29/kdiff.py`
   - 贪心流捕获/比对：`PYTHONPATH=tools/dcu .venv/bin/python .s29/capture.py
     --tag X --out .s35` / `.s29/cmp_capture.py a.json b.json`
   - 下载：`tools/dcu/s9_par_dl.py <url> <out> <total_size> <sha256> [threads]
     [chunk_mb]`（HF `resolve` 端点 Content-Length 短报先例 ⇒ 必须带
     total_size + sha256 校验）
   - e2e 探针：`.s30c/probe.sh` 是 27B 专用口径，**复制改造**为
     `.s35/probe.sh`（固定 5 prompt（3 中 2 英，各 96 tok）× 3 reps，
     输出 JSON：tok/s、TTFT、token 流）
8. **dmesg 监控**：每个 e2e WP 起跑前 `dmesg | tail -3 > .s35/dmesg_pre_<tag>`，
   跑完后 diff；**出现 XID/GPU fault 立即停服保全现场**（dmesg 全量 +
   服务日志进 `.s35/`）。
9. **每 WP 通用验收流**：数值/微测 → 服务 smoke（environ + 指纹核验）→
   e2e（N≥2 重启）→ 文档归档。改模型输出的项加零翻转硬门（S14b/BM16 流程）。
10. **卡号口径**：`HIP_VISIBLE_DEVICES` 用 torch 卡号（0/1=Z100L，
    2/3=Z100-DP4X）。

---

## WP0：资源门禁与备料（~0.5 天，可与其他 WP 部分并行）

### 0.1 GPU 自检与卡分配（第一门禁）
1. 按须知 1 自检；`hy-smi` 记 4 卡 VRAM% 进 `.s35/wp0_env.json`。
2. 空闲卡判据：VRAM <10% 且 `pgrep -af 'api_server|vllm'` 无该卡绑定进程。
3. **卡分配决策树（写入 `.s35/wp0_cards.json`）**：
   - 无空闲卡 → **停，报告用户**（附 hy-smi 证据），等待窗口；期间只做
     0.2 下载与 WP1 的**代码编写**（不跑）。
   - 仅卡 0 → WP1/2/3 单卡卡 0；WP4/WP9 标记顺延。
   - 卡 0+1 → 同上 + WP9 可行。
   - 卡 2/3 空闲 → WP4 可在卡 2 执行。
4. 判据：`wp0_cards.json` 落盘 + 决策明确。**无卡不得进入任何 e2e WP**。

### 0.2 权重下载（与 GPU 等待并行）
1. **OLMoE**：`allenai/OLMoE-1B-7B-0924`（safetensors，~13.8 GB）。
   HF API 取每文件 size + sha256（`https://huggingface.co/api/models/...`），
   逐文件 `s9_par_dl.py` 下到 `$DATA_ROOT/models/OLMoE-1B-7B-0924/`，
   sha256 全过才算完成。
2. **GGUF 35B**：`unsloth/Qwen3.6-35B-A3B-GGUF`（~20 GB）。先列仓库文件
   （HF API），按仓库 README 推荐选量化档；多档时选总量 ≤24 GiB 中精度最高
   者。同法下载到 `$DATA_ROOT/models/Qwen3.6-35B-A3B-GGUF/`。
3. 判据：两个目录 sha256 全过；`wp0_env.json` 追加下载记录（URL/size/sha/耗时）。
4. 文档 `docs/result/z100-s35-wp0-resources.md`（卡分配 + 下载台账）。

## WP1：MoE 数值门禁（关闭 R4；~1 天）

### 1.1 前置核实（对不上就停下报告）
```bash
grep -n "def fused_experts" vllm/model_executor/layers/fused_moe/fused_moe.py
sed -n '270,300p' vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py   # apply → fused_experts 调用形态
grep -n "class RoutedExperts" vllm/model_executor/layers/fused_moe/routed_experts.py
.venv/bin/python -c "import aiter" 2>&1 | head -2    # 必须 ImportError（门禁前提）
```
确认 `fused_experts` 的入参签名（w13 布局 `[E,2N,K]`、w2 `[E,K,N]`、
topk_ids/topk_weights 形状、activation 枚举），据此接线。

### 1.2 实现 `tests/kernels/test_gfx906_moe.py`（纯新增）
- 参考实现（手写 torch expert loop，**fp32 累加**）：
  `out[m] = Σ_t topk_w[m,t] · ( silu(x[m]·w13[e_t,:N,:]ᵀ) ⊙ (x[m]·w13[e_t,N:,:]ᵀ) ) · w2[e_t]ᵀ`。
- 用例矩阵：E=8, topk=2, K=512, N=256；M∈{1,8,64,256}；fp16 输入；
  silu_mul；路由用固定随机种子构造（含"同 token 选中同专家"与"均匀分散"
  两种 topk_ids 分布）。
- 断言（对 fp32 参考）：`max(|Δ|/(|ref|+1e-3)) < 2e-2` 且
  `cosine(out, ref) > 0.9995`；同一输入跑两遍**位同**；
  测试文件头部断言 aiter 不可 import（保证门禁在无 aiter 环境可跑）。
- 可选延伸（不阻塞门禁，时间富余再做）：合成 WNA16 权重（int8 gs=128）
  走 `fused_experts` 的 wna16 分支对照；GGUF 合成块对照。做了就写进文档。
- 运行：`HIP_VISIBLE_DEVICES=<空闲卡> .venv/bin/python -m pytest
  tests/kernels/test_gfx906_moe.py -v`。

### 1.3 判据 / 回滚 / 文档
- 判据：全绿（含位同）；记录实测误差分布进文档。
- 回滚：删除新文件。
- 文档 `docs/result/z100-s35-wp1-moe-gate.md`。

## WP2：Tier A e2e + 首个本地 MoE 基线（~1 天）

### 2.1 服务脚手架（复制 `.s30c/` 模式改造）
- `.s35/launch_wp2.sh`：`HIP_VISIBLE_DEVICES=<卡0> ... vllm serve
  $DATA_ROOT/models/OLMoE-1B-7B-0924 --tensor-parallel-size 1 --port 8101
  --gpu-memory-utilization 0.9 --max-model-len 8192`（dtype 跟随权重 fp16；
  不加 `--enforce-eager`，先走默认图模式）。
- `.s35/restart.sh`：仿 `.s30c/restart.sh`（pkill 用须知 3 的安全模式 →
  起服 → 等 /health → 打印 policy 指纹 + environ 核验）。
- `.s35/probe.sh`：按须知 7 改造（5 prompt × 3 reps，JSON 进 `.s35/`）。

### 2.2 任务
1. 起服（dmesg 前置快照）；图模式若 FULL 捕获失败 → 降级
   `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'`；再失败 →
   `--enforce-eager`（S7c 后安全）。**记录最终图模式**。
2. 判据 A（正确性）：/health 200；≥3 条中英文请求文本连贯（人工判读，
   截图/原文进文档）；输出无 NaN；dmesg diff 0 故障；KV cache 正常上报。
3. 判据 B（基线）：`probe.sh` 3 reps × N≥2 重启：decode tok/s、TTFT、
   conc8（`vllm bench serve --port 8101 --num-prompts 32 --max-concurrency 8`
   或 probe 内置并发档）；数字进 `.s35/wp2_olmo_baseline.json`。
4. 停服、dmesg 后置 diff。

### 2.3 回滚 / 文档
- 回滚：杀 8101 进程即可（无代码改动）。
- 文档 `docs/result/z100-s35-wp2-tier-a-baseline.md`（含图模式、基线表、
   与上游 Issue #29 传闻的对照说明）。

## WP3：Tier B GGUF 复验（生产候选定案；~1 天）

### 3.1 任务
1. `.s35/launch_wp3.sh`：`HIP_VISIBLE_DEVICES=<卡0> vllm serve
   $DATA_ROOT/models/Qwen3.6-35B-A3B-GGUF/<文件> --tensor-parallel-size 1
   --port 8102 --gpu-memory-utilization 0.9 --max-model-len 8192`
   （GGUF 量化从文件自动识别；若不识别加 `--quantization gguf`）。
   复用 `.s35/restart.sh`（改端口/启动脚本）与 `probe.sh`。
2. 判据 A（正确性）：ready + 中英文长生成（≥512 tok）连贯 + 无 NaN +
   dmesg 0 故障。**GDN 混合架构（R10）：dmesg 监控全程开**。
3. 判据 B（基线）：decode tok/s（fork 时代口径：无 MTP 37.6 / MTP 50.5
   tok/s——本机预期落在 37.6 附近，显著偏离要解释）；TTFT；N≥2 重启。
4. **卡 2 对照（仅当卡 2 空闲）**：同权重卡 2 起服（端口 8103），同 prompt
   集贪心流捕获（`.s29/capture.py`），与卡 0 臂 `cmp_capture.py` 比对；
   先测卡 0 臂自身 N=2 重启噪声底（S14b 方法论），再判分歧。
   判据：噪声底内 → DP4X 行为级放行（与 WP4 互补）；显著分歧 → 卡 2/3
   锁定 MoE，记录机制线索。
5. 停服、dmesg diff。

### 3.2 回滚 / 文档
- 回滚：杀 8102/8103 进程。
- 文档 `docs/result/z100-s35-wp3-tier-b-gguf.md`（含卡 2 对照表，若执行）。

## WP4：C++ MoE 内核 VGPR 审计（关闭 R1；~0.5 天，需卡 2）

### 4.1 前置
1. shim 冒烟（DTK 栈首次使用，**必做**）：按
   `$DATA_ROOT/f24-investigation/hipshim/README.md` 的一行冒烟 +
   `selftest/mock/run_mock_tests.sh`（无 GPU 部分）全过。
2. 卡 2 空闲（WP0 决策树）；否则本 WP 顺延（不阻塞 WP5 之后的流程，
   但 README 的 DP4X 结论保持"待验证"）。

### 4.2 任务
1. 卡 2 起 WP3 的 GGUF 服务，env 叠加：
   ```bash
   LD_PRELOAD=$DATA_ROOT/f24-investigation/hipshim/libhiplog.so \
   HIPLOG_DIR=.s35/wp4-kerntrace HIPLOG_VGPR_WARN=0
   ```
   （`WARN=0` = 全量 numRegs 表；`HIPLOG_FILTER=ggml_moe,moe_wna16,topk_softmax,moe_align,moe_sum`
   可缩小日志，但 REG/ATTR 行永不过滤。）
2. 跑完整 probe（decode + prefill 两域都触发）后停服。
3. 汇总 `$HIPLOG_DIR/hip-*.log` 的 ATTR 行 → `.s35/wp4_vgpr_table.json`：
   重点 `ggml_moe_a8`、`ggml_moe_a8_vec*`、`topk_softmax`、
   `moe_align_block_size`、`moe_sum`（`moe_wna16_gemm` 若 WP9 启动后补）。
4. 判据：全部 `numRegs ≤ 203` → **R1 关闭**，卡 2/3 放行 MoE；
   任一 ≥248 → 记录，结合 WP3 卡 2 行为级对照定案
   （DTK 栈用户态带 DP4X 支持 ⇒ 不必然故障；**两证据冲突取保守**）。
5. 长跑 30 min（卡 2，decode 连发）0 故障作为行为级佐证。

### 4.3 回滚 / 文档
- 回滚：去掉 LD_PRELOAD 即可（shim 只读不写）。
- 文档 `docs/result/z100-s35-wp4-vgpr-audit.md`（全表 + 定案）。

## WP5：MoE 性能剖析与墙位鉴定（只测量，~1 天）

### 5.1 任务
1. 对象臂：WP3 的 35B GGUF（主）；OLMoE（对照，Triton 路径）。
   每臂：起服 → `probe.sh` 热身 →
   `tools/dcu/s27_five_phase_profile.py --phase collect --profile-prefix
   s35moe --trace-dir $DATA_ROOT/s35-prof --profile-steps 20 --out .s35
   --tag <arm>`（**collect 后立即重启服务**，须知 5）。
2. 分账（`.s29/kdiff.py` 口径）：MoE 相关内核逐条 us_per_round：
   专家 GEMM（`ggml_moe_a8*`）、`topk_softmax`、`moe_align_block_size`、
   `moe_sum`、gate/共享专家线性、GDN、attention、AR。
   覆盖差值（wall − GPU busy）按 S33-WP0 三分法（skew/核间空隙/其他）。
3. **有效带宽**：decode 专家 GEMM 的权重字节/耗时
   （字节账 = 激活专家数 topk=8 × 3 × 2048 × 512 × 每权重字节，
   按 GGUF 实际量化位宽算；对照稠密侧天花板 GEMV ~650–700 GB/s、
   d2d ~771 GB/s）。
4. 产出 `.s35/wp5_moe_wall.json` + 墙位结论：decode 域与 prefill 域各
   定一个主墙（带宽 / 算力 / launch）。

### 5.2 判据 / 文档
- 判据：分账覆盖 ≥90%；两域墙位结论明确；**不改任何代码**。
- 文档 `docs/result/z100-s35-wp5-wall.md`。**本 WP 是 WP6–WP9 的立项依据**。

## WP6：Triton 路径调优配置（~1 天，WP5 之后）

### 6.1 实现 `tools/dcu/s35_moe_tune.py`
- 扫 `fused_experts` 的 tile 空间（BLOCK_SIZE_M∈{16,32,64,128}、
  BLOCK_SIZE_N∈{64,128,256}、BLOCK_SIZE_K∈{64,128,256}、
  GROUP_SIZE_M∈{1,4,8}、SPLIT_K∈{1,2}，num_stages 固定 2（ROCm 强制）），
  M 档 {1,2,4,8,16,32,64,128,256,512}；每档取最优写 JSON
  （格式对齐 `vllm/model_executor/layers/fused_moe/configs/` 现有文件）。
- 目标文件：`E=64,N=1024,device_name=AMD_GFX906.json`（OLMoE，dtype=None）。
  输出到 `.s35/tuned/`，经 `VLLM_TUNED_CONFIG_FOLDER=.s35/tuned` 注入。

### 6.2 判据 / 回滚 / 文档
- 判据：OLMoE e2e decode 改善 **≥5%**（N≥2）且其它 M 档不回退
  （微测全档表）；否则弃用（不提交配置）。
- 回滚：不设 `VLLM_TUNED_CONFIG_FOLDER`。
- 文档 `docs/result/z100-s35-wp6-tuning.md`（全档扫描表 + e2e A/B）。

## WP7：GGUF MoE decode 内核优化（主力项；~3–5 天，WP5 立项后）

### 7.1 阶段 1：微基准（先测后写）
1. 读 `csrc/quantization/gguf/moe_vec.cuh`（+`moe.cuh`/`mmq.cuh`）现有
   几何：vec 宽度、split-K、线程/块布局、LDS 用法；列出可调旋钮清单。
2. 微测脚本 `tools/dcu/s35_moe_gemv_bench.py`：真实 decode shape
   （行 = topk×M，N=512，K=2048，Q2_K/Q4_K 按实际权重），
   进程级计时（事件计时，≥100 次取中位）；输出 GB/s + 与 650–700 GB/s
   天花板的比值。**判据 A**：微测可复现（两次 <2% 差）。

### 7.2 阶段 2A：旋钮（若可及天花板 ≥80% 到此为止）
- 只改内核参数/模板实参（不改算法），build-sync `_C_stable_libtorch`
  （须知 4），微测 A/B 全档表。
- 判据：目标 shape GB/s 提升且无档回退 → 进 7.4 服务验收；
  仍 <50% 天花板 → 进 2B。

### 7.3 阶段 2B：exact-shape 新核（条件启动）
- 参照：稠密侧 `csrc/rocm/fp8_emu_gemm.cu` 的 whole-A/m1 GEMV 经验
  （整行 stage、K 循环零 barrier、uint4 预取环）+ 1Cat exact-shape 战术
  （`docs/result/z100-11g-1cat-vllm-reference-analysis-result.md`）。
- 新文件 `csrc/quantization/gguf/moe_gemv_gfx906.cu`（+CMake 注册 +
  `ops` 绑定），独立 env `VLLM_DCU_MOE_GEMV`（默认 0）+ 指纹 KNOBS。
- 派发点：`_fused_moe_gguf`（`vllm/model_executor/layers/quantization/gguf.py`
  的 decode 分支）按 env 切换；**默认路径零改动**。
- 数值：与现路径微测对照，max|Δ| 报表（fp16 域，预期 ~1e-3 级）；
  服务级走零翻转硬门（`.s29/capture.py` 双臂 + 会话内 3 reps 位同）。

### 7.4 服务验收（硬门，全过才收）
- per-kernel 微测 **≥1.5×**；e2e decode **tok/s +20%**（N≥2 重启）；
  零翻转；长跑 30 min 0 故障；dmesg 0；TTFT 不回退（±3%）。
- 任一不过 → 停，出表报告，env 保持默认 0。

### 7.5 回滚 / 文档
- 回滚：`VLLM_DCU_MOE_GEMV=0`；二进制 cp 回 `.bak-s35`。
- 文档 `docs/result/z100-s35-wp7-moe-decode-kernel.md`
  （微测全档表 / 数值表 / e2e / md5 / 回滚件位置）。

## WP8：（条件）GGUF prefill MMQ 优化

- **启动条件**：WP5 显示 prefill 域专家 GEMM 占 prefill 时间 **>40%**。
  不满足 → 本 WP 记"按图不启动"，直接终局。
- 对象：`ggml_moe_a8`（MMQ，`csrc/quantization/gguf/mmq.cuh`）；
  结构同 WP7 三阶段（微测 → 旋钮 → 新核，env `VLLM_DCU_MOE_MMQ` 默认 0）。
- 硬门：prefill tok/s +15% 且 TTFT 不回退（±3%）；零翻转。
- 文档 `docs/result/z100-s35-wp8-moe-prefill.md`。

## WP9：（条件）Tier C Qwen3-30B-A3B W8A16 TP=2 投资判定

- **启动条件**：WP4 审计通过（`moe_wna16_gemm` ≤203，或卡 2 行为级对照
  放行且 TP=2 只用卡 0/1）+ 用户在 WP5 墙位报告后拍板。不满足 → 记
  "按图不启动"。
- 权重：Qwen3-30B-A3B 的 W8A16 量化版（HF 检索 `Qwen3-30B-A3B` w8a16/
  int8 量化仓；无现成则用树内离线量化工具产出，sha256 台账）。
- 形态：`--tensor-parallel-size 2` 卡 0,1，端口 8104；
  命中 `E=128,N=768,device_name=AMD_GFX906,dtype=int8_w8a16.json`（14 档，
  启动日志确认 `Using MoE config` 命中而非 default 回退）。
- 判据：ready + 文本连贯 + dmesg 0 + decode/TTFT 基线（N≥2）+
  与稠密 27B（s30c 基线 29.48 ms/tok 口径换算）对比表 ⇒ 投产建议。
- **失败即止损**（README 失败组合的内核路径）：启动失败/输出崩坏 →
  记录机制，不恋战，回滚 = 杀进程。
- 文档 `docs/result/z100-s35-wp9-tier-c-w8a16.md`。

## 终局（全部 WP 后）

1. `docs/result/z100-s35-final.md`：WP 状态表 + 基线汇总（Tier A/B 双模型
   decode/TTFT）+ 优化项前后对照（WP6/7/8）+ 全部 md5/指纹 + 卡 2/3 定案。
2. **README 同步**（唯一允许的 README 改动）：
   - "已测试并确认可用"：按实测增补（模型 + 量化 + 卡 + 图模式 + tok/s）。
   - "已知限制"：MoE 条目改机制化说明（走哪条内核 / guard 覆盖 /
     DP4X 定案 / SKU 差异），删掉无本地依据的"显著较慢"传闻或换成实测数字。
3. 现役恢复：MoE 服务全部停掉（:8101/8102/8103/8104），生产 :8000 未动
   核验（`/health` 200 + VRAM 回到 ~91%×4）；`.s35/` 全部 JSON/日志归档。
4. 纪律重申：数值先于性能、N≥2、每 WP 独立回滚、零翻转硬门、
   不碰生产、dmesg 全程监控。

## WP 状态表（实施时逐项更新）

| WP | 判据 | 状态 | 证据 |
|---|---|---|---|
| WP0 资源门禁 | 卡分配定案 + 权重 sha256 全过 | ⬜ | `.s35/wp0_*.json` |
| WP1 数值门禁 | 全绿（误差/位同/无 aiter） | ⬜ | `docs/result/z100-s35-wp1-moe-gate.md` |
| WP2 Tier A | 正确性 + 基线（N≥2） | ⬜ | `docs/result/z100-s35-wp2-tier-a-baseline.md` |
| WP3 Tier B | 正确性 + 基线 +（卡 2 对照） | ⬜ | `docs/result/z100-s35-wp3-tier-b-gguf.md` |
| WP4 VGPR 审计 | 全 ≤203 或定案锁定 | ⬜ | `docs/result/z100-s35-wp4-vgpr-audit.md` |
| WP5 墙位鉴定 | 覆盖 ≥90% + 双域墙位 | ⬜ | `docs/result/z100-s35-wp5-wall.md` |
| WP6 调优配置 | e2e +5% 无回退 | ⬜ | `docs/result/z100-s35-wp6-tuning.md` |
| WP7 decode 内核 | per-kernel 1.5× + e2e +20% + 零翻转 | ⬜ | `docs/result/z100-s35-wp7-moe-decode-kernel.md` |
| WP8 prefill（条件） | prefill +15% 且 TTFT 不回退 | ⬜ | `docs/result/z100-s35-wp8-moe-prefill.md` |
| WP9 Tier C（条件） | 投产判定数据 | ⬜ | `docs/result/z100-s35-wp9-tier-c-w8a16.md` |
| 终局 | final 文档 + README + 现役恢复 | ⬜ | `docs/result/z100-s35-final.md` |
