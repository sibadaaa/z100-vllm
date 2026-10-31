# S36 施工图:A 轨(W2'/W3/W4)+ B 轨(MTP,冻结)

> 前置阅读:同目录 `z100-s36-design.md`(设计依据:F 模型 5/5 验证、
> 占用率判定、与 S34-WP5 的对账)。
> **部署基线**:无 MTP 服务在役(PID 见 `ps -ef | grep entrypoints`,
> 启动器 `/tmp/launch_nospec.sh`,日志 `/home/lyh/serve-nomtp-1029.log`)。
> 所有 e2e 在此臂上同口径 A/B;`/tmp/vllm_env.sh` 为该进程 env 快照。

---

## 0. 通用纪律(每个 WP 相同)

1. csrc 改动:备份在役 `.so` + md5(对齐 `.r3/so-backup/` 惯例)→ 重编译
   → 重启 → 微测 → 数值对照 → 服务 smoke → e2e;失败恢复备份并回 md5。
2. py 改动:纯 env/参数类先做(零编译);逻辑类走 feature 分支。
3. 每步留档到 `docs/perf/data/s36-<wp>-<date>.txt`(原始输出)。

---

## 1. W2' 施工:m1 核 F 分解与削减

### 1.1 第一步:F 分解(零代码,0.5 天)

```bash
# 单形状 o(N=1280,K=6144)单 kernel 时间线
bash -c 'source /tmp/vllm_env.sh
cat > /tmp/one_launch.py << "PY"
import torch, sys
torch.cuda.set_device(0)
import vllm._rocm_C
sys.path.insert(0, "/tmp")
from fill_model import make, run
a, w, w_s, lut = make(6144, 1280, 1, seed=7)
for _ in range(20): run(a, w, w_s, lut)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): run(a, w, w_s, lut)
for _ in range(50): g.replay()
torch.cuda.synchronize()
PY
/opt/dtk-26.10/rocprofiler/bin/rocprofv2 --kernel-trace -o /tmp/ftrace -- /home/lyh/z100-vllm/.venv/bin/python /tmp/one_launch.py'
```

判读:内核内部分段(LUT 段 / 首预取空泡 / ramp / 稳态斜率)。
产出:`docs/perf/data/s36-f-decomp.txt` + 四假设中命中的 1-2 个。

### 1.2 第二步:H1a 预取加深 pf[2]→pf[3](首选,~30 行)

| 位置 | 改动 |
|---|---|
| `csrc/rocm/fp8_emu_gemm.cu:721` | `uint4 pf[2][RW]` → `pf[3][RW]` |
| 同文件 723-734 预取初始化 | `for pre in 0..2` 三份字面量 |
| 同文件 829-838 主循环 peel | peel-by-two → **peel-by-three**(`VLLM_FP8_STREAM_SLICE(S0,B,REFILL)` 宏不变,调用序改三拍) |
| m1 核同构(930-959, 947 宏) | 同上三处 |
| 两处 `static_assert(kSliceChunks == 64)` | 不变(slice 几何不动,只加深流水) |

- VGPR 预算:m1<1> 40 → 约 +8(uint4×2)≈ 48 → 仍 5 波/SIMD,不降档;
  **编译后必须复核 metadata spill==0**(提取流程见 §4)。
- 位同:纯调度改动,bit 级不变。

### 1.3 第二步备选:H1b slice 1024→2048(若 H1a 不足)

- `kStreamSlice` 1024→2048(行 625)且 **两处 `kSliceChunks==64` 的
  static_assert 改 128**,lane→chunk 映射改为每 lane 2 chunk
  (pf 扩为 `pf[2][RW][2]`,dot2 链 ×2)——比 H1a 面»大,**仅当 H1a
  微测不足且分解指向 H1 时再动**。
- LDS 复核:m1 ~5KB、stream_block MS=8 ~33KB(44KiB 闸内);MS>1 全档微测。

### 1.4 第二步备选:H2/H3

- H2:`dispatch_gemv_stream_m1`(1042-1075)nt 参数 512 + 两内核加
  `__launch_bounds__(512, 3)`(m1)/`(512, 2)`(stream_block);
- H3:`fp8_lut_stage` 常量内存化(`__constant__ __half sl_lut[256]` +
  `hipMemcpyToSymbol` 于模块初始化;`gfx906_torch_bindings.cpp` 注册处)。

### 1.5 微测与门

```bash
bash -c 'source /tmp/vllm_env.sh && /home/lyh/z100-vllm/.venv/bin/python /tmp/fill_model.py'   # F 模型五点
bash -c 'source /tmp/vllm_env.sh && /home/lyh/z100-vllm/.venv/bin/python /tmp/fp8_gemv_bench.py' # 正确性+全形状
```

门:目标 o ≤20µs、qkv ≤22µs;gateup ≥570 GB/s;`torch.equal` 对拍旧核
(保留旧 .so 做参考核);spill==0。e2e:无 MTP 7469-doc 探针 step p50 −5%。

### 1.6 回滚

`cp /home/lyh/z100-vllm/.r3/so-backup/_rocm_C.s36-before.so → vllm/_rocm_C.abi3.so`
+ md5 校验 + 重启。env 层无开关(编译期常量),回滚即换 .so。

---

## 2. W3 施工:decode 长上下文分段调参(零/极少代码)

### 2.1 第零步:路由确认(0.5h)

ctx ∈ {8k, 32k, 128k} 各:`/start_profile` → 1 请求 → `/stop_profile`,
从 rank0 profiler_out 确认:decode 走 `kernel_unified_attention_3d` 还是
2D;3D 时 grid 第三维(段数)。已知短 ctx 在 3D ✓。产出三行记录。

### 2.2 第一步:段数 env 扫描(1 天,零代码)

```bash
export VLLM_GFX906_ATTN_DECODE_SEGMENTS=<S>   # S ∈ {32,64,128,256}
/home/lyh/z100-vllm/.venv/bin/python /tmp/ctx_sweep.py 16384 32768 65536
```

记录 ITL p50 vs ctx × S;同时抓 128k 单步 profile 看
`reduce_segments` 占比(防归并成新瓶颈)。

### 2.3 第二步:策略修订(按扫描结果,~10 行)

| 位置 | 改动 |
|---|---|
| `vllm/v1/attention/ops/triton_unified_attention.py:295-306` | target_grid_size 表按扫描结果修订;新增上限 env `VLLM_GFX906_ATTN_DECODE_SEG_MAX`(默认 128) |
| `vllm/v1/attention/backends/triton_attn.py:161-175` | 缓冲按新上限分配(显存增量 = threshold×heads×seg×256×4B,需核算) |

- 显存核算:现有 capacity=128 时缓冲已按 128 分配 → 提上限前先看该张量
  实际 shape(gpu_memory_utilization 0.85 下有无余量)。

### 2.4 门与回滚

门:128k ITL p50 −10%(31.4→35-38 t/s 折算);7.5k/32k 回退 ≤2%;
零翻位。回滚:env 归零(policy 回默认)+ py 改动 revert。

---

## 3. W4 施工:push AR 出口融合

**直接采纳 `docs/design/z100/s34/z100-s34-wp4-ar-fusion.md` §2-§3 的设计**
(内核草图、host 签名 `all_reduce(..., residual=, norm_weight=, quant_out=)`、
env `VLLM_DCU_CUSTOM_AR_FUSION=1`)。本轨新增:

- 时点:排在 W2'/W3 之后(步构成变化后重测收益);
- 新增证据基线:AR 8.9 ms/步(26% 设备时间,69.4µs×128,10-29 profile);
- 位同门:wp4 §4 的 T=1/T=4 双档 + `torch.equal`;bit 不同则按
  "max|Δ|≤1e-3 + top5 logprob 板"走裁决(与 wp5 同口径);
- 触点:`csrc/custom_all_reduce.cuh`(push 核第 4 步归并出口)、
  `vllm/distributed/device_communicators/custom_all_reduce.py`(签名+env,
  oneshot 门区 548-584 行)、模型层 communicator 封装。
- 影响的 .so:`_C_stable_libtorch.abi3.so`(备份 md5 见 STATUS §1.4)。

---

## 4. 通用工具:内核元数据提取(VGPR/spill 复核)

```bash
# 设备码抽取 TU(已建好,改完内核后同步内含的行区间再编译)
hipcc -O3 --offload-arch=gfx906 -I csrc -c /tmp/devmeta.cu -o /tmp/devmeta.o
python3 - << 'PY'   # fatbin → 长度前缀 bundle → gfx906 code object
import struct
d = open('/tmp/devmeta.o','rb').read()
base = 0x2000                      # .hip_fatbin 文件偏移(readelf -S 查)
n = struct.unpack_from('<Q', d, base+24)[0]
off = base + 32
for _ in range(n):
    a, b, slen = struct.unpack_from('<QQQ', d, off); off += 24
    tri = d[off:off+slen].decode(); off += slen
    if 'gfx906' in tri and b:
        open('/tmp/dm.co','wb').write(d[base+a:base+a+b])
PY
llvm-readobj --notes /tmp/dm.co | awk '/AMDGPU Metadata: ---/{f=1} f' > /tmp/dm.yaml
# 解析 .vgpr_count / .vgpr_spill_count / .group_segment_fixed_size(yaml)
```

门:改动内核 spill 必须为 0;waves/SIMD 不低于改动前。

---

## 5. B 轨(MTP,独立,冻结部署)

- B1 调查(只读,0.5 天):生产 metrics 三条
  (`spec_decode_num_drafts_total` / `num_draft_tokens_total` /
  `accepted_tokens_per_pos`)与 `vllm/v1/spec_decode/step3p5.py`
  335 行(parallel 分支)/ 387 行(顺序起草)对账;产出:
  第二 draft 未产出的成因 + 修复方案设计(独立裁决)。
- B2 部署:启动器加回
  `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`
  (其余参数与 A 轨基线一致);**在 A 轨每落一项后重测 MTP 臂**再裁决。
- 与 A 轨的隔离:B1/B2 不改任何 A 轨触点;A 轨 e2e 永远在无 MTP 臂测。

---

## 6. 顺序与工作量

```
W2'-分解(0.5d)→ W2'-H1a(1d)→ [H1b/H2/H3 按需](1d)→ 微测+e2e(0.5d)
W3-零步(0.5h)→ W3-扫描(1d)→ W3-策略(0.5d)         ← 与 W2' 并行
W4(1-2d,引用 wp4 施工图)                          ← 最后
B1(0.5d,随时,只读)
```
