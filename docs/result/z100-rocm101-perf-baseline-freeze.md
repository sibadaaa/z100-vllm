# Z100 ROCm 10.1 性能基线冻结（WP0.1）

> 施工依据：[`docs/design/z100-rocm101-perf-construction.md`](../design/z100-rocm101-perf-construction.md) §1.1 / §4.1
> 冻结时间：2026-10-24（本会话）
> **性质：只读归档。本次没有启动任何服务、没有重跑 B1–B4、没有产生新的性能数字。**
> run 目录来源：`/home/lyh/vllm-test/llm-engine-probe/runs/`

---

## 1. 信任的基线表（= 施工图 §1.1，数字不再重测）

| 编号 | 运行 | 栈 | prompt/out | TTFT / prefill | decode step p50 | TPOT p50 | 接受率 |
|---|---|---|---|---|---|---|---|
| **B1** | `20261023-135729` | rocm101，无 w8a16 | 709 / 128 | 4937 ms / 4930 ms | **192.2 ms** | 37.1 ms | 57.5% |
| **B2** | `20261023-150120` | **DTK**，w8a16 + FP16 | 709 / 128 | **1594.9 ms** / 1594 ms | **200.5 ms** | 39.9 ms | 50.6% |
| **B3** | `20261023-164704` | rocm101，w8a16 + FP16 | 手动 512 | **2393.7 ms** | **238–265 ms** | — | 15–43% |
| **B4** | `20261023-164058` | rocm101，w8a16 + FP16 | 手动 193 | 2032.8 ms | 237–257 ms | — | ~35% |

使用规则（同施工图 §1.1）：B1 是 rocm101 无 w8a16 的 decode 底线；B2 只作 DTK 参照；
B3/B4 只用于确认"功能已生效、decode 同量级"，其接受率不作为基线。

---

## 2. run_meta.json 栈身份核对（WP0.1 只读检查，2026-10-24 执行）

核对命令：`run_meta.json → m["host"]`（施工图 §4.1 的只读核对）。

### B1 `20261023-135729-serve`（rocm101，无 w8a16）

```json
{
  "started_at": "2026-10-23T13:57:29+00:00",
  "runtime": "docker",
  "image": "z100-rocm101-vllm:20261023",
  "container": "vllm-probe",
  "vendor": "hygon",
  "model": "/mnt/game/models/Qwen3.8-27B-FP8",
  "tensor_parallel_size": 4,
  "max_model_len": 32768,
  "max_num_batched_tokens": 8192,
  "probe_level": "steps"
}
```

- **host 段为空**（docker 模式不采集 host_meta）：`python / python_version / torch_version` 均为 `null`。
  施工图 §4.1 预期"B1 是 2.16.0"在 run_meta 里**无直接字段**；栈身份由 `image=z100-rocm101-vllm:20261023`
  （rocm101 镜像，与 B3/B4 时代构建的同一镜像线）佐证，判定 **rocm101 栈成立**。
- **口径警示**（影响与 E1 的可比性，判读时必须记住）：
  - B1 走 **docker** + `MAX_NUM_SEQS=128`，而 E1/E2/E3 走 **host** + `MAX_NUM_SEQS=4`；
  - B1 的启动命令带 `--enable-prefix-caching`（施工图要求 A/B 阶段关闭前缀缓存）；
  - B1 无 w8a16（`W8A16` 旋钮未设）。
  因此 B1 的 192.2 ms 只能作为"decode 底线"参照，不与 E1 做严格显著性比较（施工图 §1.3 已预期）。

### B2 `20261023-150120-serve`（DTK，w8a16 + FP16）

```json
"host": {
  "python": "/home/lyh/z100-vllm/.venv/bin/python",
  "python_version": "3.10.12",
  "torch_version": "2.11.0",
  "vllm_version": "dev"
}
```

- `serve_launch.sh` 首行工作目录：`cd /home/lyh/z100-vllm` → **DTK 树**。
- torch **2.11.0** ✅（施工图 §4.1 预期一致）。1594.9 ms / 200.5 ms 是 DTK 数字，不是 rocm101。

### B3 `20261023-164704-serve`（rocm101，w8a16 + FP16，手动 512）

```json
"host": {
  "python": "/home/lyh/rocm101-host/venv/bin/python",
  "python_version": "3.12.14",
  "torch_version": "2.16.0.dev20261016+rocm10.1",
  "vllm_version": "0.0.0.dev0"
}
```

### B4 `20261023-164058-serve`（rocm101，w8a16 + FP16，手动 193）

```json
"host": {
  "python": "/home/lyh/rocm101-host/venv/bin/python",
  "python_version": "3.12.14",
  "torch_version": "2.16.0.dev20261016+rocm10.1",
  "vllm_version": "0.0.0.dev0"
}
```

### 核对结论

| 编号 | 预期栈 | run_meta 证据 | 判定 |
|---|---|---|---|
| B1 | rocm101，torch 2.16 | host 段为空；`image=z100-rocm101-vllm:20261023` | **成立**（间接证据，已如实记录） |
| B2 | DTK，torch 2.11.0 | python=z100-vllm/.venv，torch **2.11.0** | **成立** |
| B3 | rocm101，torch 2.16 | python=rocm101-host/venv，torch **2.16.0.dev20261016+rocm10.1** | **成立** |
| B4 | rocm101，torch 2.16 | python=rocm101-host/venv，torch **2.16.0.dev20261016+rocm10.1** | **成立** |

---

## 3. T2 部署树四个 `.so` 的 md5（`/mnt/game/port/gfx906-vllm-rocm101/installed-so/`，2026-10-24 读取）

```text
9302ece8e412bbe3f0e80012e4598d08  _C.abi3.so                (mtime 2026-10-17 16:58)
ffc176afef132acdb79b762e7a4e7120  _C_stable_libtorch.abi3.so (mtime 2026-10-17 17:01)
7b4185c65a8af8a6dc027648503235ef  _moe_C_stable_libtorch.abi3.so (mtime 2026-10-17 17:04)
16f47eff8df7b5441a1084d2f8d0a947  _rocm_C.abi3.so           (mtime 2026-10-23 15:38)
```

- 前三个 `.so` 是 10-17 的构建；`_rocm_C` 是 10-23 w8a16 移植后重编的——与调研文档
  "只重编了 `_rocm_C`" 的结论一致。
- 这四个 md5 是 E1 的 `stack_identity` 基准值；P1 全量重编后由新的 md5 取代（记录到
  `so.md5` 与 stack_identity 输出）。

---

## 4. 冻结声明

1. 本文档只归档 §1.1 信任基线与只读核对结果；**没有启动服务、没有重跑 B1–B4**。
2. 后续 E1/E2/E3 判读直接引用 §1–§3，不再启动 DTK（B2 不重跑）。
3. B1 的三条口径差异（docker、MAX_NUM_SEQS=128、prefix caching on）已记录；
   E1 与 B1/B3 的 decode 比较按施工图 §1.3 的"方向性结论"口径执行。
4. E1 之前必须完成 WP0.2 的栈身份门禁（`stack_identity` exit 0）。
