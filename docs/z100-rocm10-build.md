# 用 ROCm 10.1 镜像重新编译 vLLM（2026-10-17）

## 结论

**编译成功**：四个扩展全部在 `z100-rocm10.1:24.04-r642` 镜像（AMD ROCm 10.1 /
TheRock，HIP 7.16，clang 24，Python 3.12）里编译通过，产物在
`/mnt/game/port/build-rocm10/temp/`：

| 扩展 | 大小 |
|---|---|
| `_C.abi3.so`（legacy，fork 的 gfx906/GGUF） | 42 MB |
| `_C_stable_libtorch.abi3.so` | 48 MB |
| `_moe_C_stable_libtorch.abi3.so` | 31 MB |
| `_rocm_C.abi3.so`（w8a8_sdot4 / fp8_emu / bf16_emu） | 53 MB |

四个库在 ROCm 10.1 下都能加载并注册算子
（`torch.ops._C.rms_norm_gated_gfx906`、`torch.ops._C.ggml_dequantize`、
`torch.ops._rocm_C.w8a8_sdot4_gemv`）。

**但内核无法执行**：任何 kernel launch 都报 `hipErrorInvalidImage`，且
**PyTorch 自带 kernel 同样失败**（`torch.randn(4,4,device="cuda")` 即报错），
所以不是本项目代码的问题。镜像自带探针（`/opt/z100/scripts/verify.sh`）7/7 全过，
说明栈本身正常——差异在 code object 的**打包格式**：探针是裸 `.co` ELF，
而 PyTorch/clang-24 产物是 fatbin/offload-bundle 形式，这版驱动路径加载不了。

## 环境准备（镜像里什么都要自己装）

镜像只有 ROCm 10.1 运行时 + 探针，没有 torch/triton/pip。已安装：

| 包 | 来源 | 大小 |
|---|---|---|
| torch 2.16.0.dev20261016+rocm10.1 (cp312) | `download.pytorch.org/whl/nightly/rocm10.1` | 1654 MB |
| rocm-sdk-core 10.1.0 | `stable.repo.amd.com/rocm/core/whl-next` | 400 MB |
| rocm-sdk-libraries 10.1.0（MIOpen/hipfft/hiprand/rccl/amdsmi/hipfile） | 同上 | 249 MB |
| rocm-sdk-devel 10.1.0（cmake 配置 + 头文件，解包后用） | 同上 | 665 MB |

> 网络提示：这些 CDN 对单连接限速到 30–50 KB/s，且偶发返回损坏分片。
> 最终用按 1 MiB 分块、多连接（32–48）下载 + 逐块尺寸校验 + zip 中央目录
> CRC 校验（`torch.whl` 的 `zipfile.testzip()` 必须为 None）才拿到可用 wheel。
> 缓存目录：`/mnt/game/rocm10-wheels/`（wheel 及校验脚本）。

## 编译期新增的代码适配（已提交 `5700a50c7`）

1. `cmake/hipify.py`：torch 2.16 删掉了
   `torch.utils.hipify.cuda_to_hip_mappings.CUDA_MACROS_MAP`，用 `getattr`
   守卫 DTK 时代的值形状修补。
2. `csrc/quantization/gptq/compat.cuh`：HIP ≥ 7 已自带原生
   `atomicAdd(half*, half)`，原来的 `__DTK_ARCH__` 守卫在 ROCm 10.1 上不生效
   会导致 `atomicAdd` 歧义；改为按 `HIP_VERSION >= 70000000` 关闭模拟。
3. `CMakeLists.txt`：正确 base 合并残留的**重复 `fs_io_C` target**
   （上游在非 CUDA 分支前已定义），删掉后 `add_library` 冲突消失。
4. 编译标志：`-DCMAKE_HIP_FLAGS="-mcode-object-version=4"`
   （clang 24 默认 v6，镜像的 build-probes.sh 也要求 v4）。

## 构建命令

```bash
docker exec -it z100-rocm10-build bash
# 环境与脚本（已写好）
. /mnt/game/port/setenv-rocm10.sh
bash /mnt/game/port/configure-rocm10.sh          # ROCM_PATH=/opt/rocm/core-10.1
MAX_JOBS=6 bash /mnt/game/port/build-rocm10.sh _C _C_stable_libtorch \
    _moe_C_stable_libtorch _rocm_C
```

容器 `z100-rocm10-build`：镜像 `z100-rocm10.1:24.04-r642`，
`--shm-size=16g`，挂载 `/dev/kfd /dev/mkfd /dev/dri`、`/src`(只读)、`/mnt/game`。

## 运行期阻塞（需要 DTK/AMD 侧解决）

`hipErrorInvalidImage` 出现在所有 kernel 上，包括 PyTorch 自带算子：

```
torch 2.16.0.dev20261016+rocm10.1 | hip 7.16.26385 | 4 GPUs visible
torch.randn(4,4,device="cuda") -> AcceleratorError: device kernel image is invalid
```

- 镜像自检：`/opt/z100/scripts/verify.sh` → **PASS=7 FAIL=0**
  （HSA 枚举/AQL 派发/HIP 端到端/rocBLAS 全通过）。
- 我们的 kernel 提取出的 code object 头 `Flags: 0x52f, gfx906, xnack, sramecc`
  与探针完全一致，但仍是 fatbin 打包形式。

结论：ROCm 10.1 的 **工具链可编译**（vLLM 四个扩展零错误），
但**运行**需要能加载 clang-24/fatbin 代码对象的运行时，或提供用
`-mcode-object-version=4` 且非 bundle 形式发布的 torch 构建。
