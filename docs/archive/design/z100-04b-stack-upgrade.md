# S4b 目标栈升级（torch 2.11.0+das / triton 3.6.0+das）与重建

> 上层文档：[适配方案](../海光Z100-DTK26.04-gfx906-vLLM适配方案.md) · [Todolist](../z100-todolist.md)
> 前置：[S4](z100-04-build-system-dtk.md) · [S5](z100-05-csrc-kernels.md) · 状态：⬜ 待做

## 目的

把构建基线从镜像自带的 **torch 2.10.0+das / triton 3.4.0** 换成方案固定的目标栈
**torch 2.11.0+das.opt1.dtk2604 / triton 3.6.0+das185.dtk2604**（2026-10-07 Lead 决策，用户确认"完全按原方案升"），并在新栈上重建全部扩展。两个动因：

1. `_C_stable_libtorch` 停在 `csrc/libtorch_stable/cuda_view.cu` 的 `torch::stable::Tensor::layout()` —— 镜像自带的 torch 2.10 没有该 API，而本 fork 源码是按 2.11 ABI 写的（[S5 归档 §5](../result/z100-05-csrc-kernels-result.md)）；
2. `triton 3.6.0+das185.dtk2604` 是 gfx906 attention 需要的版本（S6 的前置）。

## 前置条件

- `requirements/dcu*.txt` 已固定这两枚 cp310 轮子（S4 落地）；直链见 `requirements/dcu.txt`。
- S4 留下的容器 `z100-build`（torch 2.10 + 已编好的 `_C`/`_rocm_C`，两者都只含 gfx906）**必须保留**作为回滚基线：升级在**新容器**里做，不要就地覆盖。

## 做法

1. 新建容器 `z100-build-211`（同镜像；GPU 设备与 group 参数照抄 `z100-build`；`--memory=8g`；**不加 `--rm`**），挂 `/mnt/game`，**只读**挂工作区（`/src`）。
2. 只装目标栈的两枚轮子，**不要**顺手装 `requirements/dcu-build.txt` 的 common 依赖（那会把镜像里已验证的 DTK 栈搅乱）：

   ```bash
   pip install --no-deps <torch-2.11.0+das.opt1.dtk2604-cp310-...whl> \
                        <triton-3.6.0+das185.dtk2604.torch2100-cp310-...whl>
   python -c "import torch; print(torch.__version__, torch.version.hip)"
   python -c "import triton; print(triton.__version__)"
   ```

3. **全量同步**工作区到新副本（新容器里从头同步，不要只同步 `csrc/`），同步配方沿用 [S4 归档 §7](../result/z100-04-build-system-dtk-result.md)（排除项必须锚定，否则 `requirements/build/dcu.txt` 会被漏掉）。
4. 新构建目录 `/mnt/game/port/build-211`；导出 `PYTORCH_ROCM_ARCH=gfx906`、`MAX_JOBS<=4`；先只编 `_C _rocm_C`（**不要**裸跑全量 ninja，见 S5 归档），再单独试 `_C_stable_libtorch _moe_C_stable_libtorch`——2.11 下 `cuda_view` 的问题应当消失。
5. 跑门禁与回归（§验收），并把"新 triton 对 `gfx906` 的支持情况"作为 S6 的入口判断记进归档。

## 验收（门禁）

- `python -c "import torch;print(torch.__version__)"` = `2.11.0+das.opt1.dtk2604`；`triton.__version__` = `3.6.0...`。
- `python -c "import torch, vllm, vllm._C, vllm._rocm_C; print('IMPORT_OK')"` 成功（**必须先 `import torch`**）。
- 新 `.so` 只含 gfx906：`strings <so> | grep -oE "amdgcn-amd-amdhsa--gfx[0-9a-f]+" | sort -u`。
- `from vllm._custom_ops import ROCM_SKINNY_GEMM_ENABLED` 可导入且为 `False`。
- 回归（新栈上仍全过）：`tests/kernels/quantization/test_gguf_fixed_cols_gfx906.py` **13/13**、`tests/kernels/mamba/test_fused_sigmoid_gating.py` **5/5**（用 `python -c "import torch, pytest; ..."` 起跑，别直接 `pytest`）。
- 顺带判定：新 triton 上 `unsupported target: 'gfx906'` 是否消失。

## 失败处置

| 现象 | 处置 |
|---|---|
| 升级后 `import torch` 崩 / HIP 版本不匹配 | DTK 26.04 与 torch 2.11 不兼容 ⇒ 回退 `z100-build`（2.10 基线），把证据交 Lead 重新决策，**不要**硬凑 |
| `_C_stable_libtorch` 仍在 `cuda_view` 失败 | 记录真实报错；`_C`/`_rocm_C` 不受影响，作为遗留交 S9 |
| 构建副本与工作区不一致（例：只同步了 `csrc/`） | 全量 rsync（排除项锚定）后重新 configure，再编 |
| 内存不足 / OOM | `MAX_JOBS=2`、`ninja -j2`；构建产物只落 `/mnt/game` |

## 参考

- [S4 归档 §6（可复用构建环境表）](../result/z100-04-build-system-dtk-result.md)
- [S5 归档 §5（`_C_stable_libtorch` 阻塞与 4 类 DTK 缺陷修复）](../result/z100-05-csrc-kernels-result.md)
- [S6 Triton gfx906 支持固化](z100-06-triton-gfx906.md)
- [Todolist 全局硬约束](../z100-todolist.md)
