# P5 rollback rehearsal（2026-10-18）

目标：验证新栈（ROCm 10.1 + torch 2.16）与 DTK 栈之间的切换通道可实际使用，
且两条路径都能正常 serve；切换不影响已验证的扩展身份。

## 演练步骤与结果

| # | 动作 | 结果 |
|---|---|---|
| 1 | `tools/dcu/use_stack.sh rocm101` | copy tree 的 4 个 .so 指向 **installed snapshot**：`_C 9302ece8` / `_C_stable_libtorch ffc176af` / `_moe 7b4185c6` / `_rocm_C a250420b`（= 全部 L0–L4 结果所用身份） |
| 2 | 新栈 0.6B serve（fp16/eager/dev0） | ready 22 s；输出 ` Paris. The capital of Italy is Rome`，client rc=0，verdict=OK |
| 3 | `tools/dcu/use_stack.sh dtk` | DTK tree 的 4 个 .so 指向 DTK 构建：`_C d44117ae` / `_C_stable_libtorch e303b0b2`（F1 修复）/ `_moe 50af36b9` / `_rocm_C f205be8f` |
| 4 | DTK 0.6B serve（fp16/eager/dev0） | ready 16 s；同一 prompt 输出相同 ` Paris. The capital of Italy is Rome`，usage 13 tokens，verdict=OK |

最终身份（演练后）：

```
DTK          : d44117ae / e303b0b2 / 50af36b9 / f205be8f   (tree /home/lyh/z100-vllm)
ROCm 10.1     : 9302ece8 / ffc176af / 7b4185c6 / a250420b   (tree /mnt/game/port/gfx906-vllm-rocm101)
```

## 演练中发现并修复的问题

`use_stack.sh rocm101` 原先指向 **build tree artefacts**（`build-rocm101/temp`，
md5 `3e50e4e8/baead6fa/82278973/3e2ee355`）。这些文件与安装副本内容相同（大小一致），
但 `cmake --install` 会确定性重写 runtime path，因此 md5 不同；验证口径是**安装副本**。
已修复：

1. 新增不可变快照 `/mnt/game/port/gfx906-vllm-rocm101/installed-so/`（保存安装副本）；
2. `use_stack.sh` 的 `ROCM101_SO_DIR` 默认改为该快照；
3. 重跑 `install-rocm101.sh` 可从 build temp **精确复现**安装副本 md5
   （9302ece8/ffc176af/7b4185c6/a250420b），已实测。

## 结论

- 两栈切换/回滚通道可用，且切换后各自 serve 正常（本演练含真实请求验证）。
- 新栈身份与 L0–L4 验收所用一致；DTK 身份与 L-1 基线一致。
- 回滚判据：`use_stack.sh status` 的 4 个 md5 与上表一致 + 一次 smoke 通过。
