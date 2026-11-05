# S45-α 终局报告:N2a 激活折入采纳 + 零填充负结果 + 空隙归因

> 2026-11-01。执行 S44 移交的三件套(用户指令:"先做①,MTP 之后再做")。
> 基线 = S44 采纳态;台账 = `.s45/{s45-adopt,n2a_parity}.py/md` +
> `docs/perf/data/s45-*.txt`。

## 0. 结果一览

| 项 | 门 | 实测 | 判定 |
|---|---|---|---|
| N2a 折入 | torch.equal 位同 | M=1 max\|d\|=0(M=4 fallback 亦等);服务 board first_div=None;`mul_silu` 出账(−47 发/step) | **采纳**(`VLLM_DCU_W8A16_SILU_MUL=1`) |
| N2a e2e | 无回归 | B3 36.74 / prod 35.86 vs A2 35.19-36.94(当日噪声带内;+1% 量级被 ±7% 漂移淹没,机制证据 = 账本+位同) | PASS |
| 零填充 | 消除 48 发/step | empty 版本使贪心流分叉(zeros 承重,op 有未写元素) | **关档(负结果)** |
| 空隙归因 | — | 5.38 ms/step GPU 空闲,92% 在 piecewise 图段边界 | 落档,host 侧池子定界 |

## 1. 交付物

- csrc:`wholea_m1` 增 `smul` staging 折叠(默认 0 = 原路径逐位不变);
  新 host/schema `fp8_emu_gemm_block_w8a16_smul`(资格镜像原 M==1 wholea 臂,
  不符响亮失败);`.so` md5 `f72bb39d`→`6dee7fad`(备份在案)。
- python:`vllm::gfx906_w8a16_silu_mul_down` opaque op(M 判定在体内)+
  `Qwen2MoeMLP.forward` env 门分支(含 TP all-reduce 语义)+
  `VLLM_DCU_W8A16_SILU_MUL`(默认 0)。
- 数据:`docs/perf/data/s45-{n2a-parity,gap-attr,ledger-B3,ledger-combo}.txt`、
  `runs/s45/`。

## 2. 本轮学到了什么

1. **消费侧折入是手调内核族的正确折叠形态**:gate_up GEMV 的分块结构使
   生产侧折叠需 LDS 翻倍,而 down_proj 的整行 staging(`wholea_m1`)给了
   一处插入点;`expf ≡ libdevice.exp` 在本栈实证成立(位同门 0 差)。
   **位同测试的参考必须用与生产相同的舍入链**(单舍入)——首测 FAIL 是
   eager 双舍入参考的假阳性。
2. **上游的安全 zeros 是承重的**:`core_attn_out` 有未写元素被下游读,
   empty 化即使满足"无 padding"门也分叉。此类"消除发射"提案必须先
   证明 op 的全写契约(或做写掩码审计)。
3. **负结果归档同样是产出**:48/63 发的零填充保留,消除需 op 契约审计
   (S45+ 可选)。
4. **空隙池子定界**:5.38 ms/step 的 92% 在图段边界(host 串行),
   与发射数线性相关——fold/少发射是当前唯一已证抓手;段合并
   (splitting_ops)是下一级杠杆(S38 §2 入口)。
5. **流程教训(两次)**:kill 后必须验证 procs=0 + port free 再 launch
   (残留 worker 争用 GPU 造成 22-29 t/s 的假崩溃);docker 构建产物为
   root 属主,swap 后核对 md5。

## 3. 移交(S45β+)

- 空隙池(4.95 ms/step @ 图段边界):splitting_ops 段合并设计;
- MTP 复役(用户既定:S45 完成后)——本采纳态即起跳板;
- 可选:op 全写契约审计后重开零填充(0.29 ms/step);
- per_fused_1 折入 recurrent(0.31 ms,高危)。

## 4. 产线状态(一行级)

- `serve_deploy.sh`:S44 段(FUSION=1/_MLP=0)+ S45 段(W8A16_SILU_MUL=1);
- `.so`:`_rocm_C` `6dee7fad` / `_C_stable_libtorch` `54ccf2e2`;
- 回滚:注释 `VLLM_DCU_W8A16_SILU_MUL` + restart(+换回
  `_rocm_C.s44-before.so` = 完全 pre-S45);
- 采纳态 e2e:W1 35.86 / W2 36.38 t/s,TTFT 0.74 s。
