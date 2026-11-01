# S46 设计图:MTP k=2 复役(在 S44/S45 采纳栈上重启投机解码)

> 日期:2026-11-01。输入:S37 B2 复役裁决(**GO,+29~42% 真token吞吐**,
> accept 0.91、drafts 2.0;`docs/result/z100-s37-final.md` §5)、S34 生产
> 实测(greedy 51.6 / sampled 40.6 / warm TTFT 2.4 s,`DEPLOY.md`)、
> S44/S45α 收账(用户既定顺序:"内核侧收干净 → MTP")。用户指令:开工。
> 本轮问题:**S40-S45 五轮内核变更从未与 MTP 同跑**——复役能否兑现 S37
> GO 口径?新交互面在哪?

---

## 0. 基线快照(2026-11-01,S45 采纳态)

| 项 | 值 |
|---|---|
| 在役 | 无 MTP(`SPEC_OFF=1`)+ S44 AR 融合(hybrid)+ S45 W8A16 silu-mul 折入;`.so` = `_rocm_C` `6dee7fad` / `_C_stable_libtorch` `54ccf2e2` |
| 服务级 | W1 贪心 96:35.9-37.8 t/s(当日带);W2 采样 512:35.9-36.4;TTFT 0.74-0.78 s |
| decode 步账 | busy ~19.5 ms + AR 自旋 ~1.9 + **图段边界空隙 ~5.4** + 边界 ~1.7(S45 归因) |
| MTP 资产 | `mtp.safetensors` 在检点(草稿 1 层,k=2);`launch_mtp2.sh` 的 SPEC 契约:`SPEC_OFF=1` 现役关闭,移除即默认 `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`;accept 彩票已钉(`ROCBLAS_WORKSPACE_CONFIG`,S29 WP-A) |
| S37 GO 口径 | 真token吞吐 7.5k **53.7** / 60k **40.0** / 128k **30.8** tok/s vs 无 MTP 37.8/30.3/24.0 = **+29~42%**;drafts 2.0、验收 0.91(2026-10-30 栈) |
| S34 生产实测 | greedy 51.6 / sampled 40.6 / warm TTFT 2.4 s;accept 贪心 2.1-2.6(2026-10-27 栈) |
| 判读纪律 | **MTP 步时长(46-57 ms)与核账不可直接对比无 MTP 步(25-33 ms)**——k=2 每步产出 ~2.8 token,唯一判据 = e2e 真 token 吞吐(S37 §5 教训) |

## 1. 交互矩阵(S40-S45 × MTP 路径,复役前必答)

| 变更 | draft decode(M=1) | verify(M=3) | 判读 |
|---|---|---|---|
| S40-B1 lm_head stream(M==1 门) | 受益 | M=3 **脱门**→dot2(~900 µs vs stream 808) | 小劣化;候选微项 = 门扩 M≤3(内核本按 rows 泛化,疑似零重编) |
| S40-WP3 GEMV stack(rows=4) | ✓ | M=3 ≤ 4 ✓ | 中性偏好 |
| S41 wholea(M==1 门) | ✓ | M=3 走 stream2/tile = S34 时代行为 | 中性 |
| **S44 AR 融合(hybrid)** | ✓(M=1 ≤36) | M=3 ≤36 **融合生效**;SKIP plumbing × MTP aux hidden states(S38 写明位同保护) | **board 必验** |
| S45 smul fold(M==1 门) | ✓ 受益(草稿层 down_proj) | M=3 → fallback(物化 act)= pre-S45 行为 | 中性 |
| S36 prefill 物化(S34 后采纳) | — | 草稿预填同享物化路由 | **TTFT 成本可能 < 2.4 s**,A2 实测 |

结论:全部交互要么中性、要么已内建回退;唯一新风险面 = **S44 AR 融合在
verify 批(M=3)下的数值**,由 board 把关。

## 2. Track M:复役链(1 天,零重编)

```
M0 预检快照(0.5h,只读)
M1 复役臂 + 功能探针(0.5h,-SPEC_OFF 启动)
M2 数值门(S29 板 vs 无 MTP 基线)(0.5h)
M3 e2e ABAB(off/on ×2)(0.5d)     ← 主门
M4 账本 + 长ctx抽查(0.5h)
M5 采纳/否决 + 产线落地(0.5h)
不施工: dynk/长ctx 专项(S24 门在位,128k 已由 S37 证 +28%);
        lm_head M≤3 扩展(M1 后视 verify lm_head 实测占比决定,疑似零重编);
        k=3+ 扫描(S37 口径外,另裁)。
```

排序理由:全链 env 级(无 .so/无代码变更,lm_head 扩展除外),M3 的
e2e ABAB 是唯一硬门;失败即回 `SPEC_OFF=1`(零变更回滚)。

## 3. 门总表

| WP | 门 | 判据 |
|---|---|---|
| M1 功能 | 启动含 `spec={"method":"mtp","num_speculative_tokens":2}` 行 | probe 连贯;draft 预填无 OOM;无 capture 回退告警刷屏 |
| M2 数值 | S29 板 vs 无 MTP 基线 | **verify 无损先例**(greedy = target argmax 前缀):0 分叉或 batch 形状抖动级近并列(S34 先例 @46);无乱码/无复读 |
| M3 e2e | W1 ≥ **+29%**(S37 GO 口径下限) | ≥ ~46.4 t/s(当日无 MTP 带 36 计);W2 ≥ +15%;**TTFT ≤ 2.4 s**;臂内流一致 |
| M4 账本 | accept 与步账留档 | accept 贪心 ≥ 2.0;drafts ≈ 2.0;步成本/产出比记录(不作门) |
| M5 采纳 | 产线落地 | `SPEC_OFF` 注释 + DEPLOY.md S46 节 + 回滚行;systemd restart |

**止损**:W1 < +15%、或 TTFT > 3 s、或板塌陷/复读 → NO-GO,保持
`SPEC_OFF=1`(产线零变更)。

## 4. 预期终态与验收

- **W1 贪心 46-53 t/s**(+29-42% vs 当日带 36-37.8);W2 41-49;
  TTFT ≤ 2.4 s(S36 物化后有望更好);accept ~2.0。
- 复役后步账形态改变(步时长 ~46-57 ms × 产出 ~2.8 token)——账本口径
  按"每真 token"重归一,与 S46 前账本不可直比(记录在案)。
- 内核侧 (S33-S45) + 投机 (S46) 双收账后,服务级剩余成本 = AR 自旋
  (~1.9)+ 图段空隙(~5.4)+ 草稿预填 TTFT,均为后续专项。

## 5. 风险

- **AR 融合 × verify**:M=3 批走融合 AR(rows≤36 内)——S44 A0 只验过
  无 MTP 的 M=1/8;M2 板把关,异常则 `_MLP=0` 已是现役形态、再退
  = 关融合(env 级)。
- **accept 重启彩票**(±6-10%,S28d):`ROCBLAS_WORKSPACE_CONFIG` pin
  在位(S29 WP-A 三连重启位同验证);M3 的 ABAB 四臂同日同法仍必配对。
- **VRAM**:草稿层 KV 自同一池出(utility 0.85,S34 同参验证过
  262k maxlen);M1 启动即验。
- **TTFT +1.7 s**(S34 已接受的成本):草稿预填;S36 物化后实测可能
  更低——A2 记录实际值,不设"不变"门。
- **长 ctx**:S37 @128k 仍 +28%(30.8 vs 24.0);M4 抽查确认无回归。
