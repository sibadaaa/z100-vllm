# S46 施工图:MTP k=2 复役(零重编,env 级)

> 前置阅读:同目录 `z100-s46-design.md`(交互矩阵/门/风险)。事实基线:
> S37 B2 GO(+29~42%,accept 0.91)、S34 生产实测(51.6/40.6/2.4 s)、
> S45 采纳态在役。**本轮预期零重编**:复役 = env 级(移除 `SPEC_OFF`);
> 唯一代码候选(lm_head M≤3 门扩展)在 M4 后视数据另裁。

---

## 0. 通用纪律(承 S44/S45,差异项加粗)

1. **`.so` 台账:本轮不动 `.so`**(备份锚定即可):
   ```bash
   md5sum vllm/_rocm_C.abi3.so vllm/_C_stable_libtorch.abi3.so \
     | tee .r3/so-backup/s46-before.md5        # 期望 6dee7fad / 54ccf2e2
   ls -la /mnt/game/models/Qwen3.8-27B-FP8/mtp.safetensors
   grep -n "ROCBLAS_WORKSPACE_CONFIG" .s34/launch_mtp2.sh   # accept pin 在位
   ```
2. **启停纪律(S45 教训,强制)**:kill → sleep 12 → **验证
   `pgrep -cf "entrypoints[.]cli|VLLM::Worker"` = 0 且 port free** → 再
   launch;docker 构建产物 root 属主,任何 `.so` 操作需 sudo + 核对 md5。
3. e2e:`VLLM_PROBE_BASE=http://127.0.0.1:8000 .venv/bin/python
   runs/specbench/e2e_ts.py --tag s46-<wp>-<arm> --out runs/s46`
   (W1 96×3 + W2 512×2;ABAB 必配对,四臂同日)。
4. 板:`.s44/board.py --tag s46-<arm> --rep 3 --ref <基线板>`。
5. **MTP 步账判读**:步时长/核账不可直比无 MTP(每步产出 ~2.8 token);
   唯一吞吐判据 = e2e 真 token t/s(S37 §5)。
6. 每步原始输出落 `docs/perf/data/s46-<wp>-<名>.{txt,json}`。

先建目录与基线板:

```bash
mkdir -p .s46 runs/s46
VLLM_PROBE_BASE=http://127.0.0.1:8000 .venv/bin/python .s44/board.py \
  --tag s46-base --rep 3        # 无 MTP 基线板(现役臂即基线)
```

---

## 1. M1 复役臂 + 功能探针(0.5h)

```bash
for p in $(pgrep -f "entrypoints[.]cli"); do kill -9 "$p" 2>/dev/null; done
for p in $(pgrep -f "VLLM[:][:]Worker"); do kill -9 "$p" 2>/dev/null; done
sleep 12
pgrep -cf "entrypoints[.]cli|VLLM[:][:]Worker" | grep -q "^0$" || echo "NOT CLEAN"
.venv/bin/python .s34/launch_arm.py s46-mtp1 -SPEC_OFF   # 移除 SPEC_OFF → MTP k=2 默认
# 等 health(~300 s,草稿预填参与编译)
grep -m1 "spec=" .s34/serve-s46-mtp1.log    # 期望 spec={"method":"mtp","num_speculative_tokens":2}
KEY=$(cat /mnt/game/host-serve/api-key)
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"用三句话介绍你自己"}],"max_tokens":96}' \
  | head -c 400
```

- **门**:①日志含 `spec={"method":"mtp","num_speculative_tokens":2}` 且
  `SPEC_OFF` 未出现在主进程 environ;②输出连贯无乱码;③无 OOM、无
  capture 回退告警刷屏;④草稿预填后 VRAM ≤ 92%(hy-smi)。
- 异常处置:OOM → 降 `UTIL`(如 0.82)重试一次;capture 回退 → 记录
  告警行,与 M3 一起判读(不单独止损)。

## 2. M2 数值门(S29 板,0.5h)

```bash
VLLM_PROBE_BASE=http://127.0.0.1:8000 .venv/bin/python .s44/board.py \
  --tag s46-mtp1 --rep 3 --ref runs/s44/board-s46-base.json \
  | tee docs/perf/data/s46-m2-board-mtp.txt
```

- **门**:verify 无损先例(greedy = target argmax 前缀)→ `first_div=None`
  或**近并列分叉**(batch 形状抖动级,|Δlp| ≤ ~1e-2,S34 先例 @46);
  无乱码/无复读;96×3 长度正常。
- **塌陷式分歧/复读 = 关**(M2 FAIL → 直接 NO-GO 路径,回 SPEC_OFF)。
- 记录:MTP 下 TTFT(草稿预填成本,对照 2.4 s 上限)。

## 3. M3 e2e ABAB(off/on ×2,0.5d)—— 主门

```bash
# B1 = s46-mtp1(MTP on,已起):直接跑
VLLM_PROBE_BASE=http://127.0.0.1:8000 .venv/bin/python runs/specbench/e2e_ts.py \
  --tag s46-mtp-B1 --out runs/s46
# A1 = off
for p in ...; kill; verify; .venv/bin/python .s34/launch_arm.py s46-mtp-A1 SPEC_OFF=1
VLLM_PROBE_BASE=... e2e_ts.py --tag s46-mtp-A1 --out runs/s46
# B2 = on;A2 = off(各一轮,同日配对)
```

- **门**(对当日无 MTP 带 36-37.8 计):
  - W1 贪心 **≥ +29%**(即 ≥ ~46.4 t/s)——S37 GO 口径下限;
  - W2 采样 **≥ +15%**(≥ ~41.3 t/s);
  - **TTFT ≤ 2.4 s**;
  - 臂内流一致;板语义门维持。
- **止损**:W1 < +15%、或 TTFT > 3 s、或板塌陷 → NO-GO(保持
  `SPEC_OFF=1`,产线零变更)。
- 判读注意:四臂同日同法;accept 彩票已钉但跨臂仍必配对;W1/W2 中位
  对比,不看单 rep。

## 4. M4 账本 + 长 ctx 抽查(0.5h)

```bash
VLLM_PROBE_BASE=... VLLM_PROBE_OUT=/home/lyh/z100-vllm/runs/s46 \
  .venv/bin/python runs/s34nospec/profile_window.py
.venv/bin/python .s44/ledger_classes.py <新trace> --top 15 \
  | tee docs/perf/data/s46-m4-ledger.txt
# 长 ctx 抽查(60k / 128k 单发,S37 口径对照 40.0 / 30.8 tok/s):
python - <<'EOF'   # 见 .s46/longctx_probe.py(M4 时落档)
EOF
```

- 记录:verify 步成本 vs 产出(accept)、草稿/drafts、lm_head 在 verify
  (M=3)的实际占比(决定是否立项 M≤3 门扩展)。
- **门(软)**:128k 真 token 吞吐 ≥ 28 t/s(S37 30.8 的 −10% 容限);
  不达标仅记录,不单独否决复役(短 ctx 是主口径)。

## 5. M5 采纳/否决 + 产线落地(0.5h)

采纳(M3 全过):

- `serve_deploy.sh`:`export SPEC_OFF=1` 行注释(S45 段下)+ S46 注释块;
- `.s34/DEPLOY.md`:S46 节(回滚 = 取消注释 + restart,一行级);
- 判定落 `.s46/mtp-adopt.md`(对齐 .s41 体例);
- `sudo systemctl restart z100-vllm-mtp2` + 复核 spec 行 + 最终 e2e
  (`--tag s46-prod`);
- FP 指纹变化记录(-speculative-config 回到 CLI)。

否决:产线零变更(现役即基线),判定落 `.s46/mtp-close.md`。

---

## 6. 执行顺序与产物

```
M0 → M1 → M2 → M3(ABAB)→ M4 → M5     # 全链 env 级,1 天内
                └─(M4 数据)→ lm_head M≤3 门扩展(候选,另裁)
```

- 产物:`.s46/{mtp-adopt|mtp-close}.md`、
  `docs/perf/data/s46-{m2-board,m3-e2e,m4-ledger,longctx}.txt`、
  `runs/s46/{e2e-s46-*,board-*}.json`、`.r3/so-backup/s46-before.md5`。
- 服务变更:仅 `serve_deploy.sh` 一行注释(采纳时);`.so`/代码零改动。
