#!/usr/bin/env python3
"""S24 L6: prefix-cache hit-rate replay for agent-style conversations.

Two modes over the same server:
  grow        multi-turn agent conversation: one fixed system prompt + tool
              schema + a long document, each turn APPENDS a user message and
              re-sends the whole history (the pattern prefix caching exists
              for).
  independent every turn a fresh unrelated prompt (the anti-pattern the user
              workload currently shows at 22-34% hits).

Per turn it records TTFT, prompt_tokens and
usage.prompt_tokens_details.cached_tokens (the server runs
--enable-prompt-tokens-details), so the hit fraction is per-request evidence,
not a global counter.  Summary reports p50 TTFT per mode + mean hit fraction.

  .venv/bin/python tools/dcu/s24_prefix_replay.py --mode grow --turns 8 \
      --doc-tokens 6000 --tag l6_grow --out .s24/w1/L6
"""

import argparse
import json
import os
import statistics
import time
import urllib.request

BASE = os.environ.get("S24_BASE", "http://127.0.0.1:8000")
KEY_FILE = "/mnt/game/host-serve/api-key"

SYSTEM = (
    "你是一个严谨的运维助手。回答前先列出步骤，再给命令。"
    "所有命令都必须可在 Rocky Linux 9 上执行。"
)
TOOLS_TEXT = """available_tools:
- check_disk(host, path): returns df -h output for path
- check_mem(host): returns free -g
- check_gpu(host): returns hy-smi summary
- restart_service(host, unit): systemctl restart <unit>
- tail_log(host, path, lines): tail -n <lines> path
"""
DOC = (
    "集群巡检手册。节点 z100-a 运行 4 张海光 DCU Z100，驱动 rock-6.4.2，"
    "用户态 DTK 26.04。推理服务由 systemd 用户单元 z100-vllm-host.service "
    "托管，启动脚本 tools/dcu/serve_host_dflash2.sh，监听 8000 端口。"
    "常见故障：1) 服务 OOM：检查 hy-smi 显存占用与 gpu-memory-utilization；"
    "2) TTFT 突增：检查 prefix cache 命中率与 max-num-batched-tokens；"
    "3) decode 变慢：检查接受长度 (acceptance length) 与 spec 配置。"
) * 40  # repeated to reach ~doc-tokens


def headers():
    with open(KEY_FILE) as f:
        key = f.read().strip()
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def approx_tokens(s):
    # 1 token ~ 2.2 chars for mixed zh/en; good enough for a replay fixture.
    return max(1, int(len(s) / 2.2))


def turn_messages(mode, i):
    if mode == "grow":
        msgs = [{"role": "system", "content": SYSTEM + "\n" + TOOLS_TEXT + DOC}]
        for t in range(i):
            msgs.append({"role": "user",
                         "content": f"第{t + 1}轮：请检查节点 z100-a 的第{t + 1}项指标并给命令。"})
            msgs.append({"role": "assistant",
                         "content": f"第{t + 1}轮执行完毕：指标正常，命令已记录。"})
        msgs.append({"role": "user",
                     "content": f"第{i + 1}轮：请检查节点 z100-a 的磁盘与显存，给出巡检结论与命令。"})
        return msgs
    # independent: fresh topic every turn, no shared body beyond the system line
    topics = ["dns 配置差异分析", "时间同步漂移排查", "内核参数对比", "文件句柄上限核查",
              "网络重传统计", "磁盘 SMART 健康", "日志轮转配置", "NFS 挂载参数",
              "容器镜像清理", "备份任务巡检"]
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user",
         "content": f"请分析：{topics[i % len(topics)]}，节点 z100-b，给出步骤与命令。附上背景：{DOC[:2000]}" + str(i) * 3},
    ]


def run_turn(msgs, gen, timeout):
    body = json.dumps({
        "model": "qwen3.8-27b",
        "messages": msgs,
        "max_tokens": gen,
        "temperature": 0.0,
        "stream": True,
        # the server parses qwen3 reasoning, so early deltas are
        # reasoning_content; TTFT is the first token of ANY kind (it marks
        # prefill completion), and usage needs an explicit opt-in.
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers=headers())
    t0 = time.perf_counter()
    ttft = None
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            try:
                ev = json.loads(line[6:])
            except Exception:
                continue
            ch = (ev.get("choices") or [{}])[0]
            delta = ch.get("delta", {})
            if ttft is None and (delta.get("content") or
                                 delta.get("reasoning") or
                                 delta.get("reasoning_content")):
                ttft = time.perf_counter() - t0
            if ev.get("usage"):
                usage = ev["usage"]
    return ttft, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("grow", "independent"), default="grow")
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=1,
                    help="repetitions of the whole replay (grow: fresh each)")
    ap.add_argument("--doc-tokens", type=int, default=6000)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--tag", default="l6")
    ap.add_argument("--out", default=".")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    global DOC
    doc = DOC
    while approx_tokens(doc) < args.doc_tokens:
        doc += DOC
    doc = doc[: int(args.doc_tokens * 2.2)]

    os.makedirs(args.out, exist_ok=True)
    jsonl = os.path.join(args.out, "prefix_replay.jsonl")
    recs = []
    for rnd in range(args.rounds):
        for i in range(args.turns):
            msgs = turn_messages(args.mode, i)
            if args.mode == "grow":
                msgs = [dict(m) for m in msgs]
                # patch the configurable document into the system turn
                msgs[0]["content"] = SYSTEM + "\n" + TOOLS_TEXT + doc
            ttft, usage = run_turn(msgs, args.gen, args.timeout)
            det = (usage or {}).get("prompt_tokens_details") or {}
            rec = {
                "mode": args.mode, "round": rnd, "turn": i,
                "ttft_ms": round(ttft * 1e3, 1) if ttft else None,
                "prompt_tokens": (usage or {}).get("prompt_tokens"),
                "cached_tokens": det.get("cached_tokens"),
                "hit_frac": round(det.get("cached_tokens", 0) /
                                  max(1, (usage or {}).get("prompt_tokens", 1)), 3),
                "tag": args.tag,
            }
            print(json.dumps(rec), flush=True)
            with open(jsonl, "a") as f:
                f.write(json.dumps(rec) + "\n")
            recs.append(rec)
            time.sleep(1.0)
    ok = [r for r in recs if r["ttft_ms"]]
    if ok:
        s = {
            "mode": args.mode, "tag": args.tag, "turns": len(ok),
            "ttft_p50_ms": round(statistics.median(r["ttft_ms"] for r in ok), 1),
            "ttft_mean_ms": round(statistics.fmean(r["ttft_ms"] for r in ok), 1),
            "hit_frac_mean": round(statistics.fmean(r["hit_frac"] for r in ok), 3),
            "hit_frac_last_turn": ok[-1]["hit_frac"],
        }
        print("[SUMMARY] " + json.dumps(s), flush=True)
        with open(os.path.join(args.out, "prefix-replay-summary.json"), "a") as f:
            f.write(json.dumps(s) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
