#!/usr/bin/env python3
"""Functional "intelligence" battery for the 27B service.

- tool-call prompts: check tool_calls presence, name, JSON args
- reasoning prompts: check a known answer appears in content(+reasoning)
- language prompt: Chinese output not degraded to English

Usage: quality_battery.py --label X --api-url ... --out out.json [--max-reasoning 768]
"""
import argparse
import json
import re
import time

import httpx

SYSTEM = "You are a helpful assistant."

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a message to a person",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["recipient", "text"],
            },
        },
    },
]

TOOL_CASES = [
    ("北京今天天气怎么样？", "get_weather", {"city": ["北京"]}),
    ("帮我查一下上海现在的温度。", "get_weather", {"city": ["上海"]}),
    ("What's the weather in Paris?", "get_weather", {"city": ["Paris", "paris"]}),
    ("帮我算一下 23*47 等于多少。", "calculator", {"expression": ["23*47", "23 * 47"]}),
    ("给张三发一条消息：明天见。", "send_message", {"recipient": ["张三"], "text": ["明天见"]}),
    ("请把'今天天气很好'翻译成英文，不要调用任何工具。", None, None),
]

REASON_CASES = [
    ("一个笼子里有鸡和兔共 35 只，脚共 94 只，鸡和兔各多少只？", ["12", "23", "十二", "二十三"]),
    ("9.9 和 9.11 哪个大？直接给结论。", ["9.9"]),
    ("如果所有 bloops 都是 razzles，所有 razzles 都是 lazzles，那么所有 bloops 都是 lazzles 吗？", ["是", "yes", "Yes"]),
    ("数列 1,1,2,3,5,8,? 的下一项是多少？直接给数字。", ["13"]),
    ("3 个人 3 天喝 3 桶水，那么 9 个人 9 天喝几桶水？", ["27"]),
    ("把 'the quick brown fox' 每个单词倒序输出。", ["xof", "nworb", "kciuq"]),
]


def post(client, url, payload, timeout):
    t0 = time.time()
    try:
        r = client.post(url, json=payload, timeout=timeout)
        dt = time.time() - t0
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:300]}", "seconds": dt}
        j = r.json()
        ch = j["choices"][0]
        msg = ch["message"]
        return {
            "content": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
            "tool_calls": msg.get("tool_calls"),
            "finish_reason": ch.get("finish_reason"),
            "usage": j.get("usage", {}),
            "seconds": dt,
        }
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}", "seconds": time.time() - t0}


def check_tool(res, want_name, want_args):
    if res.get("error"):
        return False, f"error: {res['error'][:120]}"
    if want_name is None:
        ok = not res.get("tool_calls")
        return ok, "no-tool-call" if ok else f"unexpected tool {res['tool_calls'][0]['function']['name']}"
    tcs = res.get("tool_calls") or []
    if not tcs:
        return False, "no tool_call"
    tc = tcs[0]["function"]
    if tc["name"] != want_name:
        return False, f"name={tc['name']} want={want_name}"
    try:
        args = json.loads(tc["arguments"])
    except Exception:  # noqa: BLE001
        return False, f"bad JSON args: {tc['arguments'][:100]}"
    for k, allowed in (want_args or {}).items():
        v = str(args.get(k, ""))
        if not any(a.lower() in v.lower() for a in allowed):
            return False, f"arg {k}={v!r} want {allowed}"
    return True, "ok"


def check_reason(res, needles):
    if res.get("error"):
        return False, f"error: {res['error'][:120]}"
    text = (res.get("content") or "") + "\n" + (res.get("reasoning") or "")
    for n in needles:
        if n.lower() in text.lower():
            return True, "ok"
    return False, f"missing {needles}; head={(res.get('content') or res.get('reasoning') or '')[:120]!r}"


def cjk_ratio(s):
    return sum(1 for c in s if "\u4e00" <= c <= "\u9fff") / max(1, len(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--api-url", default="http://172.17.0.2:8523/v1/chat/completions")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reason-max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=float, default=900.0)
    a = ap.parse_args()

    out = {"label": a.label, "tool": [], "reason": [], "language": None}
    n_tool_ok = 0
    with httpx.Client() as client:
        for prompt, want_name, want_args in TOOL_CASES:
            payload = {
                "model": a.model,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.0,
                "tools": TOOLS,
                "tool_choice": "auto",
            }
            res = post(client, a.api_url, payload, a.timeout)
            ok, why = check_tool(res, want_name, want_args)
            n_tool_ok += ok
            out["tool"].append({"prompt": prompt, "want": want_name, "ok": ok, "why": why, "res": res})
            print(f"[tool] {'PASS' if ok else 'FAIL'} {prompt[:30]} -> {why}", flush=True)

        n_reason_ok = 0
        for prompt, needles in REASON_CASES:
            payload = {
                "model": a.model,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": prompt}],
                "max_tokens": a.reason_max_tokens,
                "temperature": 0.0,
            }
            res = post(client, a.api_url, payload, a.timeout)
            ok, why = check_reason(res, needles)
            n_reason_ok += ok
            out["reason"].append({"prompt": prompt, "needles": needles, "ok": ok, "why": why, "res": res})
            ct = (res.get("usage") or {}).get("completion_tokens")
            print(f"[reason] {'PASS' if ok else 'FAIL'} {prompt[:26]} -> {why} (ct={ct})", flush=True)

        lang_prompt = "用中文写一段关于秋天的短文，100 字左右。"
        payload = {
            "model": a.model,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": lang_prompt}],
            "max_tokens": 512,
            "temperature": 0.0,
        }
        res = post(client, a.api_url, payload, a.timeout)
        content = res.get("content") or ""
        ratio = cjk_ratio(content)
        out["language"] = {"prompt": lang_prompt, "cjk_ratio": ratio, "res": res}
        print(f"[lang] cjk_ratio={ratio:.2f} content_head={content[:60]!r}", flush=True)

    score = {
        "label": a.label,
        "tool_pass": f"{n_tool_ok}/{len(TOOL_CASES)}",
        "reason_pass": f"{n_reason_ok}/{len(REASON_CASES)}",
        "lang_cjk": round(out["language"]["cjk_ratio"], 3) if out["language"] else None,
    }
    out["score"] = score
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("SCORE:", score)
    print("saved", a.out)


if __name__ == "__main__":
    main()
