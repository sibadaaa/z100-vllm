#!/usr/bin/env python3
"""Replay prompt_text recorded in old llm_speedtest runs against a live service.

Saves old/new outputs side by side and prints symptom stats:
- empty content
- early stop (output_tokens < max_tokens)
- repetition (max repeated 20-gram ratio)
- language mixing (CJK present in reasoning but content mostly EN)
"""
import argparse
import json
import re
import time

import httpx

SYSTEM_PROMPT = "You are a helpful assistant."


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return cjk / max(1, len(text))


def repeat_score(text: str, n: int = 20) -> float:
    """Fraction of n-grams that are duplicates (rough loop detector)."""
    words = re.findall(r"\S+", text or "")
    if len(words) < n * 2:
        return 0.0
    grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    seen = set()
    dup = 0
    for g in grams:
        if g in seen:
            dup += 1
        else:
            seen.add(g)
    return dup / len(grams)


def post_one(client: httpx.Client, api_url: str, model: str, prompt: str,
             max_tokens: int, temperature: float, top_p: float,
             timeout_s: float, tools: list | None = None):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    t0 = time.time()
    try:
        r = client.post(api_url, json=payload, timeout=timeout_s)
        dt = time.time() - t0
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:500]}", "seconds": dt}
        j = r.json()
        ch = j["choices"][0]
        msg = ch["message"]
        usage = j.get("usage", {})
        return {
            "content": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
            "tool_calls": msg.get("tool_calls"),
            "finish_reason": ch.get("finish_reason"),
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "seconds": dt,
        }
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}", "seconds": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-runs", required=True, help="old run JSON(s), comma separated")
    ap.add_argument("--api-url", default="http://172.17.0.2:8523/v1/chat/completions")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-items", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--start-at", type=int, default=0)
    a = ap.parse_args()

    items = []
    for path in a.old_runs.split(","):
        d = json.load(open(path, encoding="utf-8"))
        cfg = d.get("config", {})
        max_out = cfg.get("output_length", 128)
        for res in d.get("results", []):
            if res.get("status") != "成功":
                continue
            for det in res.get("concurrent_details") or []:
                pt = det.get("prompt_text")
                if not pt:
                    continue
                items.append(
                    {
                        "src": path,
                        "prompt_len": res.get("prompt_length"),
                        "max_out": max_out,
                        "prompt_text": pt,
                        "old_content": det.get("output_content") or "",
                        "old_reasoning": det.get("reasoning_content") or "",
                        "old_out_tokens": det.get("output_tokens"),
                    }
                )
    items = items[a.start_at : a.start_at + a.max_items]
    print(f"replaying {len(items)} prompts, temperature={a.temperature}")

    out = []
    with httpx.Client() as client:
        for i, it in enumerate(items):
            r = post_one(
                client, a.api_url, a.model, it["prompt_text"], it["max_out"],
                a.temperature, a.top_p, a.timeout,
            )
            rec = dict(it)
            rec["new"] = r
            out.append(rec)
            sym = {
                "empty_content": (r.get("content") or "") == "",
                "early_stop": (r.get("completion_tokens") or 0) < it["max_out"] * 0.5,
                "repeat": round(repeat_score(r.get("reasoning") or r.get("content") or ""), 3),
                "cjk_content": round(cjk_ratio(r.get("content") or ""), 3),
                "cjk_reasoning": round(cjk_ratio(r.get("reasoning") or ""), 3),
            }
            old_sym = {
                "empty_content": it["old_content"] == "",
                "repeat": round(repeat_score(it["old_reasoning"] or it["old_content"]), 3),
                "cjk_content": round(cjk_ratio(it["old_content"]), 3),
                "cjk_reasoning": round(cjk_ratio(it["old_reasoning"]), 3),
                "old_out_tokens": it["old_out_tokens"],
            }
            print(
                f"[{i + 1}/{len(items)}] len={it['prompt_len']} "
                f"new={sym} old={old_sym}",
                flush=True,
            )
            if r.get("error"):
                print("   error:", r["error"], flush=True)
            else:
                print("   new content head:", (r.get("content") or "")[:160].replace("\n", " "), flush=True)
                print("   old content head:", it["old_content"][:160].replace("\n", " "), flush=True)

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
