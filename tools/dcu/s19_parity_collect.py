#!/usr/bin/env python3
"""S19 parity probe: greedy completions from the live server, saved for diffing
across speculative arms (DFlash vs MTP).

    python /src/tools/dcu/s19_parity_collect.py <label>
"""

import hashlib
import json
import sys
import urllib.request

PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number using "
    "iteration, then explain its time complexity in two sentences.",
    "List the first five prime numbers, then say which of them is even and why.",
    "A train travels 120 km in 1.5 hours. What is its average speed? Show the "
    "arithmetic.",
]

URL = "http://127.0.0.1:8000/v1/chat/completions"


def ask(prompt: str) -> dict:
    body = json.dumps(
        {
            "model": "qwen3.8-27b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128,
            "temperature": 0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    out = {"label": label, "results": []}
    for prompt in PROMPTS:
        d = ask(prompt)
        content = d["choices"][0]["message"]["content"]
        spec = (d.get("metrics") or {}).get("speculative_decoding") or {}
        out["results"].append(
            {
                "prompt": prompt,
                "completion_tokens": d["usage"]["completion_tokens"],
                "md5": hashlib.md5(content.encode()).hexdigest(),
                "acceptance_length": spec.get("mean_acceptance_length"),
                "content": content,
            }
        )
        print(
            f"[{label}] {out['results'][-1]['md5']} "
            f"tokens={out['results'][-1]['completion_tokens']} "
            f"acc={out['results'][-1]['acceptance_length']}"
        )
    path = f"/mnt/game/dflash/parity-{label}.json"
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
