import hashlib
import json
import sys
import time
import urllib.request

PROMPT = (
    "Write a Python function that returns the n-th Fibonacci number using "
    "iteration, then explain its time complexity in two sentences."
)

body = json.dumps(
    {
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 128,
        "temperature": 0,
        "repetition_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
).encode()
req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=body,
    headers={"Content-Type": "application/json"},
)
urllib.request.urlopen(req, timeout=600).read()  # warmup
t0 = time.time()
r = json.load(urllib.request.urlopen(req, timeout=600))
dt = time.time() - t0
c = r["choices"][0]["message"]["content"]
spec = (r.get("metrics") or {}).get("speculative_decoding") or {}
label = sys.argv[1] if len(sys.argv) > 1 else "arm"
tokens = r["usage"]["completion_tokens"]
print(
    "%s: tokens=%d wall=%.2fs tok/s=%.2f acc=%s md5=%s"
    % (
        label,
        tokens,
        dt,
        tokens / dt,
        spec.get("mean_acceptance_length"),
        hashlib.md5(c.encode()).hexdigest(),
    )
)
