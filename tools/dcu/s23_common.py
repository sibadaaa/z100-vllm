#!/usr/bin/env python3
"""Shared helpers for the O6 (S23) end-to-end verification tooling.

The production service reads its API key from a 0600 file instead of exporting
it, and every ``/v1/*`` route is behind that key while ``/metrics`` and
``/health`` are open.  The S20 probes only speak ``urllib`` and never send an
Authorization header, so instead of forking them this module patches
``urllib.request.urlopen`` to attach the key to every ``Request`` object.  That
keeps ``s20_single_probe.py`` / ``s20_conc_sweep.py`` byte-identical and still
usable against the keyed service (one measurement code path, no drift).
"""

import hashlib
import json
import os
import re
import time
import urllib.request

BASE = os.environ.get("S23_BASE", "http://127.0.0.1:8000")
KEY_FILE = os.environ.get("S23_KEY_FILE", "/mnt/game/host-serve/api-key")
MODEL = os.environ.get("S23_MODEL", "qwen3.8-27b")

# The rollback criterion from docs/result/rollback-rehearsal.md: a fixed prompt
# whose continuation every stack has produced identically so far.
SMOKE_PROMPT = "The capital of France is"


def api_key():
    """Return the API key ('' when auth is off)."""
    if os.environ.get("S23_API_KEY"):
        return os.environ["S23_API_KEY"]
    try:
        with open(KEY_FILE) as f:
            return f.read().strip()
    except OSError:
        return ""


_ORIG_URLOPEN = urllib.request.urlopen


def install_auth():
    """Make every Request built by the S20 probes carry the API key."""
    key = api_key()
    if not key or getattr(urllib.request, "_s23_auth", False):
        return key

    def urlopen(req, *args, **kwargs):
        if isinstance(req, urllib.request.Request):
            req.add_header("Authorization", f"Bearer {key}")
        return _ORIG_URLOPEN(req, *args, **kwargs)

    urllib.request.urlopen = urlopen
    urllib.request._s23_auth = True
    return key


def authed_request(path, body=None, timeout=1800):
    """Request to the keyed service; the key is attached explicitly.

    ``install_auth`` exists for the reused S20 probes, but this module must not
    rely on it: calling the captured original ``urlopen`` would skip the patch
    and silently 401 (which is exactly what a first smoke run proved).
    """
    headers = {"Authorization": f"Bearer {api_key()}"}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{BASE}{path}", data=data, headers=headers)
    return _ORIG_URLOPEN(req, timeout=timeout)


def post(path, body, timeout=1800):
    """POST JSON to the service; returns the decoded response."""
    with authed_request(path, body, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def service_up(timeout=3):
    """True when /v1/models answers 200 with a model list."""
    try:
        with authed_request("/v1/models", timeout=timeout) as resp:
            if resp.status != 200:
                return False
            return bool(json.loads(resp.read().decode()).get("data"))
    except Exception:  # noqa: BLE001 - any transport/HTTP error means not ready
        return False


def wait_ready(timeout=900, poll=5):
    """Block until the engine serves /v1/models; returns (ok, waited_s)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if service_up():
            return True, time.time() - t0
        time.sleep(poll)
    return False, time.time() - t0


def completion_body(prompt, max_tokens=16, **extra):
    """The exact greedy-completion body, in one place so it can be hashed."""
    body = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "return_token_ids": True,
        "stream": False,
    }
    body.update(extra)
    return body


def body_sha(body):
    """Stable hash of a request body (key order normalised)."""
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def completions(prompt, max_tokens=16, timeout=600, **extra):
    """Greedy completion with token ids + usage (the parity building block)."""
    body = completion_body(prompt, max_tokens=max_tokens, **extra)
    return post("/v1/completions", body, timeout=timeout)


def smoke():
    """One real chat request + one fixed-prompt completion smoke.

    The chat call proves the tool/chat path still serves and records ``usage``
    (required per arm); the completion call gives the fixed-string rollback
    criterion.  Greedy + ``ignore_eos`` so both are comparable across arms.
    """
    out = {"fixed_prompt": SMOKE_PROMPT}
    comp = completions(SMOKE_PROMPT, max_tokens=12)
    out["completion_text"] = comp["choices"][0]["text"]
    out["completion_usage"] = comp.get("usage")
    out["completion_token_ids"] = comp["choices"][0].get("token_ids")
    chat = post(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply with one word: ping"}],
            "max_tokens": 64,
            "temperature": 0.0,
        },
    )
    msg = chat["choices"][0]["message"] or {}
    out["chat_text"] = msg.get("content")
    out["chat_reasoning"] = msg.get("reasoning_content") or msg.get("reasoning")
    out["chat_usage"] = chat.get("usage")
    out["chat_finish_reason"] = chat["choices"][0].get("finish_reason")
    return out


def proc_env(pid):
    """Environ of a live process, as a dict (empty when unreadable)."""
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            raw = f.read()
    except OSError:
        return {}
    env = {}
    for item in raw.split(b"\0"):
        key, sep, val = item.partition(b"=")
        if sep:
            env[key.decode("utf-8", "ignore")] = val.decode("utf-8", "ignore")
    return env


RX_MET = {
    "gen_tok": r"^vllm:generation_tokens_total\{[^}]*\}\s+([0-9.e+]+)",
    "run": r"^vllm:num_requests_running\{[^}]*\}\s+([0-9.e+]+)",
}


def metrics():
    """Scrape the two counters the window driver gates on."""
    txt = urllib.request.urlopen(f"{BASE}/metrics", timeout=8).read().decode()
    out = {}
    for key, pat in RX_MET.items():
        m = re.search(pat, txt, re.M)
        if m:
            out[key] = float(m.group(1))
    return out


def wait_idle(timeout=180):
    """Wait until the engine reports no running and no queued request."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            m = metrics()
            if m.get("run", 0.0) == 0.0:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    return False
