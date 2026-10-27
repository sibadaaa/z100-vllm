#!/usr/bin/env python3
"""Write the arm descriptor required by plan §3 O6.1.

One JSON per arm recording what was actually launched: git rev + dirty-file
digests (the tree is shared with two other agents), the promoted ``.so`` md5, the
complete ``VLLM_*`` environment of the live engine process (read from
``/proc/<pid>/environ``, so it is the effective env and not what we intended to
set), and the arm's own switch values.

  .venv/bin/python tools/dcu/s23_descriptor.py --out <dir>/descriptor.json \
      --window w1a --arm control --so-md5 f205be8f... --pid 2622752 \
      --arm-env "VLLM_DCU_FP8_EMU_FASTPATH=0" --desc "..."
"""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SWITCHES = (
    "VLLM_DCU_FP8_EMU_FASTPATH",
    "VLLM_DCU_FP8_QUANT_FASTPATH",
    "VLLM_DCU_FP8_FUSE_QUANT",
)


def _run(*cmd):
    try:
        return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                              check=False).stdout.strip()
    except OSError:
        return ""


def dirty_files():
    """Tracked changes plus untracked python/csrc files, with md5 digests."""
    out = []
    status = _run("git", "status", "--porcelain")
    for line in status.splitlines():
        path = line[3:].strip()
        if not path:
            continue
        if line.startswith("??") and not path.endswith((".py", ".cu", ".cuh", ".cpp")):
            continue
        full = os.path.join(REPO, path)
        digest = ""
        if os.path.isfile(full):
            with open(full, "rb") as f:
                digest = hashlib.md5(f.read()).hexdigest()
        out.append({"path": path, "status": line[:2].strip(), "md5": digest})
    return out


def proc_env(pid):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--desc", default="")
    ap.add_argument("--so-md5", default="")
    ap.add_argument("--pid", default="")
    ap.add_argument("--arm-env", default="")
    ap.add_argument("--extra", action="append", default=[],
                    help="key=value pairs stored under 'extra'")
    ap.add_argument("--unit-props", default="",
                    help="systemctl show snapshot of the recreated unit")
    args = ap.parse_args()

    arm_env = {}
    for pair in args.arm_env.split():
        key, _, val = pair.partition("=")
        if key:
            arm_env[key] = val

    unit_props = {}
    if args.unit_props:
        try:
            with open(args.unit_props) as f:
                for line in f:
                    key, sep, val = line.rstrip("\n").partition("=")
                    if sep:
                        unit_props[key] = val
        except OSError:
            pass

    unit_env = {}
    if unit_props.get("Environment"):
        for kv in shlex.split(unit_props["Environment"]):
            key, _, val = kv.partition("=")
            if key:
                unit_env[key] = val

    live = proc_env(args.pid) if args.pid else {}
    env_source = "proc"
    if not live:
        # Reading /proc/<engine pid>/environ of a systemd-managed process is
        # denied in this sandbox, so fall back to the unit's own Environment=
        # property, which is where the driver injects the arm switches
        # (systemd-run --setenv).  Recorded as env_source so a reader knows.
        live = dict(unit_env)
        env_source = "unit-props"
    vllm_env = {k: v for k, v in sorted(live.items()) if k.startswith("VLLM")}
    effective = {name: live.get(name, "<unset>") for name in SWITCHES}

    mismatches = []
    for name, want in arm_env.items():
        got = live.get(name, "<unset>")
        if got != want:
            mismatches.append({"name": name, "want": want, "got": got})
    if env_source != "proc" and not live:
        mismatches.append({"name": "<env probe>", "want": "process or unit env",
                           "got": "unreadable"})

    extra = {}
    for pair in args.extra:
        key, _, val = pair.partition("=")
        extra[key] = val

    payload = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "window": args.window,
        "arm": args.arm,
        "desc": args.desc,
        "git_rev": _run("git", "rev-parse", "HEAD"),
        "git_rev_short": _run("git", "rev-parse", "--short", "HEAD"),
        "git_branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_files": dirty_files(),
        "so_md5_promoted": args.so_md5,
        "service_pid": args.pid,
        "arm_env_requested": arm_env,
        "switch_effective": effective,
        "env_mismatch": mismatches,
        "env_mismatch_ok": not mismatches,
        "env_source": env_source,
        "unit_env": unit_env,
        "vllm_env_live": vllm_env,
        "profile": extra.get("profile", arm_env.get("VLLM_DCU_FP8_DECODE_CFG", "")),
        "unit_props": unit_props,
        "extra": extra,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    status = "OK" if not mismatches else "MISMATCH"
    print(f"[descriptor] {args.window}/{args.arm} git={payload['git_rev_short']} "
          f"so={args.so_md5[:8]} pid={args.pid} switches={effective} {status}")
    for item in mismatches:
        print(f"[descriptor] MISMATCH {item['name']}: want {item['want']} "
              f"got {item['got']}", file=sys.stderr)
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
