#!/usr/bin/env python3
"""Post-restart smoke: one real chat request + the fixed rollback prompt.

Records ``usage`` for both calls (plan §3 O6.2 requires a real chat smoke with
usage after every restart) and the fixed continuation used as the rollback
identity check (``docs/result/rollback-rehearsal.md``).

  .venv/bin/python tools/dcu/s23_smoke.py --out .s23/w1/O6/w1a/control/smoke.json

Exit 0 on success, 1 when either call fails, and --expect-text / --expect-json
turn it into the rollback identity gate.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import s23_common  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--expect-json", default="",
                    help="compare this smoke against a recorded one")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    s23_common.install_auth()
    if not s23_common.wait_ready(timeout=args.timeout):
        print("[smoke] FAIL service not ready", file=sys.stderr)
        return 1
    try:
        payload = s23_common.smoke()
    except Exception as exc:  # noqa: BLE001 - any failure invalidates the arm
        print(f"[smoke] FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"[smoke] completion={payload['completion_text']!r}")
    print(f"[smoke] chat={payload['chat_text']!r} "
          f"usage={payload['chat_usage']}")
    verdict = "smoke-ok"
    if args.expect_json:
        with open(args.expect_json) as f:
            ref = json.load(f)
        same = (ref.get("completion_token_ids")
                == payload.get("completion_token_ids"))
        payload["matches_reference"] = same
        payload["reference"] = args.expect_json
        verdict = "smoke-identical" if same else "smoke-DIVERGED"
        print(f"[smoke] vs {args.expect_json}: {verdict}")
        print(f"[smoke]   ref={ref.get('completion_text')!r}")
    payload["verdict"] = verdict

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"[smoke] saved -> {args.out}")
    return 0 if verdict in ("smoke-ok", "smoke-identical") else 1


if __name__ == "__main__":
    raise SystemExit(main())
