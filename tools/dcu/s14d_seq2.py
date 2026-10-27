#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""S14d: "same prompt twice, sequentially, on a cold engine" -- the control that
separates *concurrency* from *request history*.

E5's ``conc2`` task sends the target prompt concurrently with a filler and then
sends the target again serially; the two differed by mean 1.82 nat / max 15.0 nat.
That arm cannot tell whether the cause is (a) the concurrent decoding stream or
(b) the target simply not being the first request any more.

This script removes concurrency completely:

  * request 1: the target prompt  (cold engine, first traffic)
  * request 2: the SAME target prompt again, strictly sequentially (single
    request in flight, nothing else ever queued)

  * if 1 == 2 bit-for-bit  -> request history is irrelevant; E5's conc2
    difference must be due to the concurrent stream;
  * if 1 != 2              -> the engine's numeric output depends on how many
    requests it has already served (a "first request vs later request" effect),
    which is a production-critical non-determinism of its own.

Usage:
    s14d_seq2.py --port 8623 --label E5SEQ --out /mnt/game/s14d/seq2-E5SEQ.jsonl
"""

import argparse
import json
import time
import urllib.request

TARGET = ("请用中文写一段三百字左右的说明，解释量化交易中滑点的成因、度量方法，"
          "以及降低滑点的工程手段。")


def fire(base, model, prompt, max_tokens, plp_k, slot):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
        "return_token_ids": True,
        "logprobs": True,
        "top_logprobs": 5,
        "prompt_logprobs": plp_k,
    }
    t0 = time.time()
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as resp:
        body = json.load(resp)
    choice = body["choices"][0]
    lp = choice.get("logprobs") or {}
    rows = []
    for i, entry in enumerate(lp.get("content") or []):
        top = [(t["token"], t["logprob"]) for t in entry.get("top_logprobs") or []]
        top.sort(key=lambda e: -e[1])
        rows.append({"i": i, "token_id": entry.get("token_id"), "token": entry["token"],
                     "logprob": entry["logprob"],
                     "gap": (top[0][1] - top[1][1]) if len(top) > 1 else None,
                     "top": top})
    plp_raw = body.get("prompt_logprobs") or choice.get("prompt_logprobs") or []
    plp = []
    for entry in plp_raw:
        if not entry:
            plp.append(None)
            continue
        plp.append(sorted(((int(t), v["logprob"]) for t, v in entry.items()),
                          key=lambda e: -e[1])[:plp_k])
    return {
        "slot": slot, "prompt": prompt, "finish_reason": choice.get("finish_reason"),
        "token_ids": choice.get("token_ids") or [],
        "n_tokens": len(choice.get("token_ids") or []),
        "text": choice["message"].get("content") or "",
        "n_prompt_pos": len(plp), "prompt_logprobs": plp, "rows": rows,
        "elapsed_s": round(time.time() - t0, 3),
    }


def cmp_pair(ra, rb, na, nb):
    ta, tb = ra["token_ids"], rb["token_ids"]
    first = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
    if first is None and len(ta) != len(tb):
        first = min(len(ta), len(tb))
    common = min(len(ra["rows"]), len(rb["rows"])) if first is None else first
    maxd = 0.0
    for i in range(common):
        if ra["rows"][i]["token_id"] == rb["rows"][i]["token_id"]:
            maxd = max(maxd, abs(ra["rows"][i]["logprob"] - rb["rows"][i]["logprob"]))
        else:
            break
    npos = min(len(ra["prompt_logprobs"]), len(rb["prompt_logprobs"]))
    pmax = pmean = 0.0
    pn = 0
    for i in range(npos):
        ea, eb = ra["prompt_logprobs"][i], rb["prompt_logprobs"][i]
        if not ea or not eb:
            continue
        da, db = dict(ea), dict(eb)
        sh = set(da) & set(db)
        if not sh:
            continue
        pn += 1
        d = [abs(da[t] - db[t]) for t in sh]
        pmax = max(pmax, max(d))
        pmean += sum(d) / len(d)
    pmean = pmean / pn if pn else 0.0
    print(f"{na:20s} vs {nb:20s} tokens={len(ta)}/{len(tb)} bit-identical={ta == tb} "
          f"first_div={first} maxd_common={maxd:.6f} "
          f"plp_pos={pn} plp_max={pmax:.5f} plp_mean={pmean:.5f}")
    return dict(pair=f"{na}|{nb}", n_a=len(ta), n_b=len(tb), bit_identical=ta == tb,
                first_div=first, maxd_common=maxd, plp_pos=pn, plp_max=pmax,
                plp_mean=pmean)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--plp-k", type=int, default=20)
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"

    recs = {}
    for slot in ("req1_cold", "req2_after1"):
        recs[slot] = fire(base, args.model, TARGET, args.max_tokens, args.plp_k, slot)
        recs[slot]["label"] = args.label
        print(f"[s14d-seq2] {slot}: {recs[slot]['n_tokens']} tok "
              f"{recs[slot]['elapsed_s']}s", flush=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        for k in ("req1_cold", "req2_after1"):
            fh.write(json.dumps(recs[k], ensure_ascii=False) + "\n")

    res = [cmp_pair(recs["req2_after1"], recs["req1_cold"], "req2-after", "req1-cold")]
    with open(args.out + ".summary.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
