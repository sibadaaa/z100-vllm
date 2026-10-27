#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""S14d E3: controlled conc=2 same-context consistency.

S14b registered one uncontrolled observation (13.7 nat) when a probe request was
in flight together with a decoding stream.  This makes it controlled and
prefix-cache-clean:

  1. on a COLD engine (this is the first traffic), the TARGET prompt and a
     DIFFERENT filler prompt are sent CONCURRENTLY (thread barrier) -- the
     filler creates real concurrency without sharing a prefix with the target,
  2. then the TARGET prompt is sent once more, sequentially, on the same engine.

The question: does the target's token stream / same-context logprobs change
because another request was being decoded at the same time?

Usage:
    s14d_conc2.py --port 8621 --label E5SW16 --out /mnt/game/s14d/conc2-E5SW16.jsonl
"""

import argparse
import json
import threading
import time
import urllib.request

TARGET = ("请用中文写一段三百字左右的说明，解释量化交易中滑点的成因、度量方法，"
          "以及降低滑点的工程手段。")
FILLER = ("Explain step by step how a modern transformer inference server schedules "
          "continuous batching, and why prefix caching helps throughput.")


def fire(base, model, prompt, max_tokens, plp_k, slot, out, barrier):
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
    barrier.wait()
    t0 = time.time()
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as resp:
        body = json.load(resp)
    elapsed = time.time() - t0
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
    with out["lock"]:
        out["recs"][slot] = {
            "label": out["label"], "slot": slot, "prompt": prompt,
            "finish_reason": choice.get("finish_reason"),
            "token_ids": choice.get("token_ids") or [],
            "n_tokens": len(choice.get("token_ids") or []),
            "text": choice["message"].get("content") or "",
            "n_prompt_pos": len(plp), "prompt_logprobs": plp, "rows": rows,
            "t_start": t0, "elapsed_s": round(elapsed, 3),
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
    print(f"{na:24s} vs {nb:24s} tokens={len(ta)}/{len(tb)} bit-identical={ta == tb} "
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

    out = {"recs": {}, "label": args.label, "lock": threading.Lock()}
    barrier = threading.Barrier(2)
    ths = [threading.Thread(target=fire, args=(base, args.model, TARGET, args.max_tokens,
                                              args.plp_k, "conc_target", out, barrier)),
           threading.Thread(target=fire, args=(base, args.model, FILLER, args.max_tokens,
                                              args.plp_k, "conc_filler", out, barrier))]
    for t in ths:
        t.start()
    for t in ths:
        t.join()

    seq = {"recs": {}, "label": args.label + "-seq", "lock": threading.Lock()}
    fire(base, args.model, TARGET, args.max_tokens, args.plp_k, "seq_target", seq,
         threading.Barrier(1))

    recs = {"conc_target": out["recs"]["conc_target"],
            "conc_filler": out["recs"]["conc_filler"],
            "seq_target": seq["recs"]["seq_target"]}
    with open(args.out, "w", encoding="utf-8") as fh:
        for k in ("conc_target", "conc_filler", "seq_target"):
            fh.write(json.dumps(recs[k], ensure_ascii=False) + "\n")

    print(f"# conc2 label={args.label} target={recs['conc_target']['n_tokens']} tok "
          f"({recs['conc_target']['elapsed_s']}s) filler={recs['conc_filler']['n_tokens']} tok "
          f"({recs['conc_filler']['elapsed_s']}s) seq_after={recs['seq_target']['n_tokens']} tok")
    res = [cmp_pair(recs["conc_target"], recs["seq_target"], "conc-target", "seq-after")]
    with open(args.out + ".summary.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
