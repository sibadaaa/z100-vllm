#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""S30c WP1-N: microbench + parity for the GDN M=1 native-layout decode kernel.

Production decode shape (Qwen3.5-27B-FP8, TP=4 rank): T decode tokens, 12 value
heads (48/4), 4 q/k heads (16/4), key/value head dim 128, fp32 SSM state in the
native [slots, 12, 128, 128] layout (k contiguous: stride(2)=128, stride(3)=1),
mixed_qkv [T, 2560] fp16, out [T, 1, 12, 128] fp16.

Env must be set BEFORE the first op call (host statics are process-wide):

  # serial baseline (production today)
  python tools/dcu/s30c_gdn_m1_bench.py run --impl serial --tag ctl
  # m1 native, cfg 1..4 (VLLM_GFX906_GDN_DEC_M1_CFG)
  python tools/dcu/s30c_gdn_m1_bench.py run --impl m1 --cfg 1 --tag cfg1
  python tools/dcu/s30c_gdn_m1_bench.py run --impl m1 --cfg 1 --slot-neg --tag cfg1neg
  # compare two saved runs (+ eager torch reference as third column)
  python tools/dcu/s30c_gdn_m1_bench.py compare .s30c/gdn_m1_serial1_T1.npz \
      .s30c/gdn_m1_m11_T1.npz
"""

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run")
    run.add_argument("--impl", choices=["serial", "m1"], required=True)
    run.add_argument("--cfg", type=int, default=1)
    run.add_argument("--tokens", type=int, default=1)
    run.add_argument("--slot-neg", action="store_true",
                     help="put one token on slot -1 (invalid-slot semantics)")
    run.add_argument("--iters", type=int, default=200)
    run.add_argument("--seed", type=int, default=30)
    run.add_argument("--out", default=".s30c")
    run.add_argument("--tag", default="")

    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("npz_a")
    cmp_.add_argument("npz_b")

    args = ap.parse_args()

    if args.cmd == "run":
        # Host statics (VLLM_GFX906_GDN_DEC_M1 / _CFG) are read once per
        # process: set them before torch/vllm are imported.
        os.environ["VLLM_GFX906_GDN_DEC_M1"] = "1" if args.impl == "m1" else "0"
        os.environ["VLLM_GFX906_GDN_DEC_M1_CFG"] = str(args.cfg)
        os.environ.pop("VLLM_GFX906_GDN_DEC_M1_DEBUG", None)
        os.environ.pop("VLLM_GFX906_GDN_DEC_M1_TIME", None)

    import numpy as np  # noqa: E402
    import torch  # noqa: E402

    if args.cmd == "run":
        from torch.profiler import ProfilerActivity, profile  # noqa: E402

        from vllm import _custom_ops as ops  # noqa: E402

    H, HV, KD, VD = 4, 12, 128, 128
    SLOTS = 8

    def build(tokens: int, seed: int, slot_neg: bool):
        torch.manual_seed(seed)
        dev = "cuda"
        fp16 = dict(device=dev, dtype=torch.float16)
        qkv_dim = 2 * H * KD + HV * VD
        mixed_qkv = (torch.randn(tokens, qkv_dim, **fp16) * 2.5).contiguous()
        a = (torch.randn(tokens, HV, **fp16) * 0.2).contiguous()
        b = (torch.randn(tokens, HV, **fp16) * 0.2).contiguous()
        A_log = (torch.randn(HV, **fp16) * 0.2).contiguous()
        dt_bias = (torch.randn(HV, **fp16) * 0.1).contiguous()
        state = (torch.randn(SLOTS, HV, VD, KD, device=dev,
                             dtype=torch.float32) * 0.2).contiguous()
        idx = torch.arange(1, tokens + 1, device=dev, dtype=torch.int32)
        if slot_neg:
            idx[0] = -1
        out = torch.zeros(tokens, 1, HV, VD, **fp16).contiguous()
        return dict(mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log,
                    dt_bias=dt_bias, state=state, idx=idx, out=out)

    def call(d):
        return ops.fused_recurrent_gated_delta_rule_gfx906_packed_decode(
            mixed_qkv=d["mixed_qkv"], a=d["a"], b=d["b"], A_log=d["A_log"],
            dt_bias=d["dt_bias"], state=d["state"], out=d["out"],
            state_indices=d["idx"], scale=KD ** -0.5,
            use_qk_l2norm_in_kernel=True, use_tiled_qk_head_mapping=False,
            use_transposed_state=False)

    def eager_ref(d):
        """Sequential-k fp32 reference on the native layout (CPU)."""
        tq = d["mixed_qkv"].cpu().float()
        ta = d["a"].cpu().float()
        tb = d["b"].cpu().float()
        tal = d["A_log"].cpu().float()
        tdb = d["dt_bias"].cpu().float()
        tidx = d["idx"].cpu()
        t = tq.shape[0]
        q = tq[:, :H * KD].view(t, H, KD)
        k = tq[:, H * KD:2 * H * KD].view(t, H, KD)
        v = tq[:, 2 * H * KD:].view(t, HV, VD)
        state = d["state"].cpu().clone()
        out = torch.zeros(t, HV, VD, dtype=torch.float32)
        head_ratio = HV // H
        for ti in range(t):
            slot = int(tidx[ti])
            if slot < 0:
                continue
            for hv in range(HV):
                qh = q[ti, hv // head_ratio]
                kh = k[ti, hv // head_ratio]
                qn = torch.rsqrt(float((qh * qh).sum()) + 1e-6)
                kn = torch.rsqrt(float((kh * kh).sum()) + 1e-6)
                x = float(ta[ti, hv]) + float(tdb[hv])
                sp = float(np.log1p(np.exp(x))) if x <= 20.0 else x
                decay = float(np.exp(-float(np.exp(float(tal[hv]))) * sp))
                beta_t = 1.0 / (1.0 + float(np.exp(-float(tb[ti, hv]))))
                s = state[slot, hv].double()  # fp64 ref, third column only
                dot = (s * decay * (kh * kn).unsqueeze(0)).sum(dim=1)
                vres = (v[ti, hv].double() - dot) * beta_t
                new_s = s * decay + vres.unsqueeze(1) * (kh * kn).unsqueeze(0)
                state[slot, hv] = new_s.float()
                out[ti, hv] = (new_s * (qh * qn * (KD ** -0.5))
                               .unsqueeze(0)).sum(dim=1)
        return out.to(torch.float16), state

    def event_us(fn, iters, warmup=20):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters * 1000.0

    def kernel_rows(fn, iters=22, warmup=3):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
        rows = [x for x in prof.key_averages()
                if ("m1_native" in x.key or "packed_decode" in x.key)]
        return [(x.key, x.self_device_time_total / iters, int(x.count))
                for x in rows]

    if args.cmd == "run":
        d = build(args.tokens, args.seed, args.slot_neg)
        neg = "neg" if args.slot_neg else ""
        name = f"gdn_m1_{args.impl}{args.cfg}_T{args.tokens}{neg}"
        if args.tag:
            name += f"_{args.tag}"
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, name + ".npz")

        state_before = d["state"].clone()
        call(d)  # first call initializes host statics with the right env
        torch.cuda.synchronize()
        # Parity payload: fresh state so the saved post-run state matches.
        d["state"].copy_(state_before)
        d["out"].zero_()
        call(d)
        torch.cuda.synchronize()

        ev = event_us(lambda: call(d), args.iters)
        rows = kernel_rows(lambda: call(d))
        host0 = None
        import time  # noqa: E402
        for _ in range(5):
            call(d)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            call(d)
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        host0 = (t1 - t0) / args.iters * 1e6

        np.savez_compressed(
            path,
            out=d["out"].cpu().numpy(),
            state_after=d["state"].cpu().numpy(),
            state_before=state_before.cpu().numpy(),
            mixed_qkv=d["mixed_qkv"].cpu().numpy(),
            a=d["a"].cpu().numpy(), b=d["b"].cpu().numpy(),
            A_log=d["A_log"].cpu().numpy(),
            dt_bias=d["dt_bias"].cpu().numpy(),
            idx=d["idx"].cpu().numpy(),
        )
        print(f"[{name}] saved {path}")
        print(f"  env: VLLM_GFX906_GDN_DEC_M1={os.environ['VLLM_GFX906_GDN_DEC_M1']}"
              f" CFG={os.environ['VLLM_GFX906_GDN_DEC_M1_CFG']} T={args.tokens}"
              f" slot_neg={args.slot_neg}")
        print(f"  event us/call = {ev:.1f} (iters={args.iters})")
        print(f"  enqueue us/call = {host0:.1f}")
        for key, us, cnt in rows:
            print(f"  kernel: {us:.1f} us/call x{cnt // args.iters}  {key}")
        return

    # ---- compare mode: pure torch/numpy, no vllm import needed ----
    import numpy as np  # noqa: E402
    import torch  # noqa: E402

    def load(path):
        z = np.load(path)
        return {k: torch.from_numpy(z[k]) for k in z.files}

    def metrics(got_out, got_state, ref_out, ref_state):
        do = (got_out.float() - ref_out.float()).abs().max().item()
        ds = (got_state.float() - ref_state.float()).abs().max().item()
        n_diff = int((got_out != ref_out).sum().item())
        return do, ds, n_diff

    A, B = load(args.npz_a), load(args.npz_b)
    print(f"A = {args.npz_a}")
    print(f"B = {args.npz_b}")
    do, ds, nd = metrics(A["out"], A["state_after"], B["out"], B["state_after"])
    print(f"  A-vs-B   : max|dout|={do:.3e} max|dstate|={ds:.3e} "
          f"fp16-out differing={nd}/{A['out'].numel()}")
    ea_out, ea_state = eager_ref(A)
    eb_out, eb_state = eager_ref(B)
    do, ds, nd = metrics(A["out"], A["state_after"], ea_out, ea_state)
    print(f"  A-vs-eager: max|dout|={do:.3e} max|dstate|={ds:.3e} "
          f"fp16-out differing={nd}/{A['out'].numel()}")
    do, ds, nd = metrics(B["out"], B["state_after"], eb_out, eb_state)
    print(f"  B-vs-eager: max|dout|={do:.3e} max|dstate|={ds:.3e} "
          f"fp16-out differing={nd}/{B['out'].numel()}")


if __name__ == "__main__":
    raise SystemExit(main())
