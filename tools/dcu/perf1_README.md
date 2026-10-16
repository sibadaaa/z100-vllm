# perf1 e2e A/B protocol (task-4 harness)

Owner: `verifier`. Everything below runs **inside the container** unless marked
host. All artifacts land in `/mnt/game/perf1/`.

## Files

| file | role |
|---|---|
| `tools/dcu/perf1_e2e_bench.py` | client: streaming perf sweep + cold-start probe + greedy parity capture. JSON schema is a superset of `/home/lyh/vllm-test/scripts/bench_baseline.py` (`config`/`runs`/`spec`/`summary` kept). |
| `tools/dcu/perf1_e2e_parity.py` | compares two client artifacts: prompt ids, generated token ids (exact), top-1 logprobs (tol), decoded text, within-arm repeat determinism. Exit 0 confirmed / 1 refuted / 2 incomplete. |
| `tools/dcu/perf1_arm.sh` | service arm driver: `snapshot` / `stop` / `start <arm> [KEY=VAL ...]` / `status` / `ready` / `so`. |
| `tools/dcu/perf1_e2e_suite.sh` | one arm: snapshot -> start -> cold probe -> sweep -> p512 g64 anchor -> parity -> so md5 before/after. |
| `tools/dcu/perf1_e2e_session.sh` | several arms in one lock session; last arm is left live. GPU-exclusive preflight. |
| `tools/dcu/perf1_summarize.py` | renders arm JSONs to a markdown table. |

## Arm control

The control arm is `/mnt/game/port/serve_fp8_8000.sh` **without**
`--profiler-config` (the currently running service had the profiler on; that
distorts timing). `perf1_arm.sh` only accepts this knob set:
`S11F_*`, `VLLM_DCU_*`, `VLLM_GFX906_*`, `VLLM_QWEN35_*`, `HSA_*`,
`HIP_VISIBLE_DEVICES`, `PERF1_*`.

## Running an A/B session (host shell)

```bash
docker exec z100-vllm bash -c '
exec > /mnt/game/perf1/arm_sessionN.log 2>&1
cd /src
flock -w 2400 /tmp/perf1_build.lock bash /src/tools/dcu/perf1_e2e_session.sh \
  --arm control \
  --arm <armname> VLLM_DCU_XXX=1 \
  --arm restore:control_prod
'
```

* `/tmp/perf1_build.lock` is the global GPU-exclusive lock: builds AND
  microbenchmarks AND service arms all take it. The whole session is inside it,
  so a rebuild cannot swap `_rocm_C.abi3.so` under a measuring arm.
* `restore:<name>` groups only start a service (no measurement) - always end a
  session with the production recipe so :8000 stays usable.
* `parity_only` as a bare token in an arm group skips the cold probe and sweep
  (cheap reference arm).
* Readiness: this box needs ~200 s of engine init, so
  `PERF1_READY_TIMEOUT` defaults to 420 s. Record it, do not fight it.

## What one arm measures

* cold start: first request after a fresh start, p8192 config (=7443 real
  tokens), g32 -> TTFT/prefill t/s vs steady state;
* sweep: configured prompt 128/512/2048/8192 -> real 119/451/1806/7443 tokens,
  g96, `ignore_eos`, `temperature 0.0`, concurrency 1, 3 repeats;
* anchor: p512 g64 x3 - the exact config of
  `vllm-test/results/final_verify_p512_g64_20261013-021338.json`
  (TTFT 2065 ms, prefill 218 t/s, decode 14.14 t/s, step p50 215 ms,
  tok/step 3.05);
* parity: `/v1/completions`, raw text prompt, `temperature 0`,
  `return_token_ids` + `logprobs=1`, g64, reps 2. The completions endpoint is
  used because the server runs a reasoning parser, and vLLM suppresses streamed
  `token_ids`/`logprobs` whenever a reasoning parser is active.

## Interpreting a verdict

* token ids + prompt ids identical -> **confirmed**;
* ids differ but decoded text byte-identical -> **partially confirmed**;
* text differs -> **refuted** (correctness regression until proven otherwise);
* k>2 speculation is expected to change the *id* stream vs k=2 by construction -
  judge it against the k=0/k=1 greedy reference, not against k=2.

## Evidence tied to a build

Every arm writes `arm_<arm>_code_state.txt` (git head, `git status` of
csrc/vllm/tools, source md5s), `arm_<arm>_meta.txt` (overrides + so md5 at
start), `arm_<arm>_so.txt` (`so_md5_before/after`, `so_stable`), and
`serve_<arm>.log`. `so_stable=NO` invalidates the arm.
