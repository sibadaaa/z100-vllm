# Metrics Endpoint Optimization Plan

Source: `/home/lyh/vllm-test/logs/test.log`, a `/metrics` scrape collected
immediately after startup, before inference traffic. The file has 774 lines
(about 68 KiB): 208 HELP/TYPE comment lines, 155 `*_created` series, and many
zero-valued histogram buckets and feature metrics.

## Findings

### Zero-traffic histogram overhead

Request histograms export their complete bucket ladders even when their
`_count` is zero. The log contains several hundred such bucket lines across
request latency, queue time, prefill/decode time, TTFT, ITL, TPOT, token count,
and preemption histograms.

**Plan:** document Prometheus relabeling for deployments that do not need
startup-phase bucket series. For vLLM itself, evaluate making rarely used
histograms opt-in, following the existing `--kv-cache-metrics` gate for KV
residency histograms. Preserve the histograms needed for SLO dashboards.

### `*_created` series

The 155 `*_created` lines account for about 20% of this scrape. Verify the
deployed `prometheus_client` version and document its supported setting for
disabling created series (for example, `PROMETHEUS_DISABLE_CREATED_SERIES` when
available). Changing the vLLM default requires compatibility review because it
changes the public metric output.

### Speculative decoding metrics

Speculative-decoding counters and seven per-position series are present but all
zero. `vllm/v1/spec_decode/metrics.py` creates these when a speculative config
exists. If a deployment enables the config but does not use speculative
decoding, lazily create the per-position counters on the first draft.

### Tool-call parser metrics

`vllm:tool_call_parser_invocations_total` pre-creates 12 combinations of mode,
outcome, and request type, plus their created series. Lazy registration on the
first parser invocation would remove these startup series. Dropping
`request_type` is a secondary option if endpoint identity is sufficient.
Related code is `vllm/parser/metrics.py`.

### HTTP instrumentation

`http_request_duration_highr_seconds` contributes 29 lines without endpoint
labels, while `http_request_duration_seconds` is labeled. Confirm whether the
high-resolution histogram is needed by current dashboards; remove it from the
instrumentator configuration or document its intended use if it is legacy.

### Cache configuration labels

`vllm:cache_config_info` packs about 35 configuration values into one label
set. It adds little scrape volume, but the long label value creates time-series
churn whenever configuration changes. Either expose only stable, useful fields
or document this metric as informational-only.

### Default process collectors

`python_gc_*`, `python_info`, and `process_*` are useful for debugging but are
not required by every production dashboard. Document Prometheus relabeling for
deployments that do not retain these collectors.

## Recommended work items

| Priority | Work | Location | Expected impact |
| --- | --- | --- | --- |
| P0 | Verify and document the client setting for `*_created` series | `docs/usage/metrics.md` | Up to 20% fewer lines |
| P1 | Lazy-register tool-call parser counters | `vllm/parser/metrics.py` | 24 startup series |
| P1 | Check whether `http_request_duration_highr_seconds` is required | `vllm/entrypoints/serve/instrumentator/metrics.py` | 29 startup lines |
| P2 | Lazy-register spec-decode per-position counters | `vllm/v1/spec_decode/metrics.py` | 7+ startup series |
| P2 | Reduce or document `cache_config_info` labels | `vllm/v1/metrics/loggers.py` | Less label churn |
| P3 | Document relabeling for unused feature/process series | `docs/usage/metrics.md` | Deployment-dependent |

## Implementation Status

| Priority | Work item | Where | Result |
| --- | --- | --- | --- |
| P0 | Verify and document the client setting for `*_created` series | `docs/usage/metrics.md` | `PROMETHEUS_DISABLE_CREATED_SERIES=1` verified against `prometheus_client` 0.24.1; covers 155 of the 774 baseline lines |
| P1 | Lazy-register tool-call parser counters | `vllm/parser/metrics.py` | All 28 baseline lines absent until the first parsed tool call |
| P1 | Check whether `http_request_duration_highr_seconds` is required | `vllm/entrypoints/serve/instrumentator/metrics.py` | Required, kept, documented (decision below) |
| P2 | Lazy-register spec-decode per-position counters | `vllm/v1/spec_decode/metrics.py` | 16 of the 18 baseline lines absent until the first draft |
| P2 | Reduce or document `cache_config_info` labels | `vllm/config/cache.py`, `docs/usage/metrics.md` | 36 to 33 labels, documented as informational-only |
| P3 | Document relabeling for unused feature/process series | `docs/usage/metrics.md` | Relabeling recipes for `python_*`, `process_*`, `cache_config_info`, `http_request_duration_highr_seconds` |

### Decisions

- `*_created` series stay enabled by default. The switch is applied through the
  `prometheus_client` environment variable before the server starts; changing
  the vLLM default would alter the public metric output. The variable is
  supported since `prometheus_client` 0.14.0 and vLLM requires `>= 0.18.0`, so it
  is available on every supported install.
- `http_request_duration_highr_seconds` is kept. The endpoint-labelled
  `http_request_duration_seconds` histogram only has the 0.1/0.5/1/+Inf buckets,
  which is too coarse for HTTP latency percentiles, and the Rust API server
  registers the same family (`rust/src/metrics/src/api_server.rs`), asserted by
  `rust/src/server/src/routes/tests.rs`. Dropping it from the Python server would
  remove the only percentile source and diverge from the Rust server. It is
  documented, with a relabeling recipe for deployments that never chart it.
- Zero-traffic histogram buckets are documented rather than removed. The
  per-request histogram children are created when the metrics logger is
  initialized so that dashboards see a stable series set as soon as the server
  is ready, which `tests/v1/metrics/test_histogram_buckets.py` asserts.
  Prometheus relabeling selects series rather than sample values, so it cannot
  drop individual empty buckets; the documentation therefore covers dropping
  whole families and the flags that gate the optional ones.

## Verification

1. Re-scrape `/metrics` after each change and compare line and series counts
   with this baseline.
2. Check all reference Grafana queries after any change to created series.
3. For lazy registration, verify the metric appears after its first qualifying
   event and remains queryable thereafter.

### Results

End-to-end: the `:8000` deployment (Qwen3.8-27B-FP8 + DFlash2 k7, TP4) was
restarted on the patched checkout with the same argv and environment as the run
that produced the baseline, and `/metrics` was scraped after the engine became
ready and exactly one `GET /v1/models`, matching how the baseline was taken.

- baseline, old code, startup: `/home/lyh/vllm-test/logs/test.log`
- after restart, new code, startup:
  `/home/lyh/vllm-test/logs/metrics-after-restart-20261015-154800.log`
- after the first requests:
  `/home/lyh/vllm-test/logs/metrics-after-traffic2-20261015-154825.log`

| Metric | Baseline | After restart | Delta |
| --- | --- | --- | --- |
| scrape lines | 774 | 729 | -45 (-5.8%) |
| series lines | 565 | 527 | -38 |
| `HELP`/`TYPE` lines | 209 | 202 | -7 |
| payload size | 68,724 B | 62,881 B | -5,843 B (-8.5%) |
| `*_created` lines | 155 (20.0%) | 132 | -23 |
| `vllm:tool_call_parser_invocations_*` | 24 series | 0 until the first parsed tool call | -24 |
| `vllm:spec_decode_num_accepted_tokens_per_pos_*` | 14 series | 0 until the first draft | -14 |
| `vllm:cache_config_info` labels | 36 | 33 | -3 |
| `http_request_duration_highr_seconds_*` | 25 series | 25 | 0, kept on purpose |

`PROMETHEUS_DISABLE_CREATED_SERIES` was verified directly against the deployed
`prometheus_client` 0.24.1 (setting it removes every `*_created` line,
including the `HELP`/`TYPE` pairs). Applied to the startup scrape above it would
leave 597 lines, 22.9% below the baseline.

Lazy registration was verified on the live server. After one plain completion
and one tool-call chat request:

- `vllm:spec_decode_num_accepted_tokens_per_pos_total` carries all seven
  `position` label values with the acceptance ladder (18/15/12/12/9/7/7 at the
  time of the scrape), matching the per-position rates the engine logs.
- `vllm:tool_call_parser_invocations_total` carries only the one observed
  combination (`chat_completions`/`non_streaming`/`tool_call`) instead of the 12
  pre-created ones.
- Both survive later scrapes, i.e. they materialize once and stay queryable.

The code changes were also measured before restarting, with an in-process
harness (`/home/lyh/vllm-test/scripts/metrics_ab_scrape.py`) that registers
exactly what the API server registers at startup: 100 lines down to 56 (-44%)
for the touched families, 68 down to 47 with created series disabled. The live
restart above supersedes those numbers; the harness remains useful for
re-measuring without a restart.

Operational helpers used here (outside the repository):

- `/home/lyh/vllm-test/scripts/restart_s19_fp8.sh stop|start|wait|status` -
  restarts the arm from the captured argv/environment.
- `/home/lyh/vllm-test/scripts/metrics_compare.py <before> <after>` - prints the
  line/series/created/family comparison tables used above.

Unit tests, run from a writable copy of the repository inside the deployment
container (the checkout is mounted read-only at `/src`, so a copy is needed for
pytest to write `__pycache__`):

```bash
cd /tmp/vllm-verify && PYTHONPATH=/tmp/vllm-verify python3 -m pytest \
    tests/parser/test_metrics.py tests/v1/spec_decode/test_metrics.py -q
cd /tmp/vllm-verify && PYTHONPATH=/tmp/vllm-verify python3 -m pytest \
    tests/v1/metrics tests/parser --ignore=tests/parser/cohere -q
cd /tmp/vllm-verify && PYTHONPATH=/tmp/vllm-verify python3 -m pytest \
    tests/config/test_config_utils.py -q
```

`tests/parser/cohere` is skipped because the optional `cohere_melody` package is
not installed in the container.
