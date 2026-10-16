# Production Metrics

vLLM exposes a number of metrics that can be used to monitor the health of the
system. These metrics are exposed via the `/metrics` endpoint on the vLLM
OpenAI compatible API server.

You can start the server using Python, or using [Docker](../deployment/docker.md):

```bash
vllm serve unsloth/Llama-3.2-1B-Instruct
```

Then query the endpoint to get the latest metrics from the server:

??? console "Output"

    ```console
    $ curl http://0.0.0.0:8000/metrics

    # HELP vllm:iteration_tokens_total Histogram of number of tokens per engine_step.
    # TYPE vllm:iteration_tokens_total histogram
    vllm:iteration_tokens_total_sum{model_name="unsloth/Llama-3.2-1B-Instruct"} 0.0
    vllm:iteration_tokens_total_bucket{le="1.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="8.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="16.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="32.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="64.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="128.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="256.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="512.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    ...
    ```

The following metrics are exposed:

## General Metrics

--8<-- "gen:metrics-general"

## Speculative Decoding Metrics

--8<-- "gen:metrics-spec-decode"

## NIXL KV Connector Metrics

--8<-- "gen:metrics-nixl"

## Model Flops Utilization (MFU) Performance Metrics

These metrics are available via `--enable-mfu-metrics`:

--8<-- "gen:metrics-mfu"

## Reducing Scrape Volume

A freshly started server exports several hundred series before it has handled
any request: the per-request histograms publish their full bucket ladder even
though nothing has been observed yet, and `prometheus_client` adds one
`*_created` gauge per counter, histogram and summary child. The sections below
describe how to trim that payload, and which parts of it vLLM cannot remove
without changing its public metric output.

### `*_created` series

`prometheus_client` exports a `<name>_created` gauge next to every counter,
histogram and summary. These gauges only carry the registration timestamp;
immediately after startup they can account for roughly 20% of the lines in a
scrape. Set `PROMETHEUS_DISABLE_CREATED_SERIES` to disable them:

```bash
PROMETHEUS_DISABLE_CREATED_SERIES=1 vllm serve unsloth/Llama-3.2-1B-Instruct
```

The client reads the variable when `prometheus_client` is imported, so it must
be present in the server process environment (for example `docker run -e`, a
Kubernetes `env:` entry, or the systemd unit). It is supported by
`prometheus_client >= 0.14.0` ([added in 0.14.0](https://github.com/prometheus/client_python/releases/tag/v0.14.0))
and vLLM requires `>= 0.18.0`, so it is always available. Disabling the
`*_created` gauges does not affect the values of the metrics themselves. vLLM
keeps the client default (created series enabled) because turning them off
changes the public metric output; dashboards that chart `*_created` directly
must be updated first.

### Metrics registered on first use

Two families are registered lazily, so a server that never triggers them does
not export their (zero-valued) series at startup:

- `vllm:tool_call_parser_invocations_total` appears on the first tool-call
  parser invocation, that is, only with `--tool-call-parser` and only once a
  request actually goes through the parser. It is labelled with `model_name`,
  `mode`, `outcome` and `request_type`, so up to 12 combinations plus their
  `_created` gauges are absent until then.
- `vllm:spec_decode_num_accepted_tokens_per_pos` gains one `position` label
  value per draft position on the first draft. The `vllm:spec_decode_num_drafts`,
  `vllm:spec_decode_num_draft_tokens` and `vllm:spec_decode_num_accepted_tokens`
  counters are still registered as soon as speculative decoding is configured,
  and remain zero until the first draft.

### Zero-valued histogram buckets

Every histogram child exports its complete bucket ladder, including buckets
whose `le` bound was never reached and histograms whose `_count` is still zero.
vLLM creates the per-request histogram children when the metrics logger is
initialized (rather than on the first request) so that dashboards and alerts
can scrape a stable set of series from the moment the server is ready. The
result is a few hundred zero-valued bucket lines at startup that turn into real
data as soon as traffic arrives. Prometheus `metric_relabel_configs` select
whole series and cannot test sample values, so individual empty buckets cannot
be filtered - only complete families can be dropped (see below).

### Dropping families in Prometheus

Relabeling at scrape time removes series from the TSDB and from dashboards
without changing the vLLM process. The exporter still serializes them, so this
reduces stored cardinality rather than scrape payload size:

```yaml
scrape_configs:
  - job_name: vllm
    static_configs:
      - targets: ["vllm:8000"]
    metric_relabel_configs:
      # Client and process collectors, if no dashboard uses them.
      - source_labels: [__name__]
        regex: "python_(gc_.*|info)|process_.*"
        action: drop
      # Informational config dump, if it is not used to join across instances.
      - source_labels: [__name__]
        regex: "vllm:cache_config_info"
        action: drop
      # Unlabeled high-resolution HTTP latency histogram (see below).
      - source_labels: [__name__]
        regex: "http_request_duration_highr_seconds.*"
        action: drop
```

!!! note
    The built-in `python_gc_*`, `python_info` and `process_*` collectors are not
    exported when `--api-server-count > 1`, because `prometheus_client` does not
    support them in multiprocess mode.

Optional families that a deployment can simply not enable:

| Flag | Families |
| --- | --- |
| `--kv-cache-metrics` | `vllm:kv_block_lifetime_seconds`, `vllm:kv_block_idle_before_evict_seconds`, `vllm:kv_block_reuse_gap_seconds` (22 buckets each) |
| `--enable-mfu-metrics` | `vllm:estimated_flops_per_gpu_total`, `vllm:estimated_read_bytes_per_gpu_total`, `vllm:estimated_write_bytes_per_gpu_total` |

### HTTP request metrics

`prometheus_fastapi_instrumentator` registers two latency histograms for the
OpenAI API server:

- `http_request_duration_seconds` is labelled with `handler`, `method` and
  `status`, and uses very few buckets (0.1, 0.5, 1, +Inf) to keep label
  cardinality low.
- `http_request_duration_highr_seconds` has no labels and 22 buckets, which
  makes it the only source of usable HTTP latency percentiles (the labelled
  histogram is too coarse for p50/p90/p99 of a single endpoint).

The high-resolution histogram is kept: it is the metric the instrumentator
recommends for SLIs, the Rust API server registers the same family, and its
series count is constant because it carries no labels. Deployments that do not
chart HTTP latency percentiles can drop it with the relabeling rule above.

### `vllm:cache_config_info`

`vllm:cache_config_info` is an informational metric: one gauge per engine, set
to 1, whose labels describe the cache configuration. It is meant for joining
configuration onto other series in Prometheus, not for alerting, and a
configuration change produces a new series rather than updating the old one.
Bookkeeping fields that describe how the config was constructed
(`_block_size_resolved`, `user_specified_block_size`,
`user_specified_mamba_block_size`) are excluded from its labels. Drop the
family if the configuration is already recorded elsewhere.

## Deprecation Policy

Note: when metrics are deprecated in version `X.Y`, they are hidden in version `X.Y+1`
but can be re-enabled using the `--show-hidden-metrics-for-version=X.Y` escape hatch,
and are then removed in version `X.Y+2`.
