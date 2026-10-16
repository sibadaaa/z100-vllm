# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The per-position spec-decode counters are materialized on the first draft, so
enabling speculative decoding without drafting anything does not export one
zero-valued series per draft position at startup."""

from types import SimpleNamespace

import pytest
from prometheus_client import REGISTRY, generate_latest

from vllm.v1.metrics.prometheus import unregister_vllm_metrics
from vllm.v1.spec_decode.metrics import SpecDecodingProm, SpecDecodingStats

pytestmark = pytest.mark.cpu_test

_MODEL_LABELS = {0: ["test-model"]}
_PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"


@pytest.fixture
def cleanup_metrics():
    yield
    unregister_vllm_metrics()


def _scrape() -> str:
    return generate_latest(REGISTRY).decode()


def _lines(prefix: str) -> list[str]:
    return [line for line in _scrape().splitlines() if line.startswith(prefix)]


def _prom(**kwargs) -> SpecDecodingProm:
    # __init__ only checks that a speculative config exists; the position count
    # comes from the observed stats.
    return SpecDecodingProm(
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        labelnames=["model_name"],
        per_engine_labelvalues=_MODEL_LABELS,
        **kwargs,
    )


def test_per_pos_counters_absent_until_first_draft(cleanup_metrics):
    _prom()

    text = _scrape()
    # The family and the aggregate counters exist as soon as spec decoding is
    # configured...
    assert "# HELP vllm:spec_decode_num_accepted_tokens_per_pos_total" in text
    assert 'vllm:spec_decode_num_drafts_total{model_name="test-model"} 0.0' in text
    # ...but no position has been observed yet.
    assert _lines(f"{_PER_POS}{{") == []
    assert "vllm:spec_decode_num_accepted_tokens_per_pos_created" not in text


def test_per_pos_counters_appear_on_first_draft(cleanup_metrics):
    prom = _prom()
    stats = SpecDecodingStats.new(num_spec_tokens=3)
    stats.observe_draft(num_draft_tokens=3, num_accepted_tokens=2)
    prom.observe(stats)

    assert sorted(_lines(f"{_PER_POS}{{")) == [
        f'{_PER_POS}{{model_name="test-model",position="0"}} 1.0',
        f'{_PER_POS}{{model_name="test-model",position="1"}} 1.0',
        f'{_PER_POS}{{model_name="test-model",position="2"}} 0.0',
    ]

    # The series remain queryable and keep accumulating.
    prom.observe(stats)
    assert sorted(_lines(f"{_PER_POS}{{")) == [
        f'{_PER_POS}{{model_name="test-model",position="0"}} 2.0',
        f'{_PER_POS}{{model_name="test-model",position="1"}} 2.0',
        f'{_PER_POS}{{model_name="test-model",position="2"}} 0.0',
    ]


def test_diffusion_does_not_export_per_pos_counters(cleanup_metrics):
    prom = SpecDecodingProm(
        speculative_config=None,
        labelnames=["model_name"],
        per_engine_labelvalues=_MODEL_LABELS,
        is_diffusion=True,
    )
    stats = SpecDecodingStats.new(num_spec_tokens=4)
    stats.observe_draft(num_draft_tokens=4, num_accepted_tokens=0)
    prom.observe(stats)

    text = _scrape()
    assert "vllm:spec_decode_num_accepted_tokens_per_pos" not in text
    assert "vllm:diffusion_num_denoising_steps_total" in text


def test_nothing_exported_without_spec_decoding(cleanup_metrics):
    prom = SpecDecodingProm(
        speculative_config=None,
        labelnames=["model_name"],
        per_engine_labelvalues=_MODEL_LABELS,
    )
    prom.observe(SpecDecodingStats.new(num_spec_tokens=3))

    assert "vllm:spec_decode" not in _scrape()
