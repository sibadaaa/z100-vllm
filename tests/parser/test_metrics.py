# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The tool-call parser counter is registered on first use, so a server that
never parses a tool call does not export zero-valued series for every
mode/outcome/request-type combination at startup."""

import pytest
from prometheus_client import REGISTRY, generate_latest

from vllm.parser import metrics as parser_metrics
from vllm.v1.metrics.prometheus import unregister_vllm_metrics

pytestmark = pytest.mark.cpu_test

_INVOCATIONS = "vllm:tool_call_parser_invocations"


@pytest.fixture
def parser_metric_state(monkeypatch):
    monkeypatch.setattr(parser_metrics, "_model_name", None)
    monkeypatch.setattr(parser_metrics, "_tool_call_parser_invocations", None)
    yield
    unregister_vllm_metrics()


def _scrape() -> str:
    return generate_latest(REGISTRY).decode()


def _invocation_lines() -> list[str]:
    prefix = f"{_INVOCATIONS}_total{{"
    return [line for line in _scrape().splitlines() if line.startswith(prefix)]


def test_nothing_exported_before_init(parser_metric_state):
    """Recording without a configured tool-call parser is a no-op."""
    parser_metrics.record_tool_parser_invocation(
        is_tool_called=True, is_streaming=False, request=object()
    )
    assert _INVOCATIONS not in _scrape()


def test_registered_on_first_invocation(parser_metric_state):
    parser_metrics.init_parser_metrics(model_name="test-model")
    assert _INVOCATIONS not in _scrape()

    parser_metrics.record_tool_parser_invocation(
        is_tool_called=True, is_streaming=False, request=object()
    )
    assert _invocation_lines() == [
        'vllm:tool_call_parser_invocations_total{mode="non_streaming",'
        'model_name="test-model",outcome="tool_call",request_type="other"} 1.0'
    ]


def test_only_observed_combinations_are_exported(parser_metric_state):
    parser_metrics.init_parser_metrics(model_name="test-model")
    for is_tool_called, is_streaming in [(True, False), (True, False), (False, True)]:
        parser_metrics.record_tool_parser_invocation(
            is_tool_called=is_tool_called,
            is_streaming=is_streaming,
            request=object(),
        )

    assert sorted(_invocation_lines()) == [
        'vllm:tool_call_parser_invocations_total{mode="non_streaming",'
        'model_name="test-model",outcome="tool_call",request_type="other"} 2.0',
        'vllm:tool_call_parser_invocations_total{mode="streaming",'
        'model_name="test-model",outcome="no_tool_call",request_type="other"} 1.0',
    ]
