# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus metrics for the parsers."""

from __future__ import annotations

from enum import Enum
from typing import cast

from prometheus_client import REGISTRY, Counter

_model_name: str | None = None

_TOOL_CALL_PARSER_INVOCATIONS_TOTAL = "vllm:tool_call_parser_invocations_total"
_tool_call_parser_invocations: Counter | None = None


class ToolCallOutcome(Enum):
    TOOL_CALL = "tool_call"
    NO_TOOL_CALL = "no_tool_call"


class RequestType(Enum):
    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"
    OTHER = "other"


def init_parser_metrics(*, model_name: str) -> None:
    """Cache the model label shared by parser metrics.

    The counters themselves are registered on the first recorded invocation so
    that a server which never parses a tool call does not export a zero-valued
    series for every mode/outcome/request-type combination at startup.
    """
    global _model_name
    _model_name = model_name


def _tool_call_parser_invocations_counter() -> Counter:
    global _tool_call_parser_invocations
    if _tool_call_parser_invocations is None:
        try:
            _tool_call_parser_invocations = Counter(
                name=_TOOL_CALL_PARSER_INVOCATIONS_TOTAL,
                documentation=(
                    "Total number of ToolParser invocations. "
                    "Non-streaming increments once per choice; "
                    "streaming increments once per delta."
                ),
                labelnames=["model_name", "mode", "outcome", "request_type"],
            )
        except ValueError:
            _tool_call_parser_invocations = cast(
                Counter,
                REGISTRY._names_to_collectors[_TOOL_CALL_PARSER_INVOCATIONS_TOTAL],
            )
    return _tool_call_parser_invocations


def record_tool_parser_invocation(
    *,
    is_tool_called: bool | Exception,
    is_streaming: bool,
    request: object,
) -> None:
    """Increment the tool-call parser invocation counter when parser metrics
    are enabled, registering it on the first invocation. Currently parser
    failures are treated as no tool calls.

    TODO: To accurately track parser failures, add a new ToolCallOutcome and
    more importantly, ensure exceptions are propagated out of the ToolParsers
    instead of being caught internally. This would require going through
    ToolParser implementation on a case-by-case basis.
    """
    if _model_name is None:
        return

    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

    match request:
        case ChatCompletionRequest():
            request_type = RequestType.CHAT_COMPLETIONS
        case ResponsesRequest():
            request_type = RequestType.RESPONSES
        case _:
            request_type = RequestType.OTHER

    match is_tool_called:
        case bool():
            outcome = (
                ToolCallOutcome.TOOL_CALL
                if is_tool_called
                else ToolCallOutcome.NO_TOOL_CALL
            )
        case _:
            outcome = ToolCallOutcome.NO_TOOL_CALL

    _tool_call_parser_invocations_counter().labels(
        model_name=_model_name,
        mode="streaming" if is_streaming else "non_streaming",
        outcome=outcome.value,
        request_type=request_type.value,
    ).inc()
