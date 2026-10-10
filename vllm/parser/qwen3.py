# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 parser for tool calls and reasoning.

Qwen3 XML tool call format::

    <tool_call>
    <function=func_name>
    <parameter=key>value</parameter>
    </function>
    </tool_call>

The argument body consists of ``<parameter=NAME>VALUE</parameter>`` tags.
The ``_qwen3_arg_converter`` parses these into a JSON object.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

import regex as re

from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
CHATML_TURN_BOUNDARIES = frozenset(("<|im_start|>", "<|im_end|>"))
FUNC_PREFIX = "<function="
FUNC_END = "</function>"
PARAM_START = "<parameter="
PARAM_END = "</parameter>"

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]*")

# Quantized Qwen3 checkpoints occasionally drop the ``function=`` marker and
# open a tool call with the bare name, e.g. ``<tool_call>\n<bash>\n``. The
# grammar has no terminal for that spelling, so the call used to be swallowed
# and the response ended as an empty ``stop``. The requested tool names make
# the intent unambiguous, so the opener is rewritten to the canonical form.
_LOOSE_OPENER_RE = re.compile(r"^(?P<indent>\s*)<(?P<name>[A-Za-z_][A-Za-z0-9_.\-]*)>")


def _tool_names(tools: list[Tool] | None) -> frozenset[str]:
    """Names of the tools requested for this request."""
    names: set[str] = set()
    for tool in tools or ():
        function = getattr(tool, "function", None)
        name = getattr(function, "name", None) if function is not None else None
        if name is None:
            name = getattr(tool, "name", None)
        if isinstance(name, str) and name:
            names.add(name)
    return frozenset(names)


def repair_loose_opener(text: str, tool_names: frozenset[str]) -> str:
    """Rewrite a bare ``<name>`` tool-call opener to ``<function=name>``.

    ``text`` is the region following a ``<tool_call>`` marker.
    """
    if not tool_names:
        return text
    match = _LOOSE_OPENER_RE.match(text)
    if match is None or match.group("name") not in tool_names:
        return text
    name = match.group("name")
    return f"{match.group('indent')}<function={name}>" + text[match.end() :]


def _opener_settled(region: str, tool_names: frozenset[str]) -> bool:
    """Whether text following ``<tool_call>`` can be parsed without lookahead.

    The region stays open while it may still grow into an opener: a prefix of
    the canonical ``<function=`` marker, or ``<`` plus a prefix of a requested
    tool name.
    """
    stripped = region.lstrip()
    if not stripped:
        return False
    if not stripped.startswith("<"):
        return True
    if FUNC_PREFIX.startswith(stripped):
        return stripped == FUNC_PREFIX
    match = _IDENT_RE.match(stripped[1:])
    if match is None:
        return True
    if ">" in stripped[1 + match.end() :]:
        return True
    name = match.group(0)
    return not any(tool_name.startswith(name) for tool_name in tool_names)


def _partial_marker_len(text: str) -> int:
    """Length of a trailing partial ``<tool_call>`` that must be held back."""
    for keep in range(len(TOOL_CALL_START) - 1, 0, -1):
        if text.endswith(TOOL_CALL_START[:keep]):
            return keep
    return 0


_PARAM_RE = re.compile(
    r"<\s*parameter\s*=\s*([^>]*)>"
    r"(.*?)"
    r"(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))",
    re.DOTALL,
)
_PARTIAL_PARAM_RE = re.compile(r"<\s*parameter\s*=\s*([^>]+)>(.*)$", re.DOTALL)


def _trim_wrapping_newlines(value: str) -> str:
    """Strip one leading and one trailing newline (the Qwen3 template markup)."""
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def _qwen3_arg_converter(raw_args: str, partial: bool) -> str:
    params: dict[str, object] = {}

    for match in _PARAM_RE.finditer(raw_args):
        name = match.group(1)
        value = match.group(2)
        params[name] = _trim_wrapping_newlines(value)

    if partial:
        remaining = _PARAM_RE.sub("", raw_args)
        m = _PARTIAL_PARAM_RE.search(remaining)
        if m:
            name = m.group(1)
            value = m.group(2)
            if name:
                params[name] = _trim_wrapping_newlines(value)

    return json.dumps(params, ensure_ascii=False)


@functools.cache
def qwen3_config(
    thinking: bool = True,
    *,
    name: str = "qwen3",
    think_start: str = THINK_START,
    think_end: str = THINK_END,
    tool_start: str = TOOL_CALL_START,
    tool_end: str = TOOL_CALL_END,
    turn_boundary_tokens: frozenset[str] = frozenset(),
) -> ParserEngineConfig:
    return ParserEngineConfig(
        name=name,
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        wait_for_reasoning=thinking,
        turn_boundary_tokens=turn_boundary_tokens,
        terminals={
            # Reasoning terminals
            "THINK_START": think_start,
            "THINK_END": think_end,
            # Tool call terminals
            "TOOL_START": tool_start,
            "TOOL_END": tool_end,
            "FUNC_PREFIX": FUNC_PREFIX,
            "FUNC_END": FUNC_END,
            "PARAM_START": PARAM_START,
            "PARAM_END": PARAM_END,
            "CLOSE_ANGLE": ">",
        },
        token_id_terminals={
            "THINK_START": think_start,
            "THINK_END": think_end,
            "TOOL_START": tool_start,
            "TOOL_END": tool_end,
        },
        transitions={
            # -- Reasoning transitions --
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
                (),
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            # Absorb duplicate </think> — model may emit it after
            # already transitioning to CONTENT; drop it silently.
            (ParserState.CONTENT, "THINK_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            # Tool call directly from reasoning (implicit end)
            (ParserState.REASONING, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
            ),
            # -- Tool call transitions --
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
            ),
            # Fallback: <function= without a preceding <tool_call>
            (ParserState.CONTENT, "FUNC_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_PREAMBLE, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_PREAMBLE, "FUNC_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (),
            ),
            (ParserState.TOOL_NAME, "CLOSE_ANGLE"): Transition(
                ParserState.TOOL_ARGS,
                (),
            ),
            # Malformed: </function> while still in TOOL_NAME (no closing >)
            (ParserState.TOOL_NAME, "FUNC_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_ARGS, "FUNC_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_ARGS, "PARAM_START"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.ARG_VALUE_CHUNK,),
            ),
            (ParserState.TOOL_ARGS, "PARAM_END"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.ARG_VALUE_CHUNK,),
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            # Consecutive tool call without closing </tool_call>
            (ParserState.TOOL_BETWEEN, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_BETWEEN, "FUNC_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
        },
        arg_converter=_qwen3_arg_converter,
        stream_arg_deltas=True,
        strip_trailing_reasoning_whitespace=False,
        tool_args_json=False,
    )


class Qwen3Parser(ParserEngine):
    """Qwen3 parser: ``<think>``/``</think>`` reasoning +
    ``<tool_call>`` XML tool calls in a single engine.

    - ``<tool_call>`` as implicit reasoning end (a grammar transition, so it
      also feeds ``is_reasoning_end`` and the structured-output gate)

    Subclasses that share the grammar but differ only in the four wrapper
    token strings (reasoning + tool-call) override the class attributes
    below; everything else is inherited unchanged.
    """

    CONFIG_NAME = "qwen3"
    THINK_START = THINK_START
    THINK_END = THINK_END
    TOOL_START = TOOL_CALL_START
    TOOL_END = TOOL_CALL_END
    TURN_BOUNDARIES: frozenset[str] = CHATML_TURN_BOUNDARIES

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self.thinking_enabled = chat_kwargs.get("enable_thinking", True)
        kwargs.setdefault(
            "parser_engine_config",
            qwen3_config(
                thinking=self.thinking_enabled,
                name=self.CONFIG_NAME,
                think_start=self.THINK_START,
                think_end=self.THINK_END,
                tool_start=self.TOOL_START,
                tool_end=self.TOOL_END,
                turn_boundary_tokens=self.TURN_BOUNDARIES,
            ),
        )
        super().__init__(
            tokenizer,
            tools,
            **kwargs,
        )
        self._tool_names = _tool_names(tools)
        self._names_source = tools
        self._pending: list[tuple[str, tuple[int, ...]]] = []
        self._pending_text = ""
        self._open_region = False

    def _reset(self, initial_state: ParserState | None = None) -> None:
        super()._reset(initial_state=initial_state)
        self._pending = []
        self._pending_text = ""
        self._open_region = False

    def _feed(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
    ) -> list[SemanticEvent]:
        if self._tools is not self._names_source:
            self._names_source = self._tools
            self._tool_names = _tool_names(self._tools)
        if not self._tool_names:
            return self._engine.feed(delta_text, delta_token_ids)
        if delta_text:
            self._pending.append((delta_text, tuple(delta_token_ids)))
            self._pending_text += delta_text
        events: list[SemanticEvent] = []
        for part_text, part_ids in self._release_parts():
            events.extend(self._engine.feed(part_text, part_ids))
        return events

    def _release_parts(self) -> list[tuple[str, tuple[int, ...]]]:
        """Split what has arrived into engine-feedable parts.

        A tool-call marker is fed as soon as it is complete — with its own
        token ids, so the engine keeps treating it as a marker rather than as
        prose. Only the region right after it is held, and only while it could
        still grow into an opener, so a bare ``<bash>`` can be rewritten to
        ``<function=bash>``.
        """
        parts: list[tuple[str, tuple[int, ...]]] = []
        while self._pending_text:
            if self._open_region:
                if not _opener_settled(self._pending_text, self._tool_names):
                    break
                start = self._pending_text.find(TOOL_CALL_START)
                if start == 0:
                    self._open_region = False
                    continue
                cut = len(self._pending_text) if start < 0 else start
                text, token_ids = self._take(cut)
                parts.append((repair_loose_opener(text, self._tool_names), token_ids))
                self._open_region = False
                continue
            start = self._pending_text.find(TOOL_CALL_START)
            if start >= 0:
                parts.append(self._take(start + len(TOOL_CALL_START)))
                self._open_region = True
                continue
            captured = len(self._pending_text) - _partial_marker_len(self._pending_text)
            if captured <= 0:
                break
            text, token_ids = self._take(captured)
            if not text:
                break
            parts.append((text, token_ids))
        return parts

    def _take(self, cut: int) -> tuple[str, tuple[int, ...]]:
        """Remove and return the first *cut* characters of the buffer."""
        head: list[str] = []
        head_token_ids: list[int] = []
        kept: list[tuple[str, tuple[int, ...]]] = []
        remaining = cut
        for segment, segment_ids in self._pending:
            if remaining <= 0:
                kept.append((segment, segment_ids))
                continue
            if remaining >= len(segment):
                head.append(segment)
                head_token_ids.extend(segment_ids)
                remaining -= len(segment)
                continue
            chars, count = self._aligned_prefix(segment, segment_ids, remaining)
            if chars > 0:
                head.append(segment[:chars])
                head_token_ids.extend(segment_ids[:count])
                rest = (segment[chars:], segment_ids[count:])
            else:
                rest = (segment, segment_ids)
            kept.append(rest)
            remaining = 0
        self._pending = kept
        self._pending_text = "".join(segment for segment, _ in kept)
        return "".join(head), tuple(head_token_ids)

    def _aligned_prefix(
        self,
        segment: str,
        segment_ids: tuple[int, ...],
        limit: int,
    ) -> tuple[int, int]:
        """Largest ``(chars, token_ids)`` prefix of a segment within *limit*."""
        if not segment_ids:
            return min(limit, len(segment)), 0
        chars = 0
        count = 0
        for token_id in segment_ids:
            piece = self.model_tokenizer.decode([token_id])
            if not isinstance(piece, str):
                piece = ""
            if chars + len(piece) > limit:
                break
            chars += len(piece)
            count += 1
        return chars, count

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if not self.thinking_enabled:
            return None, model_output
        return super().extract_reasoning(model_output, request)
