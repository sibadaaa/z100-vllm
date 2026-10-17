# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

from vllm import envs
from collections.abc import Mapping
from typing import Any, Literal, cast

from vllm.config import VllmConfig
from vllm.exceptions import VLLMValidationError
from vllm.inputs import (
    EngineInput,
    PromptType,
    SingletonInput,
    split_enc_dec_input,
)
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.cache import processor_cache_from_config
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalUUIDDict
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing import EncDecMultiModalProcessor
from vllm.multimodal.utils import argsort_mm_positions
from vllm.renderers import BaseRenderer, renderer_from_config
from vllm.renderers.inputs.preprocess import parse_model_prompt
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.tasks import GENERATION_TASKS, POOLING_TASKS, SupportedTask
from vllm.tokenizers import TokenizerLike
from vllm.utils import length_from_prompt_token_ids_or_embeds, random_uuid
from vllm.utils.async_utils import make_async
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.kv_hints import KvHintsEnvelope
from vllm.v1.metrics.stats import MultiModalCacheStats
from vllm.v1.structured_output.backend_guidance import validate_guidance_grammar
from vllm.v1.structured_output.backend_lm_format_enforcer import (
    validate_structured_output_request_lm_format_enforcer,
)
from vllm.v1.structured_output.backend_outlines import (
    validate_structured_output_request_outlines,
)
from vllm.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar

logger = init_logger(__name__)


class InputProcessor:
    def __init__(
        self,
        vllm_config: VllmConfig,
        renderer: BaseRenderer | None = None,
        *,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.structured_outputs_config = vllm_config.structured_outputs_config

        self.generation_config_fields = self.model_config.try_get_generation_config()

        self.mm_registry = mm_registry
        self.mm_processor_cache = processor_cache_from_config(vllm_config)

        # Callers (e.g. AsyncLLM) pass an already-built renderer; only build
        # one (and its tokenizer) when not provided.
        self.renderer = renderer = renderer or renderer_from_config(vllm_config)
        tokenizer = self.renderer.tokenizer
        if self.model_config.is_multimodal_model:
            renderer.mm_processor = mm_registry.create_processor(
                self.model_config,
                tokenizer=tokenizer,
            )

        self.input_preprocessor = InputPreprocessor(
            vllm_config,
            renderer=renderer,
            mm_registry=mm_registry,
        )

        # Raw-prompt preprocessing (tokenization and multimodal processing)
        # is blocking, so async callers run it on the renderer's thread pool
        # to keep their event loop responsive.
        self.process_inputs_async = make_async(
            self.process_inputs, executor=self.renderer._executor
        )

    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self.input_preprocessor.tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: TokenizerLike | None) -> None:
        self.input_preprocessor.renderer.tokenizer = tokenizer

    def _validate_logprobs(
        self,
    ) -> Callable[[SamplingParams], None]:
        """Load the custom logits processor classes and return the per-request
        params validator for the active model runner."""
        custom_logitsprocs = self.model_config.logits_processors
        if self.vllm_config.use_v2_model_runner:
            if self.model_config.runner_type == "pooling" and not custom_logitsprocs:
                return lambda _: None

            from vllm.v1.worker.gpu.sample.logits_processor import (
                build_custom_logits_processors_params_validator,
            )

    def _validate_supported_sampling_params(
        self,
        params: SamplingParams,
    ) -> None:
        # NOTE: per-request user logits processors were removed from
        # SamplingParams upstream; there is no field left to reject here.
        # Penalties/bad words under async scheduling + spec decode are synced
        # from upstream #30495 (multi-placeholder folding plus spec_token_ids
        # backfill) and verified: greedy MTP + penalties is coherent over
        # 2200+ tokens and async matches sync token-for-token. Note two
        # inherent (non-bug) behaviors, identical without speculative
        # decoding: (1) strong repetition_penalty + temperature=1.0 +
        # very long generations (>~1800 tokens) naturally drift into token
        # salad because all frequent tokens get penalized; (2) with spec
        # decode, penalties apply only to the target distribution, so draft
        # acceptance decays as the penalized history grows.
        # Structured outputs remain gated under ASYNC scheduling + spec
        # decode: the async scheduler books spec_token_ids as -1
        # placeholders and grammar_bitmask() feeds them to xgrammar, which
        # rejects -1 and crashes the request (assert in
        # StructuredOutputManager.grammar_bitmask). Upstream avoids this
        # with a draft-ids CPU copy for structured requests; this fork's
        # draft tensor is consumed by the input scatter, so the backfill
        # path has to be ported first.
        if (
            self.vllm_config.speculative_config is not None
            and self.vllm_config.scheduler_config.async_scheduling
            and params.structured_outputs
        ):
            raise ValueError(
                "structured outputs are not yet supported together with "
                "async scheduling + speculative decoding on this build; "
                "use --no-async-scheduling or retry without "
                "structured_outputs."
            )

    def resolve_watermarking(self, params: SamplingParams | BeamSearchParams) -> bool:
        """Return whether watermarking will be applied to this request.

        ``None`` and ``True`` request watermarking, while ``False`` opts out.
        If watermarking is unavailable or incompatible with the request, emit
        a warning and return ``False``.
        """
        return self.vllm_config._check_supports_watermarking(params)

    def apply_watermarking(self, params: SamplingParams, watermarking: bool) -> None:
        """Store the resolved `watermarking` on request-local params.

        Also records whether watermarking was requested on a watermarking
        engine but could not be applied, as opposed to an explicit opt-out.
        """
        # Sticky, so resolving an already resolved request keeps the flag.
        params._watermarking_skipped = params._watermarking_skipped or (
            self.vllm_config.watermark_config is not None
            and params.watermarking is not False
            and not watermarking
        )
        params.watermarking = watermarking

    def _validate_params(
        self,
        params: SamplingParams | PoolingParams,
        supported_tasks: tuple[SupportedTask, ...],
    ) -> bool | None:
        """Raise `ValueError` if SamplingParams or PoolingParams is not valid.

        Returns the resolved `watermarking` value for `SamplingParams` (`None`
        for `PoolingParams`). The caller applies it to its private copy of the
        params so that a caller-owned params object is never mutated.
        """
        watermarking: bool | None = None
        if isinstance(params, SamplingParams):
            supported_generation_tasks = [
                task for task in supported_tasks if task in GENERATION_TASKS
            ]
            if not supported_generation_tasks:
                raise VLLMValidationError("This model does not support generation")

            params.verify(
                self.model_config,
                self.speculative_config,
                self.structured_outputs_config,
                self.tokenizer,
            )
            if params.prompt_logprob_token_ids is not None:
                if not self.vllm_config.use_v2_model_runner:
                    raise VLLMValidationError(
                        "prompt_logprob_token_ids requires the V2 model runner "
                        "(VLLM_USE_V2_MODEL_RUNNER=1).",
                        parameter="prompt_logprob_token_ids",
                    )
                if self.vllm_config.cache_config.kv_sharing_fast_prefill:
                    raise VLLMValidationError(
                        "prompt_logprob_token_ids is incorrect with "
                        "--kv-sharing-fast-prefill; disable it for scoring.",
                        parameter="prompt_logprob_token_ids",
                    )

        self._validate_logprobs(params)
        self._validate_sampling_params(params)
        self._validate_supported_sampling_params(params)

            if self.model_config.is_diffusion:
                # Without --diffusion-config the served canvas is unknown here;
                # the ids and the read-only normalisation are still checked.
                validate_diffusion_sampling_params(
                    params,
                    canvas_length=(
                        self.diffusion_config.canvas_length
                        if self.diffusion_config is not None
                        else None
                    ),
                    vocab_size=self.model_config.get_vocab_size(),
                )

        def _validate_single_prompt(single_prompt: dict | str) -> None:
            if not isinstance(single_prompt, dict):
                return
            mm_data = single_prompt.get("multi_modal_data")
            mm_uuids = single_prompt.get("multi_modal_uuids")
            if not mm_data or not mm_uuids:
                return

            for modality, items in mm_data.items():
                if modality in mm_uuids:
                    data_len = len(items) if isinstance(items, list) else 1
                    uuid_len = (
                        len(mm_uuids[modality])
                        if isinstance(mm_uuids[modality], list)
                        else 1
                    )
                    if uuid_len != data_len:
                        raise ValueError(
                            f"multi_modal_uuids for modality '{modality}' "
                            "must have same length as data: got "
                            f"{uuid_len} uuids vs "
                            f"{data_len} items."
                        )
                else:
                    raise ValueError(
                        f"multi_modal_uuids for modality '{modality}' must "
                        "be provided if multi_modal_data is provided."
                    )
            if params.thinking_token_budget is not None and (
                self.vllm_config.reasoning_config is None
                or not self.vllm_config.reasoning_config.enabled
            ):
                raise VLLMValidationError(
                    "thinking_token_budget is set but reasoning_config is "
                    "not configured. Please set --reasoning-parser "
                    "and/or --reasoning-config to use thinking_token_budget."
                )
            if (
                params.trace_decode_token_ids
                and not self.model_config.enable_trace_replay
            ):
                raise VLLMValidationError(
                    "trace_decode_token_ids is set but trace replay is not "
                    "enabled. Start the engine with --enable-trace-replay "
                    "to use it."
                )
            # Resolved last so that feature gates above report the missing
            # feature rather than its incompatibility with watermarking.
            watermarking = self.resolve_watermarking(params)
        elif isinstance(params, PoolingParams):
            supported_pooling_tasks = [
                task for task in supported_tasks if task in POOLING_TASKS
            ]
            if not supported_pooling_tasks:
                raise VLLMValidationError("This model does not support pooling")

        # Handle explicit encoder/decoder prompts or singleton prompt
        if isinstance(prompt, dict) and "encoder_prompt" in prompt:
            enc = prompt.get("encoder_prompt")
            dec = prompt.get("decoder_prompt")
            if enc is not None:
                _validate_single_prompt(cast(dict | str, enc))
            if dec is not None:
                _validate_single_prompt(cast(dict | str, dec))
        else:
            raise TypeError(
                f"params must be either SamplingParams or PoolingParams, "
                f"but got {type(params).__name__}"
            )

        return watermarking

    def _normalize_trace_replay_params(
        self, sampling_params: SamplingParams, prompt_len: int
    ) -> None:
        """Apply trace replay's generation semantics to request-local params."""
        trace_token_ids = sampling_params.trace_decode_token_ids
        assert trace_token_ids
        assert sampling_params.max_tokens is not None

        max_trace_len = max(self.model_config.max_model_len - prompt_len, 1)
        trace_token_ids = trace_token_ids[:max_trace_len]
        sampling_params.trace_decode_token_ids = trace_token_ids

        # Apply this after the generation config so its EOS token cannot stop
        # replay before the trace is exhausted.
        sampling_params.max_tokens = min(
            len(trace_token_ids), sampling_params.max_tokens
        )
        sampling_params.min_tokens = 0
        sampling_params.ignore_eos = True
        sampling_params._eos_token_id = None
        sampling_params.stop = []
        sampling_params.stop_token_ids = []
        sampling_params._all_stop_token_ids = set()

    def _validate_lora(self, lora_request: LoRARequest | None) -> None:
        if lora_request is None:
            return

        # LoRA request passed in while LoRA is not enabled
        if not self.lora_config:
            raise ValueError(
                f"Got lora_request {lora_request} but LoRA is not enabled!"
            )

        if self.tokenizer is not None:
            logger.warning_once(
                "vLLM has deprecated support for supporting different "
                "tokenizers for different LoRAs. By default, vLLM uses base "
                "model's tokenizer. If you are using a LoRA "
                "with its own tokenizer, consider specifying `--tokenizer "
                "[lora_path]` to use the LoRA tokenizer."
            )

    def _validate_structured_output(self, params: SamplingParams) -> None:
        if not params.structured_outputs or not self.structured_outputs_config:
            return

        if self.model_config.skip_tokenizer_init and params.structured_outputs:
            raise ValueError(
                "Structured outputs requires a tokenizer so it can't be used with 'skip_tokenizer_init'"  # noqa: E501
            )

        backend = self.structured_outputs_config.backend
        if _backend := params.structured_outputs._backend:
            # Request-level backend selection is not supported.
            # The values may differ if `params` is reused and was set
            # to a specific backend based on `auto` behavior in a previous
            # request. We remember that it was set as a result of `auto`
            # using the `_backend_was_auto` field set in the params.
            if backend != _backend and not (
                backend == "auto" and params.structured_outputs._backend_was_auto
            ):
                raise ValueError(
                    "Request-level structured output backend selection is not "
                    f"supported. The request specified '{_backend}', but vLLM "
                    f"was initialised with '{backend}'. This error can be "
                    "resolved by removing '_backend' from the request."
                )
        else:
            params.structured_outputs._backend = backend

        # Request content validation
        if (
            isinstance(params.structured_outputs.choice, list)
            and not params.structured_outputs.choice
        ):
            # It is invalid for choice to be an empty list
            raise ValueError(
                f"Choice '{params.structured_outputs.choice}' cannot be an empty list"  # noqa: E501
            )
        # Reject empty string grammar early to avoid engine-side crashes
        if (
            isinstance(params.structured_outputs.grammar, str)
            and params.structured_outputs.grammar.strip() == ""
        ):
            raise ValueError("structured_outputs.grammar cannot be an empty string")

        if backend.startswith("xgrammar"):
            # xgrammar with no fallback
            validate_xgrammar_grammar(params)
        elif backend.startswith("guidance"):
            # TODO: ideally we would have the LLTokenizer here as Lark syntax
            # allows <|special_token|> and similar, see
            # https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md#special-tokens
            # Without tokenizer these are disallowed in grammars.
            if isinstance(self.tokenizer, MistralTokenizer):
                raise ValueError(
                    "Mistral tokenizer is not supported for the 'guidance' "
                    "structured output backend. Please use ['xgrammar', 'outlines'] "
                    "backends or tokenizer_mode='hf' instead."
                )
            validate_guidance_grammar(params, tokenizer=None)
        elif backend == "outlines":
            # outlines backend
            validate_structured_output_request_outlines(params)
        elif backend == "lm-format-enforcer":
            # lm format enforcer backend
            if isinstance(self.tokenizer, MistralTokenizer):
                raise ValueError(
                    "Mistral tokenizer is not supported for the 'lm-format-enforcer' "
                    "structured output backend. Please use ['xgrammar', 'outlines'] "
                    "backends or tokenizer_mode='hf' instead."
                )
            validate_structured_output_request_lm_format_enforcer(params)
        else:
            # NOTE: backend must be "auto" here, because we have
            # checked supported_backends above.
            # In this mode, we set opinionated defaults based on what we think
            # will satisfy the most use cases without having to worry about
            # this setting. We include fallback behavior here, but not with any
            # other setting where a specific backend was specified.
            try:
                validate_xgrammar_grammar(params)
                params.structured_outputs._backend = "xgrammar"
            except ValueError:
                # The request either failed validation
                # or includes some jsonschema feature(s) that
                # are not supported in xgrammar.
                if isinstance(self.tokenizer, MistralTokenizer):
                    # Fall back to outlines if the tokenizer is Mistral
                    validate_structured_output_request_outlines(params)
                    params.structured_outputs._backend = "outlines"
                else:
                    # Fall back to guidance by default.
                    validate_guidance_grammar(params, tokenizer=None)
                    params.structured_outputs._backend = "guidance"
            # Remember that this backend was set automatically
            params.structured_outputs._backend_was_auto = True

    def _maybe_build_mm_uuids(
        self,
        request_id: str,
        prompt: PromptType | EngineInput,
    ) -> MultiModalUUIDDict | None:
        """Build per-item multimodal hash overrides when enabled. In this case,
        multimodal data items are identified by their request id, modality and
        index rather than their content.

        Returns a dictionary of modality -> list[str] of overrides, or None if
        disabled or no multimodal data is present.
        """

        def _extract_mm_data(p: PromptType):
            if isinstance(p, dict) and "encoder_prompt" in p:
                enc = p.get("encoder_prompt")
                if isinstance(enc, dict):
                    return enc.get("multi_modal_data")
                return None
            if isinstance(p, dict):
                return p.get("multi_modal_data")
            return None

        mm_data = _extract_mm_data(prompt)
        if not mm_data:
            return None

        mm_uuids: dict[str, list[str | None] | str] = {}
        for modality, data in mm_data.items():
            # Hash each item for embedding inputs.
            n = (
                len(data)
                if isinstance(data, list) or MultiModalDataParser.is_embeddings(data)
                else 1
            )
            mm_uuids[modality] = [f"{request_id}-{modality}-{i}" for i in range(n)]
        return mm_uuids

    @staticmethod
    def assign_request_id(request: EngineCoreRequest):
        """Replace the externally supplied request ID with an internal request ID
        that adds 8 random characters in order to ensure uniqueness.
        """
        if request.external_req_id is not None:
            raise ValueError(
                "The external_req_id field should not be set on EngineCoreRequests"
                " passed to vLLM; use the request_id field."
            )
        request.external_req_id = request.request_id
        if envs.VLLM_DISABLE_REQUEST_ID_RANDOMIZATION:
            logger.warning_once(
                "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION is set and will be "
                "removed in a future release. Duplicate externally-provided "
                "request IDs may cause failures and/or subtle correctness errors."
            )
        else:
            request.request_id = f"{request.external_req_id}-{random_uuid():.8}"

    def process_inputs(
        self,
        request_id: str,
        prompt: PromptType | EngineInput,
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        supported_tasks: tuple[Any, ...] | None = None,
        resumable: bool = False,
        session_id: str | None = None,
        kv_hints: KvHintsEnvelope | None = None,
    ) -> EngineCoreRequest:
        watermarking = self._validate_params(params, supported_tasks)
        self._validate_lora(lora_request)
        self._validate_params(params)

        data_parallel_size = self.vllm_config.parallel_config.data_parallel_size
        if data_parallel_rank is not None and not (
            0 <= data_parallel_rank < data_parallel_size
        ):
            raise ValueError(
                f"data_parallel_rank {data_parallel_rank} "
                f"is out of range [0, {data_parallel_size})."
            )

        if isinstance(prompt, dict) and "type" in prompt:
            # Already-rendered EngineInput (output of Renderer.render_cmpl()
            # or render_chat()): pass it through untouched. Running it through
            # input_preprocessor.preprocess() again would treat the renderer's
            # MultiModalInput as a raw prompt, silently dropping mm_kwargs /
            # mm_placeholders in _process_tokens.
            if tokenization_kwargs:
                logger.warning_once(
                    "Passing tokenization_kwargs to InputProcessor is "
                    "deprecated and will be removed in v0.18. You should "
                    "instead pass them to Renderer.render_cmpl() or "
                    "Renderer.render_chat()."
                )

            if arrival_time is None:
                arrival_time = prompt.get("arrival_time", time.time())  # type: ignore[assignment]

            processed_inputs: EngineInput = prompt  # type: ignore[assignment]
        else:
            logger.warning_once(
                "Passing raw prompts to InputProcessor is deprecated "
                "and will be removed in v0.18. You should instead pass "
                "the outputs of Renderer.render_cmpl() or Renderer.render_chat()."
            )

            if arrival_time is None:
                arrival_time = time.time()

            # Optionally generate multimodal hash overrides to avoid hashing
            # multimodal data items by their content as their identifiers.

            # NOTE: when users explicitly turn off BOTH prefix caching and
            # input processing caching, no multimodal features or embeddings
            # will be reused across requests, therefore identifying
            # multimodal data items by their content is no longer necessary,
            # and we create uuids with request id-modality-index as
            # multimodal hash overrides.
            if (
                self.model_config.multimodal_config
                and self.model_config.multimodal_config.mm_processor_cache_gb == 0
                and not self.cache_config.enable_prefix_caching
            ):
                mm_uuids = self._maybe_build_mm_uuids(request_id, prompt)
            else:
                # Otherwise, use user-provided uuids as multimodal hash
                # overrides if provided.
                self._validate_multi_modal_uuids(prompt)
                if isinstance(prompt, dict):
                    mm_uuids = cast(
                        MultiModalUUIDDict | None, prompt.get("multi_modal_uuids")
                    )
                else:
                    mm_uuids = None

            # Process inputs, which includes:
            # 1. Tokenize text prompt, with LoRA request if one exists.
            # 2. For multimodal models with a merged preprocessor, preprocess
            #   multimodal data and expand prompt token ids accordingly.
            processed_inputs = self.input_preprocessor.preprocess(
                prompt,
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )
        from vllm.platforms import current_platform

        current_platform.validate_request(
            processed_inputs=processed_inputs,
            params=params,
        )

        eos_token_id = self.input_preprocessor.renderer.get_eos_token_id()

        encoder_inputs, decoder_inputs = split_enc_dec_input(processed_inputs)
        self._validate_model_inputs(encoder_inputs, decoder_inputs)

        # Mypy can be conservative for TypedDict unions; normalize access.
        if decoder_inputs["type"] == "embeds":
            prompt_token_ids = None
            prompt_embeds = decoder_inputs["prompt_embeds"]
        else:
            prompt_token_ids = decoder_inputs["prompt_token_ids"]
            prompt_embeds = None

        sampling_params = None
        pooling_params = None
        if isinstance(params, SamplingParams):
            # TODO: can we avoid cloning here in multiproc case?
            sampling_params = params.clone()
            # Resolve on the request-local copy: `params` may be shared across
            # prompts, requests and even engines by the caller.
            assert watermarking is not None
            self.apply_watermarking(sampling_params, watermarking)
            prompt_len = length_from_prompt_token_ids_or_embeds(
                prompt_token_ids, prompt_embeds
            )
            if not 0 <= sampling_params.routed_experts_prompt_start <= prompt_len:
                raise VLLMValidationError(
                    f"routed_experts_prompt_start must be between 0 and "
                    f"the prompt length ({prompt_len}), inclusive.",
                    parameter="routed_experts_prompt_start",
                    value=sampling_params.routed_experts_prompt_start,
                )
            rows = sampling_params.prompt_logprob_token_ids
            if rows is not None:
                start = sampling_params.prompt_logprob_start or 0
                num_rows = max(prompt_len - 1 - start, 0)
                if len(rows) != num_rows:
                    raise VLLMValidationError(
                        f"prompt_logprob_token_ids has {len(rows)} rows, but the "
                        f"prompt has {num_rows} scored rows "
                        f"(prompt_len - 1 - prompt_logprob_start).",
                        parameter="prompt_logprob_token_ids",
                        value=len(rows),
                    )
            # If unset max tokens, then generate up to the max_model_len.
            if sampling_params.max_tokens is None:
                seq_len = length_from_prompt_token_ids_or_embeds(
                    prompt_token_ids, prompt_embeds
                )
                sampling_params.max_tokens = self.model_config.max_model_len - seq_len
                # min_tokens is not checked while max_tokens is unset.
                if sampling_params.min_tokens > sampling_params.max_tokens:
                    raise VLLMValidationError(
                        f"min_tokens must be less than or equal to "
                        f"max_tokens={sampling_params.max_tokens}, got "
                        f"{sampling_params.min_tokens}."
                    )
            sampling_params.update_from_generation_config(
                self.generation_config_fields, eos_token_id
            )
            if self.tokenizer is not None:
                sampling_params.update_from_tokenizer(self.tokenizer)
        else:
            pooling_params = params.clone()

        # Multimodal related.
        mm_features: list[MultiModalFeatureSpec] | None = None

        if decoder_inputs["type"] == "multimodal":
            decoder_mm_inputs = decoder_inputs["mm_kwargs"]
            decoder_mm_positions = decoder_inputs["mm_placeholders"]
            decoder_mm_hashes = decoder_inputs["mm_hashes"]

            # Merge and flatten multimodal placeholders, hashes and inputs
            # from dictionaries to lists, and sort them by each item's position
            # in the input sequence.
            sorted_mm_idxs = argsort_mm_positions(decoder_mm_positions)

            mm_features = []
            for modality, idx in sorted_mm_idxs:
                mm_features.append(
                    MultiModalFeatureSpec(
                        data=decoder_mm_inputs[modality][idx],
                        modality=modality,
                        identifier=decoder_mm_hashes[modality][idx],
                        mm_position=decoder_mm_positions[modality][idx],
                    )
                )

        return EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            prompt_embeds=prompt_embeds,
            mm_features=mm_features,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            cache_salt=decoder_inputs.get("cache_salt"),
            priority=priority,
            data_parallel_rank=data_parallel_rank,
            trace_headers=trace_headers,
            resumable=resumable,
            session_id=session_id,
            kv_hints=kv_hints,
        )

    def _validate_model_inputs(
        self, encoder_inputs: SingletonInput | None, decoder_inputs: SingletonInput
    ):
        if encoder_inputs is not None:
            self._validate_model_input(encoder_inputs, prompt_type="encoder")

        self._validate_model_input(decoder_inputs, prompt_type="decoder")

    def _validate_model_input(
        self,
        prompt_inputs: SingletonInput,
        *,
        prompt_type: Literal["encoder", "decoder"],
    ):
        model_config = self.model_config

        prompt_ids = (
            None
            if prompt_inputs["type"] == "embeds"
            else prompt_inputs["prompt_token_ids"]
        )
        prompt_embeds = (
            prompt_inputs["prompt_embeds"]
            if prompt_inputs["type"] == "embeds"
            else None
        )
        prompt_len = length_from_prompt_token_ids_or_embeds(prompt_ids, prompt_embeds)
        if prompt_inputs["type"] == "embeds":
            is_token_ids = prompt_inputs.get("is_token_ids")
            if is_token_ids is not None and len(is_token_ids) != prompt_len:
                raise VLLMValidationError(
                    "prompt_is_token_ids must have the same length as prompt_embeds "
                    f"(expected {prompt_len}, got {len(is_token_ids)}).",
                    parameter="prompt_is_token_ids",
                )
        if not prompt_ids:
            if prompt_type == "encoder" and model_config.is_multimodal_model:
                pass  # Mllama may have empty encoder inputs for text-only data
            elif prompt_inputs["type"] == "embeds":
                pass  # Prompt embeds should not have prompt_ids.
            else:
                raise ValueError(f"The {prompt_type} prompt cannot be empty")

        tokenizer = self.tokenizer
        if tokenizer is not None:
            max_input_id = max(prompt_ids or [], default=0)

            # NOTE: tokenizer.max_token_id is the tokenizer’s vocab size while
            # self.model_config.get_vocab_size() is the model’s vocab size.
            # For Qwen3 models, the language model has extra tokens that do
            # not exist in the tokenizer, and vice versa for multimodal
            # placeholder tokens in some multimodal models.
            # See https://github.com/QwenLM/Qwen3/issues/29#issuecomment-1933720399 # noqa: E501
            # and https://github.com/vllm-project/vllm/pull/22471#discussion_r2312251421 # noqa: E501

            # Here we take the max of the two to determine if a token id is
            # truly out-of-vocabulary.
            if max_input_id > max(
                tokenizer.max_token_id, self.model_config.get_vocab_size() - 1
            ):
                raise ValueError(f"Token id {max_input_id} is out of vocabulary")

        max_prompt_len = self.model_config.max_model_len
        if prompt_len > max_prompt_len:
            if prompt_type == "encoder" and model_config.is_multimodal_model:
                mm_registry = self.input_preprocessor.mm_registry
                mm_processor = mm_registry.create_processor(
                    model_config,
                    tokenizer=tokenizer,
                )
                assert isinstance(mm_processor, EncDecMultiModalProcessor)

                if mm_processor.pad_dummy_encoder_prompt:
                    return  # Skip encoder length check for Whisper

            if model_config.is_multimodal_model:
                suggestion = (
                    "Make sure that `max_model_len` is no smaller than the "
                    "number of text tokens plus multimodal tokens. For image "
                    "inputs, the number of image tokens depends on the number "
                    "of images, and possibly their aspect ratios as well."
                )
            else:
                suggestion = (
                    "Make sure that `max_model_len` is no smaller than the "
                    "number of text tokens."
                )

            raise ValueError(
                f"The {prompt_type} prompt (length {prompt_len}) is "
                f"longer than the maximum model length of {max_prompt_len}. "
                f"{suggestion}"
            )

            # TODO: Find out how many placeholder tokens are there so we can
            # check that chunked prefill does not truncate them
            # max_batch_len = self.scheduler_config.max_num_batched_tokens

        if (
            prompt_len == max_prompt_len
            and prompt_type == "decoder"
            and not model_config.is_multimodal_model
            and self.model_config.runner_type != "pooling"
        ):
            suggestion = (
                "Make sure that `max_model_len` is no smaller than the "
                "number of text tokens (prompt + requested output tokens)."
            )
            raise ValueError(
                f"The {prompt_type} prompt (length {prompt_len}) plus the number of "
                f"requested output tokens (at least 1) is longer than the maximum "
                f"model length of {max_prompt_len}. {suggestion}"
            )

    def _validate_model_input(
        self,
        prompt_input: SingletonInput,
        prompt_type: Literal["encoder", "decoder"],
    ) -> None:
        prompt_ids = (
            None
            if prompt_input["type"] == "embeds"
            else prompt_input["prompt_token_ids"]
        )
        prompt_embeds = (
            prompt_input["prompt_embeds"] if prompt_input["type"] == "embeds" else None
        )

        prompt_len = length_from_prompt_token_ids_or_embeds(prompt_ids, prompt_embeds)
        self._validate_prompt_len(prompt_len, prompt_type)

        if prompt_input["type"] == "embeds":
            is_token_ids = prompt_input.get("is_token_ids")
            if is_token_ids is not None and len(is_token_ids) != prompt_len:
                raise VLLMValidationError(
                    "prompt_is_token_ids must have the same length as prompt_embeds "
                    f"(expected {prompt_len}, got {len(is_token_ids)}).",
                    parameter="prompt_is_token_ids",
                )

        if prompt_input["type"] == "multimodal":
            decoder_mm_positions = prompt_input["mm_placeholders"]
            for modality, mm_positions in decoder_mm_positions.items():
                for mm_position in mm_positions:
                    num_embeds = mm_position.get_num_embeds()
                    if num_embeds > self.mm_encoder_cache_size:
                        raise VLLMValidationError(
                            f"The {prompt_type} prompt contains a(n) {modality} item "
                            f"with {num_embeds} embedding tokens, which exceeds the "
                            f"pre-allocated encoder cache size "
                            f"{self.mm_encoder_cache_size}. Please reduce the input "
                            f"size or increase the encoder cache size "
                            f"by setting --limit-mm-per-prompt at startup."
                        )

        # Shared by generate, embedding and pooling requests.
        if prompt_ids:
            self.renderer.validate_token_ids(prompt_ids)

    def _validate_model_inputs(
        self,
        encoder_input: SingletonInput | None,
        decoder_input: SingletonInput,
    ):
        if encoder_input is not None:
            self._validate_model_input(encoder_input, prompt_type="encoder")

        self._validate_model_input(decoder_input, prompt_type="decoder")
