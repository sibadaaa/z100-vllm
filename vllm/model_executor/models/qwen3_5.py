# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3.5 Series compatible with HuggingFace weights."""

from collections.abc import Iterable
from typing import ClassVar

import torch
from torch import nn
from transformers import (
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5TextConfig,
)

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.utils import (
    is_model_fused_shared_expert_compatible,
)
from vllm.model_executor.layers.fusion.mm_input_norm import build_mm_input_norm
from vllm.model_executor.layers.layernorm import GemmaRMSNorm as Qwen3_5RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
    maybe_remap_moe_expert_param_name,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.sequence import IntermediateTensors
from vllm.tokenizers.registry import cached_tokenizer_from_config

from .interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
    _require_is_multimodal,
)
from .qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from .qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextModel,
    Qwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts,
)
from .qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
    mrope_positions_factor,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    _merge_multimodal_embeddings,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_fuse_shared_experts,
    maybe_prefix,
)

logger = init_logger(__name__)


class Qwen3_5ProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5Config)


class Qwen3_5MoeProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        # transformers 5.x renames the top-level Qwen3.5-MoE config class to
        # Qwen3_5MoeTextConfig for text-only models, while transformers ≤4.x
        # returns Qwen3_5MoeConfig (the multimodal wrapper).  Accept both so
        # that vLLM works regardless of which transformers version is installed.
        return self.ctx.get_hf_config((Qwen3_5MoeConfig, Qwen3_5MoeTextConfig))


class Qwen3_5DecoderLayer(Qwen3NextDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super(Qwen3NextDecoderLayer, self).__init__()

        config = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        is_moe_layer = config.model_type == "qwen3_5_moe_text"
        self.use_attn_reduce_scatter_for_moe = (
            parallel_config.use_sequence_parallel_moe
            and parallel_config.pipeline_parallel_size == 1
            and is_moe_layer
        )

        if self.layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
                reduce_results=not self.use_attn_reduce_scatter_for_moe,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                reduce_results=not self.use_attn_reduce_scatter_for_moe,
                mrope_positions_factor=mrope_positions_factor(vllm_config),
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        # NOTE: Determine the MLP type based on the model type
        # Qwen3.5 use all layers for MLP / Qwen3.5-MoE use sparse MoE blocks
        if config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        elif config.model_type == "qwen3_5_text":
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            raise ValueError(f"Invalid model_type {config.model_type}")

        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3_5Model(Qwen3NextModel):
    # Qwen3.5 ships the GDN in_proj checkpoints separately (qwen3-next
    # pre-fuses them); fuse them on top of the qwen3-next QKV/gate_up mapping.
    hf_to_vllm_mapper = Qwen3NextModel.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_stacked={
            ".in_proj_qkv": (".in_proj_qkvz", (0, 1, 2)),
            ".in_proj_z": (".in_proj_qkvz", 3),
            ".in_proj_b": (".in_proj_ba", 0),
            ".in_proj_a": (".in_proj_ba", 1),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3NextModel, self).__init__()

        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig = (
            vllm_config.model_config.hf_text_config
        )
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config
        self.quant_config = vllm_config.quant_config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            return Qwen3_5DecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.is_fused_shared_expert_enabled = is_model_fused_shared_expert_compatible(
            self.layers,
            Qwen3NextSparseMoeBlock,
            "mlp",
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # FSE must match construction (Qwen3NextSparseMoeBlock): reroute the
        # shared expert into the extra fused slot only when AITER FSE is both
        # requested and compatible with the quant spec.
        if "moe" in self.config.model_type:
            weights = maybe_fuse_shared_experts(
                weights,
                enabled=self.is_fused_shared_expert_enabled,
                n_routed_experts=self.config.num_experts,
                n_shared_experts=1,
                ckpt_prefix="mlp.shared_expert",
            )

        force_split = bool(
            getattr(first_linear_attn, "split_projections", False)
            if first_linear_attn is not None
            else False
        )

        has_split_qkv = bool(
            first_linear_attn is not None
            and hasattr(first_linear_attn, "in_proj_qkv")
        )
        has_split_ba = bool(
            first_linear_attn is not None
            and hasattr(first_linear_attn, "in_proj_b")
            and hasattr(first_linear_attn, "in_proj_a")
        )
        has_fused_ba = bool(
            first_linear_attn is not None
            and hasattr(first_linear_attn, "in_proj_ba")
            and first_linear_attn.in_proj_ba is not None
        )
        force_split_ba = os.getenv("VLLM_QWEN35_FORCE_BA_SPLIT", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if force_split_ba:
            has_split_ba = True
            has_fused_ba = False
        use_split_gdn_proj = has_split_qkv

        _append_qwen35_load_debug(
            f"MODE force_split={force_split} has_split_qkv={has_split_qkv} has_split_ba={has_split_ba} has_fused_ba={has_fused_ba}"
        )

        if has_fused_ba and not has_split_ba and not force_split:
            reverse_ba = os.getenv("VLLM_QWEN35_REVERSE_BA_SHARDS", "0").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if reverse_ba:
                stacked_params_mapping.extend(
                    [
                        ("in_proj_ba", "in_proj_a", 0),
                        ("in_proj_ba", "in_proj_b", 1),
                    ]
                )
            else:
                stacked_params_mapping.extend(
                    [
                        ("in_proj_ba", "in_proj_b", 0),
                        ("in_proj_ba", "in_proj_a", 1),
                    ]
                )

        if not has_split_qkv and not force_split:
            stacked_params_mapping.extend(
                [
                    ("in_proj_qkvz", "in_proj_qkv", (0, 1, 2)),
                    ("in_proj_qkvz", "in_proj_z", 3),
                ]
            )

        modules_dict = dict(self.named_modules())
        loaded_params: set[str] = set()
        # Track partition sizes for merged parameters to calculate correct offsets
        partition_sizes: dict[str, dict[int, int]] = {}
        expert_params_mapping = self.get_expert_mapping()
        is_fused_expert = False
        fused_expert_params_mapping = _make_qwen35_fused_expert_params_mapping(self)
        num_experts = (
            self.config.num_experts if hasattr(self.config, "num_experts") else 0
        )

        def get_split_sizes_for_param(
            param_name: str, shard_ids: tuple[int, ...]
        ) -> list[int] | None:
            module_name, _, _ = param_name.rpartition(".")
            if not module_name:
                return None

            module = modules_dict.get(module_name)
            output_sizes = getattr(module, "output_sizes", None)
            if output_sizes is None:
                return None

            max_shard_id = max(shard_ids)
            if max_shard_id >= len(output_sizes):
                return None

            return [output_sizes[int(shard_id)] for shard_id in shard_ids]

        def load_split_gguf_shards(
            param_name,
            param,
            loaded_weight,
            shard_ids,
            weight_loader,
        ):
            split_sizes = get_split_sizes_for_param(param_name, shard_ids)

            if getattr(param, "is_gguf_weight_type", False):
                for sid in shard_ids:
                    param.shard_weight_type[sid] = loaded_weight.item()
                return

            if getattr(param, "is_gguf_weight", False):
                output_dim = getattr(param, "output_dim", 0)
                if split_sizes is None:
                    shard_size = loaded_weight.size(output_dim) // len(shard_ids)
                    split_sizes = [shard_size] * len(shard_ids)
                assert sum(split_sizes) == loaded_weight.size(output_dim), (
                    f"Cannot split GGUF shard for {param_name}: expected logical sizes "
                    f"{split_sizes}, got shape {tuple(loaded_weight.shape)} over dim "
                    f"{output_dim}"
                )
                tp_size = get_tensor_model_parallel_world_size()
                tp_rank = get_tensor_model_parallel_rank()
                split_offset = 0

                for sid, shard_size in zip(shard_ids, split_sizes, strict=True):
                    assert shard_size % tp_size == 0, (
                        f"GGUF shard size {shard_size} for {param_name} is not divisible "
                        f"by tensor parallel size {tp_size}"
                    )
                    local_shard = shard_size // tp_size
                    start_idx = tp_rank * local_shard
                    shard = loaded_weight.narrow(output_dim, split_offset, shard_size)
                    shard = shard.narrow(output_dim, start_idx, local_shard)
                    shard = shard.to(device=param.device)
                    param.shard_id.append(sid)
                    param.shard_id_map[sid] = len(param.data_container)
                    param.data_container.append(shard)
                    split_offset += shard_size
                return

            if loaded_weight.ndim == 0 or loaded_weight.numel() == 1:
                for sid in shard_ids:
                    weight_loader(param, loaded_weight, sid)
                return

            output_dim = getattr(param, "output_dim", 0)
            if split_sizes is None:
                shard_size = loaded_weight.size(output_dim) // len(shard_ids)
                split_sizes = [shard_size] * len(shard_ids)
            assert sum(split_sizes) == loaded_weight.size(output_dim), (
                f"Cannot split GGUF shard for {param_name}: expected logical sizes "
                f"{split_sizes}, got shape {tuple(loaded_weight.shape)} over dim "
                f"{output_dim}"
            )
            split_offset = 0
            tp_size = get_tensor_model_parallel_world_size()
            for sid, shard_size in zip(shard_ids, split_sizes, strict=True):
                shard = loaded_weight.narrow(output_dim, split_offset, shard_size)
                if hasattr(param, "load_merged_column_weight"):
                    param.load_merged_column_weight(
                        loaded_weight=shard,
                        shard_id=sid,
                        shard_offset=split_offset // tp_size,
                        shard_size=shard_size // tp_size,
                    )
                elif "loaded_shard_id" in signature(weight_loader).parameters:
                    weight_loader(param, shard, loaded_shard_id=sid)
                else:
                    weight_loader(param, shard, sid)
                split_offset += shard_size

        def load_single_gguf_shard(param, loaded_weight, shard_id):
            if getattr(param, "is_gguf_weight_type", False):
                param.shard_weight_type[shard_id] = loaded_weight.item()
                return

            if getattr(param, "is_gguf_weight", False):
                output_dim = getattr(param, "output_dim", 0)
                tp_size = get_tensor_model_parallel_world_size()
                tp_rank = get_tensor_model_parallel_rank()
                local_shard = loaded_weight.size(output_dim) // tp_size
                start_idx = tp_rank * local_shard
                shard = loaded_weight.narrow(output_dim, start_idx, local_shard)
                shard = shard.to(device=param.device)
                param.shard_id.append(shard_id)
                param.shard_id_map[shard_id] = len(param.data_container)
                param.data_container.append(shard)
                return

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if name.startswith("mtp."):
                continue

            if self.quant_config is not None and self.quant_config.get_name() == "gguf":
                module_name, _, param_leaf = name.rpartition(".")
                module = modules_dict.get(module_name)
                if param_leaf == "weight" and isinstance(module, Qwen3_5RMSNorm):
                    loaded_weight = loaded_weight - 1.0

                if name.endswith("A_log"):
                    loaded_weight = torch.log(-loaded_weight)

            # Remapping the name of FP8 kv-scale.
            if name.endswith("scale"):
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                if f".{weight_name}." not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                # name = apply_attn_prefix(name, params_dict)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                if "linear_attn.in_proj" in name and any(
                    token in name
                    for token in ("in_proj_qkv", "in_proj_z", "in_proj_ba", "in_proj_b", "in_proj_a")
                ):
                    _append_qwen35_load_debug(
                        "STACKED "
                        f"name={name} shard_id={shard_id} "
                        f"is_gguf_weight={getattr(param, 'is_gguf_weight', False)} "
                        f"is_gguf_weight_type={getattr(param, 'is_gguf_weight_type', False)} "
                        f"loaded_shape={tuple(loaded_weight.shape) if hasattr(loaded_weight, 'shape') else 'NA'}"
                    )
                if isinstance(shard_id, tuple):
                    if getattr(param, "is_gguf_weight", False) or getattr(
                        param, "is_gguf_weight_type", False
                    ):
                        load_split_gguf_shards(
                            name, param, loaded_weight, shard_id, weight_loader
                        )
                    else:
                        weight_loader(param, loaded_weight, shard_id)
                    break
                if shard_id is not None and (
                    getattr(param, "is_gguf_weight", False)
                    or getattr(param, "is_gguf_weight_type", False)
                ):
                    load_single_gguf_shard(param, loaded_weight, shard_id)
                    break
                # Check if weight_loader accepts shard_id parameter
                import inspect

                sig = inspect.signature(weight_loader)
                if len(sig.parameters) >= 3:
                    weight_loader(param, loaded_weight, shard_id)
                elif (
                    hasattr(param, "load_merged_column_weight") and shard_id is not None
                ):
                    # Handle merged column parallel parameters (e.g., gate_up_proj)
                    # Calculate shard_offset by accumulating previous partition sizes
                    if isinstance(shard_id, int):
                        # For gate_up_proj: shard_id 0 = gate, shard_id 1 = up
                        # Use the sharded size from checkpoint, accounting for TP
                        tp_size = get_tensor_model_parallel_world_size()
                        shard_size = loaded_weight.size(param.output_dim) // tp_size

                        # Track this partition's size for offset calculation
                        if name not in partition_sizes:
                            partition_sizes[name] = {}
                        partition_sizes[name][shard_id] = shard_size

                        # Calculate offset as sum of all previous partition sizes
                        shard_offset = sum(
                            size
                            for sid, size in partition_sizes[name].items()
                            if sid < shard_id
                        )

                        param.load_merged_column_weight(
                            loaded_weight,
                            shard_offset=shard_offset,
                            shard_size=shard_size,
                        )
                    elif hasattr(param, "load_qkv_weight"):
                        # For QKV parameters with string shard_id
                        num_heads = getattr(self.config, "num_attention_heads", None)
                        if num_heads:
                            # Calculate shard_size and offset for QKV
                            # Use the sharded size from checkpoint, accounting for TP
                            tp_size = get_tensor_model_parallel_world_size()
                            shard_size = loaded_weight.size(param.output_dim) // tp_size

                            # Track this partition's size for offset calculation
                            if name not in partition_sizes:
                                partition_sizes[name] = {}
                            # Convert string shard_id to integer for tracking
                            qkv_shard_map = {"q": 0, "k": 1, "v": 2}
                            shard_id_int = qkv_shard_map.get(shard_id, 0)
                            partition_sizes[name][shard_id_int] = shard_size

                            # Calculate offset as sum of all previous partition sizes
                            shard_offset = sum(
                                size
                                for sid, size in partition_sizes[name].items()
                                if sid < shard_id_int
                            )

                            param.load_qkv_weight(
                                loaded_weight,
                                shard_offset=shard_offset,
                                shard_size=shard_size,
                                shard_id=shard_id,
                                num_heads=num_heads,
                            )
                        else:
                            weight_loader(param, loaded_weight)
                    else:
                        try:
                            weight_loader(param, loaded_weight)
                        except AssertionError as exc:
                            raise AssertionError(
                                f"Failed loading {name}: param shape {tuple(param.shape)} "
                                f"vs weight shape {tuple(loaded_weight.shape)}"
                            ) from exc
                else:
                    try:
                        loaded_weight = _maybe_unsqueeze_shared_expert_gate(
                            name, param, loaded_weight
                        )
                        weight_loader(param, loaded_weight)
                    except AssertionError as exc:
                        raise AssertionError(
                            f"Failed loading {name}: param shape {tuple(param.shape)} "
                            f"vs weight shape {tuple(loaded_weight.shape)}"
                        ) from exc
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    # Post-#41184 the expert params live under
                    # ``experts.routed_experts.*``; remap old-style names.
                    name_mapped = maybe_remap_moe_expert_param_name(
                        name_mapped, params_dict
                    )
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    if is_fused_expert:
                        # qwen3.5 no need to transpose
                        # loaded_weight = loaded_weight.transpose(-1, -2)
                        if "experts.gate_up_proj" in name:
                            loaded_weight = loaded_weight.chunk(2, dim=-2)
                            success_w1 = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[0],
                                "w1",
                                num_experts,
                            )
                            success_w3 = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[1],
                                "w3",
                                num_experts,
                            )
                            success = success_w1 and success_w3
                        else:
                            # down_proj
                            success = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                        if success:
                            name = name_mapped
                            break
                    else:
                        # Skip loading extra bias for GPTQ models.
                        if (
                            name_mapped.endswith(".bias")
                            or name_mapped.endswith("_bias")
                        ) and name_mapped not in params_dict:
                            continue
                        param = params_dict[name_mapped]
                        weight_loader = param.weight_loader
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        if _is_qwen35_gguf_projection_aux_param(name):
                            _append_qwen35_load_debug(f"MISSING name={name}")
                            continue
                        logger.warning_once(
                            f"Parameter {name} not found in params_dict, skip loading"
                        )
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    if "linear_attn.in_proj" in name and any(
                        token in name
                        for token in ("in_proj_qkv", "in_proj_z", "in_proj_ba", "in_proj_b", "in_proj_a")
                    ):
                        _append_qwen35_load_debug(
                            "DIRECT "
                            f"name={name} "
                            f"is_gguf_weight={getattr(param, 'is_gguf_weight', False)} "
                            f"is_gguf_weight_type={getattr(param, 'is_gguf_weight_type', False)} "
                            f"loaded_shape={tuple(loaded_weight.shape) if hasattr(loaded_weight, 'shape') else 'NA'}"
                        )
                    if getattr(param, "is_gguf_weight", False) or getattr(
                        param, "is_gguf_weight_type", False
                    ):
                        if getattr(param, "is_gguf_weight_type", False):
                            param.weight_type = loaded_weight.item()
                        else:
                            tp_size = get_tensor_model_parallel_world_size()
                            tp_rank = get_tensor_model_parallel_rank()
                            module_name, _, _ = name.rpartition(".")
                            module = modules_dict[module_name]
                            if isinstance(module, RowParallelLinear):
                                # Row-parallel GGUF weights (GDN out_proj,
                                # attention o_proj, MLP down_proj) keep the
                                # full output dim and are sharded along the
                                # packed input dim, mirroring
                                # RowParallelLinear.weight_loader: the raw
                                # quant layout is block-linear in the input
                                # dim, so halving it halves the block count
                                # per rank. Narrowing dim 0 (the column
                                # convention) instead silently produces a
                                # half-width unreduced output and crashes the
                                # first TP>1 forward.
                                shard_dim = getattr(param, "input_dim", 1)
                                local_shard = (
                                    loaded_weight.size(shard_dim) // tp_size
                                )
                                loaded_weight = loaded_weight.narrow(
                                    shard_dim,
                                    tp_rank * local_shard,
                                    local_shard,
                                )
                            elif isinstance(module, MergedColumnParallelLinear):
                                # A fused GGUF tensor feeding a merged column
                                # layer (GDN in_proj_qkv = [q|k|v]) must be
                                # sharded per logical section; a contiguous
                                # dim-0 halving would hand rank 0 all of
                                # q,k plus half of v while the forward
                                # unpacks [q/2|k/2|v/2] per rank.
                                shards = []
                                offset = 0
                                for section in module.output_sizes:
                                    local_shard = section // tp_size
                                    shards.append(
                                        loaded_weight.narrow(
                                            0,
                                            offset + tp_rank * local_shard,
                                            local_shard,
                                        )
                                    )
                                    offset += section
                                loaded_weight = torch.cat(shards, dim=0)
                            else:
                                output_dim = getattr(param, "output_dim", 0)
                                local_shard = (
                                    loaded_weight.size(output_dim) // tp_size
                                )
                                loaded_weight = loaded_weight.narrow(
                                    output_dim,
                                    tp_rank * local_shard,
                                    local_shard,
                                )
                            loaded_weight = loaded_weight.to(device=param.device)
                            loaded_param = torch.nn.Parameter(
                                loaded_weight.contiguous(), requires_grad=False
                            )
                            for attr_name, attr_value in vars(param).items():
                                setattr(loaded_param, attr_name, attr_value)
                            module_name, _, param_leaf = name.rpartition(".")
                            module = modules_dict[module_name]
                            module.register_parameter(param_leaf, loaded_param)
                            params_dict[name] = loaded_param
                        loaded_params.add(name)
                        continue
                    try:
                        loaded_weight = _maybe_unsqueeze_shared_expert_gate(
                            name, param, loaded_weight
                        )
                        weight_loader(param, loaded_weight)
                    except AssertionError as exc:
                        raise AssertionError(
                            f"Failed loading {name}: param shape {tuple(param.shape)} "
                            f"vs weight shape {tuple(loaded_weight.shape)}"
                        ) from exc
            loaded_params.add(name)
        return loaded_params


class Qwen3_5ForCausalLMBase(
    nn.Module,
    HasInnerState,
    IsHybrid,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        # GDN fused projections.
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }
    # Maps PEFT embed/lm_head LoRA targets onto vLLM embedding wrappers.
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    # Some community text-only checkpoints keep the extraneous
    # `model.language_model.` prefix inherited from the VL training stack.
    # Strip it so both prefixed and clean checkpoints load correctly.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"model.language_model.": "model.", "mtp.": None},
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config

        scheduler_config = vllm_config.scheduler_config
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3_5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_logits_local(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1), 0


class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase):
    pass


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLMBase, QwenNextMixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # set MoE hyperparameters
        self.set_moe_parameters()


########################################################
# Qwen3_5-Dense
########################################################


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid):
    supports_multimodal_pruning = True

    hf_to_vllm_mapper = (
        Qwen3VLForConditionalGeneration.hf_to_vllm_mapper
        | WeightsMapper(orig_to_new_prefix={"mtp.": None})
    )

    packed_modules_mapping = Qwen3VLForConditionalGeneration.packed_modules_mapping | {
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # protocols have not __init__ method, so we need to use nn.Module.__init__
        nn.Module.__init__(self)
        config: Qwen3_5Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.get_multimodal_config()

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )
        self.video_pruning_rate = self.multimodal_config.video_pruning_rate
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

        # attributes needed by EVS-related functions inherited from Qwen3-VL
        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                input_norm=build_mm_input_norm(self.model_config),
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5ForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for Qwen3.5.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen3VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
            intermediate_tensors: Intermediate tensors from previous pipeline
                stages.
            inputs_embeds: Pre-computed input embeddings.
            **kwargs: Additional keyword arguments including:
                - pixel_values: Pixel values to be fed to a model.
                    `None` if no images are passed.
                - image_grid_thw: Tensor `(n_images, 3)` of image 3D grid in
                    LLM. `None` if no images are passed.
                - pixel_values_videos: Pixel values of videos to be fed to a
                    model. `None` if no videos are passed.
                - video_grid_thw: Tensor `(n_videos, 3)` of video 3D grid in
                    LLM. `None` if no videos are passed.

        """
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.compute_logits_local(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()


########################################################
# Qwen3_5-MoE
########################################################


class Qwen3_5_MoeMixtureOfExperts(MixtureOfExperts):
    language_model: Qwen3_5ForCausalLM
    num_local_physical_experts: int

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.language_model.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.moe_layers = []
        example_moe = None
        for layer in self.language_model.model.layers:
            if isinstance(layer, Qwen3_5DecoderLayer) and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError(
                "No Qwen3_5 layer found in the language_model.model.layers."
            )

        # Set MoE hyperparameters
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5MoeProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5MoeForConditionalGeneration(
    Qwen3_5ForConditionalGeneration, Qwen3_5_MoeMixtureOfExperts
):
    # For MoE LoRA weights loading
    is_3d_moe_weight: ClassVar[bool] = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # protocols have not __init__ method, so we need to use nn.Module.__init__
        nn.Module.__init__(self)
        config: Qwen3_5MoeConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.get_multimodal_config()

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )
        self.video_pruning_rate = self.multimodal_config.video_pruning_rate
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

        # attributes needed by EVS-related functions inherited from Qwen3-VL
        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                input_norm=build_mm_input_norm(self.model_config),
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5MoeForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # set MoE hyperparameters
        self.set_moe_parameters()
