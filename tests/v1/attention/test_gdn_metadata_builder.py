# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers import LlamaConfig

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


class _CacheConfigStub:
    # "none" keeps mamba_get_block_table_tensor's identity path, which is what
    # these builder-level tests want to exercise.
    mamba_cache_mode = "none"


class _VllmConfigStub:
    cache_config = _CacheConfigStub()


def _create_stateless_builder(full_cuda_graph: bool = False):
    builder = object.__new__(GDNAttentionMetadataBuilder)
    builder.use_spec_decode = False
    builder.num_spec = 0
    builder.use_full_cuda_graph = full_cuda_graph
    builder._building_for_capture = False
    # build() resolves the mamba state indices through these two; the upstream
    # sync moved that lookup into build(), so `object.__new__` builders need
    # them set explicitly.
    builder.kv_cache_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
    )
    builder.vllm_config = _VllmConfigStub()
    return builder


def test_decode_does_not_transfer_context_lens(monkeypatch):
    """A decode-only build must not transfer per-request context lengths.

    ``CommonAttentionMetadata`` no longer carries an exact CPU copy of
    ``num_computed_tokens`` (dropped by the upstream b44895cf sync), so the
    one-token-chunk classification still needs a single DtoH transfer per
    *step*: ``build()`` runs once per step and its metadata is reused by every
    GDN layer, so this is not a per-layer sync.  More than one transfer here
    means the builder started syncing per request or per layer.
    """
    builder = _create_stateless_builder()

    batch = BatchSpec(seq_lens=[80, 96], query_lens=[1, 1])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)

    transfers: list = []
    real_cpu = torch.Tensor.cpu

    def guarded(self, *args, **kwargs):
        transfers.append(tuple(self.shape))
        return real_cpu(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", guarded)
    meta = builder.build(common_prefix_len=0, common_attn_metadata=common)

    assert len(transfers) <= 1, transfers
    assert all(shape == (batch.batch_size,) for shape in transfers), transfers
    assert meta.num_prefills == 0
    assert meta.has_initial_state is None


def test_non_spec_query_start_has_cpu_twin():
    """The builder publishes a CPU twin of `non_spec_query_start_loc`.

    That is the tensor every GDN layer hands to `causal_conv1d_fn`, so the
    gfx906 per-sequence fallback can read its token boundaries on the host
    instead of issuing one DtoH sync per boundary per layer.  It must hold for a
    pure-prefill batch and for a mixed decode+prefill batch (no speculative
    decoding in either), where the CPU and device copies are the two halves of
    the same buffer.
    """
    for seq_lens, query_lens, want_prefills in (
        ([100, 1], [1, 1], 1),  # mixed: one decode row + one stateless chunk
        ([200, 96], [64, 64], 2),  # all-prefill chunked
    ):
        builder = _create_stateless_builder()
        common = create_common_attn_metadata(
            BatchSpec(seq_lens=seq_lens, query_lens=query_lens), BLOCK_SIZE, DEVICE
        )

        meta = builder.build(common_prefix_len=0, common_attn_metadata=common)

        assert meta.num_prefills == want_prefills
        mirror = meta.non_spec_query_start_loc_cpu
        assert mirror is not None
        assert mirror.device.type == "cpu"
        device_loc = meta.non_spec_query_start_loc
        assert device_loc is not None
        assert mirror.shape == device_loc.shape
        assert mirror.tolist() == device_loc.tolist()


def test_full_cudagraph_spec_metadata_uses_request_count(tmp_path):
    num_speculative_tokens = 3
    LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_hidden_layers=1,
        num_key_value_heads=4,
        vocab_size=128,
    ).save_pretrained(tmp_path)
    vllm_config = create_vllm_config(
        model_name=str(tmp_path),
        block_size=BLOCK_SIZE,
    )
    vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=num_speculative_tokens,
    )
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
        ),
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )
    batch = BatchSpec(seq_lens=[80, 96], query_lens=[4, 4])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    meta = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_decode_draft_tokens_cpu=torch.tensor([3, 3], dtype=torch.int32),
        num_accepted_tokens=torch.ones(batch.batch_size, dtype=torch.int32),
    )

    assert meta.num_spec_decodes == batch.batch_size
    assert meta.num_spec_decode_tokens == batch.compute_num_tokens()
    assert meta.spec_state_indices_tensor is not None
    assert meta.spec_state_indices_tensor.shape == (
        batch.batch_size,
        num_speculative_tokens + 1,
    )
    assert meta.spec_sequence_masks is not None
    assert meta.spec_sequence_masks.shape == (batch.batch_size,)
    assert meta.spec_query_start_loc is not None
    assert meta.spec_query_start_loc.shape == (batch.batch_size + 1,)
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.shape == (batch.batch_size,)


def _build_non_spec(batch: BatchSpec):
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    builder = _create_stateless_builder()
    return builder, common, builder.build(common_prefix_len=0, common_attn_metadata=common)


@pytest.mark.parametrize(
    ("seq_len", "query_len", "num_prefills"),
    [
        pytest.param(1, 1, 1, id="first-chunk"),
        pytest.param(65, 1, 0, id="resumed-chunk"),
        pytest.param(0, 0, 0, id="padding"),
    ],
)
def test_one_token_chunk_classification(
    seq_len: int,
    query_len: int,
    num_prefills: int,
):
    """Only a real first chunk with no prior state needs state initialization."""
    _, _, meta = _build_non_spec(
        BatchSpec(seq_lens=[100, seq_len], query_lens=[1, query_len])
    )

    assert meta.num_prefills == num_prefills
    assert meta.num_decodes == 2 - num_prefills
    assert meta.num_prefill_tokens == num_prefills
    assert meta.num_decode_tokens == 1 + query_len - num_prefills
    if num_prefills:
        assert meta.has_initial_state is not None
        assert meta.has_initial_state.tolist() == [True, False]
    else:
        assert meta.has_initial_state is None


def test_one_token_first_chunk_excludes_padding():
    """Neither padding requests nor padding tokens count as prefill work."""
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[100, 1, 0, 0], query_lens=[1, 1, 0, 0]),
        BLOCK_SIZE,
        DEVICE,
    )
    builder = _create_stateless_builder()
    meta = builder.build(common_prefix_len=0, common_attn_metadata=common)

    assert meta.num_decodes == 1
    assert meta.num_prefills == 1
    assert meta.num_decode_tokens == 1
    assert meta.num_prefill_tokens == 1


def test_cudagraph_capture_batch_stays_decode_only(tmp_path):
    """Capture rows have no history, but must still select decode kernels."""
    LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_hidden_layers=1,
        num_key_value_heads=4,
        vocab_size=128,
    ).save_pretrained(tmp_path)
    vllm_config = create_vllm_config(
        model_name=str(tmp_path),
        block_size=BLOCK_SIZE,
    )
    vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
        ),
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )
    batch = BatchSpec(seq_lens=[1] * 4, query_lens=[1] * 4)
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    meta = builder.build_for_cudagraph_capture(common)

    assert meta.num_prefills == 0
    assert meta.num_decodes == 4
    assert meta.has_initial_state is None
    staged = meta.non_spec_state_indices_tensor
    assert staged is not None
    assert staged.data_ptr() == builder.non_spec_state_indices_tensor.data_ptr()
    torch.testing.assert_close(staged, common.block_table_tensor[:, 0])
