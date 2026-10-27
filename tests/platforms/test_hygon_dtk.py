# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Hygon DCU (DTK) platform helpers.

These tests are GPU-free: they drive the sysfs/selection logic with fake
trees and monkeypatched environment variables.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

hygon_dtk = pytest.importorskip("vllm.platforms.hygon_dtk")


@pytest.fixture(autouse=True)
def clear_caches():
    hygon_dtk._is_hygon_dtk_cached.cache_clear()
    hygon_dtk._dcu_drm_cards.cache_clear()
    yield
    hygon_dtk._is_hygon_dtk_cached.cache_clear()
    hygon_dtk._dcu_drm_cards.cache_clear()


def _fake_torch_props(gcn_arch_name: str) -> types.ModuleType:
    """Minimal ``torch`` whose device 0 reports ``gcn_arch_name``."""
    torch = types.ModuleType("torch")
    cuda = types.ModuleType("torch.cuda")
    cuda.get_device_properties = lambda index: types.SimpleNamespace(
        gcnArchName=gcn_arch_name
    )
    torch.cuda = cuda
    return torch


@pytest.mark.parametrize(
    ("arch", "expected"),
    [
        ("gfx906", True),
        ("gfx90a", True),
        ("gfx942", True),
        ("gfx1100", True),
        ("GFX906", True),
        # HIP target-feature suffixes must be tolerated.
        ("gfx906:sramecc+:xnack-", True),
        # The bogus value a DTK amdsmi shim can return.
        ("gfx18446744073709551615", False),
        ("gfx", False),
        ("", False),
        (None, False),
        ("not-an-arch", False),
    ],
)
def test_is_valid_gcn_arch(arch, expected):
    assert hygon_dtk.is_valid_gcn_arch(arch) is expected


def test_arch_from_gfx_target_version():
    # Verified on 4x Hygon DCU (device ids 0x54B7 / 0x55B7).
    assert hygon_dtk.arch_from_gfx_target_version(90106) == "gfx906"
    # Unknown packings must not be guessed.
    assert hygon_dtk.arch_from_gfx_target_version(12345) is None


@pytest.mark.parametrize(
    ("arch", "expected"),
    [
        # Hygon SMI/driver naming for the Z100: remapped to the ISA name.
        ("gfx916", "gfx906"),
        ("gfx916:sramecc+:xnack-", "gfx906"),
        (" gfx916 ", "gfx906"),
        ("GFX916", "gfx906"),
        ("GFX916:sramecc+", "gfx906"),
        ("gfx906:sramecc+:xnack-", "gfx906"),
        # The lowercase "a" of gfx90a must survive normalization.
        ("gfx90a", "gfx90a"),
        ("GFX90A", "gfx90a"),
        ("gfx942", "gfx942"),
        ("gfx1100", "gfx1100"),
        # Unusable values are rejected, not guessed.
        ("gfx18446744073709551615", None),
        ("gfx", None),
        ("", None),
        ("   ", None),
        (None, None),
        ("not-an-arch", None),
    ],
)
def test_normalize_gcn_arch(arch, expected):
    assert hygon_dtk.normalize_gcn_arch(arch) == expected


def test_gcn_arch_from_kfd_still_reports_gfx906(tmp_path, monkeypatch):
    """The KFD path feeds the same normalization (table already says gfx906)."""
    node = tmp_path / "nodes" / "0"
    node.mkdir(parents=True)
    (node / "properties").write_text("simd_count 240\ngfx_target_version 90106\n")
    monkeypatch.setattr(
        hygon_dtk, "_KFD_TOPOLOGY_GLOB", str(tmp_path / "nodes" / "*" / "properties")
    )
    assert hygon_dtk.gcn_arch_from_kfd(0) == "gfx906"


def test_is_valid_gcn_arch_accepts_gfx916_before_alias():
    """The regex is purely syntactic; remapping is normalize_gcn_arch's job."""
    assert hygon_dtk.is_valid_gcn_arch("gfx916") is True
    assert hygon_dtk.normalize_gcn_arch("gfx916") == "gfx906"


def test_gcn_arch_from_torch_normalizes_gfx916(monkeypatch):
    """DTK torch may report gfx916; the DTK helper must canonicalize it."""
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch_props("gfx916:sramecc+:xnack-"),
    )
    assert hygon_dtk.gcn_arch_from_torch(0) == "gfx906"


def test_gcn_arch_override_reports_canonical_arch(monkeypatch):
    monkeypatch.setenv("VLLM_DCU_GCN_ARCH", "gfx916")
    assert hygon_dtk.gcn_arch() == "gfx906"


def test_read_kfd_topology_skips_cpu_node(tmp_path, monkeypatch):
    """KFD node 0 is the CPU and also reports gfx_target_version 0."""
    nodes = tmp_path / "nodes"
    (nodes / "0").mkdir(parents=True)
    (nodes / "0" / "properties").write_text(
        "cpu_cores_count 12\nsimd_count 0\ngfx_target_version 0\n"
    )
    for index in (1, 2, 3, 4):
        node = nodes / str(index)
        node.mkdir()
        (node / "properties").write_text("simd_count 240\ngfx_target_version 90106\n")

    monkeypatch.setattr(
        hygon_dtk, "_KFD_TOPOLOGY_GLOB", str(nodes / "*" / "properties")
    )
    assert hygon_dtk._read_kfd_topology() == [90106] * 4
    assert hygon_dtk.gcn_arch_from_kfd(0) == "gfx906"
    assert hygon_dtk.gcn_arch_from_kfd(3) == "gfx906"
    assert hygon_dtk.gcn_arch_from_kfd(4) is None


def test_is_hygon_dtk_from_env(monkeypatch):
    monkeypatch.delenv("DTKROOT", raising=False)
    monkeypatch.delenv("HYHAL_PATH", raising=False)
    monkeypatch.setattr(
        hygon_dtk, "_DTK_INFO_VERSION", Path("/nonexistent/dtk-version")
    )
    monkeypatch.setattr(hygon_dtk.glob, "glob", lambda pattern: [])

    assert hygon_dtk.is_hygon_dtk() is False

    hygon_dtk._is_hygon_dtk_cached.cache_clear()
    monkeypatch.setenv("DTKROOT", "/opt/dtk")
    assert hygon_dtk.is_hygon_dtk() is True


def test_gcn_arch_prefers_explicit_override(monkeypatch):
    monkeypatch.setenv("VLLM_DCU_GCN_ARCH", "gfx906:sramecc+")
    monkeypatch.setattr(hygon_dtk, "gcn_arch_from_torch", lambda index=0: "gfx928")
    assert hygon_dtk.gcn_arch() == "gfx906"


def test_gcn_arch_rejects_invalid_override(monkeypatch):
    monkeypatch.setenv("VLLM_DCU_GCN_ARCH", "gfx18446744073709551615")
    with pytest.raises(ValueError, match="not a valid GCN arch"):
        hygon_dtk.gcn_arch()


def test_gcn_arch_falls_back_from_torch_to_kfd(monkeypatch):
    monkeypatch.delenv("VLLM_DCU_GCN_ARCH", raising=False)
    calls = []

    def fake_torch(index=0):
        calls.append("torch")
        return None

    def fake_kfd(index=0):
        calls.append("kfd")
        return "gfx906"

    monkeypatch.setattr(hygon_dtk, "gcn_arch_from_torch", fake_torch)
    monkeypatch.setattr(hygon_dtk, "gcn_arch_from_kfd", fake_kfd)
    assert hygon_dtk.gcn_arch() == "gfx906"
    assert calls == ["torch", "kfd"]


def _make_fake_sysfs(root: Path) -> None:
    """card0 = non-DCU; card1/card2 = DCU with out-of-order BDFs."""
    drm = root / "drm"
    drm.mkdir(parents=True)

    def card(
        name: str,
        vendor: str,
        vram: str | None,
        unique_id: str | None,
        bdf: str | None,
    ):
        card_dir = drm / name
        card_dir.mkdir()
        if bdf is None:
            device = card_dir / "device"
            device.mkdir()
        else:
            target = root / bdf
            target.mkdir()
            device = card_dir / "device"
            device.symlink_to(target, target_is_directory=True)
        (device / "vendor").write_text(vendor)
        if vram is not None:
            (device / "mem_info_vram_total").write_text(vram)
        if unique_id is not None:
            (device / "unique_id").write_text(unique_id)

    # An NVIDIA board, plus a DRM connector node that must be ignored.
    card("card0", "0x10de", None, None, None)
    (drm / "card0-DP-1").mkdir()
    # DCU boards: bus order 2e:00 must come before 38:00.
    card("card1", "0x1d94", "34342961152", "6f09645be0141861", "0000:38:00.0")
    card("card2", "0x1d94", "34342961152", "6f09645d36544061", "0000:2e:00.0")


def test_dcu_drm_cards_filters_and_orders_by_bdf(tmp_path, monkeypatch):
    _make_fake_sysfs(tmp_path)
    monkeypatch.setattr(
        hygon_dtk, "_DRM_CARD_GLOB", str(tmp_path / "drm" / "card[0-9]*")
    )

    cards = hygon_dtk._dcu_drm_cards()
    assert [Path(c).parent.name for c in cards] == ["card2", "card1"]
    # The non-DCU card0 and the card0-DP-1 connector are excluded.
    assert all("card0" != Path(c).parent.name for c in cards)


def test_dcu_sysfs_reads_follow_bdf_order(tmp_path, monkeypatch):
    _make_fake_sysfs(tmp_path)
    monkeypatch.setattr(
        hygon_dtk, "_DRM_CARD_GLOB", str(tmp_path / "drm" / "card[0-9]*")
    )

    assert hygon_dtk.dcu_device_memory_total_bytes(0) == 34342961152
    assert hygon_dtk.dcu_device_unique_id(0) == "6f09645d36544061"
    assert hygon_dtk.dcu_device_unique_id(1) == "6f09645be0141861"
    # Out-of-range must be reported as unknown, not as an exception.
    assert hygon_dtk.dcu_device_memory_total_bytes(9) is None
    assert hygon_dtk.dcu_device_unique_id(9) is None


def test_rocminfo_candidates_include_dtk(monkeypatch):
    monkeypatch.delenv("DTKROOT", raising=False)
    candidates = hygon_dtk.rocminfo_candidates()
    assert "/opt/rocm/bin/rocminfo" in candidates
    assert "/opt/dtk/bin/rocminfo" in candidates
    assert len(candidates) == len(set(candidates))


@pytest.mark.parametrize(
    ("policy", "probe_result", "expected"),
    [
        ("1", False, True),
        ("on", False, True),
        ("force", False, True),
        ("0", True, False),
        ("off", True, False),
        ("auto", True, True),
        ("auto", False, False),
        ("", True, False),
        ("", False, False),
    ],
)
def test_gfx906_bf16_enabled(monkeypatch, policy, probe_result, expected):
    monkeypatch.setenv("VLLM_DCU_BF16", policy)
    monkeypatch.setattr(hygon_dtk, "dcu_bf16_probe", lambda: probe_result)
    assert hygon_dtk.gfx906_bf16_enabled() is expected


def test_gfx906_supported_dtypes_keep_fp16_first(monkeypatch):
    torch = pytest.importorskip("torch")
    rocm = pytest.importorskip("vllm.platforms.rocm")
    monkeypatch.setattr(rocm, "_ON_GFX906", True)

    monkeypatch.setattr(rocm, "gfx906_bf16_enabled", lambda: False)
    assert rocm._rocm_supported_dtypes() == [torch.float16, torch.float32]

    monkeypatch.setattr(rocm, "gfx906_bf16_enabled", lambda: True)
    # bf16 is offered, but fp16 stays first so --dtype auto is unchanged.
    assert rocm._rocm_supported_dtypes() == [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]


def test_gfx906_bf16_default_off(monkeypatch):
    """With VLLM_DCU_BF16 unset the default policy is off, not the probe."""
    monkeypatch.delenv("VLLM_DCU_BF16", raising=False)
    monkeypatch.setattr(hygon_dtk, "dcu_bf16_probe", lambda: True)
    assert hygon_dtk.gfx906_bf16_enabled() is False


def test_bf16_policy_warns_on_unrecognized_value(monkeypatch):
    """An unrecognized VLLM_DCU_BF16 value falls back to 'off' with a warning."""
    monkeypatch.setenv("VLLM_DCU_BF16", "disable")
    assert hygon_dtk.bf16_policy() == "off"
    # Empty string (unset) falls back to 'off' silently.
    monkeypatch.setenv("VLLM_DCU_BF16", "")
    assert hygon_dtk.bf16_policy() == "off"


def test_use_custom_allreduce_gfx906_opt_in(monkeypatch):
    """gfx906 joins the custom allreduce only via VLLM_DCU_CUSTOM_AR=1."""
    rocm = pytest.importorskip("vllm.platforms.rocm")
    monkeypatch.setattr(rocm, "_GCN_ARCH", "gfx906")
    monkeypatch.setattr(rocm, "_ON_GFX906", True)

    monkeypatch.setattr(rocm, "dcu_custom_ar_enabled", lambda: False)
    assert rocm.RocmPlatform.use_custom_allreduce() is False
    monkeypatch.setattr(rocm, "dcu_custom_ar_enabled", lambda: True)
    assert rocm.RocmPlatform.use_custom_allreduce() is True

    # MI300-class arches keep their unconditional opt-in.
    monkeypatch.setattr(rocm, "_GCN_ARCH", "gfx942")
    monkeypatch.setattr(rocm, "_ON_GFX906", False)
    monkeypatch.setattr(rocm, "dcu_custom_ar_enabled", lambda: False)
    assert rocm.RocmPlatform.use_custom_allreduce() is True


def test_should_custom_ar_dcu_pcie_branch(monkeypatch):
    """PCIe-only multi-rank gate: forced algo, world size, and opt-in flag."""
    torch = pytest.importorskip("torch")
    ca = pytest.importorskip(
        "vllm.distributed.device_communicators.custom_all_reduce"
    )

    def make(world_size):
        # Bypass __init__: only the attributes should_custom_ar reads matter.
        obj = ca.CustomAllreduce.__new__(ca.CustomAllreduce)
        obj.disabled = False
        obj.world_size = world_size
        obj.max_size = 8192 * 1024
        obj.batch_invariant = False
        obj.fully_connected = False
        return obj

    # 2048 B fp16, 16-byte aligned, so the upstream dtype/alignment gates pass.
    x = torch.zeros(1024, dtype=torch.float16)

    monkeypatch.setattr(ca, "dcu_custom_ar_enabled", lambda: True)

    # No forced algo: the C++ dispatch would launch nothing (garbage out),
    # so the RCCL all-reduce is kept.
    monkeypatch.setattr(ca, "_dcu_custom_ar_algo", lambda: "")
    assert make(4).should_custom_ar(x) is False

    # Forced algo: custom kernels are used.
    monkeypatch.setattr(ca, "_dcu_custom_ar_algo", lambda: "2stage")
    assert make(4).should_custom_ar(x) is True

    # World sizes 3/5/7 throw in the C++ dispatch and must never reach it.
    assert make(3).should_custom_ar(x) is False

    # The algo value must match the C++ strcmp values exactly (no case fold).
    monkeypatch.setattr(ca, "_dcu_custom_ar_algo", lambda: "2Stage")
    assert make(4).should_custom_ar(x) is False

    # Without the opt-in, the upstream gate (False for >2 PCIe GPUs) holds.
    monkeypatch.setattr(ca, "dcu_custom_ar_enabled", lambda: False)
    monkeypatch.setattr(ca, "_dcu_custom_ar_algo", lambda: "2stage")
    assert make(4).should_custom_ar(x) is False


def test_gfx906_coerce_model_dtype(monkeypatch):
    """gfx906_coerce_model_dtype: bf16 -> fp16 unless the user opts in."""
    torch = pytest.importorskip("torch")
    rocm = pytest.importorskip("vllm.platforms.rocm")
    monkeypatch.setattr(rocm, "_ON_GFX906", True)

    # Default (bf16 off): bfloat16 coerces to float16, the rest passes through.
    monkeypatch.setattr(rocm, "gfx906_bf16_enabled", lambda: False)
    assert rocm.gfx906_coerce_model_dtype(torch.bfloat16) is torch.float16
    assert rocm.gfx906_coerce_model_dtype(torch.float16) is torch.float16
    assert rocm.gfx906_coerce_model_dtype(torch.float32) is torch.float32

    # Opt-in (VLLM_DCU_BF16=1 / probe pass): bfloat16 stays bfloat16.
    monkeypatch.setattr(rocm, "gfx906_bf16_enabled", lambda: True)
    assert rocm.gfx906_coerce_model_dtype(torch.bfloat16) is torch.bfloat16

    # Off gfx906: no coercion, other arches keep their bf16 support.
    monkeypatch.setattr(rocm, "_ON_GFX906", False)
    monkeypatch.setattr(rocm, "gfx906_bf16_enabled", lambda: False)
    assert rocm.gfx906_coerce_model_dtype(torch.bfloat16) is torch.bfloat16


def test_get_gcn_arch_normalizes_amdsmi_hygon_alias(monkeypatch):
    """An amdsmi shim reporting gfx916 must yield gfx906, not (9, 1)."""
    rocm = pytest.importorskip("vllm.platforms.rocm")
    monkeypatch.setattr(rocm, "is_hygon_dtk", lambda: False)
    monkeypatch.setattr(rocm, "_HAS_AMDSMI", False)
    monkeypatch.setattr(
        rocm, "_query_gcn_arch_from_amdsmi", lambda: "gfx916:sramecc+"
    )
    assert rocm._get_gcn_arch() == "gfx906"


def test_get_gcn_arch_rejects_bogus_amdsmi_and_uses_torch(monkeypatch):
    rocm = pytest.importorskip("vllm.platforms.rocm")
    monkeypatch.setattr(rocm, "is_hygon_dtk", lambda: False)
    monkeypatch.setattr(rocm, "_HAS_AMDSMI", False)
    monkeypatch.setattr(
        rocm, "_query_gcn_arch_from_amdsmi", lambda: "gfx18446744073709551615"
    )
    monkeypatch.setattr(
        rocm.torch.cuda,
        "get_device_properties",
        lambda index: types.SimpleNamespace(gcnArchName="gfx916"),
    )
    # The bogus amdsmi value is dropped, torch's gfx916 is canonicalized, and
    # the resulting capability is (9, 0) instead of the bogus (9, 1).
    assert rocm._get_gcn_arch() == "gfx906"
    assert rocm._capability_from_gcn_arch("gfx906") == (9, 0)
