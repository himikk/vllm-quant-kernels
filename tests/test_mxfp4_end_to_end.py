"""Phase 3: end-to-end smoke test of _quantized_forward in MXFP4 mode.

Constructs a fake `lm_head` namespace (same pattern as bench/bench_speed.py),
calls `proc._init_mxfp4(lm_head)` directly, then runs `_quantized_forward`
on a fp16 hidden-state tensor and compares against the fp16 cuBLAS baseline.

This validates the full wire-up from env-var → quantize → kernel dispatch
without requiring a full vLLM model load.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_quant_kernels._quant import QuantizedLogitsProcessor


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "qwen3p5_122b_lm_head.safetensors"


def _is_thor() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 10


pytestmark = pytest.mark.skipif(not _is_thor(), reason="MXFP4 requires Thor (sm_110+)")


def _make_proc():
    """Build a QuantizedLogitsProcessor without triggering full vLLM init."""
    # Bypass __init__ — only _init_* / _*_forward methods are used.
    proc = QuantizedLogitsProcessor.__new__(QuantizedLogitsProcessor)
    proc.scale = 1.0
    proc.org_vocab_size = None
    proc.soft_cap = None
    proc.logits_as_input = False
    proc.use_all_gather = False
    return proc


def _make_lm_head(w: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(weight=SimpleNamespace(data=w), bias=None)


def test_mxfp4_end_to_end_random_weights():
    """Round-trip a small fake lm_head through _init_mxfp4 + _mxfp4_forward."""
    torch.manual_seed(0)
    device = "cuda"
    M, K = 4096, 256                          # tiny lm_head, K % 32 == 0
    w = torch.randn(M, K, device=device, dtype=torch.float16) * 0.05
    lm_head = _make_lm_head(w)
    proc = _make_proc()

    proc._init_mxfp4(lm_head, w, __import__("sys"))
    assert hasattr(lm_head, "_mxfp4_w")
    assert lm_head._mxfp4_w.shape == (M, K // 2)
    assert lm_head._mxfp4_w_scales.shape == (M, K // 32)
    assert lm_head._mxfp4_K == K

    for batch in [1, 4, 32]:
        x = torch.randn(batch, K, device=device, dtype=torch.float16) * 0.5
        out = proc._mxfp4_forward(x, lm_head, embedding_bias=None)
        assert out.shape == (batch, M)
        assert torch.isfinite(out).all()
        ref = torch.nn.functional.linear(x, w)
        cos = torch.nn.functional.cosine_similarity(
            out.float().flatten(), ref.float().flatten(), dim=0,
        ).item()
        print(f"B={batch:3d}  MXFP4 cos vs fp16-cublas (random weights): {cos:.4f}")
        # Random weights: per-group max ≈ 0.15, so quant error dominates; cos > 0.95
        # is a sanity bar, not an accuracy bar — that's measured by the bake-off.
        assert cos > 0.95, f"MXFP4 cos too low on random weights: {cos:.4f}"


@pytest.mark.skipif(not FIXTURE_PATH.exists(), reason="Qwen lm_head fixture missing")
def test_mxfp4_end_to_end_real_lm_head():
    """End-to-end with the real Qwen3.5-122B lm_head — full plugin path."""
    from safetensors.torch import load_file
    device = "cuda"

    tensors = load_file(str(FIXTURE_PATH), device=device)
    w = next(iter(tensors.values())).to(torch.float16)
    M, K = w.shape

    lm_head = _make_lm_head(w)
    proc = _make_proc()
    proc._init_mxfp4(lm_head, w, __import__("sys"))

    torch.manual_seed(42)
    for batch in [1, 4, 32]:
        x = torch.randn(batch, K, device=device, dtype=torch.float16) * 0.5
        out = proc._mxfp4_forward(x, lm_head, embedding_bias=None)
        assert out.shape == (batch, M)
        assert torch.isfinite(out).all()
        ref = torch.nn.functional.linear(x, w)
        cos = torch.nn.functional.cosine_similarity(
            out.float().flatten(), ref.float().flatten(), dim=0,
        ).item()
        rel = (out.float() - ref.float()).norm() / ref.float().norm()
        # Top-1 agreement
        top1_kernel = out.argmax(dim=-1)
        top1_ref = ref.argmax(dim=-1)
        top1_match = (top1_kernel == top1_ref).float().mean().item()
        print(f"B={batch:3d}  cos={cos:.4f}  rel={rel*100:.2f}%  top1_match={top1_match*100:.0f}%")
        # Sanity bar — accuracy gap was already documented in Phase 1.
        assert cos > 0.99, f"MXFP4 cos too low on real weights: {cos:.4f}"
