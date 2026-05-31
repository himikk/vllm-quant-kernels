"""End-to-end accuracy tests against the real Qwen3.5-122B lm_head tensor.

These tests load the actual `lm_head.weight` extracted from
Intel/Qwen3.5-122B-A10B-int4-AutoRound and compare each quantization mode
against a FP16/BF16 cuBLAS reference. The fixture is 1.5 GB and is not
committed; regenerate it with `tests/fixtures/extract_lm_head.py`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


FIXTURE = Path(__file__).parent / "fixtures" / "qwen3p5_122b_lm_head.safetensors"


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(
        not FIXTURE.exists(),
        reason=(
            f"Fixture not found: {FIXTURE}.\n"
            f"Run tests/fixtures/extract_lm_head.py to generate it."
        ),
    ),
]


def _load_lm_head_weight() -> torch.Tensor:
    """Load the real lm_head weight to CUDA, cast to fp16 for reference."""
    from safetensors import safe_open
    with safe_open(str(FIXTURE), framework="pt", device="cuda") as f:
        w = f.get_tensor("lm_head.weight")
    # Convert bf16 -> fp16 so the cuBLAS reference and the kernel inputs match
    return w.to(torch.float16)


def _make_fake_lm_head(w: torch.Tensor):
    """Wrap a tensor as a minimal lm_head-like object for QuantizedLogitsProcessor."""
    from types import SimpleNamespace

    class _FakeWeight:
        def __init__(self, t: torch.Tensor) -> None:
            self.data = t
            self.dtype = t.dtype
            self.shape = t.shape
            self.device = t.device

    return SimpleNamespace(weight=_FakeWeight(w))


def _activation_for(K: int, batch: int) -> torch.Tensor:
    """Realistic decode-step activation: layer-norm output, mean 0, std ~1."""
    # Seed for reproducibility across runs
    g = torch.Generator(device="cuda").manual_seed(0xC0DE)
    return torch.randn(batch, K, device="cuda", dtype=torch.float16, generator=g)


def _precision_summary(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float, float]:
    """Return (max_abs, mean_abs, rel_err, cos_sim) of out vs ref."""
    diff = (out.float() - ref.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    ref_mag = ref.float().abs().mean().item()
    rel_err = mean_err / (ref_mag + 1e-8)
    cos_sim = F.cosine_similarity(
        out.float().reshape(out.shape[0], -1),
        ref.float().reshape(ref.shape[0], -1),
        dim=1,
    ).mean().item()
    return max_err, mean_err, rel_err, cos_sim


@pytest.fixture(scope="module")
def real_w() -> torch.Tensor:
    """The actual fp16-casted Qwen3.5-122B lm_head weight on CUDA."""
    return _load_lm_head_weight()


@pytest.fixture(scope="module")
def real_w_stats(real_w: torch.Tensor) -> dict:
    """Distribution stats of the real weight — for diagnostic prints."""
    w = real_w
    abs_w = w.float().abs()
    return {
        "shape": tuple(w.shape),
        "dtype": str(w.dtype),
        "abs_mean": abs_w.mean().item(),
        "abs_max": abs_w.max().item(),
        "abs_p999": torch.quantile(abs_w.flatten()[:1_000_000], 0.999).item(),
        "row_max_min": abs_w.amax(dim=1).min().item(),
        "row_max_max": abs_w.amax(dim=1).max().item(),
    }


class TestRealLmHeadStats:
    """Print weight distribution stats once."""

    def test_print_stats(self, real_w_stats: dict) -> None:
        print("\n[real lm_head] " + ", ".join(f"{k}={v}" for k, v in real_w_stats.items()))


class TestRealLmHeadAllModes:
    """For each mode, quantize and run forward pass; compare to fp16 reference."""

    @pytest.fixture(scope="class")
    def reference(self, real_w: torch.Tensor):
        """Pre-compute fp16 reference outputs for several batch sizes."""
        K = real_w.shape[1]
        outs = {}
        for b in (1, 4):
            x = _activation_for(K, b)
            outs[b] = (x, F.linear(x, real_w))
        return outs

    def _run_mode(self, env_var: str, real_w: torch.Tensor, reference) -> list[tuple[int, float, float, float]]:
        """Run all reference batches for a given mode; return per-batch metrics."""
        from vllm_quant_kernels._quant import QuantizedLogitsProcessor

        old = os.environ.get(env_var)
        os.environ[env_var] = "1"
        results = []
        try:
            lm_head = _make_fake_lm_head(real_w)
            processor = QuantizedLogitsProcessor(real_w.shape[0])
            processor._init_int8(lm_head)
            processor._int8_initialized = True

            for batch in (1, 4):
                x, ref = reference[batch]
                out = processor._quantized_forward(x, lm_head, None)
                max_e, mean_e, rel_e, cos = _precision_summary(out, ref)
                print(
                    f"\n[{env_var}] batch={batch}  max={max_e:.4f}  mean={mean_e:.4f}  "
                    f"rel={rel_e * 100:.3f}%  cos_sim={cos:.6f}"
                )
                assert out.shape == ref.shape
                assert not torch.isnan(out).any()
                assert not torch.isinf(out).any()
                results.append((batch, max_e, rel_e, cos))
        finally:
            if old is None:
                os.environ.pop(env_var, None)
            else:
                os.environ[env_var] = old
        return results

    def test_w8a16(self, real_w, reference):
        # W8A16: weights INT8 per-row, activations FP16. Tightest tolerances.
        for batch, max_e, rel_e, cos in self._run_mode(
            "VLLM_USE_INT8_LMHEAD", real_w, reference
        ):
            assert rel_e < 0.015, f"w8a16 batch={batch} rel={rel_e * 100:.3f}%"
            assert cos > 0.9999, f"w8a16 batch={batch} cos={cos:.6f}"

    def test_w8a8(self, real_w, reference):
        cap = torch.cuda.get_device_capability()
        if cap[0] < 10:
            pytest.skip("W8A8 requires Thor (sm_110+)")
        for batch, max_e, rel_e, cos in self._run_mode(
            "VLLM_USE_W8A8_LMHEAD", real_w, reference
        ):
            assert rel_e < 0.04, f"w8a8 batch={batch} rel={rel_e * 100:.3f}%"
            assert cos > 0.999, f"w8a8 batch={batch} cos={cos:.6f}"

    def test_fp8(self, real_w, reference):
        cap = torch.cuda.get_device_capability()
        if cap[0] < 10:
            pytest.skip("FP8 requires Thor (sm_110+)")
        for batch, max_e, rel_e, cos in self._run_mode(
            "VLLM_USE_FP8_LMHEAD", real_w, reference
        ):
            assert rel_e < 0.08, f"fp8 batch={batch} rel={rel_e * 100:.3f}%"
            assert cos > 0.997, f"fp8 batch={batch} cos={cos:.6f}"

    def test_mxfp8(self, real_w, reference):
        cap = torch.cuda.get_device_capability()
        if cap[0] < 10:
            pytest.skip("MXFP8 requires Thor (sm_110+)")
        for batch, max_e, rel_e, cos in self._run_mode(
            "VLLM_USE_MXFP8_LMHEAD", real_w, reference
        ):
            assert rel_e < 0.08, f"mxfp8 batch={batch} rel={rel_e * 100:.3f}%"
            assert cos > 0.997, f"mxfp8 batch={batch} cos={cos:.6f}"
