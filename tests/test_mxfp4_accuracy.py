"""Phase 1: MXFP4 accuracy bake-off on the real Qwen3.5-122B lm_head.

We quantize the real weight matrix to MXFP4 (E2M1 + per-32 E8M0 group scales),
then run logits = X @ dequant(W).T against an FP16 cuBLAS reference and report
cos-sim, relative-error, and top-K overlap.

This test does NOT need a Triton kernel — it uses a pure-PyTorch dequant
reference. The goal is a GO/NO-GO accuracy decision before writing the kernel.
"""

import os
from pathlib import Path

import pytest
import torch

from vllm_quant_kernels._quant import _quantize_to_mxfp4, MXFP4_GROUP_SIZE


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "qwen3p5_122b_lm_head.safetensors"

MXFP4_MAX = 6.0  # max magnitude representable in E2M1

# E2M1 quantization grid: 8 non-negative magnitudes — used by the dequant
# reference. Note: _quantize_to_mxfp4 lives in src/vllm_quant_kernels/_quant.py
# and uses midpoints (faster); this dequant table is the inverse map.
_E2M1_MAGNITUDES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)


def _is_thor() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 10


def _dequantize_mxfp4(
    packed: torch.Tensor,
    scales_uint8: torch.Tensor,
    group_size: int = MXFP4_GROUP_SIZE,
) -> torch.Tensor:
    """Pure-PyTorch dequant: MXFP4 -> fp32 tensor of shape (R, K).

    Mirrors the OCP MX V1 spec exactly so we have a ground-truth reference
    independent of any Triton kernel.
    """
    R, K_half = packed.shape
    K = K_half * 2
    G = scales_uint8.shape[1]
    assert G == K // group_size

    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    # Interleave: even = low, odd = high
    nibbles = torch.stack([low, high], dim=-1).view(R, K)            # (R, K) uint8

    idx = (nibbles & 0x07).to(torch.long)                            # 0..7
    sign = (nibbles >> 3) & 0x01                                     # 0 or 1
    grid = _E2M1_MAGNITUDES.to(packed.device)
    mag = grid[idx]                                                  # (R, K) fp32
    signed = torch.where(sign.bool(), -mag, mag)                     # (R, K) fp32

    # Scale per group: 2 ** (scales_uint8 - 127)
    k_exp = scales_uint8.to(torch.float32) - 127.0                   # (R, G)
    scale = torch.pow(2.0, k_exp).unsqueeze(-1)                      # (R, G, 1)
    out = (signed.view(R, G, group_size) * scale).view(R, K)
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mxfp4_quant_dequant_error():
    """dequant(quant(x)) should be within ~14% per-element rel-error on Gaussian data.

    The worst-case relative error from rounding to the E2M1 grid is 1/4 (between
    {4, 6}: midpoint 5 maps to 6, giving |5-6|/5 = 20%). On Gaussian data after
    per-group scaling the mean rel-error should be well under that.
    """
    torch.manual_seed(0)
    device = "cuda"
    K = 256
    R = 16
    x = torch.randn(R, K, device=device, dtype=torch.float32) * 0.1

    packed, scales = _quantize_to_mxfp4(x)
    deq = _dequantize_mxfp4(packed, scales)
    cos = _cos_sim(deq, x)
    rel = _rel_err(deq, x)
    assert cos > 0.99, f"MXFP4 dequant cos-sim too low: {cos:.4f}"
    assert rel < 0.14, f"MXFP4 dequant rel-error too high: {rel:.4f}"
    print(f"MXFP4 dequant on Gaussian: cos={cos:.4f}  rel={rel*100:.2f}%")


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _rel_err(approx: torch.Tensor, ref: torch.Tensor) -> float:
    diff = (approx.float() - ref.float()).norm()
    return (diff / ref.float().norm()).item()


def _topk_overlap(approx: torch.Tensor, ref: torch.Tensor, k: int) -> float:
    """Average overlap of top-k indices per row, computed on GPU."""
    ap = approx.topk(k, dim=-1).indices.sort(dim=-1).values
    rf = ref.topk(k, dim=-1).indices.sort(dim=-1).values
    # Per-row intersection size via broadcasted equality
    eq = (ap.unsqueeze(-1) == rf.unsqueeze(-2)).any(dim=-1)         # (B, k) bool
    return (eq.float().sum(dim=-1) / k).mean().item()


def _quantize_weight_chunked(
    w: torch.Tensor, m_chunk: int = 4096
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a large weight to MXFP4 and return (packed, scales, dequant).

    Chunked along the M (rows) axis to keep peak memory bounded.
    """
    M, K = w.shape
    packed = torch.empty((M, K // 2), dtype=torch.uint8, device=w.device)
    scales = torch.empty((M, K // MXFP4_GROUP_SIZE), dtype=torch.uint8, device=w.device)
    deq = torch.empty_like(w, dtype=torch.float16)
    for i in range(0, M, m_chunk):
        j = min(i + m_chunk, M)
        p, s = _quantize_to_mxfp4(w[i:j])
        packed[i:j] = p
        scales[i:j] = s
        deq[i:j] = _dequantize_mxfp4(p, s).to(torch.float16)
    return packed, scales, deq


@pytest.mark.skipif(not FIXTURE_PATH.exists(), reason="Qwen lm_head fixture missing")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mxfp4_accuracy_on_qwen_lm_head():
    """GO/NO-GO accuracy bake-off vs FP16 cuBLAS on the real lm_head.

    W4A16 only — isolates the weight-quantization error. W4A8 will be measured
    later by the actual Triton kernel.
    """
    from safetensors.torch import load_file
    device = "cuda"

    tensors = load_file(str(FIXTURE_PATH), device=device)
    w_fp16 = next(iter(tensors.values())).to(torch.float16)         # (M=248320, K=3072)
    M, K = w_fp16.shape
    print(f"\nlm_head shape: {tuple(w_fp16.shape)}, dtype={w_fp16.dtype}")
    print(f"  abs_max={w_fp16.abs().max().item():.4f}")
    # Full p99.9 via kthvalue (avoids torch.quantile's 16M-element limit).
    flat = w_fp16.float().abs().flatten()
    k = int(round(0.999 * flat.numel()))
    print(f"  p99.9  ={flat.kthvalue(k).values.item():.4f}")
    del flat

    # Quantize weight to MXFP4 in chunks of 4096 rows.
    _packed, _scales, w_deq = _quantize_weight_chunked(w_fp16, m_chunk=4096)
    w_err = (w_deq - w_fp16).abs()
    print(f"  weight dequant max  err = {w_err.max().item():.4f}")
    print(f"  weight dequant mean err = {w_err.mean().item():.5f}")

    # Sweep a few batch sizes against FP16 cuBLAS reference (W4A16).
    torch.manual_seed(42)
    for batch in [1, 4, 32]:
        x_fp16 = torch.randn(batch, K, device=device, dtype=torch.float16) * 0.5
        ref = torch.nn.functional.linear(x_fp16, w_fp16)            # (B, M)
        w4a16 = torch.nn.functional.linear(x_fp16, w_deq)

        cos = _cos_sim(w4a16, ref)
        rel = _rel_err(w4a16, ref)
        t5 = _topk_overlap(w4a16, ref, 5)
        t50 = _topk_overlap(w4a16, ref, 50)
        print(f"  B={batch:3d} W4A16  cos={cos:.6f}  rel={rel*100:5.2f}%  "
              f"top5={t5*100:5.1f}%  top50={t50*100:5.1f}%")

    assert torch.isfinite(w_deq).all()
