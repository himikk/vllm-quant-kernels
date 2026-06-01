"""Phase 2: verify the MXFP4 Triton kernel matches the PyTorch dequant reference.

Builds the kernel via the factory, runs it once on small tensors, then compares
against `linear(dequant_x, dequant_w)` computed in fp16. Tolerance is set by
the inherent FP4 rounding error — we don't expect bit-exact agreement, only
that the kernel output stays within the same fp16 accumulator margin as a
naive dequant-then-matmul reference.
"""

import pytest
import torch

from vllm_quant_kernels._kernels import _make_thor_mxfp4_gemm_kernel
from vllm_quant_kernels._quant import _quantize_to_mxfp8

from test_mxfp4_accuracy import _quantize_to_mxfp4, _dequantize_mxfp4


def _is_thor() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 10


pytestmark = pytest.mark.skipif(not _is_thor(), reason="MXFP4 requires Thor (sm_110+)")


def _run_mxfp4_kernel(
    w_packed: torch.Tensor,    # (M, K//2) uint8
    w_scales: torch.Tensor,    # (M, K//32) uint8
    x_e4m3: torch.Tensor,      # (N, K) uint8 view of E4M3
    x_scales: torch.Tensor,    # (N, K//32) uint8
    M: int, N: int, K: int,
) -> torch.Tensor:
    """Run the MXFP4 kernel and return output (N, M) fp16."""
    mxfp4_gemm, _n_bucket = _make_thor_mxfp4_gemm_kernel()

    out = torch.empty((N, M), dtype=torch.float16, device=w_packed.device)

    def grid(meta):
        import triton
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    mxfp4_gemm[grid](
        out, w_packed, x_e4m3, w_scales, x_scales,
        M, N, K, _n_bucket(N),
        out.stride(1), out.stride(0),                      # stride_om, stride_on
        w_packed.stride(0), w_packed.stride(1),            # stride_wm, stride_wk
        x_e4m3.stride(0), x_e4m3.stride(1),                # stride_xn, stride_xk
        w_scales.stride(0), w_scales.stride(1),            # stride_wsm, stride_wsg
        x_scales.stride(0), x_scales.stride(1),            # stride_xsn, stride_xsg
        GROUP_SIZE=32,
    )
    return out


def _reference_w4a8(
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    x_e4m3_uint8: torch.Tensor,
    x_scales: torch.Tensor,
    M: int, N: int, K: int,
) -> torch.Tensor:
    """PyTorch reference: dequant both operands and run fp32 matmul.

    Mirrors what the Triton kernel computes (modulo fp32 -> fp16 cast at end).
    """
    # Dequant weights to fp32 then fp16
    w_deq = _dequantize_mxfp4(w_packed, w_scales).to(torch.float16)         # (M, K)

    # Dequant activations
    x_e4m3 = x_e4m3_uint8.view(torch.float8_e4m3fn).float()
    x_groups = x_e4m3.view(N, K // 32, 32)
    x_k_exp = x_scales.float() - 127.0
    x_deq = (x_groups * torch.pow(2.0, x_k_exp).unsqueeze(-1)).view(N, K).to(torch.float16)

    return torch.nn.functional.linear(x_deq, w_deq)


def test_mxfp4_kernel_matches_reference():
    """The kernel output must match `linear(dequant_x, dequant_w)` within fp16 tol."""
    torch.manual_seed(0)
    device = "cuda"

    # Use a problem big enough to actually exercise the autotune (multi-tile in M)
    # but small enough to keep the test fast.
    M, N, K = 512, 4, 256
    GROUP_SIZE = 32

    # Random weights in fp16, scaled to typical lm_head magnitude.
    w_fp16 = torch.randn(M, K, device=device, dtype=torch.float16) * 0.05
    w_packed, w_scales = _quantize_to_mxfp4(w_fp16)

    # Random activations, quantize to MXFP8 via existing helper.
    x_fp16 = torch.randn(N, K, device=device, dtype=torch.float16) * 0.5
    x_e4m3_uint8, x_scales = _quantize_to_mxfp8(x_fp16)

    # Kernel output
    out_kernel = _run_mxfp4_kernel(
        w_packed, w_scales, x_e4m3_uint8, x_scales,
        M, N, K,
    )
    torch.cuda.synchronize()

    # Reference
    out_ref = _reference_w4a8(
        w_packed, w_scales, x_e4m3_uint8, x_scales,
        M, N, K,
    )

    assert out_kernel.shape == out_ref.shape == (N, M)
    assert torch.isfinite(out_kernel).all()

    # Compare in fp32 — looking for "close enough"
    diff = (out_kernel.float() - out_ref.float()).abs()
    max_abs = diff.max().item()
    rel = diff.norm() / out_ref.float().norm()

    a = out_kernel.float().flatten()
    b = out_ref.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()

    print(f"\nMXFP4 kernel vs reference:")
    print(f"  max_abs_diff = {max_abs:.6f}")
    print(f"  rel_diff     = {rel.item()*100:.3f}%")
    print(f"  cos_sim      = {cos:.6f}")
    print(f"  out_kernel   range = [{out_kernel.min().item():.3f}, {out_kernel.max().item():.3f}]")
    print(f"  out_ref      range = [{out_ref.min().item():.3f}, {out_ref.max().item():.3f}]")

    # The Triton kernel and the PyTorch reference both implement the same
    # mathematical operation (FP32 accumulation of dequantized FP4×FP8 products
    # cast to fp16 at the end). Any difference is from:
    #   1. tcgen05 hardware vs SW dequant order (negligible — fp32 accum)
    #   2. summation order across the K axis (fp32 is associative enough at K=256)
    # Allow 1% relative + cos > 0.9999.
    assert cos > 0.9999, f"cos-sim too low: {cos}"
    assert rel.item() < 0.01, f"rel diff too high: {rel.item()*100:.3f}%"
