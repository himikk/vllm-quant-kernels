"""Phase 0 smoke test: verify Triton's tl.dot_scaled accepts 'e2m1' on Thor.

This is a feasibility gate, not a correctness/accuracy test. It builds a tiny
MXFP4 GEMM kernel (W4A8: e2m1 weights, e4m3 activations) and runs it once.
If it compiles and produces finite outputs, MXFP4 is feasible on this device.
"""

import pytest
import torch
import triton
import triton.language as tl


def _is_thor() -> bool:
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 10


pytestmark = pytest.mark.skipif(not _is_thor(), reason="MXFP4 requires Thor (sm_110+)")


@triton.jit
def _mxfp4_smoke_kernel(
    out_ptr,
    w_ptr,      # (M, K/2) uint8, e2m1 packed
    x_ptr,      # (N, K)   uint8, e4m3
    ws_ptr,     # (M, K/32) uint8, e8m0
    xs_ptr,     # (N, K/32) uint8, e8m0
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,     # logical K (in fp4 elements)
    GROUP_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)        # logical K range, for e4m3 activations
    rk2 = tl.arange(0, BLOCK_K // 2)  # packed K range, for e2m1 weights
    rg = tl.arange(0, BLOCK_K // GROUP_SIZE)

    rmask = rm < M
    nmask = rn < N

    # w storage stride: (M, K/2) uint8 contiguous → stride_m = K/2, stride_k = 1
    w_blk = w_ptr + rm[:, None] * (K // 2) + rk2[None, :]
    # x storage stride: (N, K) uint8 contiguous → stride_n = K, stride_k = 1
    x_blk = x_ptr + rn[:, None] * K + rk[None, :]
    # scale strides: (*, K/32)
    ws_blk = ws_ptr + rm[:, None] * (K // GROUP_SIZE) + rg[None, :]
    xs_blk = xs_ptr + rn[:, None] * (K // GROUP_SIZE) + rg[None, :]

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    n_groups_per_tile = BLOCK_K // GROUP_SIZE
    K_groups = K // GROUP_SIZE
    K_packed = K // 2

    for k0 in range(0, K, BLOCK_K):
        kmask = (k0 + rk) < K
        k2mask = (k0 // 2 + rk2) < K_packed
        wi = tl.load(
            w_blk,
            mask=rmask[:, None] & k2mask[None, :],
            other=0,
        )
        xi = tl.load(
            x_blk,
            mask=nmask[:, None] & kmask[None, :],
            other=0,
        )
        g0 = k0 // GROUP_SIZE
        gmask = (g0 + rg) < K_groups
        ws_tile = tl.load(
            ws_blk,
            mask=rmask[:, None] & gmask[None, :],
            other=0,
        )
        xs_tile = tl.load(
            xs_blk,
            mask=nmask[:, None] & gmask[None, :],
            other=0,
        )
        acc = tl.dot_scaled(
            xi, xs_tile, "e4m3",
            tl.trans(wi), ws_tile, "e2m1",
            acc=acc,
            out_dtype=tl.float32,
        )
        w_blk += (BLOCK_K // 2)
        x_blk += BLOCK_K
        ws_blk += n_groups_per_tile
        xs_blk += n_groups_per_tile

    # Write output (N, M) fp16
    out_blk = out_ptr + rn[:, None] * M + rm[None, :]
    tl.store(
        out_blk,
        acc.to(tl.float16),
        mask=nmask[:, None] & rmask[None, :],
    )


def test_mxfp4_dot_scaled_compiles_and_runs():
    """The kernel must JIT-compile and produce finite outputs."""
    torch.manual_seed(0)
    device = "cuda"

    M, N, K = 128, 32, 128       # small problem, BLOCK_K covers all of K
    GROUP_SIZE = 32

    # Random weights packed as e2m1: each uint8 holds two fp4 values.
    # Use a non-zero bit pattern (don't make them all map to fp4=0).
    w_packed = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=device)
    # Activations as e4m3 (uint8 view): avoid the NaN code (0xFF) and -inf (0x7F).
    x_e4m3 = torch.randint(1, 0x7F, (N, K), dtype=torch.uint8, device=device)
    # Scales: e8m0 around bias 127 (== 2^0 = 1.0).
    ws = torch.full((M, K // GROUP_SIZE), 127, dtype=torch.uint8, device=device)
    xs = torch.full((N, K // GROUP_SIZE), 127, dtype=torch.uint8, device=device)

    out = torch.zeros((N, M), dtype=torch.float16, device=device)

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 32, 128
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    _mxfp4_smoke_kernel[grid](
        out, w_packed, x_e4m3, ws, xs,
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE=GROUP_SIZE,
        num_warps=4, num_stages=2,
    )
    torch.cuda.synchronize()

    assert out.shape == (N, M)
    assert torch.isfinite(out).all(), f"out has non-finite values: {out}"
    assert (out != 0).any(), "output is all zeros — kernel likely no-op"
    print(f"MXFP4 smoke output range: [{out.min().item():.3f}, {out.max().item():.3f}]")


if __name__ == "__main__":
    test_mxfp4_dot_scaled_compiles_and_runs()
    print("PHASE 0 PASS: tl.dot_scaled('e2m1') compiles and runs on this device.")
