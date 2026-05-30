"""Triton kernels for INT8 lm_head quantization."""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_K": 256}, num_warps=8, num_stages=2),
    ],
    key=["M", "K", "NUM_BATCH"],
)
@triton.jit
def int8_gmv(
    out_ptr,
    w_ptr,
    x_ptr,
    s_ptr,
    M,
    K,
    stride_ob,
    stride_xb,
    NUM_BATCH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Batched INT8→FP16 GEMV — single weight tile load, up to 4 batch elements fused.

    Each program block processes all batch elements for a set of output rows.
    Weight tile is loaded once and reused across batch inputs.
    """
    pid_m = tl.program_id(0)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < M

    acc0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for ks in range(0, K, BLOCK_K):
        co = ks + tl.arange(0, BLOCK_K)
        km = co < K

        # Load weight tile once
        w = tl.load(
            w_ptr + rows[:, None] * K + co[None, :],
            mask=rmask[:, None] & km[None, :],
            other=0,
        ).to(tl.float32)

        # Reuse weight tile for each batch element
        x0 = tl.load(x_ptr + 0 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
        acc0 += tl.sum(w * x0[None, :], axis=1)
        if NUM_BATCH > 1:
            x1 = tl.load(x_ptr + 1 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
            acc1 += tl.sum(w * x1[None, :], axis=1)
        if NUM_BATCH > 2:
            x2 = tl.load(x_ptr + 2 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
            acc2 += tl.sum(w * x2[None, :], axis=1)
        if NUM_BATCH > 3:
            x3 = tl.load(x_ptr + 3 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
            acc3 += tl.sum(w * x3[None, :], axis=1)

    # Scale and store
    s = tl.load(s_ptr + rows, mask=rmask, other=1.0).to(tl.float32)
    tl.store(out_ptr + 0 * stride_ob + rows, (acc0 * s).to(tl.float16), mask=rmask)
    if NUM_BATCH > 1:
        tl.store(out_ptr + 1 * stride_ob + rows, (acc1 * s).to(tl.float16), mask=rmask)
    if NUM_BATCH > 2:
        tl.store(out_ptr + 2 * stride_ob + rows, (acc2 * s).to(tl.float16), mask=rmask)
    if NUM_BATCH > 3:
        tl.store(out_ptr + 3 * stride_ob + rows, (acc3 * s).to(tl.float16), mask=rmask)


def _n_bucket(n: int) -> int:
    """Bucket batch size for autotune key selection.

    Bucket boundaries tuned for production batch sizes (max_num_seqs=8, no MTP).
    """
    if n < 4:
        return 0
    if n < 8:
        return 1
    if n < 32:
        return 2
    return 3


# ---------------------------------------------------------------------------
# Thor-specific GEMM kernel (sm_110+)
# ---------------------------------------------------------------------------

def _make_thor_gemm_kernel():
    """Build and return the Thor tcgen05 GEMM kernel.

    Only called on Thor (sm_110+) devices.
    """
    _cfgs = []
    # Small-BN tiles for production (batch 2-8).
    for _BM in [128, 256]:
        for _BN in [16, 32]:
            for _BK in [64, 128]:
                for _nw in [4, 8]:
                    for _ns in [2, 3]:
                        _cfgs.append(
                            triton.Config(
                                {"BLOCK_M": _BM, "BLOCK_N": _BN, "BLOCK_K": _BK},
                                num_warps=_nw,
                                num_stages=_ns,
                            )
                        )
    # Large-BN tiles for high-concurrency / MTP.
    for _BM, _BN in [(128, 128), (128, 256), (256, 128)]:
        for _BK in [64, 128]:
            for _ns in [2, 3]:
                _cfgs.append(
                    triton.Config(
                        {"BLOCK_M": _BM, "BLOCK_N": _BN, "BLOCK_K": _BK},
                        num_warps=8,
                        num_stages=_ns,
                    )
                )

    @triton.autotune(
        configs=_cfgs,
        key=["M", "K", "N_BUCKET"],
        cache_results=True,
        rep=500,
    )
    @triton.jit
    def int8_gemm(
        out_ptr,
        w_ptr,
        x_ptr,
        s_ptr,
        M,
        N,
        K,
        N_BUCKET,
        stride_om,
        stride_on,
        stride_wm,
        stride_wk,
        stride_xn,
        stride_xk,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """INT8→FP16 GEMM with tcgen05 on Thor (sm_110+)."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        rmask = rm < M
        nmask = rn < N

        w_blk = w_ptr + (rm[:, None] * stride_wm + rk[None, :] * stride_wk)
        x_blk = x_ptr + (rn[:, None] * stride_xn + rk[None, :] * stride_xk)
        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            kmask = (k0 + rk) < K
            wi = tl.load(
                w_blk,
                mask=rmask[:, None] & kmask[None, :],
                other=0,
            )
            wf = wi.to(tl.float16)
            x = tl.load(
                x_blk,
                mask=nmask[:, None] & kmask[None, :],
                other=0.0,
            )
            x = x.to(tl.float16)
            acc += tl.dot(x, tl.trans(wf), out_dtype=tl.float32)
            w_blk += BLOCK_K * stride_wk
            x_blk += BLOCK_K * stride_xk

        s = tl.load(s_ptr + rm, mask=rmask, other=1.0).to(tl.float32)
        acc = acc * s[None, :]
        out_blk = out_ptr + rn[:, None] * stride_on + rm[None, :] * stride_om
        tl.store(
            out_blk,
            acc.to(tl.float16),
            mask=nmask[:, None] & rmask[None, :],
        )

    return int8_gemm, _n_bucket


# ---------------------------------------------------------------------------
# Thor-specific FP8 GEMM kernel (sm_110+)
# Uses native float8e4nv tensor cores — no int8→fp16 upcast needed.
# ---------------------------------------------------------------------------

def _make_thor_fp8_gemm_kernel():
    """Build and return the Thor FP8 GEMM kernel.

    Weights are stored as float8e4nv (1 byte/element, same memory as INT8).
    tl.dot operates natively on fp8, hitting tcgen05 FP8 tensor cores.
    Only called on Thor (sm_110+) devices.
    """
    _cfgs = []
    for _BM in [128, 256]:
        for _BN in [16, 32]:
            for _BK in [64, 128]:
                for _nw in [4, 8]:
                    for _ns in [2, 3]:
                        _cfgs.append(
                            triton.Config(
                                {"BLOCK_M": _BM, "BLOCK_N": _BN, "BLOCK_K": _BK},
                                num_warps=_nw,
                                num_stages=_ns,
                            )
                        )
    for _BM, _BN in [(128, 128), (128, 256), (256, 128)]:
        for _BK in [64, 128]:
            for _ns in [2, 3]:
                _cfgs.append(
                    triton.Config(
                        {"BLOCK_M": _BM, "BLOCK_N": _BN, "BLOCK_K": _BK},
                        num_warps=8,
                        num_stages=_ns,
                    )
                )

    @triton.autotune(
        configs=_cfgs,
        key=["M", "K", "N_BUCKET"],
        cache_results=True,
        rep=500,
    )
    @triton.jit
    def fp8_gemm(
        out_ptr,
        w_ptr,
        x_ptr,
        s_ptr,
        M,
        N,
        K,
        N_BUCKET,
        stride_om,
        stride_on,
        stride_wm,
        stride_wk,
        stride_xn,
        stride_xk,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """FP8→FP16 GEMM using native float8e4nv tensor cores on Thor (sm_110+)."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        rmask = rm < M
        nmask = rn < N

        w_blk = w_ptr + (rm[:, None] * stride_wm + rk[None, :] * stride_wk)
        x_blk = x_ptr + (rn[:, None] * stride_xn + rk[None, :] * stride_xk)
        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            kmask = (k0 + rk) < K
            # Load weight stored as int8, bitcast to fp8e4nv (same bits, different type)
            wi = tl.load(
                w_blk,
                mask=rmask[:, None] & kmask[None, :],
                other=0,
            ).to(tl.float8e4nv, bitcast=True)
            # Load activation as fp16, cast to fp8 for the dot
            x = tl.load(
                x_blk,
                mask=nmask[:, None] & kmask[None, :],
                other=0.0,
            ).to(tl.float8e4nv)
            acc += tl.dot(x, tl.trans(wi), out_dtype=tl.float32)
            w_blk += BLOCK_K * stride_wk
            x_blk += BLOCK_K * stride_xk

        s = tl.load(s_ptr + rm, mask=rmask, other=1.0).to(tl.float32)
        acc = acc * s[None, :]
        out_blk = out_ptr + rn[:, None] * stride_on + rm[None, :] * stride_om
        tl.store(
            out_blk,
            acc.to(tl.float16),
            mask=nmask[:, None] & rmask[None, :],
        )

    return fp8_gemm, _n_bucket


# ---------------------------------------------------------------------------
# Thor-specific W8A8 INT8×INT8 GEMM kernel (sm_110+)
# Both weights and activations are INT8; accumulator is INT32.
# Dequantized at the end: out = acc * w_scale[row] * x_scale
# ---------------------------------------------------------------------------

def _make_thor_w8a8_gemm_kernel():
    """Build and return the Thor W8A8 INT8×INT8 GEMM kernel.

    Uses tl.dot(int8, int8, out_dtype=tl.int32) which maps to tcgen05.mma
    with INT8 operands on sm_110a — the native INT8 tensor core path.
    Only called on Thor (sm_110+) devices.
    """
    _cfgs = []
    # Small-BN tiles for production (batch 2-8).
    for _BM in [128, 256]:
        for _BN in [16, 32]:
            for _BK in [64, 128]:
                for _nw in [4, 8]:
                    for _ns in [2, 3]:
                        _cfgs.append(
                            triton.Config(
                                {"BLOCK_M": _BM, "BLOCK_N": _BN, "BLOCK_K": _BK},
                                num_warps=_nw,
                                num_stages=_ns,
                            )
                        )
    # Large-BN tiles for high-concurrency / MTP.
    for _BM, _BN in [(128, 128), (128, 256), (256, 128)]:
        for _BK in [64, 128]:
            for _ns in [2, 3]:
                _cfgs.append(
                    triton.Config(
                        {"BLOCK_M": _BM, "BLOCK_N": _BN, "BLOCK_K": _BK},
                        num_warps=8,
                        num_stages=_ns,
                    )
                )

    @triton.autotune(
        configs=_cfgs,
        key=["M", "K", "N_BUCKET"],
        cache_results=True,
        rep=500,
    )
    @triton.jit
    def int8_w8a8_gemm(
        out_ptr,
        w_ptr,
        x_ptr,
        ws_ptr,
        xs,
        M,
        N,
        K,
        N_BUCKET,
        stride_om,
        stride_on,
        stride_wm,
        stride_wk,
        stride_xn,
        stride_xk,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """INT8×INT8→INT32 GEMM with tcgen05 on Thor (sm_110+).

        w: (M, K) int8, per-row quantized weights
        x: (N, K) int8, per-tensor quantized activations
        ws: (M,) float32, per-row weight dequant scale
        xs: float32 scalar, activation dequant scale
        out: (N, M) float16
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        rmask = rm < M
        nmask = rn < N

        w_blk = w_ptr + (rm[:, None] * stride_wm + rk[None, :] * stride_wk)
        x_blk = x_ptr + (rn[:, None] * stride_xn + rk[None, :] * stride_xk)
        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.int32)

        for k0 in range(0, K, BLOCK_K):
            kmask = (k0 + rk) < K
            wi = tl.load(
                w_blk,
                mask=rmask[:, None] & kmask[None, :],
                other=0,
            )
            xi = tl.load(
                x_blk,
                mask=nmask[:, None] & kmask[None, :],
                other=0,
            )
            acc += tl.dot(xi, tl.trans(wi), out_dtype=tl.int32)
            w_blk += BLOCK_K * stride_wk
            x_blk += BLOCK_K * stride_xk

        ws = tl.load(ws_ptr + rm, mask=rmask, other=1.0).to(tl.float32)
        result = acc.to(tl.float32) * ws[None, :] * xs
        out_blk = out_ptr + rn[:, None] * stride_on + rm[None, :] * stride_om
        tl.store(
            out_blk,
            result.to(tl.float16),
            mask=nmask[:, None] & rmask[None, :],
        )

    return int8_w8a8_gemm, _n_bucket
