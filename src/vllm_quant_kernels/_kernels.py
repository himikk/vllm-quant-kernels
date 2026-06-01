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
        # Weights are streamed once and never reused within the kernel call —
        # evict_first frees L2 for activations and scales which ARE reused.
        w = tl.load(
            w_ptr + rows[:, None] * K + co[None, :],
            mask=rmask[:, None] & km[None, :],
            other=0,
            eviction_policy="evict_first",
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
# Shared-memory pre-check for autotune configs.
# ---------------------------------------------------------------------------
# Thor sm_110a has 228 KB of dynamic shared memory per block (compute_capabilities.md
# Table 31, cc 10.x/11.0: "Max shared memory per thread block = 227 KB"). Triton
# adds a small static overhead (~16 B per pipeline stage), so we keep a 4 KB
# safety margin. Without this guard, Triton emits one "Autotuning failed with
# out of resource: shared memory" warning per OOM config at model-load time —
# correct behavior but noisy.
_THOR_SMEM_LIMIT_BYTES = 228 * 1024 - 4 * 1024


def _fits_smem(BM: int, BN: int, BK: int, ns: int,
               w_bytes: int, x_bytes: int) -> bool:
    """Return True if the (BM, BN, BK, num_stages) tile fits in Thor SMEM.

    Triton's software pipeline stages each hold one weight tile and one
    activation tile in SMEM. The dominant term is `(BM*BK*w_bytes + BN*BK*x_bytes) * ns`.
    We ignore per-stage barriers / scale tensors (≤ 1 KB) and rely on the
    safety margin baked into `_THOR_SMEM_LIMIT_BYTES`.
    """
    per_stage = BM * BK * w_bytes + BN * BK * x_bytes
    return per_stage * ns <= _THOR_SMEM_LIMIT_BYTES


def _filter_configs(cfgs: list, w_bytes: int, x_bytes: int) -> list:
    """Drop configs whose static SMEM exceeds Thor's per-block limit.

    Triton's autotuner would otherwise try each oversized config, catch the
    `OutOfResource: shared memory` exception, and log a warning. Pre-filtering
    here keeps model-load logs clean and shaves a few config-launch attempts.
    """
    out = []
    for cfg in cfgs:
        BM = cfg.kwargs["BLOCK_M"]
        BN = cfg.kwargs["BLOCK_N"]
        BK = cfg.kwargs["BLOCK_K"]
        ns = cfg.num_stages
        if _fits_smem(BM, BN, BK, ns, w_bytes, x_bytes):
            out.append(cfg)
    return out


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
                    for _ns in [2, 3, 4, 5]:
                        _cfgs.append(
                            triton.Config(
                                {"BLOCK_M": _BM, "BLOCK_N": _BN, "BLOCK_K": _BK},
                                num_warps=_nw,
                                num_stages=_ns,
                            )
                        )
    # Large-BN tiles for high-concurrency / MTP.
    for _BM, _BN in [(128, 128)]:
        for _BK in [128, 256]:
            for _ns in [3, 4, 5, 6]:
                _cfgs.append(
                    triton.Config(
                        {"BLOCK_M": _BM, "BLOCK_N": _BN, "BLOCK_K": _BK},
                        num_warps=8,
                        num_stages=_ns,
                    )
                )

    # int8_gemm: weights are int8 (1 B) but activations are fp16 (2 B).
    _cfgs = _filter_configs(_cfgs, w_bytes=1, x_bytes=2)

    @triton.autotune(
        configs=_cfgs,
        key=["M", "K", "N_BUCKET"],
        cache_results=True,
        rep=100,
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
                eviction_policy="evict_first",
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
                    for _ns in [2, 3, 4, 5]:
                        _cfgs.append(
                            triton.Config(
                                {"BLOCK_M": _BM, "BLOCK_N": _BN, "BLOCK_K": _BK},
                                num_warps=_nw,
                                num_stages=_ns,
                            )
                        )
    for _BM, _BN in [(128, 128)]:
        for _BK in [128, 256]:
            for _ns in [3, 4, 5, 6]:
                _cfgs.append(
                    triton.Config(
                        {"BLOCK_M": _BM, "BLOCK_N": _BN, "BLOCK_K": _BK},
                        num_warps=8,
                        num_stages=_ns,
                    )
                )

    # fp8_gemm: both weights and activations are 1 B (fp8 e4m3 / int8 view).
    _cfgs = _filter_configs(_cfgs, w_bytes=1, x_bytes=1)

    @triton.autotune(
        configs=_cfgs,
        key=["M", "K", "N_BUCKET"],
        cache_results=True,
        rep=100,
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
                eviction_policy="evict_first",
            ).to(tl.float8e4nv, bitcast=True)
            # Load activation (also stored as int8 view of fp8e4nv, pre-cast on host)
            x = tl.load(
                x_blk,
                mask=nmask[:, None] & kmask[None, :],
                other=0,
            ).to(tl.float8e4nv, bitcast=True)
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
                    for _ns in [2, 3, 4, 5]:
                        _cfgs.append(
                            triton.Config(
                                {"BLOCK_M": _BM, "BLOCK_N": _BN, "BLOCK_K": _BK},
                                num_warps=_nw,
                                num_stages=_ns,
                            )
                        )
    # Large-BN tiles for high-concurrency / MTP.
    for _BM, _BN in [(128, 128)]:
        for _BK in [128, 256]:
            for _ns in [3, 4, 5, 6]:
                _cfgs.append(
                    triton.Config(
                        {"BLOCK_M": _BM, "BLOCK_N": _BN, "BLOCK_K": _BK},
                        num_warps=8,
                        num_stages=_ns,
                    )
                )

    # int8_w8a8_gemm: int8 weights + int8 activations, 1 B each.
    _cfgs = _filter_configs(_cfgs, w_bytes=1, x_bytes=1)

    @triton.autotune(
        configs=_cfgs,
        key=["M", "K", "N_BUCKET"],
        cache_results=True,
        rep=100,
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
                eviction_policy="evict_first",
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


# ---------------------------------------------------------------------------
# Thor-specific MXFP8 GEMM kernel (sm_110+)
# Uses tl.dot_scaled with e4m3 inputs and e8m0 per-group scales (group size 32).
# Maps to tcgen05.mma with MX format operands on sm_110a — native hardware path.
# ---------------------------------------------------------------------------

MXFP8_GROUP_SIZE = 32


def _make_thor_mxfp8_gemm_kernel():
    """Build and return the Thor MXFP8 GEMM kernel.

    Weights stored as fp8 e4m3 (1 byte/element), per-32-element group e8m0 scales.
    Activations quantized to fp8 e4m3 + e8m0 scales at runtime.
    Only called on Thor (sm_110+) devices.
    """
    _cfgs = []
    # MXFP8 scales group K by 32 elements — BLOCK_K must be a multiple of 32.
    for _BM in [128, 256]:
        for _BN in [16, 32]:
            for _BK in [64, 128]:
                for _nw in [4, 8]:
                    for _ns in [2, 3, 4, 5]:
                        _cfgs.append(
                            triton.Config(
                                {"BLOCK_M": _BM, "BLOCK_N": _BN, "BLOCK_K": _BK},
                                num_warps=_nw,
                                num_stages=_ns,
                            )
                        )
    for _BM, _BN in [(128, 128)]:
        for _BK in [128, 256]:
            for _ns in [3, 4, 5, 6]:
                _cfgs.append(
                    triton.Config(
                        {"BLOCK_M": _BM, "BLOCK_N": _BN, "BLOCK_K": _BK},
                        num_warps=8,
                        num_stages=_ns,
                    )
                )

    # mxfp8_gemm: fp8 weights + fp8 activations (1 B each). Per-group e8m0
    # scales are tiny (~1/32 the size of the operand tile) — ignore them.
    _cfgs = _filter_configs(_cfgs, w_bytes=1, x_bytes=1)

    @triton.autotune(
        configs=_cfgs,
        key=["M", "K", "N_BUCKET"],
        cache_results=True,
        rep=100,
    )
    @triton.jit
    def mxfp8_gemm(
        out_ptr,
        w_ptr,
        x_ptr,
        ws_ptr,
        xs_ptr,
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
        stride_wsm,
        stride_wsg,
        stride_xsn,
        stride_xsg,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
    ):
        """MXFP8 GEMM with native tcgen05.mma on Thor (sm_110+).

        w:  (M, K) e4m3 stored as uint8 view
        x:  (N, K) e4m3 stored as uint8 view
        ws: (M, K//32) uint8 e8m0 per-group weight scales
        xs: (N, K//32) uint8 e8m0 per-group activation scales
        out: (N, M) float16
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        rg = tl.arange(0, BLOCK_K // GROUP_SIZE)
        rmask = rm < M
        nmask = rn < N

        w_blk = w_ptr + (rm[:, None] * stride_wm + rk[None, :] * stride_wk)
        x_blk = x_ptr + (rn[:, None] * stride_xn + rk[None, :] * stride_xk)
        ws_blk = ws_ptr + (rm[:, None] * stride_wsm + rg[None, :] * stride_wsg)
        xs_blk = xs_ptr + (rn[:, None] * stride_xsn + rg[None, :] * stride_xsg)

        # acc shape is (N, M) to match our output layout
        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

        n_groups_per_tile = BLOCK_K // GROUP_SIZE
        K_groups = K // GROUP_SIZE

        for k0 in range(0, K, BLOCK_K):
            kmask = (k0 + rk) < K
            wi = tl.load(
                w_blk,
                mask=rmask[:, None] & kmask[None, :],
                other=0,
                eviction_policy="evict_first",
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
                eviction_policy="evict_first",
            )
            xs_tile = tl.load(
                xs_blk,
                mask=nmask[:, None] & gmask[None, :],
                other=0,
            )
            # tl.dot_scaled: lhs is activations (N, K), rhs is weights (K, M)
            # rhs_scale shape: [N=M, K//32] — already in [M, K//32] layout
            acc = tl.dot_scaled(
                xi, xs_tile, "e4m3",
                tl.trans(wi), ws_tile, "e4m3",
                acc=acc,
                out_dtype=tl.float32,
            )
            w_blk += BLOCK_K * stride_wk
            x_blk += BLOCK_K * stride_xk
            ws_blk += n_groups_per_tile * stride_wsg
            xs_blk += n_groups_per_tile * stride_xsg

        out_blk = out_ptr + rn[:, None] * stride_on + rm[None, :] * stride_om
        tl.store(
            out_blk,
            acc.to(tl.float16),
            mask=nmask[:, None] & rmask[None, :],
        )

    return mxfp8_gemm, _n_bucket


# ---------------------------------------------------------------------------
# Thor-specific MXFP4 GEMM kernel (sm_110+)
# Weights in E2M1 (4-bit) with per-32 E8M0 group scales; activations in E4M3
# (8-bit) with per-32 E8M0 group scales. tl.dot_scaled lowers to
# tcgen05.mma.kind::mxf4 on sm_110a — native hardware path for FP4.
# This is W4A8: 2 FP4 weight bytes per fp16 weight, activations bandwidth
# negligible for lm_head (M >> N).
# ---------------------------------------------------------------------------

MXFP4_GROUP_SIZE = 32


def _make_thor_mxfp4_gemm_kernel():
    """Build and return the Thor MXFP4 GEMM kernel.

    Weights stored as packed E2M1 (2 elements / byte) along K, with per-32
    E8M0 scales. Activations quantized to E4M3 + E8M0 scales at runtime.
    Only called on Thor (sm_110+) devices.
    """
    _cfgs = []
    # MXFP4 scales group K by 32 elements — BLOCK_K (logical) must be a multiple of 32.
    for _BM in [128, 256]:
        for _BN in [16, 32]:
            for _BK in [64, 128, 256]:
                for _nw in [4, 8]:
                    for _ns in [2, 3, 4, 5]:
                        _cfgs.append(
                            triton.Config(
                                {"BLOCK_M": _BM, "BLOCK_N": _BN, "BLOCK_K": _BK},
                                num_warps=_nw,
                                num_stages=_ns,
                            )
                        )
    for _BM, _BN in [(128, 128)]:
        for _BK in [128, 256]:
            for _ns in [3, 4, 5, 6]:
                _cfgs.append(
                    triton.Config(
                        {"BLOCK_M": _BM, "BLOCK_N": _BN, "BLOCK_K": _BK},
                        num_warps=8,
                        num_stages=_ns,
                    )
                )

    # SMEM cost: weights are FP4 packed (0.5 B / element), activations E4M3 (1 B).
    # The shared _fits_smem helper takes integer w_bytes, so inline the check:
    # per_stage = BM * BK / 2 + BN * BK   bytes
    def _fits_mxfp4(cfg) -> bool:
        BM = cfg.kwargs["BLOCK_M"]
        BN = cfg.kwargs["BLOCK_N"]
        BK = cfg.kwargs["BLOCK_K"]
        ns = cfg.num_stages
        per_stage = (BM * BK) // 2 + BN * BK
        return per_stage * ns <= _THOR_SMEM_LIMIT_BYTES

    _cfgs = [c for c in _cfgs if _fits_mxfp4(c)]

    @triton.autotune(
        configs=_cfgs,
        key=["M", "K", "N_BUCKET"],
        cache_results=True,
        rep=100,
    )
    @triton.jit
    def mxfp4_gemm(
        out_ptr,
        w_ptr,      # (M, K // 2) uint8 — two E2M1 nibbles per byte
        x_ptr,      # (N, K)      uint8 — E4M3 view
        ws_ptr,     # (M, K // 32) uint8 — E8M0 scales for weights
        xs_ptr,     # (N, K // 32) uint8 — E8M0 scales for activations
        M,
        N,
        K,          # logical K (in FP4 elements)
        N_BUCKET,
        stride_om,
        stride_on,
        stride_wm,  # in uint8 (= K // 2)
        stride_wk,  # in uint8 (= 1)
        stride_xn,
        stride_xk,
        stride_wsm,
        stride_wsg,
        stride_xsn,
        stride_xsg,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,  # logical K block size
        GROUP_SIZE: tl.constexpr,
    ):
        """MXFP4 GEMM with native tcgen05.mma.kind::mxf4 on Thor (sm_110+).

        w:  (M, K // 2) uint8 (two E2M1 nibbles per byte, lower bits = first elem)
        x:  (N, K)      uint8 (E4M3 view)
        ws: (M, K // 32) uint8 E8M0 per-group weight scales
        xs: (N, K // 32) uint8 E8M0 per-group activation scales
        out: (N, M) float16
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)              # logical K range — activations
        rk2 = tl.arange(0, BLOCK_K // 2)        # packed K range — weights
        rg = tl.arange(0, BLOCK_K // GROUP_SIZE)
        rmask = rm < M
        nmask = rn < N

        # w storage: (M, K//2) uint8.
        w_blk = w_ptr + (rm[:, None] * stride_wm + rk2[None, :] * stride_wk)
        x_blk = x_ptr + (rn[:, None] * stride_xn + rk[None, :] * stride_xk)
        ws_blk = ws_ptr + (rm[:, None] * stride_wsm + rg[None, :] * stride_wsg)
        xs_blk = xs_ptr + (rn[:, None] * stride_xsn + rg[None, :] * stride_xsg)

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
                eviction_policy="evict_first",
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
                eviction_policy="evict_first",
            )
            xs_tile = tl.load(
                xs_blk,
                mask=nmask[:, None] & gmask[None, :],
                other=0,
            )
            # lhs is activations (N, K) e4m3; rhs is weights (K, M) e2m1 after trans.
            # rhs_scale shape: [N=M, K//32] (per docstring — NOT transposed).
            acc = tl.dot_scaled(
                xi, xs_tile, "e4m3",
                tl.trans(wi), ws_tile, "e2m1",
                acc=acc,
                out_dtype=tl.float32,
            )
            w_blk += (BLOCK_K // 2) * stride_wk
            x_blk += BLOCK_K * stride_xk
            ws_blk += n_groups_per_tile * stride_wsg
            xs_blk += n_groups_per_tile * stride_xsg

        out_blk = out_ptr + rn[:, None] * stride_on + rm[None, :] * stride_om
        tl.store(
            out_blk,
            acc.to(tl.float16),
            mask=nmask[:, None] & rmask[None, :],
        )

    return mxfp4_gemm, _n_bucket
