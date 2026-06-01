"""INT8 quantized LogitsProcessor for vLLM."""

import torch
from typing import Optional

from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)

from ._kernels import (
    int8_gmv,
    _make_thor_gemm_kernel,
    _make_thor_fp8_gemm_kernel,
    _make_thor_w8a8_gemm_kernel,
    _make_thor_mxfp8_gemm_kernel,
    _make_thor_mxfp4_gemm_kernel,
    MXFP8_GROUP_SIZE,
    MXFP4_GROUP_SIZE,
)


# E2M1 grid magnitudes and the midpoints between consecutive ones — used by
# _quantize_to_mxfp4 to do round-to-nearest without materializing an
# (R, K, 8) diff tensor.
_E2M1_MIDPOINTS = torch.tensor(
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
    dtype=torch.float32,
)
_E2M1_MAX = 6.0


def _quantize_to_mxfp8(t: torch.Tensor, group_size: int = MXFP8_GROUP_SIZE):
    """Quantize a 2D tensor to MXFP8: e4m3 values + e8m0 per-group scales.

    Args:
        t: (R, K) fp32/fp16/bf16, K must be divisible by group_size.

    Returns:
        (values_uint8_view, scales_uint8) where:
          values_uint8_view: (R, K) uint8 view of float8_e4m3fn data
          scales_uint8:      (R, K // group_size) uint8 (e8m0 encoding,
                             scale = 2^(scales_uint8 - 127))
    """
    R, K = t.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    t_fp32 = t.float()
    groups = t_fp32.view(R, K // group_size, group_size)
    group_max = groups.abs().amax(dim=-1).clamp(min=1e-30)  # (R, K//group_size)

    # e8m0: scale = 2^(uint8 - 127). Solve 2^k * 448 >= group_max → k >= log2(group_max / 448)
    # Use ceil so the largest element fits in [-448, 448].
    k_exp = torch.ceil(torch.log2(group_max / 448.0))
    k_exp = k_exp.clamp(min=-127.0, max=127.0)
    scales_uint8 = (k_exp + 127.0).to(torch.uint8)              # (R, K//group_size)
    scales_fp32 = torch.pow(2.0, k_exp).unsqueeze(-1)           # (R, K//group_size, 1)

    quantized = (groups / scales_fp32).clamp(-448.0, 448.0)
    quantized = quantized.to(torch.float8_e4m3fn).view(R, K)
    return quantized.view(torch.uint8), scales_uint8


def _quantize_to_mxfp4(t: torch.Tensor, group_size: int = MXFP4_GROUP_SIZE):
    """Quantize a 2D tensor to MXFP4: E2M1 values + E8M0 per-group scales.

    Args:
        t: (R, K) fp32/fp16/bf16. K must be divisible by group_size and even.

    Returns:
        (packed_uint8, scales_uint8) where:
          packed_uint8: (R, K // 2) uint8 — two E2M1 nibbles per byte, with
                         the first (even-index) element in the lower 4 bits.
          scales_uint8: (R, K // group_size) uint8, e8m0 encoded
                         (decoded scale = 2 ** (scales_uint8 - 127)).
    """
    R, K = t.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    assert K % 2 == 0, f"K={K} must be even (FP4 packing)"

    t_fp32 = t.detach().to(torch.float32)
    groups = t_fp32.view(R, K // group_size, group_size)
    group_max = groups.abs().amax(dim=-1).clamp(min=1e-30)          # (R, G)

    # Scale chosen so the largest element in each group maps into [-6, 6].
    k_exp = torch.ceil(torch.log2(group_max / _E2M1_MAX))
    k_exp = k_exp.clamp(min=-127.0, max=127.0)
    scales_uint8 = (k_exp + 127.0).to(torch.uint8)                  # (R, G)
    scales_fp32 = torch.pow(2.0, k_exp).unsqueeze(-1)               # (R, G, 1)

    scaled = (groups / scales_fp32).view(R, K)                      # (R, K) in [-6, 6]

    sign_bit = (scaled < 0).to(torch.uint8) << 3                    # (R, K)
    mag = scaled.abs().clamp(max=_E2M1_MAX)

    midpoints = _E2M1_MIDPOINTS.to(mag.device)
    idx = torch.bucketize(mag, midpoints).to(torch.uint8)           # (R, K)  0..7

    nibbles = sign_bit | idx                                        # (R, K) in [0..15]

    # Pack two nibbles per byte, lower bits = first element (even index).
    even = nibbles[:, 0::2]
    odd = nibbles[:, 1::2]
    packed = (odd << 4) | even                                      # (R, K // 2)
    return packed.contiguous(), scales_uint8.contiguous()


@PluggableLayer.register_oot(name="LogitsProcessor")
class QuantizedLogitsProcessor(LogitsProcessor):
    """vLLM LogitsProcessor with quantized lm_head matmul.

    Quantizes lm_head.weight to INT8 at first call, then uses a Triton kernel
    for the matmul. Falls back to the parent implementation for already-quantized
    weights or small vocab models.
    """

    _VOCAB_THRESHOLD = 100_000

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        # Initialize quantization once per lm_head
        if not hasattr(self, "_int8_initialized"):
            self._int8_initialized = True
            self._init_int8(lm_head)

        # If not quantized (already INT8 or small vocab), fall through
        if (not hasattr(lm_head, "_int8_w")
                and not hasattr(lm_head, "_fp8_w")
                and not hasattr(lm_head, "_w8a8_w")
                and not hasattr(lm_head, "_mxfp8_w")
                and not hasattr(lm_head, "_mxfp4_w")):
            return super()._get_logits(hidden_states, lm_head, embedding_bias)

        return self._quantized_forward(hidden_states, lm_head, embedding_bias)

    def _init_int8(self, lm_head: VocabParallelEmbedding) -> None:
        """Quantize lm_head.weight if applicable.

        Dtype selected by env vars (checked in priority order):
          VLLM_USE_MXFP4_LMHEAD=1 — MX FP4 e2m1 weights (4-bit) + per-32-element
                                     e8m0 scales, e4m3 activations. Native
                                     tcgen05.mma.kind::mxf4 MMA. Thor only.
                                     EXPERIMENTAL: lower accuracy than MXFP8
                                     (cos ~0.993 vs 0.9993). Opt-in only.
          VLLM_USE_MXFP8_LMHEAD=1 — MX FP8 e4m3 with per-32-element e8m0 scales,
                                    native tcgen05.mma MX format, Thor only
          VLLM_USE_FP8_LMHEAD=1   — per-row FP8 e4m3, scale = row_max / 448, Thor only
          VLLM_USE_W8A8_LMHEAD=1  — per-row INT8 weights + per-tensor INT8 activations
                                     at runtime, Thor only (falls back to W8A16)
          VLLM_USE_INT8_LMHEAD=1  — per-row INT8 weight-only, scale = row_max / 127
        """
        import os
        w = lm_head.weight.data

        # Only quantize FP16/BF16 weights with large vocab
        if w.dtype not in (torch.bfloat16, torch.float16):
            return
        if w.shape[0] <= self._VOCAB_THRESHOLD:
            return

        import sys as _sys
        cap = torch.cuda.get_device_capability(w.device)
        is_thor = cap[0] >= 10

        def _env_bool(name):
            val = os.environ.get(name, "")
            return val.lower() in ("1", "true", "yes", "on")

        use_mxfp4 = _env_bool("VLLM_USE_MXFP4_LMHEAD")
        use_mxfp8 = _env_bool("VLLM_USE_MXFP8_LMHEAD")
        use_fp8   = _env_bool("VLLM_USE_FP8_LMHEAD")
        use_w8a8  = _env_bool("VLLM_USE_W8A8_LMHEAD")

        if use_mxfp4:
            if not is_thor:
                print(
                    "[vllm-quant-kernels] VLLM_USE_MXFP4_LMHEAD requires Thor (sm_110+),"
                    " falling back to int8",
                    file=_sys.stderr, flush=True,
                )
            elif w.shape[1] % MXFP4_GROUP_SIZE != 0:
                print(
                    f"[vllm-quant-kernels] VLLM_USE_MXFP4_LMHEAD requires K divisible "
                    f"by {MXFP4_GROUP_SIZE} (got K={w.shape[1]}), falling back to int8",
                    file=_sys.stderr, flush=True,
                )
            elif w.shape[1] % 2 != 0:
                print(
                    f"[vllm-quant-kernels] VLLM_USE_MXFP4_LMHEAD requires even K "
                    f"(got K={w.shape[1]}), falling back to int8",
                    file=_sys.stderr, flush=True,
                )
            else:
                self._init_mxfp4(lm_head, w, _sys)
                return

        if use_mxfp8:
            if not is_thor:
                print(
                    "[vllm-quant-kernels] VLLM_USE_MXFP8_LMHEAD requires Thor (sm_110+),"
                    " falling back to int8",
                    file=_sys.stderr, flush=True,
                )
            elif w.shape[1] % MXFP8_GROUP_SIZE != 0:
                print(
                    f"[vllm-quant-kernels] VLLM_USE_MXFP8_LMHEAD requires K divisible "
                    f"by {MXFP8_GROUP_SIZE} (got K={w.shape[1]}), falling back to int8",
                    file=_sys.stderr, flush=True,
                )
            else:
                self._init_mxfp8(lm_head, w, _sys)
                return

        if use_fp8:
            if not is_thor:
                print(
                    "[vllm-quant-kernels] VLLM_USE_FP8_LMHEAD requires Thor (sm_110+),"
                    " falling back to int8",
                    file=_sys.stderr, flush=True,
                )
            else:
                self._init_fp8(lm_head, w, _sys)
                return

        if use_w8a8:
            if not is_thor:
                print(
                    "[vllm-quant-kernels] VLLM_USE_W8A8_LMHEAD requires Thor (sm_110+),"
                    " falling back to w8a16 int8",
                    file=_sys.stderr, flush=True,
                )
            else:
                self._init_w8a8(lm_head, w, _sys)
                return

        self._init_int8_weights(lm_head, w, _sys, is_thor)

    def _init_int8_weights(self, lm_head, w, _sys, is_thor: bool) -> None:
        """Quantize to INT8 and store on lm_head."""
        lm_head._int8_use_thor = is_thor
        scales = w.float().abs().amax(dim=1) / 127.0
        scales = scales.clamp(min=1e-12)
        w_int8 = (w.float() / scales.unsqueeze(1)).round().clamp(-127, 127).to(
            torch.int8
        )

        lm_head._int8_w = w_int8
        lm_head._int8_scales = scales.to(torch.float16)

        # NOTE: We intentionally do NOT free lm_head.weight here.
        # When tie_word_embeddings=True, lm_head.weight is the same tensor as
        # embed_tokens.weight. Zeroing it out would corrupt the embedding table
        # and cause 'weight must be 2-D' errors on the next forward pass.

        saved_mb = w.numel() * w.element_size() // 1024 // 1024
        kernel = "thor/int8" if lm_head._int8_use_thor else "spark/int8"
        print(
            f"[vllm-quant-kernels] lm_head quantized: "
            f"fp16 {w_int8.shape[0]}×{w_int8.shape[1]} → int8 "
            f"(saved {saved_mb} MB, kernel={kernel})",
            file=_sys.stderr,
            flush=True,
        )

        lm_head._int8_block_m = 128
        lm_head._int8_block_k = 256

        if lm_head._int8_use_thor:
            lm_head._int8_gemm_kernel, lm_head._int8_n_bucket = _make_thor_gemm_kernel()

    def _init_w8a8(self, lm_head, w, _sys) -> None:
        """Quantize to INT8 for W8A8 path and store on lm_head.

        Weights are quantized per-row (same as W8A16).
        Activations are quantized per-tensor at runtime in _w8a8_forward().
        Thor only — uses tcgen05.mma INT8 tensor cores via tl.dot(int8, int8).
        """
        scales = w.float().abs().amax(dim=1) / 127.0
        scales = scales.clamp(min=1e-12)
        w_int8 = (w.float() / scales.unsqueeze(1)).round().clamp(-127, 127).to(
            torch.int8
        )

        lm_head._w8a8_w = w_int8
        lm_head._w8a8_scales = scales.to(torch.float32)
        lm_head._w8a8_gemm_kernel, lm_head._w8a8_n_bucket = _make_thor_w8a8_gemm_kernel()

        saved_mb = w.numel() * w.element_size() // 1024 // 1024
        print(
            f"[vllm-quant-kernels] lm_head quantized: "
            f"fp16 {w_int8.shape[0]}×{w_int8.shape[1]} → int8 w8a8 "
            f"(saved {saved_mb} MB, kernel=thor/w8a8)",
            file=_sys.stderr,
            flush=True,
        )

    def _init_fp8(self, lm_head, w, _sys) -> None:
        """Quantize to FP8 e4m3 and store on lm_head. Thor only."""
        scales = w.float().abs().amax(dim=1) / 448.0
        scales = scales.clamp(min=1e-12)
        w_fp8 = (w.float() / scales.unsqueeze(1)).clamp(-448, 448).to(
            torch.float8_e4m3fn
        )

        # Store fp8 weights as int8 view — Triton loads int8 and bitcasts to fp8e4nv
        lm_head._fp8_w = w_fp8.view(torch.int8)
        lm_head._fp8_scales = scales.to(torch.float16)

        saved_mb = w.numel() * w.element_size() // 1024 // 1024
        print(
            f"[vllm-quant-kernels] lm_head quantized: "
            f"fp16 {w_fp8.shape[0]}×{w_fp8.shape[1]} → fp8e4m3 "
            f"(saved {saved_mb} MB, kernel=thor/fp8)",
            file=_sys.stderr,
            flush=True,
        )

        lm_head._fp8_gemm_kernel, lm_head._fp8_n_bucket = _make_thor_fp8_gemm_kernel()

    def _init_mxfp8(self, lm_head, w, _sys) -> None:
        """Quantize to MXFP8 (e4m3 + per-32-element e8m0 scales). Thor only.

        Storage:
          lm_head._mxfp8_w        — (M, K) uint8 view of float8_e4m3fn
          lm_head._mxfp8_w_scales — (M, K//32) uint8, e8m0 encoded
        """
        w_uint8, w_scales = _quantize_to_mxfp8(w)
        lm_head._mxfp8_w = w_uint8
        lm_head._mxfp8_w_scales = w_scales

        saved_mb = w.numel() * w.element_size() // 1024 // 1024
        scale_mb = w_scales.numel() // 1024 // 1024
        print(
            f"[vllm-quant-kernels] lm_head quantized: "
            f"fp16 {w_uint8.shape[0]}×{w_uint8.shape[1]} → mxfp8 e4m3 "
            f"(saved {saved_mb} MB, scales {scale_mb} MB, kernel=thor/mxfp8)",
            file=_sys.stderr,
            flush=True,
        )

        lm_head._mxfp8_gemm_kernel, lm_head._mxfp8_n_bucket = _make_thor_mxfp8_gemm_kernel()

    def _init_mxfp4(self, lm_head, w, _sys) -> None:
        """Quantize to MXFP4 (E2M1 weights + per-32-element E8M0 scales). Thor only.

        Storage:
          lm_head._mxfp4_w        — (M, K // 2) uint8, two E2M1 nibbles per byte
          lm_head._mxfp4_w_scales — (M, K // 32) uint8, e8m0 encoded
          lm_head._mxfp4_K        — original (logical) K, since _mxfp4_w.shape[1] is K/2

        Activations are quantized to MXFP8 (E4M3 + E8M0 scales) at runtime in
        _mxfp4_forward — this is a W4A8 mixed-precision GEMM.
        """
        w_packed, w_scales = _quantize_to_mxfp4(w)
        lm_head._mxfp4_w = w_packed
        lm_head._mxfp4_w_scales = w_scales
        lm_head._mxfp4_K = int(w.shape[1])           # logical K, not packed K/2

        # Bandwidth math vs original fp16:
        #   fp16: M*K*2 B
        #   mxfp4: M*(K/2) B for values + M*(K/32) B for scales = M*K*(1/2 + 1/32) B
        # ⇒ ~70% reduction (vs ~50% for MXFP8).
        orig_mb = w.numel() * w.element_size() // 1024 // 1024
        new_mb = (w_packed.numel() + w_scales.numel()) // 1024 // 1024
        print(
            f"[vllm-quant-kernels] lm_head quantized: "
            f"fp16 {w.shape[0]}×{w.shape[1]} → mxfp4 e2m1 "
            f"({orig_mb} MB → {new_mb} MB, kernel=thor/mxfp4 [EXPERIMENTAL])",
            file=_sys.stderr,
            flush=True,
        )

        lm_head._mxfp4_gemm_kernel, lm_head._mxfp4_n_bucket = _make_thor_mxfp4_gemm_kernel()

    def _quantized_forward(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run quantized forward pass — dispatches to mxfp4, mxfp8, fp8, w8a8, or int8 kernel."""
        if hasattr(lm_head, "_mxfp4_w"):
            return self._mxfp4_forward(hidden_states, lm_head, embedding_bias)
        if hasattr(lm_head, "_mxfp8_w"):
            return self._mxfp8_forward(hidden_states, lm_head, embedding_bias)
        if hasattr(lm_head, "_fp8_w"):
            return self._fp8_forward(hidden_states, lm_head, embedding_bias)
        if hasattr(lm_head, "_w8a8_w"):
            return self._w8a8_forward(hidden_states, lm_head, embedding_bias)
        return self._int8_forward(hidden_states, lm_head, embedding_bias)

    def _fp8_forward(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """FP8 forward — Thor only, uses native fp8 tensor cores."""
        M, K = lm_head._fp8_w.shape
        x = hidden_states.view(-1, K)
        batch = x.shape[0]
        out = torch.empty(batch, M, dtype=torch.float16, device=x.device)

        # Pre-cast activation to fp8e4m3 on the host so the kernel only does a
        # bitcast in the inner loop (no per-iteration .to() cast).
        x_fp8 = x.to(torch.float8_e4m3fn).view(torch.int8)
        xc = x_fp8.contiguous() if not x_fp8.is_contiguous() else x_fp8
        grid = lambda meta: (
            (M     + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
            (batch + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
        )
        lm_head._fp8_gemm_kernel[grid](
            out, lm_head._fp8_w, xc, lm_head._fp8_scales,
            M, batch, K, lm_head._fp8_n_bucket(batch),
            out.stride(1), out.stride(0),
            lm_head._fp8_w.stride(0), lm_head._fp8_w.stride(1),
            xc.stride(0), xc.stride(1),
        )

        logits = out.view(hidden_states.shape[:-1] + (M,))
        if embedding_bias is not None:
            logits = logits + embedding_bias
        return logits

    def _mxfp8_forward(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """MXFP8 forward — Thor only, uses tcgen05.mma with MX format operands.

        Activations are quantized to e4m3 + per-32-element e8m0 scales at runtime.
        """
        M, K = lm_head._mxfp8_w.shape
        x = hidden_states.view(-1, K)
        batch = x.shape[0]
        out = torch.empty(batch, M, dtype=torch.float16, device=x.device)

        # Quantize activations to MXFP8
        x_uint8, x_scales = _quantize_to_mxfp8(x)
        x_uint8 = x_uint8.contiguous() if not x_uint8.is_contiguous() else x_uint8
        x_scales = x_scales.contiguous() if not x_scales.is_contiguous() else x_scales

        grid = lambda meta: (
            (M     + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
            (batch + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
        )
        lm_head._mxfp8_gemm_kernel[grid](
            out,
            lm_head._mxfp8_w,
            x_uint8,
            lm_head._mxfp8_w_scales,
            x_scales,
            M, batch, K, lm_head._mxfp8_n_bucket(batch),
            out.stride(1), out.stride(0),
            lm_head._mxfp8_w.stride(0), lm_head._mxfp8_w.stride(1),
            x_uint8.stride(0), x_uint8.stride(1),
            lm_head._mxfp8_w_scales.stride(0), lm_head._mxfp8_w_scales.stride(1),
            x_scales.stride(0), x_scales.stride(1),
            GROUP_SIZE=MXFP8_GROUP_SIZE,
        )

        logits = out.view(hidden_states.shape[:-1] + (M,))
        if embedding_bias is not None:
            logits = logits + embedding_bias
        return logits

    def _mxfp4_forward(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """MXFP4 forward (W4A8) — Thor only.

        Weights are E2M1 packed (2 nibbles/byte) with per-32 E8M0 scales.
        Activations are quantized to MXFP8 (E4M3 + per-32 E8M0 scales) at runtime.
        tl.dot_scaled lowers to tcgen05.mma.kind::mxf4 — native FP4×FP8 MMA.
        """
        K = lm_head._mxfp4_K                                      # logical K
        M = lm_head._mxfp4_w.shape[0]
        x = hidden_states.view(-1, K)
        batch = x.shape[0]
        out = torch.empty(batch, M, dtype=torch.float16, device=x.device)

        # Quantize activations to MXFP8 (E4M3 + E8M0 group scales).
        x_uint8, x_scales = _quantize_to_mxfp8(x)
        x_uint8 = x_uint8.contiguous() if not x_uint8.is_contiguous() else x_uint8
        x_scales = x_scales.contiguous() if not x_scales.is_contiguous() else x_scales

        grid = lambda meta: (
            (M     + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
            (batch + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
        )
        lm_head._mxfp4_gemm_kernel[grid](
            out,
            lm_head._mxfp4_w,                                      # (M, K//2) uint8
            x_uint8,                                               # (batch, K) uint8 e4m3
            lm_head._mxfp4_w_scales,                               # (M, K//32) uint8
            x_scales,                                              # (batch, K//32) uint8
            M, batch, K, lm_head._mxfp4_n_bucket(batch),
            out.stride(1), out.stride(0),
            lm_head._mxfp4_w.stride(0), lm_head._mxfp4_w.stride(1),
            x_uint8.stride(0), x_uint8.stride(1),
            lm_head._mxfp4_w_scales.stride(0), lm_head._mxfp4_w_scales.stride(1),
            x_scales.stride(0), x_scales.stride(1),
            GROUP_SIZE=MXFP4_GROUP_SIZE,
        )

        logits = out.view(hidden_states.shape[:-1] + (M,))
        if embedding_bias is not None:
            logits = logits + embedding_bias
        return logits

    def _w8a8_forward(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """W8A8 forward — Thor only, uses native INT8×INT8 tcgen05 tensor cores."""
        M, K = lm_head._w8a8_w.shape
        x = hidden_states.view(-1, K)
        batch = x.shape[0]
        out = torch.empty(batch, M, dtype=torch.float16, device=x.device)

        # Quantize activations per-tensor to INT8
        x_fp32 = x.float()
        x_scale = x_fp32.abs().amax() / 127.0
        x_scale = x_scale.clamp(min=1e-12)
        x_int8 = (x_fp32 / x_scale).round().clamp(-127, 127).to(torch.int8)
        xc = x_int8.contiguous() if not x_int8.is_contiguous() else x_int8

        grid = lambda meta: (
            (M     + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
            (batch + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
        )
        lm_head._w8a8_gemm_kernel[grid](
            out, lm_head._w8a8_w, xc, lm_head._w8a8_scales,
            x_scale.item(),
            M, batch, K, lm_head._w8a8_n_bucket(batch),
            out.stride(1), out.stride(0),
            lm_head._w8a8_w.stride(0), lm_head._w8a8_w.stride(1),
            xc.stride(0), xc.stride(1),
        )

        logits = out.view(hidden_states.shape[:-1] + (M,))
        if embedding_bias is not None:
            logits = logits + embedding_bias
        return logits

    def _int8_forward(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """INT8 forward pass with device-specific dispatch."""
        M, K = lm_head._int8_w.shape
        x = hidden_states.view(-1, K)
        batch = x.shape[0]
        out = torch.empty(batch, M, dtype=torch.float16, device=x.device)

        grid = lambda meta: ((M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],)

        if batch == 1:
            # BW-bound, fastest for single token
            int8_gmv[grid](
                out,
                lm_head._int8_w,
                x.to(torch.float16),
                lm_head._int8_scales,
                M,
                K,
                out.stride(0),
                x.stride(0),
                NUM_BATCH=1,
            )
        elif hasattr(lm_head, "_int8_gemm_kernel"):
            # Thor: tcgen05 GEMM (batch >= 2)
            xc = x.contiguous() if not x.is_contiguous() else x
            grid2 = lambda meta: (
                (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
                (batch + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
            )
            lm_head._int8_gemm_kernel[grid2](
                out,
                lm_head._int8_w,
                xc,
                lm_head._int8_scales,
                M,
                batch,
                K,
                lm_head._int8_n_bucket(batch),
                out.stride(1),
                out.stride(0),
                lm_head._int8_w.stride(0),
                lm_head._int8_w.stride(1),
                xc.stride(0),
                xc.stride(1),
            )
        else:
            # Spark: fused batch path for batch <= 4, per-row loop for batch > 4
            if batch <= 4:
                int8_gmv[grid](
                    out,
                    lm_head._int8_w,
                    x.to(torch.float16),
                    lm_head._int8_scales,
                    M,
                    K,
                    out.stride(0),
                    x.stride(0),
                    NUM_BATCH=batch,
                )
            else:
                for b in range(batch):
                    int8_gmv[grid](
                        out[b : b + 1],
                        lm_head._int8_w,
                        x[b : b + 1].to(torch.float16),
                        lm_head._int8_scales,
                        M,
                        K,
                        M,
                        K,
                        NUM_BATCH=1,
                    )

        logits = out.view(hidden_states.shape[:-1] + (M,))

        if embedding_bias is not None:
            logits = logits + embedding_bias

        return logits
