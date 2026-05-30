"""INT8 quantized LogitsProcessor for vLLM."""

import torch
from typing import Optional

from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)

from ._kernels import int8_gmv, _make_thor_gemm_kernel, _make_thor_fp8_gemm_kernel, _make_thor_w8a8_gemm_kernel


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
        if not hasattr(lm_head, "_int8_w") and not hasattr(lm_head, "_fp8_w") and not hasattr(lm_head, "_w8a8_w"):
            return super()._get_logits(hidden_states, lm_head, embedding_bias)

        return self._quantized_forward(hidden_states, lm_head, embedding_bias)

    def _init_int8(self, lm_head: VocabParallelEmbedding) -> None:
        """Quantize lm_head.weight to FP8, W8A8, or W8A16 if applicable.

        Dtype selected by env vars (checked in priority order):
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

        use_fp8  = _env_bool("VLLM_USE_FP8_LMHEAD")
        use_w8a8 = _env_bool("VLLM_USE_W8A8_LMHEAD")

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

    def _quantized_forward(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run quantized forward pass — dispatches to fp8, w8a8, or int8 kernel."""
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

        xc = x.contiguous() if not x.is_contiguous() else x
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
