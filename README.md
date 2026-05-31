# vllm-quant-kernels

Out-of-tree (OOT) vLLM plugin that replaces the `lm_head` matmul with quantized Triton kernels optimized for NVIDIA Jetson Thor (sm_110a / Blackwell).

The `lm_head` projection (hidden states → vocab logits) is typically the largest single matmul in the decode step for large-vocabulary models. For Qwen3.5-122B this is a `248320 × 3072` weight matrix — quantizing it to INT8 saves ~1.4 GB of VRAM and reduces per-token memory bandwidth by 2×.

## Kernels

Four kernel paths are available, selected by environment variable:

| Env var | Kernel | Devices | Description |
|---------|--------|---------|-------------|
| `VLLM_USE_MXFP8_LMHEAD=1` | `mxfp8_gemm` | Thor (sm_110a) | **MXFP8** — weights and activations both stored as FP8 e4m3 with per-32-element e8m0 scales (OCP MX format). Uses `tl.dot_scaled` → native `tcgen05.mma` MX format hardware. |
| `VLLM_USE_W8A8_LMHEAD=1` | `int8_w8a8_gemm` | Thor (sm_110a) | **W8A8** — weights quantized per-row INT8, activations quantized per-tensor INT8 at runtime. Uses `tl.dot(int8, int8, out_dtype=int32)` → native tcgen05 INT8 tensor cores. |
| `VLLM_USE_INT8_LMHEAD=1` | `int8_gemm` / `int8_gmv` | Thor + others | **W8A16** — weights INT8, activations FP16. GEMM via tcgen05 on Thor; GEMV fused kernel on other devices. |
| `VLLM_USE_FP8_LMHEAD=1` | `fp8_gemm` | Thor (sm_110a) | **W8A16-FP8** — weights quantized to FP8 e4m3 with per-row FP16 scale, activations cast to FP8 at runtime. Native FP8 tensor cores. |

Priority order when multiple vars are set: MXFP8 > FP8 > W8A8 > INT8.

MXFP8, FP8, and W8A8 require Thor (sm_110a). If enabled on a non-Thor device, the plugin prints a warning and falls back to W8A16 INT8.

### Batch=1 fast path (W8A16 INT8 only)

For single-token decode (`batch=1`), the W8A16 INT8 path uses `int8_gmv` — a bandwidth-optimized GEMV that loads the weight tile once and reuses it across fused batch elements. This path is not compute-bound and does not benefit from tensor core INT8. The other modes (MXFP8, FP8, W8A8) use their 2D GEMM kernel for all batch sizes.

## Accuracy (Qwen3.5-122B shape: 248320 × 3072, random Gaussian weights)

Measured on Thor vs. FP16 cuBLAS reference:

| Mode | batch | rel error | cos sim |
|------|-------|-----------|---------|
| W8A8 | 1 | 1.23% | 0.999924 |
| W8A8 | 4 | 1.30% | 0.999915 |
| W8A16 INT8 | 1 | 0.85% | 0.999964 |
| W8A16 INT8 | 4 | 0.85% | 0.999964 |
| MXFP8 | 1 | 3.74% | 0.999301 |
| MXFP8 | 4 | 3.78% | 0.999288 |
| FP8 | 1 | 3.76% | 0.999291 |
| FP8 | 4 | 3.76% | 0.999294 |

**Note on MXFP8 vs FP8 accuracy:** On random Gaussian weights, MXFP8 gives essentially the same accuracy as plain FP8. The e8m0 group scale rounds up to the next power of 2 (~30% overhead per group) which offsets the finer 32-element group granularity. MXFP8 typically wins on weight distributions with highly heterogeneous within-row dynamic range (e.g., attention QKV with channel-wise outliers); for `lm_head` projections it is roughly equivalent. Use MXFP8 when you want hardware-native MX format support (single `tcgen05.mma` instruction with built-in scaling) or when working with weight distributions that benefit from per-group scales.

## Requirements

- Python 3.10+
- PyTorch 2.4+
- Triton 3.0+
- vLLM (any version with `PluggableLayer` OOT support)
- CUDA 13.0+ and sm_110a for W8A8, FP8, and MXFP8 paths

## Installation

```bash
pip install -e .
```

The package registers itself as a vLLM plugin via `project.entry-points`:

```
vllm.general_plugins.quant_kernels = vllm_quant_kernels:register
```

vLLM loads all registered plugins on startup. No code changes to vLLM are required.

## Usage

Set one environment variable before launching vLLM:

```bash
# MXFP8 — uses tcgen05.mma MX format on Thor (sm_110a)
export VLLM_USE_MXFP8_LMHEAD=1

# W8A8 INT8×INT8 — best throughput on Thor for INT8
export VLLM_USE_W8A8_LMHEAD=1

# W8A16 INT8 weight-only — works on any device, highest accuracy
export VLLM_USE_INT8_LMHEAD=1

# FP8 — per-row scale, simplest FP8 path
export VLLM_USE_FP8_LMHEAD=1
```

On startup, the plugin prints which path is active:

```
[vllm-quant-kernels] Registered QuantizedLogitsProcessor as OOT replacement for LogitsProcessor (dtype=mxfp8).
[vllm-quant-kernels] lm_head quantized: fp16 248320×3072 → mxfp8 e4m3 (saved 1455 MB, scales 22 MB, kernel=thor/mxfp8)
```

## Conditions for quantization

The plugin only quantizes `lm_head` when all of the following are true:

- The weight dtype is `float16` or `bfloat16`
- The vocabulary size exceeds 100,000 rows
- The appropriate env var is set to a truthy value (`1`, `true`, `yes`, `on`)

Models with smaller vocabularies or already-quantized weights (INT8, GPTQ, AWQ, etc.) are passed through unchanged to the standard vLLM path.

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

Tests require a CUDA GPU. W8A8 and FP8 precision tests automatically skip on non-Thor devices.

## Implementation notes

**Kernel factory pattern.** Each Thor kernel is constructed lazily via a factory function (`_make_thor_w8a8_gemm_kernel()` etc.) called once at model-load time. This defers Triton JIT compilation and autotuning to initialization rather than the first request.

**Autotune cache.** Triton caches autotuned configs to `~/.triton/cache/`. On first run, autotuning over 30 configs × 500 reps takes several minutes for the Qwen3.5-122B shape. Subsequent runs use the cached result instantly.

**W8A8 activation quantization.** Activations are quantized per-tensor (single scale = `abs_max / 127`) immediately before the kernel launch in `_w8a8_forward()`. The INT32 accumulator is dequantized at the end of the kernel: `out = acc * w_scale_row * x_scale_scalar`.

**tie_word_embeddings safety.** The original FP16 weights are intentionally not freed after quantization. When `tie_word_embeddings=True`, `lm_head.weight` and `embed_tokens.weight` share the same tensor; zeroing it would corrupt the embedding table.

**MX format details.** The MXFP8 path uses `tl.dot_scaled` with `e4m3` operand format and `e8m0` scales (uint8 storage, decoded as `2^(value - 127)`). Group size is 32 elements per the OCP MX specification. On sm_110a, this maps to `tcgen05.mma` with `.kind::mxf8f6f4` and `.block32` — a single hardware instruction with built-in per-group dequantization. Scale storage for the Qwen3.5-122B `lm_head` is ~22 MB (`248320 × 96` uint8) versus the original ~730 MB FP16 weight matrix.
