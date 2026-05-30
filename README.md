# vllm-quant-kernels

Out-of-tree (OOT) vLLM plugin that replaces the `lm_head` matmul with quantized Triton kernels optimized for NVIDIA Jetson Thor (sm_110a / Blackwell).

The `lm_head` projection (hidden states → vocab logits) is typically the largest single matmul in the decode step for large-vocabulary models. For Qwen3.5-122B this is a `248320 × 3072` weight matrix — quantizing it to INT8 saves ~1.4 GB of VRAM and reduces per-token memory bandwidth by 2×.

## Kernels

Three kernel paths are available, selected by environment variable:

| Env var | Kernel | Devices | Description |
|---------|--------|---------|-------------|
| `VLLM_USE_W8A8_LMHEAD=1` | `int8_w8a8_gemm` | Thor (sm_110a) | **W8A8** — weights quantized per-row INT8, activations quantized per-tensor INT8 at runtime. Uses `tl.dot(int8, int8, out_dtype=int32)` → native tcgen05 INT8 tensor cores. |
| `VLLM_USE_INT8_LMHEAD=1` | `int8_gemm` / `int8_gmv` | Thor + others | **W8A16** — weights INT8, activations FP16. GEMM via tcgen05 on Thor; GEMV fused kernel on other devices. |
| `VLLM_USE_FP8_LMHEAD=1` | `fp8_gemm` | Thor (sm_110a) | **W8A16-FP8** — weights quantized to FP8 e4m3, activations cast to FP8 at runtime. Native FP8 tensor cores. |

Priority order when multiple vars are set: FP8 > W8A8 > INT8.

FP8 and W8A8 require Thor (sm_110a). If enabled on a non-Thor device, the plugin prints a warning and falls back to W8A16 INT8.

### Batch=1 fast path (all INT8 modes)

For single-token decode (`batch=1`), all modes use `int8_gmv` — a bandwidth-optimized GEMV that loads the weight tile once and reuses it across fused batch elements. This path is not compute-bound and does not benefit from tensor core INT8.

## Accuracy (Qwen3.5-122B shape: 248320 × 3072)

Measured on Thor vs. FP16 cuBLAS reference:

| Mode | batch | rel error | cos sim |
|------|-------|-----------|---------|
| W8A8 | 1 | 1.23% | 0.999924 |
| W8A8 | 4 | 1.30% | 0.999915 |
| W8A16 INT8 | 1 | 0.85% | 0.999964 |
| W8A16 INT8 | 4 | 0.85% | 0.999964 |
| FP8 | 1 | 3.76% | 0.999291 |
| FP8 | 4 | 3.76% | 0.999294 |

W8A8 adds ~0.4% relative error over W8A16 from the per-tensor activation quantization. FP8 is wider due to the 3-bit mantissa of e4m3.

## Requirements

- Python 3.10+
- PyTorch 2.4+
- Triton 3.0+
- vLLM (any version with `PluggableLayer` OOT support)
- CUDA 13.0+ and sm_110a for W8A8 and FP8 paths

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
# W8A8 INT8×INT8 — recommended for Thor, best throughput
export VLLM_USE_W8A8_LMHEAD=1

# W8A16 INT8 weight-only — works on any device
export VLLM_USE_INT8_LMHEAD=1

# FP8 — highest memory savings, wider accuracy tolerance
export VLLM_USE_FP8_LMHEAD=1
```

On startup, the plugin prints which path is active:

```
[vllm-quant-kernels] Registered QuantizedLogitsProcessor as OOT replacement for LogitsProcessor (dtype=w8a8).
[vllm-quant-kernels] lm_head quantized: fp16 248320×3072 → int8 w8a8 (saved 1455 MB, kernel=thor/w8a8)
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
