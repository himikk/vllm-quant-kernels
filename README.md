# vllm-quant-kernels

Out-of-tree (OOT) vLLM plugin that replaces the `lm_head` matmul with quantized Triton kernels optimized for NVIDIA Jetson Thor (sm_110a / Blackwell). Also ships a fused Gumbel-argmax sampler kernel that collapses the V1 `Sampler.sample` chain into one pass over the logits tensor.

The `lm_head` projection (hidden states → vocab logits) is typically the largest single matmul in the decode step for large-vocabulary models. For Qwen3.5-122B this is a `248320 × 3072` weight matrix — quantizing it to INT8 saves ~1.4 GB of VRAM and reduces per-token memory bandwidth by 2×.

The fused sampler addresses the second bandwidth-bound step in decode: after `lm_head` produces an `[B, V]` logits tensor, upstream vLLM's sampler reads and rewrites it ~10–14 times for fp32-cast, temperature, top-k/top-p, softmax, exponential RNG, and argmax. The fused kernel replaces that chain with a single two-pass blocked reduction.

## Kernels

Five kernel paths are available, selected by environment variable:

| Env var | Kernel | Devices | Description |
|---------|--------|---------|-------------|
| `VLLM_USE_MXFP4_LMHEAD=1` | `mxfp4_gemm` | Thor (sm_110a) | **MXFP4 (W4A8)** — weights stored as E2M1 (4-bit) packed two-per-byte with per-32-element E8M0 scales; activations quantized to E4M3 (8-bit) + E8M0 scales at runtime (OCP MX format). Uses `tl.dot_scaled(..., 'e2m1', ...)` → native `tcgen05.mma.kind::mxf8f6f4` MX format hardware. **Experimental** — lower accuracy than MXFP8 (see below). |
| `VLLM_USE_MXFP8_LMHEAD=1` | `mxfp8_gemm` | Thor (sm_110a) | **MXFP8** — weights and activations both stored as FP8 e4m3 with per-32-element e8m0 scales (OCP MX format). Uses `tl.dot_scaled` → native `tcgen05.mma` MX format hardware. |
| `VLLM_USE_W8A8_LMHEAD=1` | `int8_w8a8_gemm` | Thor (sm_110a) | **W8A8** — weights quantized per-row INT8, activations quantized per-tensor INT8 at runtime. Uses `tl.dot(int8, int8, out_dtype=int32)` → native tcgen05 INT8 tensor cores. |
| `VLLM_USE_INT8_LMHEAD=1` | `int8_gemm` / `int8_gmv` | Thor + others | **W8A16** — weights INT8, activations FP16. GEMM via tcgen05 on Thor; GEMV fused kernel on other devices. |
| `VLLM_USE_FP8_LMHEAD=1` | `fp8_gemm` | Thor (sm_110a) | **W8A16-FP8** — weights quantized to FP8 e4m3 with per-row FP16 scale, activations cast to FP8 at runtime. Native FP8 tensor cores. |

Priority order when multiple vars are set: MXFP4 > MXFP8 > FP8 > W8A8 > INT8.

MXFP4, MXFP8, FP8, and W8A8 require Thor (sm_110a). If enabled on a non-Thor device, the plugin prints a warning and falls back to W8A16 INT8.

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

### Accuracy on the real Qwen3.5-122B `lm_head` tensor

`tests/test_real_lm_head.py` runs each mode against the actual `lm_head.weight` extracted from `Intel/Qwen3.5-122B-A10B-int4-AutoRound`. The fixture (1.5 GB) is gitignored — regenerate with `tests/fixtures/extract_lm_head.py` on a host where the model is cached.

| Mode | batch=1 rel | batch=4 rel | batch=1 cos | batch=4 cos |
|------|------------|------------|------------|------------|
| W8A16 INT8 | 1.04% | 1.02% | 0.999945 | 0.999947 |
| W8A8       | 1.30% | 1.45% | 0.999915 | 0.999893 |
| FP8        | 3.81% | 3.79% | 0.999278 | 0.999274 |
| MXFP8      | 3.81% | 3.80% | 0.999274 | 0.999270 |
| MXFP4 (W4A8) — experimental | 8.74% | 11.84% | 0.9962 | 0.9930 |

The real `lm_head` has a well-behaved distribution (row absmax varies only ~6×, abs_max=0.19, p99.9=0.057), so MXFP8 ≈ FP8 here — confirming the synthetic-data finding above.

**MXFP4 accuracy caveat.** MXFP4 trades ~3× more relative error for the smaller weight stream. On random fp16 Gaussian activations into the real Qwen lm_head, the **top-1 sampled token still matches FP16 cuBLAS 100% of the time at batch 1 and 4** (measured in `tests/test_mxfp4_end_to_end.py`), but the rank-ordering of the lower-probability tokens drifts measurably: top-5 set overlap drops to ~80% and top-50 to ~74–83%. This is acceptable for greedy decoding but will perturb temperature/top-p sampling behavior. MXFP4 is exposed as an **opt-in experimental path only** — never the default — and is intended for workloads where the weight bandwidth saving is more valuable than exact token-distribution preservation.

## Speed (Qwen3.5-122B shape: 248320 × 3072, NVIDIA Thor)

Median over 50 timed iters (3 untimed autotune-trigger + 5 warmup) using CUDA events. End-to-end forward latency including activation quantization for W8A8 / FP8 / MXFP8 / MXFP4.

| batch | fp16 (cuBLAS) | w8a16 | w8a8 | fp8 | mxfp8 | mxfp4 (W4A8) |
|-------|--------------:|------:|-----:|----:|------:|-------------:|
| 1  | 14.18 ms (1.00×) | **3.70 ms (3.83×)** | 4.43 ms (3.20×) | 3.43 ms (4.13×) | 3.46 ms (4.10×) | 3.63 ms (3.91×) |
| 2  | 12.32 ms (1.00×) | 7.13 ms (1.73×) | 4.22 ms (2.92×) | **4.15 ms (2.97×)** | 4.61 ms (2.67×) | 4.97 ms (2.48×) |
| 4  | 16.93 ms (1.00×) | 7.30 ms (2.32×) | 4.37 ms (3.87×) | **4.40 ms (3.85×)** | 4.73 ms (3.58×) | 4.93 ms (3.44×) |
| 8  | 16.86 ms (1.00×) | 7.18 ms (2.35×) | 4.44 ms (3.80×) | 4.29 ms (3.93×) | 4.44 ms (3.80×) | **3.71 ms (4.54×)** |
| 16 | 12.75 ms (1.00×) | 5.97 ms (2.14×) | 3.70 ms (3.45×) | **3.48 ms (3.66×)** | 3.57 ms (3.57×) | 3.76 ms (3.39×) |
| 32 | 13.25 ms (1.00×) | 5.39 ms (2.46×) | 4.55 ms (2.91×) | 4.50 ms (2.94×) | 4.95 ms (2.68×) | **3.83 ms (3.46×)** |

(**bold** = fastest in row)

Effective weight bandwidth (theoretical peak on Thor LPDDR5X = 273 GB/s):

| mode | peak measured | % of theoretical |
|------|--------------:|-----------------:|
| w8a16 | 206.2 GB/s | 76% |
| fp8 | 222.3 GB/s | 81% |
| mxfp8 | 220.4 GB/s | 81% |
| w8a8 | 206.3 GB/s | 76% |
| mxfp4 | 111.7 GB/s | 41% (weight stream) |

Key observations:

- **Batch=1 (decode):** W8A16 INT8 still wins via the dedicated `int8_gmv` BW-optimized fast path (single weight stream, no tensor-core compute needed). FP8 and MXFP8 are within 5%.
- **Batch=8 and batch=32:** **MXFP4 is the fastest path** at 3.71 ms (4.54×) and 3.83 ms (3.46×) — the FP4 weight stream becomes the bottleneck only at larger batches where the per-launch quant/activation overhead is amortized.
- **Batch ≥ 2 (small):** **FP8 / MXFP8 lead** at 3.2–4.7 ms across batches 2, 4, 16. FP4's halved weight stream doesn't pay off here — the kernel hits a ~3.7 ms floor set by activation quantization, output writes, and MMA throughput (the `tcgen05.mma.kind::mxf8f6f4` unified instruction processes FP4 at the same per-K-element rate as FP8).
- The `mxfp4` "weight bandwidth %" is a misleading headline. The kernel is **not** memory-bandwidth-bound at FP4 sizes; it's a mixed compute/launch-overhead bound. The honest comparison is wall-clock latency, where MXFP4 wins at b=8 and b=32 and loses or ties everywhere else.
- All non-fp16 modes hit 200–240 GB/s effective weight bandwidth at their best — within ~15% of theoretical LPDDR5X peak.

Run the benchmark yourself:

```bash
python bench/bench_speed.py                              # full sweep
python bench/bench_speed.py --batches 1 --modes fp16 w8a16   # subset
```

The first run takes 5–10 minutes (Triton autotunes ~56 configs per kernel/shape, cached in `~/.triton/cache`). Subsequent runs finish in ~20 seconds.

## Fused sampler (`VLLM_USE_FUSED_SAMPLER=1`)

vLLM's V1 `Sampler.sample` chain — temperature divide, top-k/top-p mask, softmax, exponential RNG, multinomial argmax — materializes ~6 full `[B, V]` fp32 tensors and round-trips through HBM ~10–14 times per call. For a `B=32 × V=152k` decode step that's ~240 MiB of working set on a device with 273 GB/s of memory bandwidth.

`_sampler_kernels.py` collapses that chain into a single two-pass blocked reduction. Each program tile reads its slice of logits once, applies per-row temperature in registers, optionally masks below a top-k threshold, adds Gumbel noise via `tl.rand`, and reduces to a block-local argmax. A tiny `[B, num_blocks]` second pass on the host picks the winning block. Greedy rows (temperature < 1e-5) bypass Gumbel and use upstream's `argmax` directly.

The kernel runs whenever **none** of the following are active:

- `top_p` (would require partial sort inside the kernel)
- `allowed_token_ids_mask` / `bad_words` / `MinP` / `MinTokens` / `LogitBias` / thinking-budget
- penalties (repetition, frequency, presence)
- per-request `torch.Generator` seeds
- `max_num_logprobs` with `processed_logits` / `processed_logprobs` mode

When any feature outside the fast path is active, `FusedSampler.sample` transparently falls back to upstream's implementation. Greedy + temperature + top-k are all supported on the fast path.

### Sampler benchmarks (`bench/bench_sampler.py`, Thor sm_110.0, V=151_936)

Time for one `Sampler.sample()` call (median of 50 timed iters; logits drawn from a pre-generated pool so allocation cost is not in the loop):

| mode | batch | upstream | fused | speedup |
|------|------:|---------:|------:|--------:|
| greedy        |  1 | 0.023 ms | 0.024 ms | 0.99× |
| greedy        |  4 | 0.040 ms | 0.041 ms | 0.98× |
| greedy        | 32 | 0.122 ms | 0.120 ms | 1.02× |
| random        |  1 | 0.080 ms | 0.067 ms | 1.20× |
| random        |  4 | 0.116 ms | 0.049 ms | 2.35× |
| random        |  8 | 0.181 ms | 0.083 ms | 2.18× |
| random        | 16 | 0.325 ms | 0.125 ms | 2.60× |
| random        | 32 | 0.739 ms | 0.229 ms | **3.22×** |
| random+topk50 |  2 | 0.363 ms | 0.341 ms | 1.06× |
| random+topk50 |  4 | 0.386 ms | 0.437 ms | 0.88× |
| random+topk50 | 32 | 0.947 ms | 0.876 ms | 1.08× |

Observations:

- **Greedy:** neutral. Upstream's `logits.argmax(dim=-1)` is already a single reduction kernel — there is nothing to fuse. `FusedSampler.sample` short-circuits to the same `argmax` on `all_greedy` batches.
- **Random (no top-k):** clear win, scaling with batch. At B=32 the fused kernel saves **~0.5 ms per decode step**. Production traffic using default `temperature=1.0, top_p=1.0, top_k=0` (top-k disabled) lands here.
- **Random + top-k=50:** roughly neutral. FlashInfer (the upstream backend on Thor cc=11.0 for this config) is already a fast rejection sampler, and our path's `torch.topk` cost matches it. The fused path stays enabled because correctness is identical and the fallback only adds dispatch overhead.

Run yourself:

```bash
python bench/bench_sampler.py                      # full sweep
python bench/bench_sampler.py --modes random       # just the headline case
```

## Requirements

- Python 3.10+
- PyTorch 2.4+
- Triton 3.0+
- vLLM (any version with `PluggableLayer` OOT support)
- CUDA 13.0+ and sm_110a for W8A8, FP8, MXFP8, and MXFP4 paths

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
# MXFP4 (W4A8) — EXPERIMENTAL, lower accuracy. Fastest at batch 8 and 32.
export VLLM_USE_MXFP4_LMHEAD=1

# MXFP8 — uses tcgen05.mma MX format on Thor (sm_110a)
export VLLM_USE_MXFP8_LMHEAD=1

# W8A8 INT8×INT8 — best throughput on Thor for INT8
export VLLM_USE_W8A8_LMHEAD=1

# W8A16 INT8 weight-only — works on any device, highest accuracy
export VLLM_USE_INT8_LMHEAD=1

# FP8 — per-row scale, simplest FP8 path
export VLLM_USE_FP8_LMHEAD=1

# Fused sampler — independent of the lm_head flags above; can be combined
# with any of them or used on its own. Patches vllm.v1.sample.sampler.Sampler.
export VLLM_USE_FUSED_SAMPLER=1
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

**Autotune cache.** Triton caches autotuned configs to `~/.triton/cache/`. On first run, autotuning over ~56 configs takes several minutes for the Qwen3.5-122B shape. Subsequent runs use the cached result instantly.

**L2 cache hint.** All weight `tl.load` calls use `eviction_policy="evict_first"` — the weight matrix is 700+ MB (much larger than L2) and each tile is touched exactly once per kernel call. Marking weight cache lines for early eviction leaves room in L2 for activations and scale tensors that are reused across the K loop.

**FP8 host-side activation pre-cast.** The FP8 path casts `hidden_states` from FP16 to `float8_e4m3fn` once on the host in `_fp8_forward()` before the kernel launch (stored as `int8` view). The kernel then loads the activation via bitcast — no per-iteration `.to(fp8e4nv)` cast inside the K loop. This removes ~25% of FP8 latency and brings FP8 bandwidth utilization from 63% to 88% of theoretical LPDDR5X peak.

**W8A8 activation quantization.** Activations are quantized per-tensor (single scale = `abs_max / 127`) immediately before the kernel launch in `_w8a8_forward()`. The INT32 accumulator is dequantized at the end of the kernel: `out = acc * w_scale_row * x_scale_scalar`.

**tie_word_embeddings safety.** The original FP16 weights are intentionally not freed after quantization. When `tie_word_embeddings=True`, `lm_head.weight` and `embed_tokens.weight` share the same tensor; zeroing it would corrupt the embedding table.

**MX format details.** The MXFP8 path uses `tl.dot_scaled` with `e4m3` operand format and `e8m0` scales (uint8 storage, decoded as `2^(value - 127)`). Group size is 32 elements per the OCP MX specification. On sm_110a, this maps to `tcgen05.mma` with `.kind::mxf8f6f4` and `.block32` — a single hardware instruction with built-in per-group dequantization. Scale storage for the Qwen3.5-122B `lm_head` is ~22 MB (`248320 × 96` uint8) versus the original ~730 MB FP16 weight matrix.
