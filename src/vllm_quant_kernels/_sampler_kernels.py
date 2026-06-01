"""Triton kernel for fused Gumbel-argmax sampling.

Replaces the chain of fp32-cast + temperature + softmax + top-k/top-p sort +
exponential-RNG + division + argmax in vLLM's V1 Sampler with a single two-pass
blocked reduction over the logits tensor.

Algorithm:
  For each row i (= request) and each block j of vocab:
    - load logits_block (already fp32 from vLLM's Sampler).
    - if temperature[i] < EPS:  treat as greedy, no division, no noise.
    - else:                     y = logits / temperature[i] + Gumbel(seed[i], pos)
    - if top_k_threshold is provided, mask y < threshold[i] -> -inf
    - reduce: local_max, local_argmax over the block.
    - write local_max[i, j], local_argmax[i, j].
  Then host does a tiny argmax across blocks (~vocab/BLOCK_SIZE elements per row).

The Gumbel trick: argmax(logits/T + g_k) where g_k ~ Gumbel(0,1) samples from
softmax(logits/T). g_k = -log(-log(u)) with u ~ Uniform(0,1].

Structural template adapted from vllm/v1/worker/gpu/sample/gumbel.py (the V2
model runner) — the V2 sampler ships this kernel upstream; V1-classic (which
the int4 Qwen model uses on Thor) does not, hence this plugin.
"""

import torch
import triton
import triton.language as tl


# Smallest positive normal fp32, used to clamp uniform[0,1) away from zero so
# that -log(u) stays finite. Matches upstream gumbel.py:14.
_FP32_TINY = tl.constexpr(float.fromhex("0x1p-126"))

# Sentinel temperature value indicating greedy (skip Gumbel, plain argmax).
_GREEDY_EPS = tl.constexpr(1e-5)


@triton.jit
def _gumbel_block_argmax_kernel(
    # Outputs (per (row, block) tile)
    local_argmax_ptr,        # [B, num_blocks] int64
    local_argmax_stride,
    local_max_ptr,           # [B, num_blocks] fp32
    local_max_stride,
    # Inputs
    logits_ptr,              # [B, V] fp32 (already cast by Sampler.forward)
    logits_stride,
    temperature_ptr,         # [B] fp32 (or None — see HAS_TEMPERATURE)
    seeds_ptr,               # [B] int32 (per-row RNG seed for Gumbel)
    topk_threshold_ptr,      # [B] fp32 (or None — see HAS_TOPK)
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    HAS_TEMPERATURE: tl.constexpr,
    HAS_TOPK: tl.constexpr,
):
    """One program per (row, block) tile. Each program reduces a single tile.

    Greedy rows (temperature ~ 0) bypass the Gumbel noise — equivalent to
    Sampler.greedy_sample. The greedy/random selection is per-row, matching
    vLLM's torch.where(temp < EPS, greedy, random) at sampler.py:291-296.
    """
    row = tl.program_id(0)
    block_idx = tl.program_id(1)

    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    logits = tl.load(
        logits_ptr + row * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)

    if HAS_TEMPERATURE:
        temp = tl.load(temperature_ptr + row).to(tl.float32)
        is_greedy = temp < _GREEDY_EPS
    else:
        temp = 1.0
        is_greedy = False

    # Top-k mask: drop entries below the per-row k-th largest threshold.
    # The threshold is provided in the SAME space as the raw (pre-temperature,
    # pre-Gumbel) logits, because top-k membership is invariant under both
    # the (monotonic) division by T > 0 and the (rank-preserving in
    # expectation, but per-sample stochastic) Gumbel addition. Applying the
    # mask here means non-top-k tokens cannot win even with a lucky Gumbel
    # draw.
    if HAS_TOPK:
        thr = tl.load(topk_threshold_ptr + row).to(tl.float32)
        in_topk = logits >= thr
    else:
        in_topk = True

    # Temperature division. Skipped for greedy rows (would 0-divide).
    if HAS_TEMPERATURE:
        if not is_greedy:
            logits = logits / temp

    # Gumbel noise. Only added for non-greedy rows so the greedy branch is
    # exactly equivalent to logits.argmax(dim=-1).
    if HAS_TEMPERATURE:
        if not is_greedy:
            seed = tl.load(seeds_ptr + row)
            u = tl.rand(seed, block)
            u = tl.maximum(u, _FP32_TINY)
            logits = logits + (-tl.log(-tl.log(u)))

    # Combine OOB mask with top-k mask: anything outside [0, vocab_size) OR
    # below the top-k threshold gets -inf.
    if HAS_TOPK:
        logits = tl.where(mask & in_topk, logits, float("-inf"))
    else:
        logits = tl.where(mask, logits, float("-inf"))

    # Block-local argmax. Triton's tl.max returns the index within the tile.
    value, idx = tl.max(logits, axis=0, return_indices=True)
    token_id = block_idx * BLOCK_SIZE + idx

    tl.store(local_argmax_ptr + row * local_argmax_stride + block_idx, token_id)
    tl.store(local_max_ptr + row * local_max_stride + block_idx, value)


def fused_gumbel_sample(
    logits: torch.Tensor,                 # [B, V] fp32
    temperature: torch.Tensor | None,     # [B] fp32 (None => greedy/all-T=1)
    seeds: torch.Tensor,                  # [B] int32
    topk_threshold: torch.Tensor | None = None,  # [B] fp32 or None
    block_size: int = 1024,
) -> torch.Tensor:
    """Fused Gumbel-argmax sampling over `[B, V]` logits.

    Returns: `[B]` int64 sampled token IDs.

    Contract:
      - `logits` is the post-logits-processor, post-fp32-cast tensor (i.e., the
        exact same tensor Sampler.sample receives).
      - `temperature[i] < 1e-5` means request i is greedy. The kernel handles
        per-row mixed greedy/random in a single launch.
      - `seeds[i]` is the RNG seed for row i. For Python random (no explicit
        per-request torch.Generator), the caller derives this from
        torch.cuda.default_generators using a counter the same way
        `q.exponential_()` would consume.
      - `topk_threshold[i]` is the k-th largest value of (logits[i]/temp[i]),
        in the SAME space as the kernel's internal computation. Caller must
        compute it from the post-temperature logits. None means no top-k.
    """
    B, V = logits.shape
    assert logits.dtype == torch.float32, (
        f"Fused sampler requires fp32 logits, got {logits.dtype}"
    )
    assert logits.is_contiguous(), "Fused sampler requires contiguous logits"
    assert seeds.shape == (B,) and seeds.dtype == torch.int32
    if temperature is not None:
        assert temperature.shape == (B,) and temperature.dtype == torch.float32
    if topk_threshold is not None:
        assert topk_threshold.shape == (B,) and topk_threshold.dtype == torch.float32

    num_blocks = triton.cdiv(V, block_size)
    local_argmax = torch.empty(B, num_blocks, dtype=torch.int64, device=logits.device)
    local_max    = torch.empty(B, num_blocks, dtype=torch.float32, device=logits.device)

    _gumbel_block_argmax_kernel[(B, num_blocks)](
        local_argmax, local_argmax.stride(0),
        local_max,    local_max.stride(0),
        logits, logits.stride(0),
        temperature if temperature is not None else logits,  # dummy ptr, unused
        seeds,
        topk_threshold if topk_threshold is not None else logits,  # dummy
        V,
        BLOCK_SIZE=block_size,
        HAS_TEMPERATURE=temperature is not None,
        HAS_TOPK=topk_threshold is not None,
        num_warps=4,
        num_stages=2,
    )

    # Tiny final reduction: pick winning block per row, gather the token id.
    # Working set is [B, num_blocks] = at most ~150 fp32 values per row at
    # V=152k, BLOCK_SIZE=1024 — negligible vs the [B,V] traffic we saved.
    winner_block = local_max.argmax(dim=-1, keepdim=True)
    sampled = local_argmax.gather(dim=-1, index=winner_block).view(-1)
    return sampled
