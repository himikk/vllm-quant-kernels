"""Fused lm_head + sampling replacement for vLLM V1-classic Sampler.

Overrides only `Sampler.sample()` — the small inner method called by
`Sampler.forward()`. Falls back to the parent (unfused) implementation when
features outside the fast path are active (penalties, MinP, logit_bias, etc.).

Activation: `VLLM_USE_FUSED_SAMPLER=1` env var.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import torch

from ._sampler_kernels import fused_gumbel_sample

if TYPE_CHECKING:
    from vllm.config.model import LogprobsMode
    from vllm.v1.sample.metadata import SamplingMetadata


_SAMPLING_EPS = 1e-5


def _fast_path_ok(sampling_metadata: "SamplingMetadata") -> bool:
    """Predicate: are all features outside our fast path inactive?

    The fused kernel handles:
      - per-row temperature (incl. mixed greedy/random via per-row Gumbel skip)
      - top-k (via host-computed per-row threshold)

    Falls back when any of these are active:
      - top-p (would require partial sort inside the kernel; not implemented)
      - allowed_token_ids_mask, bad_words
      - any logitsprocs (MinP, MinTokens, LogitBias, ...)
      - thinking budget tracking
      - per-request `generators` (custom torch.Generator seeds)
    """
    sm = sampling_metadata
    if sm.top_p is not None:
        return False
    if sm.allowed_token_ids_mask is not None:
        return False
    if sm.bad_words_token_ids:
        return False
    if sm.logitsprocs.argmax_invariant or sm.logitsprocs.non_argmax_invariant:
        return False
    if sm.thinking_budget_state_holder is not None and (
        sm.thinking_budget_state_holder.has_tracked_requests()
    ):
        return False
    if sm.generators:
        # Per-request torch.Generator seeds — would need an extra dict-driven
        # patching pass. Defer to upstream's slower per-row exponential loop.
        return False
    return True


def _compute_topk_threshold(
    logits: torch.Tensor,           # [B, V] fp32, raw (pre-temperature)
    top_k: torch.Tensor,            # [B] int32, with V (==vocab) meaning "disabled"
    vocab_size: int,
) -> torch.Tensor | None:
    """Compute per-row top-k threshold via a single torch.topk.

    Returns None if every row has top_k disabled (top_k >= vocab_size).

    Returned threshold[i] = the (top_k[i])-th largest value of logits[i],
    in the same space as the kernel's input logits (i.e., raw, pre-temperature).
    Top-k membership is invariant under positive division and Gumbel addition
    (in expectation), so masking in raw-logit space is correct.

    Edge case: vLLM stores disabled rows as `top_k[i] = vocab_size`. We can't
    pass per-row k to torch.topk in one shot, so we use `k = max(top_k)` and
    then per-row scatter to find the actual k-th-largest. When max k == V the
    work is unavoidable; we short-circuit that case to None.
    """
    max_k = int(top_k.max().item())
    if max_k >= vocab_size:
        return None  # all rows disabled
    # Top-k over the full row, K = global max. Cost: O(V log K). For K<=50
    # this is far cheaper than a full sort.
    # NOTE(thor): torch.topk uses radix-select on CUDA — single-pass over V.
    topk_vals, _ = torch.topk(logits, max_k, dim=-1, sorted=True)
    # threshold[i] = topk_vals[i, top_k[i] - 1].  Rows with top_k[i] == V
    # would be out-of-bounds; clamp to last valid column and we'll mask less
    # aggressively for those rows (effectively disabling top-k for them).
    idx = (top_k.clamp(max=max_k) - 1).clamp(min=0).to(torch.int64).unsqueeze(-1)
    threshold = topk_vals.gather(dim=-1, index=idx).squeeze(-1).contiguous()
    return threshold


def _make_seeds(batch_size: int, device: torch.device) -> torch.Tensor:
    """Derive per-row int32 seeds for Gumbel noise.

    We deliberately bypass `sampling_metadata.generators` (the dict of
    per-request torch.Generator instances) — when that dict is non-empty the
    fast-path predicate has already returned False. For the empty-dict case
    (the common one), we draw from torch.cuda's default RNG stream once per
    sampler call. This matches the entropy source of
    `topk_topp_sampler.random_sample` which calls `q.exponential_()` on the
    default generator.
    """
    # torch.randint pulls one int32 per row from the default CUDA generator;
    # this is the same RNG the unfused path consumes.
    seeds = torch.randint(
        low=0,
        high=2**31 - 1,
        size=(batch_size,),
        dtype=torch.int32,
        device=device,
    )
    return seeds


def install_fused_sampler() -> None:
    """Monkey-patch vllm.v1.sample.sampler.Sampler with the fused variant.

    Must run before `vllm.v1.worker.gpu_model_runner` is imported (i.e., from
    a `vllm.general_plugins` entry-point at engine init). After this returns,
    `GPUModelRunner.__init__` will construct our FusedSampler subclass.
    """
    import vllm.v1.sample.sampler as upstream_module

    upstream_sampler = upstream_module.Sampler

    class FusedSampler(upstream_sampler):
        """Drop-in Sampler that fuses temperature + (top-k) + Gumbel argmax.

        Subclasses the upstream Sampler so all methods we don't override
        (`compute_logprobs`, `gather_logprobs`, `apply_logits_processors`,
        `apply_temperature`, `gather_specific_token_logprobs`, ...) keep
        working unchanged. Callers like `RejectionSampler` that reach in for
        static methods are unaffected.
        """

        # We override `sample` rather than `forward` so logprob handling, the
        # fp32 cast, logits-processor chain, and the SamplerOutput packaging
        # in `forward` all remain upstream's responsibility. This keeps the
        # patch surgical and minimizes drift on vLLM upgrades.
        def sample(
            self,
            logits: torch.Tensor,
            sampling_metadata: "SamplingMetadata",
            logprobs_mode_override: "LogprobsMode | None" = None,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            if not _fast_path_ok(sampling_metadata):
                return super().sample(
                    logits, sampling_metadata, logprobs_mode_override
                )

            # Logprob requests are forwarded to upstream so we don't have to
            # re-implement the processed_logits / processed_logprobs branch
            # of the Gumbel kernel. The hot decode path doesn't request them.
            logprobs_mode = logprobs_mode_override or self.logprobs_mode
            wants_processed_logprobs = (
                (
                    sampling_metadata.max_num_logprobs is not None
                    or sampling_metadata.logprob_token_ids
                )
                and logprobs_mode in ("processed_logits", "processed_logprobs")
            )
            if wants_processed_logprobs:
                return super().sample(
                    logits, sampling_metadata, logprobs_mode_override
                )

            # All-greedy short-circuit. Upstream's Sampler.sample also
            # short-circuits this case (sampler.py:256-266). We do the same
            # here because our kernel's two-pass block reduction has more
            # launch overhead than a single torch.argmax for this trivial
            # case — measured ~40us regression at B=1 without this branch.
            if sampling_metadata.all_greedy:
                return logits.argmax(dim=-1).view(-1), None

            assert sampling_metadata.temperature is not None
            B, V = logits.shape

            # Top-k threshold (one full pass over logits via radix-select).
            # We compute it on the RAW logits (pre-temperature) so the
            # threshold expresses set membership, not magnitude. Membership
            # in the top-k of `logits` is identical to membership in the
            # top-k of `logits / T` for any T > 0 (monotonic transform).
            topk_threshold = None
            if sampling_metadata.top_k is not None:
                topk_threshold = _compute_topk_threshold(
                    logits, sampling_metadata.top_k, V
                )

            # Per-row Gumbel seeds. For greedy rows (temperature < EPS) the
            # kernel ignores the seed entirely (no Gumbel noise added), so it
            # doesn't matter what we put there.
            seeds = _make_seeds(B, logits.device)

            sampled = fused_gumbel_sample(
                logits=logits,
                temperature=sampling_metadata.temperature.to(torch.float32),
                seeds=seeds,
                topk_threshold=topk_threshold,
            )
            # No processed_logprobs in the fast path (handled by early-return
            # above).
            return sampled, None

    upstream_module.Sampler = FusedSampler
    print(
        "[vllm-quant-kernels] FusedSampler installed "
        "(vllm.v1.sample.sampler.Sampler patched).",
        file=sys.stderr,
        flush=True,
    )


def env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "")
    if val == "":
        return default
    return val.lower() in ("1", "true", "yes", "on")
