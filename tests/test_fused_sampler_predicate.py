"""Tests for FusedSampler integration with vLLM's Sampler class.

These tests exercise the monkey-patching + predicate gating mechanism by
constructing real SamplingMetadata and calling Sampler.forward on both the
fused and unfused implementations.
"""

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


def _make_sampling_metadata(
    B: int,
    V: int,
    *,
    all_greedy: bool = False,
    all_random: bool = True,
    top_p: torch.Tensor | None = None,
    top_k: torch.Tensor | None = None,
    generators: dict | None = None,
    max_num_logprobs: int | None = None,
    allowed_token_ids_mask: torch.Tensor | None = None,
    bad_words_token_ids: dict | None = None,
    temperature_val: float = 1.0,
):
    """Build a minimal SamplingMetadata for testing the fast path."""
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm.v1.sample.logits_processor.state import LogitsProcessors

    if all_greedy:
        temp_t = torch.zeros(B, dtype=torch.float32, device="cuda")
    else:
        temp_t = torch.full((B,), temperature_val, dtype=torch.float32, device="cuda")

    sm = SamplingMetadata(
        temperature=temp_t,
        all_greedy=all_greedy,
        all_random=all_random and not all_greedy,
        top_p=top_p,
        top_k=top_k,
        generators=generators or {},
        max_num_logprobs=max_num_logprobs,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(B, dtype=torch.float32, device="cuda"),
        presence_penalties=torch.zeros(B, dtype=torch.float32, device="cuda"),
        repetition_penalties=torch.ones(B, dtype=torch.float32, device="cuda"),
        output_token_ids=[[] for _ in range(B)],
        allowed_token_ids_mask=allowed_token_ids_mask,
        bad_words_token_ids=bad_words_token_ids or {},
        logitsprocs=LogitsProcessors(),
        logprob_token_ids=None,
        spec_token_ids=None,
        thinking_budget_state_holder=None,
    )
    return sm


@pytest.fixture
def fused_sampler_class():
    """Install the FusedSampler (idempotent) and return the patched class."""
    import vllm.v1.sample.sampler as upstream_module
    from vllm_quant_kernels._sampler import install_fused_sampler

    # Stash original so we can restore after the test session.
    original = upstream_module.Sampler
    install_fused_sampler()
    yield upstream_module.Sampler
    upstream_module.Sampler = original


class TestPatch:
    def test_install_replaces_sampler(self, fused_sampler_class):
        import vllm.v1.sample.sampler as upstream_module
        assert upstream_module.Sampler is fused_sampler_class
        assert upstream_module.Sampler.__name__ == "FusedSampler"

    def test_install_is_subclass(self, fused_sampler_class):
        # Must inherit from upstream Sampler so RejectionSampler-style
        # `sampler.compute_logprobs(...)` calls keep working.
        sampler = fused_sampler_class()
        assert hasattr(sampler, "compute_logprobs")
        assert hasattr(sampler, "gather_logprobs")
        assert hasattr(sampler, "apply_temperature")


class TestFastPathGreedy:
    """All-greedy batches must produce the same tokens as upstream."""

    def test_greedy_matches_upstream(self, fused_sampler_class):
        B, V = 8, 4096
        torch.manual_seed(101)
        logits = torch.randn(B, V, dtype=torch.float32, device="cuda")

        # Reference: original argmax (what upstream's greedy_sample does).
        expected = logits.argmax(dim=-1)

        sm = _make_sampling_metadata(B, V, all_greedy=True, all_random=False)
        sampler = fused_sampler_class()
        # Bypass Sampler.forward (which casts to fp32 again, no-op here) and
        # call .sample directly with already-fp32 logits.
        sampled, _ = sampler.sample(logits.clone(), sm)
        assert sampled.shape == (B,)
        assert torch.equal(sampled.long(), expected)


class TestFastPathRandom:
    """Random sampling should produce tokens distributed like softmax(logits/T)."""

    def test_random_distribution_smoke(self, fused_sampler_class):
        B, V = 64, 256  # B big enough to get statistics, V small enough to count
        torch.manual_seed(202)
        single_logits = torch.randn(V, dtype=torch.float32, device="cuda")
        logits = single_logits.unsqueeze(0).expand(B, V).contiguous()

        sm = _make_sampling_metadata(B, V, all_random=True, temperature_val=1.0)
        sampler = fused_sampler_class()
        sampled, _ = sampler.sample(logits.clone(), sm)

        # Sanity: all tokens in valid range, not constant (i.e. not just argmax).
        assert (sampled >= 0).all() and (sampled < V).all()
        unique = set(sampled.cpu().tolist())
        assert len(unique) >= 5, f"distribution too concentrated: {unique}"


class TestFallback:
    """Features outside the fast path must transparently fall back to upstream."""

    def test_top_p_falls_back(self, fused_sampler_class):
        """top_p is not in the fast path. Verify fallback by checking that
        the call still returns valid tokens (and doesn't crash)."""
        B, V = 4, 4096
        torch.manual_seed(303)
        logits = torch.randn(B, V, dtype=torch.float32, device="cuda")
        top_p = torch.full((B,), 0.9, dtype=torch.float32, device="cuda")

        sm = _make_sampling_metadata(B, V, all_random=True, top_p=top_p)
        sampler = fused_sampler_class()
        sampled, _ = sampler.sample(logits.clone(), sm)
        assert sampled.shape == (B,)
        assert (sampled >= 0).all() and (sampled < V).all()

    def test_max_num_logprobs_processed_falls_back(self, fused_sampler_class):
        """processed_logprobs mode is not in the fast path."""
        B, V = 4, 4096
        torch.manual_seed(404)
        logits = torch.randn(B, V, dtype=torch.float32, device="cuda")

        sm = _make_sampling_metadata(B, V, all_random=True, max_num_logprobs=5)
        sampler = fused_sampler_class(logprobs_mode="processed_logprobs")
        # Should not raise — falls back to upstream which handles logprobs.
        sampled, processed = sampler.sample(logits.clone(), sm)
        assert sampled.shape == (B,)

    def test_generators_fall_back(self, fused_sampler_class):
        """Per-request torch.Generator seeds force fallback path."""
        B, V = 4, 4096
        torch.manual_seed(505)
        logits = torch.randn(B, V, dtype=torch.float32, device="cuda")
        # Empty generators dict is fine — we test with one actual entry.
        gen = torch.Generator(device="cuda").manual_seed(42)
        sm = _make_sampling_metadata(B, V, all_random=True, generators={0: gen})

        sampler = fused_sampler_class()
        sampled, _ = sampler.sample(logits.clone(), sm)
        assert sampled.shape == (B,)


class TestTopKInFastPath:
    """Top-k requests stay on the fast path and must obey top-k membership."""

    def test_topk_membership_via_fast_path(self, fused_sampler_class):
        B, V, k = 8, 4096, 10
        torch.manual_seed(606)
        logits = torch.randn(B, V, dtype=torch.float32, device="cuda")
        top_k = torch.full((B,), k, dtype=torch.int32, device="cuda")

        # Pre-compute the true top-k set for verification.
        _, topk_idx = torch.topk(logits, k, dim=-1)
        topk_set_per_row = [set(topk_idx[i].cpu().tolist()) for i in range(B)]

        sm = _make_sampling_metadata(B, V, all_random=True, top_k=top_k)
        sampler = fused_sampler_class()
        sampled, _ = sampler.sample(logits.clone(), sm)

        for i in range(B):
            tok = int(sampled[i].item())
            assert tok in topk_set_per_row[i], (
                f"row {i}: sampled token {tok} not in top-{k} set "
                f"{sorted(topk_set_per_row[i])}"
            )
