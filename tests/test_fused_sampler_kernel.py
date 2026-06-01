"""Tests for the fused Gumbel-argmax sampler kernel.

The kernel is tested in isolation (no vLLM Sampler wrapper) to keep tests
fast and independent of the vLLM version installed. The wrapper class is
tested separately in test_fused_sampler_predicate.py.
"""

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


# Mirrors Qwen3.5-122B lm_head shape (org_vocab=151_936) but reduced for
# test speed. The kernel is shape-agnostic — these dimensions exercise the
# multi-block reduction path.
V_SMALL = 4096
V_LARGE = 152_064  # exact Qwen org_vocab_size


def _make_logits(B: int, V: int, seed: int = 0) -> torch.Tensor:
    """Random fp32 logits with a non-trivial dynamic range (~ standard Normal)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(B, V, generator=g, dtype=torch.float32, device="cuda")


class TestKernelImport:
    def test_kernel_importable(self):
        from vllm_quant_kernels._sampler_kernels import fused_gumbel_sample
        assert fused_gumbel_sample is not None


class TestGreedyCorrectness:
    """Greedy rows (temperature < EPS) must exactly match torch.argmax."""

    @pytest.mark.parametrize("B,V", [(1, V_SMALL), (4, V_SMALL), (8, V_LARGE), (32, V_LARGE)])
    def test_greedy_exact_match(self, B, V):
        from vllm_quant_kernels._sampler_kernels import fused_gumbel_sample

        logits = _make_logits(B, V, seed=42)
        temperature = torch.zeros(B, dtype=torch.float32, device="cuda")  # all greedy
        seeds = torch.zeros(B, dtype=torch.int32, device="cuda")  # ignored on greedy rows

        sampled = fused_gumbel_sample(logits, temperature, seeds)
        expected = logits.argmax(dim=-1)

        assert sampled.shape == (B,)
        assert sampled.dtype == torch.int64
        assert torch.equal(sampled, expected), (
            f"greedy mismatch: kernel={sampled[:8].tolist()} "
            f"expected={expected[:8].tolist()}"
        )

    def test_greedy_no_temperature_tensor(self):
        """temperature=None still gives plain argmax (HAS_TEMPERATURE=False branch)."""
        from vllm_quant_kernels._sampler_kernels import fused_gumbel_sample

        B, V = 4, V_SMALL
        logits = _make_logits(B, V, seed=1)
        seeds = torch.zeros(B, dtype=torch.int32, device="cuda")

        sampled = fused_gumbel_sample(logits, temperature=None, seeds=seeds)
        expected = logits.argmax(dim=-1)
        assert torch.equal(sampled, expected)


class TestRandomDistribution:
    """Random rows must sample from softmax(logits/T) within chi-squared tolerance."""

    def test_random_matches_softmax_distribution(self):
        """Run kernel N times with temp=1.0 on a fixed row of small vocab,
        compare empirical token frequencies to true softmax probabilities."""
        from vllm_quant_kernels._sampler_kernels import fused_gumbel_sample

        V = 32  # small vocab so each token has enough samples
        N = 8000
        torch.manual_seed(7)
        true_logits = torch.randn(V, dtype=torch.float32, device="cuda")
        true_probs = torch.softmax(true_logits, dim=-1).cpu().numpy()

        # Batch all N draws as separate "rows" so the kernel runs once.
        logits = true_logits.unsqueeze(0).expand(N, V).contiguous()
        temperature = torch.ones(N, dtype=torch.float32, device="cuda")
        seeds = torch.randint(0, 2**31 - 1, (N,), dtype=torch.int32, device="cuda")

        sampled = fused_gumbel_sample(logits, temperature, seeds).cpu().numpy()
        counts = [int((sampled == k).sum()) for k in range(V)]
        empirical = [c / N for c in counts]

        # Chi-squared goodness-of-fit. With df = V-1 = 31, chi-sq 99% crit ≈ 52.2.
        # Allow generous slack — we're sampling 8k from 32 categories.
        expected_counts = [true_probs[k] * N for k in range(V)]
        chi_sq = sum(
            (counts[k] - expected_counts[k]) ** 2 / expected_counts[k]
            for k in range(V)
        )
        assert chi_sq < 60.0, (
            f"chi-sq={chi_sq:.2f} > 60 (df=31); empirical={empirical[:5]}... "
            f"true={true_probs[:5].tolist()}"
        )

    def test_random_temperature_scales_distribution(self):
        """Higher temperature => flatter distribution. Verify the empirical
        fraction of argmax-hits matches the softmax(logits/T) probability at
        both low and high T."""
        from vllm_quant_kernels._sampler_kernels import fused_gumbel_sample

        V = 16
        N = 4000
        torch.manual_seed(11)
        true_logits = torch.randn(V, dtype=torch.float32, device="cuda")
        argmax_token = int(true_logits.argmax().item())

        def empirical_argmax_fraction(temp_val: float) -> float:
            logits = true_logits.unsqueeze(0).expand(N, V).contiguous()
            temperature = torch.full((N,), temp_val, dtype=torch.float32, device="cuda")
            seeds = torch.randint(0, 2**31 - 1, (N,), dtype=torch.int32, device="cuda")
            sampled = fused_gumbel_sample(logits, temperature, seeds)
            return float((sampled == argmax_token).float().mean().item())

        def theoretical_argmax_prob(temp_val: float) -> float:
            return float(torch.softmax(true_logits / temp_val, dim=-1).max().item())

        for temp_val in (0.1, 1.0, 5.0):
            emp = empirical_argmax_fraction(temp_val)
            theo = theoretical_argmax_prob(temp_val)
            # Standard error for N draws of a Bernoulli(theo) is sqrt(theo*(1-theo)/N).
            # ~3 sigma allowance plus 0.01 floor for very-small-prob cases.
            tol = 3.0 * (theo * (1 - theo) / N) ** 0.5 + 0.01
            assert abs(emp - theo) < tol, (
                f"T={temp_val}: empirical={emp:.4f}, theoretical={theo:.4f}, "
                f"diff={abs(emp - theo):.4f} > tol={tol:.4f}"
            )


class TestTopK:
    """Top-k threshold masking must restrict samples to the top-k set."""

    @pytest.mark.parametrize("k", [1, 5, 50])
    def test_topk_membership(self, k):
        """Every sampled token must be among the per-row top-k tokens."""
        from vllm_quant_kernels._sampler_kernels import fused_gumbel_sample

        B, V, N = 8, V_LARGE, 4
        # N draws per row to stress randomness — concatenate B*N row replicas.
        torch.manual_seed(23)
        logits_row = torch.randn(B, V, dtype=torch.float32, device="cuda")
        logits = logits_row.repeat_interleave(N, dim=0).contiguous()
        BN = B * N

        # threshold = kth-largest value of each row
        topk_vals, topk_idx = torch.topk(logits, k, dim=-1, sorted=True)
        threshold = topk_vals[:, k - 1].contiguous()

        temperature = torch.ones(BN, dtype=torch.float32, device="cuda")
        seeds = torch.randint(0, 2**31 - 1, (BN,), dtype=torch.int32, device="cuda")
        sampled = fused_gumbel_sample(logits, temperature, seeds, topk_threshold=threshold)

        # Membership check.
        topk_set = set()  # (row, token) tuples that are in top-k
        topk_idx_cpu = topk_idx.cpu()
        for i in range(BN):
            topk_set.update((i, t) for t in topk_idx_cpu[i].tolist())

        sampled_cpu = sampled.cpu().tolist()
        bad = [(i, sampled_cpu[i]) for i in range(BN) if (i, sampled_cpu[i]) not in topk_set]
        assert not bad, f"top-{k}: {len(bad)} samples outside top-k set, first 5: {bad[:5]}"

    def test_topk_1_equals_argmax(self):
        """top-k=1 must give the greedy answer regardless of temperature."""
        from vllm_quant_kernels._sampler_kernels import fused_gumbel_sample

        B, V = 8, V_LARGE
        torch.manual_seed(31)
        logits = torch.randn(B, V, dtype=torch.float32, device="cuda")
        expected = logits.argmax(dim=-1)

        topk_vals, _ = torch.topk(logits, 1, dim=-1)
        threshold = topk_vals[:, 0].contiguous()
        temperature = torch.ones(B, dtype=torch.float32, device="cuda")
        seeds = torch.randint(0, 2**31 - 1, (B,), dtype=torch.int32, device="cuda")

        sampled = fused_gumbel_sample(logits, temperature, seeds, topk_threshold=threshold)
        assert torch.equal(sampled, expected)


class TestMixedBatch:
    """Greedy and random rows in the same batch must coexist correctly."""

    def test_mixed_greedy_random(self):
        from vllm_quant_kernels._sampler_kernels import fused_gumbel_sample

        B, V = 4, V_LARGE
        logits = _make_logits(B, V, seed=99)

        # Row 0 and 2 greedy, row 1 and 3 random.
        temperature = torch.tensor([0.0, 1.0, 0.0, 0.8], dtype=torch.float32, device="cuda")
        seeds = torch.randint(0, 2**31 - 1, (B,), dtype=torch.int32, device="cuda")

        sampled = fused_gumbel_sample(logits, temperature, seeds)
        expected_greedy = logits.argmax(dim=-1)

        # Greedy rows must exactly match argmax.
        assert sampled[0].item() == expected_greedy[0].item()
        assert sampled[2].item() == expected_greedy[2].item()
        # Random rows: just sanity-check they're in range. Distribution checked elsewhere.
        assert 0 <= sampled[1].item() < V
        assert 0 <= sampled[3].item() < V
