"""Speed benchmark for the fused sampler kernel.

Times Sampler.sample() — the inner method called from Sampler.forward() that
turns fp32 logits into sampled token IDs — comparing the upstream vLLM
implementation against the FusedSampler from this plugin.

Two configurations are measured per batch size:
    - greedy:           temperature[i] = 0       (argmax-only path)
    - random + top-k:   temperature[i] = 1.0, top_k[i] = 50

Random + top-p, MinP, penalties, etc. would force the FusedSampler to fall
back to upstream, so they are excluded here.

Run:

    .venv/bin/python vllm-quant-kernels/bench/bench_sampler.py

Defaults match the Qwen3.5-122B vocab (org_vocab_size=151_936). The kernel is
shape-agnostic — change `--V` to bench other vocabs.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Callable

import torch


DEFAULT_V = 151_936  # Qwen3.5 org_vocab_size
DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32)

WARMUP_ITERS = 5
TIMED_ITERS = 50
AUTOTUNE_TRIGGER_ITERS = 3


def _bench(fn: Callable[[], object], iters: int) -> list[float]:
    """Per-iter latency in ms via CUDA events."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _warmup_and_time(fn: Callable[[], object]) -> float:
    for _ in range(AUTOTUNE_TRIGGER_ITERS):
        fn()
    torch.cuda.synchronize()
    _bench(fn, WARMUP_ITERS)
    samples = _bench(fn, TIMED_ITERS)
    return statistics.median(samples)


def _make_metadata(B: int, V: int, *, mode: str, top_k_val: int = 50):
    """Build SamplingMetadata for the given configuration.

    mode is one of:
      - "greedy":         temperature=0, all_greedy=True
      - "random":         temperature=1.0, top_k=None
      - "random_topk":    temperature=1.0, top_k[i] = top_k_val
    """
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm.v1.sample.logits_processor.state import LogitsProcessors

    if mode == "greedy":
        temp_t = torch.zeros(B, dtype=torch.float32, device="cuda")
        all_greedy, all_random = True, False
    else:
        temp_t = torch.ones(B, dtype=torch.float32, device="cuda")
        all_greedy, all_random = False, True

    top_k_t = None
    if mode == "random_topk":
        top_k_t = torch.full((B,), top_k_val, dtype=torch.int32, device="cuda")

    return SamplingMetadata(
        temperature=temp_t,
        all_greedy=all_greedy,
        all_random=all_random,
        top_p=None,
        top_k=top_k_t,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(B, dtype=torch.float32, device="cuda"),
        presence_penalties=torch.zeros(B, dtype=torch.float32, device="cuda"),
        repetition_penalties=torch.ones(B, dtype=torch.float32, device="cuda"),
        output_token_ids=[[] for _ in range(B)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
        logprob_token_ids=None,
        spec_token_ids=None,
        thinking_budget_state_holder=None,
    )


def _bench_sampler(sampler, B: int, V: int, mode: str) -> float:
    """Time one sampler configuration.

    Upstream `sample()` mutates logits in-place via `apply_temperature`'s
    `logits.div_(...)`; our FusedSampler does not. To keep the benchmark
    apples-to-apples we time a regenerate-and-call cycle, but use a pinned
    pool of pre-generated logits batches so the per-iter cost of producing
    a fresh `[B, V] fp32` tensor doesn't dominate at high (B, V).
    """
    # Generate a small pool of independent logits tensors; we cycle through
    # them so each call gets a fresh (unmutated) buffer with negligible
    # allocation cost. Each tensor is ~19 MiB at the Qwen V=152k shape.
    POOL = 8
    logits_pool = [
        torch.randn(B, V, dtype=torch.float32, device="cuda") for _ in range(POOL)
    ]
    sm = _make_metadata(B, V, mode=mode)

    counter = {"i": 0}

    def fn():
        # Index round-robin into the pool. Avoids the full-pass clone() we'd
        # otherwise need to give upstream a fresh-each-time tensor.
        idx = counter["i"] & (POOL - 1)
        counter["i"] += 1
        sampler.sample(logits_pool[idx], sm)

    return _warmup_and_time(fn)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--V", type=int, default=DEFAULT_V, help="vocab size")
    parser.add_argument(
        "--batches", type=int, nargs="+", default=list(DEFAULT_BATCHES),
        help="batch sizes to sweep",
    )
    parser.add_argument(
        "--modes", nargs="+",
        default=["greedy", "random", "random_topk"],
        choices=["greedy", "random", "random_topk"],
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    cap = torch.cuda.get_device_capability(device)
    name = torch.cuda.get_device_name(device)
    print(f"GPU: {name} (cc={cap[0]}.{cap[1]})")
    print(f"Vocab: V={args.V}    Batches: {args.batches}")
    print(f"Iters: {AUTOTUNE_TRIGGER_ITERS} autotune + {WARMUP_ITERS} warmup + {TIMED_ITERS} timed")
    print()

    import vllm.v1.sample.sampler as upstream_module
    UpstreamSampler = upstream_module.Sampler

    # Install fused sampler, capture FusedSampler class, then restore upstream
    # so the two are independent for fair benchmarking.
    from vllm_quant_kernels._sampler import install_fused_sampler
    install_fused_sampler()
    FusedSampler = upstream_module.Sampler
    upstream_module.Sampler = UpstreamSampler  # restore

    fused = FusedSampler()
    upstream = UpstreamSampler()

    print("=" * 78)
    print(f"  {'mode':<13} {'batch':>5}  {'upstream':>10}   {'fused':>10}   {'speedup':>9}")
    print("=" * 78)

    for mode in args.modes:
        for batch in args.batches:
            t0 = time.time()
            up_ms    = _bench_sampler(upstream, batch, args.V, mode=mode)
            fused_ms = _bench_sampler(fused,    batch, args.V, mode=mode)
            speedup  = up_ms / fused_ms
            wall = time.time() - t0
            print(
                f"  {mode:<13} {batch:>5}  "
                f"{up_ms:8.3f} ms   {fused_ms:8.3f} ms   "
                f"{speedup:6.2f}x   ({wall:4.1f}s wall)"
            )
        print("-" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
