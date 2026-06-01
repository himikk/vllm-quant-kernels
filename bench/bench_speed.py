"""Speed benchmark for vllm-quant-kernels.

Measures end-to-end forward latency of each quantization mode against a fp16
cuBLAS baseline at the Qwen3.5-122B lm_head shape (M=248320, K=3072), across
a small sweep of batch sizes.

Run:

    .venv/bin/python vllm-quant-kernels/bench/bench_speed.py

The first run is slow (Triton autotunes ~30 configs per shape, persisted in
~/.triton/cache). Subsequent runs reuse the cache and finish in seconds.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from types import SimpleNamespace
from typing import Callable

import torch


# Default shape: Qwen3.5-122B lm_head (vocab=248320, hidden=3072)
DEFAULT_M = 248320
DEFAULT_K = 3072
DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32)

WARMUP_ITERS = 5
TIMED_ITERS = 50
AUTOTUNE_TRIGGER_ITERS = 3  # extra untimed iters before warmup to flush JIT


class _FakeWeight:
    def __init__(self, t: torch.Tensor) -> None:
        self.data = t
        self.dtype = t.dtype
        self.shape = t.shape
        self.device = t.device


def _make_fake_lm_head(w: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(weight=_FakeWeight(w))


def _bench(fn: Callable[[], torch.Tensor], iters: int) -> list[float]:
    """Return per-iter latencies in milliseconds using CUDA events."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _warmup_and_time(fn: Callable[[], torch.Tensor]) -> tuple[float, float, float]:
    """Return (median_ms, min_ms, max_ms) for the callable."""
    # Trigger any Triton autotune (untimed).
    for _ in range(AUTOTUNE_TRIGGER_ITERS):
        fn()
    torch.cuda.synchronize()
    # Warmup proper.
    _bench(fn, WARMUP_ITERS)
    # Measure.
    samples = _bench(fn, TIMED_ITERS)
    return statistics.median(samples), min(samples), max(samples)


def _bench_fp16(w: torch.Tensor, batch: int) -> tuple[float, float, float]:
    K = w.shape[1]
    x = torch.randn(batch, K, device="cuda", dtype=torch.float16)
    return _warmup_and_time(lambda: torch.nn.functional.linear(x, w))


def _bench_mode(env_var: str, w: torch.Tensor, batch: int) -> tuple[float, float, float]:
    """Bench a single quantized mode for a given batch. Sets env var, initializes
    quantized state once, then times only _quantized_forward()."""
    from vllm_quant_kernels._quant import QuantizedLogitsProcessor

    old = os.environ.get(env_var)
    os.environ[env_var] = "1"
    try:
        lm_head = _make_fake_lm_head(w)
        proc = QuantizedLogitsProcessor(w.shape[0])
        proc._init_int8(lm_head)
        proc._int8_initialized = True
        K = w.shape[1]
        x = torch.randn(batch, K, device="cuda", dtype=torch.float16)
        return _warmup_and_time(lambda: proc._quantized_forward(x, lm_head, None))
    finally:
        if old is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = old


def _bytes_per_weight(env_var: str) -> float:
    """Weight storage size per element, for memory-bandwidth estimates."""
    if env_var == "fp16":
        return 2.0
    if env_var == "VLLM_USE_MXFP4_LMHEAD":
        # 0.5 byte e2m1 value + 1/32 byte scale = 17/32 = 0.53125 bytes/elem
        return 0.5 + 1.0 / 32.0
    if env_var == "VLLM_USE_MXFP8_LMHEAD":
        # 1 byte e4m3 value + 1/32 byte scale ≈ 1.03 bytes/elem
        return 1.0  # under-counts scales by ~3 %, close enough for a headline number
    # INT8 and FP8 are all 1 byte per weight.
    return 1.0


def _fmt_row(name: str, batch: int, median_ms: float, fp16_ms: float,
             M: int, K: int, env_var: str) -> str:
    flops = 2.0 * M * K * batch  # FMA = 2 flops
    tflops = flops / (median_ms * 1e-3) / 1e12
    # Effective weight bandwidth (weight matrix re-streamed every call).
    weight_bytes = M * K * _bytes_per_weight(env_var)
    gbps = weight_bytes / (median_ms * 1e-3) / 1e9
    speedup = fp16_ms / median_ms
    return (
        f"  {name:<8} batch={batch:>3}  "
        f"{median_ms:7.3f} ms   "
        f"{tflops:6.2f} TFLOP/s   "
        f"{gbps:7.1f} GB/s   "
        f"{speedup:5.2f}x"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int, default=DEFAULT_M, help="vocab size")
    parser.add_argument("--K", type=int, default=DEFAULT_K, help="hidden dim")
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCHES),
        help="batch sizes to sweep",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["fp16", "w8a16", "w8a8", "fp8", "mxfp8", "mxfp4"],
        choices=["fp16", "w8a16", "w8a8", "fp8", "mxfp8", "mxfp4"],
        help="modes to bench",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    cap = torch.cuda.get_device_capability(device)
    is_thor = cap[0] >= 10
    name = torch.cuda.get_device_name(device)
    print(f"GPU: {name} (cc={cap[0]}.{cap[1]}, is_thor={is_thor})")
    print(f"Shape: M={args.M}, K={args.K}    Batches: {args.batches}")
    print(f"Iters: {AUTOTUNE_TRIGGER_ITERS} autotune-trigger + {WARMUP_ITERS} warmup + {TIMED_ITERS} timed (median reported)")
    print()

    # Synthetic fp16 weights at the target shape. We only care about speed,
    # not output values, so random data is fine.
    torch.manual_seed(0)
    w = torch.randn(args.M, args.K, device=device, dtype=torch.float16) * 0.02

    mode_to_env = {
        "fp16":   "fp16",  # special-cased
        "w8a16":  "VLLM_USE_INT8_LMHEAD",
        "w8a8":   "VLLM_USE_W8A8_LMHEAD",
        "fp8":    "VLLM_USE_FP8_LMHEAD",
        "mxfp8":  "VLLM_USE_MXFP8_LMHEAD",
        "mxfp4":  "VLLM_USE_MXFP4_LMHEAD",
    }

    thor_only = {"w8a8", "fp8", "mxfp8", "mxfp4"}

    print("=" * 78)
    print(f"  {'mode':<8} {'batch':>9}  {'median':>9}   {'compute':>14}   {'wt BW':>9}   {'vs fp16':>7}")
    print("=" * 78)

    # Bench per batch, fp16 first so we have the baseline.
    for batch in args.batches:
        fp16_ms = None
        for mode in args.modes:
            if mode in thor_only and not is_thor:
                print(f"  {mode:<8} batch={batch:>3}  (skipped: requires Thor sm_110+)")
                continue
            env_var = mode_to_env[mode]
            t0 = time.time()
            if mode == "fp16":
                median, mn, mx = _bench_fp16(w, batch)
                fp16_ms = median
            else:
                median, mn, mx = _bench_mode(env_var, w, batch)
            elapsed = time.time() - t0
            ref_ms = fp16_ms if fp16_ms is not None else median
            print(_fmt_row(mode, batch, median, ref_ms, args.M, args.K, env_var)
                  + f"   ({elapsed:5.1f}s wall)")
        print("-" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
