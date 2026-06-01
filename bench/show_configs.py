"""Print the autotune-selected config for each Thor kernel at each batch size.

Use after running the speed benchmark — the Triton in-process cache holds the
winning configs and exposes them via the autotuner's `cache` dict.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch


M = 248320
K = 3072
BATCHES = (1, 2, 4, 8, 16, 32)


class _FakeWeight:
    def __init__(self, t):
        self.data = t
        self.dtype = t.dtype
        self.shape = t.shape
        self.device = t.device


def _make_fake_lm_head(w):
    return SimpleNamespace(weight=_FakeWeight(w))


def _run_one(env_var: str, w: torch.Tensor, batch: int):
    from vllm_quant_kernels._quant import QuantizedLogitsProcessor

    os.environ[env_var] = "1"
    try:
        lm_head = _make_fake_lm_head(w)
        proc = QuantizedLogitsProcessor(w.shape[0])
        proc._init_int8(lm_head)
        proc._int8_initialized = True
        x = torch.randn(batch, K, device="cuda", dtype=torch.float16)
        # Run once to trigger autotune
        proc._quantized_forward(x, lm_head, None)
        torch.cuda.synchronize()
        return lm_head
    finally:
        os.environ.pop(env_var, None)


def _print_kernel_cache(kernel_attr_name: str, lm_head, mode: str, batch: int):
    kernel = getattr(lm_head, kernel_attr_name, None)
    if kernel is None:
        print(f"  {mode} batch={batch}: no kernel attr {kernel_attr_name}")
        return
    cache = getattr(kernel, "cache", {})
    for key, cfg in cache.items():
        print(f"  {mode:<6} batch={batch:>3}  key={key}  -> {cfg}")


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 1
    cap = torch.cuda.get_device_capability(0)
    if cap[0] < 10:
        print(f"Not a Thor device (cc={cap[0]}.{cap[1]}); nothing to show.")
        return 0

    torch.manual_seed(0)
    w = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.02

    modes = [
        ("w8a16", "VLLM_USE_INT8_LMHEAD", "_int8_gemm_kernel"),
        ("w8a8",  "VLLM_USE_W8A8_LMHEAD", "_w8a8_gemm_kernel"),
        ("fp8",   "VLLM_USE_FP8_LMHEAD",  "_fp8_gemm_kernel"),
        ("mxfp8", "VLLM_USE_MXFP8_LMHEAD", "_mxfp8_gemm_kernel"),
    ]

    print(f"M={M}, K={K}")
    print("=" * 110)

    for mode, env, attr in modes:
        for batch in BATCHES:
            if mode == "w8a16" and batch == 1:
                # w8a16 batch=1 uses the int8_gmv kernel, not the 2D one
                continue
            lm_head = _run_one(env, w, batch)
            _print_kernel_cache(attr, lm_head, mode, batch)
        print("-" * 110)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
