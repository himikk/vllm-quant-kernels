"""Extract the lm_head tensor from Intel/Qwen3.5-122B-A10B-int4-AutoRound.

This produces the file `qwen3p5_122b_lm_head.safetensors` next to this script.
The file is gitignored because it is 1.5 GB; tests that need it skip gracefully
when it is missing.

Run from a host where the model is cached in the HF hub directory.
"""

from __future__ import annotations

import os
from pathlib import Path

HF_HUB = Path(os.environ.get("HF_HUB", os.path.expanduser("~/.cache/huggingface/hub")))
MODEL = "models--Intel--Qwen3.5-122B-A10B-int4-AutoRound"
SHARD = "model-00014-of-00014.safetensors"


def main() -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    snapshots_dir = HF_HUB / MODEL / "snapshots"
    if not snapshots_dir.exists():
        raise SystemExit(f"Model snapshots directory not found: {snapshots_dir}")

    snapshot_dirs = list(snapshots_dir.iterdir())
    if not snapshot_dirs:
        raise SystemExit(f"No snapshots inside {snapshots_dir}")

    src = snapshot_dirs[0] / SHARD
    if not src.exists():
        raise SystemExit(f"Expected shard not found: {src}")

    dst = Path(__file__).parent / "qwen3p5_122b_lm_head.safetensors"

    with safe_open(src, framework="pt") as f:
        t = f.get_tensor("lm_head.weight")

    print(
        f"Extracted lm_head.weight  shape={tuple(t.shape)}  dtype={t.dtype}  "
        f"{t.numel() * t.element_size() / 1024 ** 2:.1f} MB"
    )

    save_file({"lm_head.weight": t}, str(dst))
    print(f"Saved to {dst}")


if __name__ == "__main__":
    main()
