"""Optimized quantized kernels for vLLM."""

__version__ = "0.1.0"


def register() -> None:
    """Register quantized kernel plugins with vLLM."""
    from ._registration import register_all

    register_all()
