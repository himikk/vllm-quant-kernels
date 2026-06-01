"""Registration of quantized kernel replacements with vLLM."""

import os

from vllm.model_executor.custom_op import PluggableLayer

_ENV_INT8   = "VLLM_USE_INT8_LMHEAD"
_ENV_W8A8   = "VLLM_USE_W8A8_LMHEAD"
_ENV_FP8    = "VLLM_USE_FP8_LMHEAD"
_ENV_MXFP8  = "VLLM_USE_MXFP8_LMHEAD"
_ENV_MXFP4  = "VLLM_USE_MXFP4_LMHEAD"
_ENV_SAMPLER = "VLLM_USE_FUSED_SAMPLER"


def _env_bool(name: str, default: bool = False) -> bool:
    """Check if an environment variable is set to a truthy value."""
    val = os.environ.get(name, "")
    if val == "":
        return default
    return val.lower() in ("1", "true", "yes", "on")


def _silence_unknown_env_warning() -> None:
    """Add our env vars to vLLM's whitelist to suppress the 'Unknown' warning."""
    try:
        import vllm.envs as vllm_envs
        for var in (_ENV_INT8, _ENV_W8A8, _ENV_FP8, _ENV_MXFP8, _ENV_MXFP4, _ENV_SAMPLER):
            if var not in vllm_envs.environment_variables:
                vllm_envs.environment_variables[var] = lambda: os.environ.get(var, "")
    except Exception:
        pass  # non-fatal: vLLM version may not have this dict


def register_all() -> None:
    """Register all OOT replacements if an env var is set."""
    import sys
    _silence_unknown_env_warning()

    use_mxfp4   = _env_bool(_ENV_MXFP4)
    use_mxfp8   = _env_bool(_ENV_MXFP8)
    use_fp8     = _env_bool(_ENV_FP8)
    use_w8a8    = _env_bool(_ENV_W8A8)
    use_int8    = _env_bool(_ENV_INT8)
    use_sampler = _env_bool(_ENV_SAMPLER)

    if (not use_mxfp4 and not use_mxfp8 and not use_fp8 and not use_w8a8
            and not use_int8 and not use_sampler):
        print(
            f"[vllm-quant-kernels] Plugin loaded but disabled "
            f"(set {_ENV_MXFP4}=1, {_ENV_MXFP8}=1, {_ENV_FP8}=1, {_ENV_W8A8}=1, "
            f"{_ENV_INT8}=1 to enable quantized LM head, or {_ENV_SAMPLER}=1 to "
            f"enable the fused sampler).",
            file=sys.stderr, flush=True,
        )
        return

    # Quantized lm_head — priority: mxfp4 > mxfp8 > fp8 > w8a8 > int8.
    if use_mxfp4 or use_mxfp8 or use_fp8 or use_w8a8 or use_int8:
        if use_mxfp4:
            quant_dtype = "mxfp4"
        elif use_mxfp8:
            quant_dtype = "mxfp8"
        elif use_fp8:
            quant_dtype = "fp8"
        elif use_w8a8:
            quant_dtype = "w8a8"
        else:
            quant_dtype = "int8"

        from ._quant import QuantizedLogitsProcessor  # noqa: F401
        print(
            f"[vllm-quant-kernels] Registered QuantizedLogitsProcessor as OOT "
            f"replacement for LogitsProcessor (dtype={quant_dtype}).",
            file=sys.stderr, flush=True,
        )

    # Fused sampler — patches vllm.v1.sample.sampler.Sampler. Independent of
    # the lm_head quantization flags; the two can be combined freely.
    if use_sampler:
        from ._sampler import install_fused_sampler
        install_fused_sampler()
