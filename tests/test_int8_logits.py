"""Tests for vllm-quant-kernels package."""

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


class TestInt8GmvKernel:
    """Tests for the int8_gmv batched GEMV kernel."""

    def test_kernel_exists(self):
        """Test that int8_gmv kernel is importable."""
        from vllm_quant_kernels._kernels import int8_gmv

        assert int8_gmv is not None

    def test_kernel_shape(self):
        """Test int8_gmv produces correct output shape."""
        from vllm_quant_kernels._kernels import int8_gmv

        M, K = 32000, 4096
        batch = 2

        w_int8 = torch.randint(-128, 127, (M, K), device="cuda", dtype=torch.int8)
        x_fp16 = torch.randn(batch, K, device="cuda", dtype=torch.float16)
        scales = torch.randn(M, device="cuda", dtype=torch.float16).clamp(min=1e-12)
        out = torch.empty(batch, M, device="cuda", dtype=torch.float16)

        grid = lambda meta: ((M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],)
        int8_gmv[grid](
            out,
            w_int8,
            x_fp16,
            scales,
            M,
            K,
            out.stride(0),
            x_fp16.stride(0),
            NUM_BATCH=batch,
        )

        assert out.shape == (batch, M)
        assert out.dtype == torch.float16
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_kernel_batch_1(self):
        """Test int8_gmv with batch size 1."""
        from vllm_quant_kernels._kernels import int8_gmv

        M, K = 32000, 4096
        batch = 1

        w_int8 = torch.randint(-128, 127, (M, K), device="cuda", dtype=torch.int8)
        x_fp16 = torch.randn(batch, K, device="cuda", dtype=torch.float16)
        scales = torch.randn(M, device="cuda", dtype=torch.float16).clamp(min=1e-12)
        out = torch.empty(batch, M, device="cuda", dtype=torch.float16)

        grid = lambda meta: ((M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],)
        int8_gmv[grid](
            out,
            w_int8,
            x_fp16,
            scales,
            M,
            K,
            out.stride(0),
            x_fp16.stride(0),
            NUM_BATCH=batch,
        )

        assert out.shape == (batch, M)
        assert not torch.isnan(out).any()


class TestQuantizedLogitsProcessor:
    """Tests for QuantizedLogitsProcessor OOT replacement."""

    def test_class_importable(self):
        """Test that QuantizedLogitsProcessor is importable."""
        from vllm_quant_kernels._quant import QuantizedLogitsProcessor

        assert QuantizedLogitsProcessor is not None

    def test_inherits_from_logits_processor(self):
        """Test that QuantizedLogitsProcessor inherits from LogitsProcessor."""
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm_quant_kernels._quant import QuantizedLogitsProcessor

        assert issubclass(QuantizedLogitsProcessor, LogitsProcessor)

    def test_quantization_skips_small_vocab(self):
        """Test that small vocab models skip quantization."""
        import os
        os.environ["VLLM_USE_INT8_LMHEAD"] = "1"

        from vllm_quant_kernels._registration import register_all
        from vllm_quant_kernels._quant import QuantizedLogitsProcessor

        # Register (should not raise)
        register_all()

        processor = QuantizedLogitsProcessor(32000)
        assert processor._VOCAB_THRESHOLD == 100_000

    def test_quantization_skips_non_fp(self):
        """Test that non-FP16/BF16 weights are not quantized."""
        import os
        os.environ["VLLM_USE_INT8_LMHEAD"] = "1"

        from vllm_quant_kernels._registration import register_all
        from vllm_quant_kernels._quant import QuantizedLogitsProcessor

        register_all()

        processor = QuantizedLogitsProcessor(32000)

        # Create a mock lm_head with int8 weights
        class MockLmHead:
            def __init__(self):
                self.weight = torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0]],
                    device="cuda",
                    dtype=torch.int8,
                )

        mock_lm_head = MockLmHead()
        processor._init_int8(mock_lm_head)

        # Should not have _int8_w for int8 input
        assert not hasattr(mock_lm_head, "_int8_w")


class TestInt8Precision:
    """Compare INT8 quantized lm_head output against fp16 cuBLAS reference.

    Uses the real Qwen3.5-122B lm_head shape (248320 × 3072) to exercise
    the exact autotune key that matters in production.
    """

    M = 248_320  # Qwen3.5-122B vocab size (from run log)
    K = 3_072    # Qwen3.5-122B hidden dim

    @pytest.fixture(scope="class")
    def quant_lm_head(self):
        """Quantize a random fp16 weight once; share across both tests."""
        from types import SimpleNamespace
        from vllm_quant_kernels._quant import QuantizedLogitsProcessor

        w_fp16 = torch.randn(self.M, self.K, device="cuda", dtype=torch.float16)

        class _FakeWeight:
            def __init__(self, t):
                self.data = t
                self.dtype = t.dtype
                self.shape = t.shape
                self.device = t.device

        lm_head = SimpleNamespace(weight=_FakeWeight(w_fp16))

        processor = QuantizedLogitsProcessor(self.M)
        processor._init_int8(lm_head)

        assert hasattr(lm_head, "_int8_w"), "quantization should have triggered"
        return lm_head, w_fp16

    def _precision_check(self, quant_lm_head, batch):
        import torch.nn.functional as F
        from vllm_quant_kernels._quant import QuantizedLogitsProcessor

        lm_head, w_fp16 = quant_lm_head
        x = torch.randn(batch, self.K, device="cuda", dtype=torch.float16)

        # Reference: fp16 cuBLAS
        ref = F.linear(x, w_fp16)  # (batch, M)

        # Quantized path — fresh processor, skip re-init
        processor = QuantizedLogitsProcessor(self.M)
        processor._int8_initialized = True
        out = processor._quantized_forward(x, lm_head, None)  # (batch, M)

        diff = (out.float() - ref.float()).abs()
        max_err  = diff.max().item()
        mean_err = diff.mean().item()
        ref_mag  = ref.float().abs().mean().item()
        rel_err  = mean_err / (ref_mag + 1e-8)
        cos_sim  = F.cosine_similarity(
            out.float().reshape(batch, -1),
            ref.float().reshape(batch, -1),
            dim=1,
        ).mean().item()

        print(
            f"\nbatch={batch}  max_abs={max_err:.4f}  mean_abs={mean_err:.4f}"
            f"  rel={rel_err * 100:.3f}%  cos_sim={cos_sim:.6f}"
        )
        return max_err, rel_err, cos_sim

    def test_precision_batch1(self, quant_lm_head):
        """Single-token decode path (NUM_BATCH=1 branch).

        Observed on Thor/sm_110: max_abs~2.5, rel~0.85%, cos_sim~0.999964.
        Tolerances have ~40% headroom above observed values.
        """
        max_err, rel_err, cos_sim = self._precision_check(quant_lm_head, batch=1)
        assert max_err < 3.5,   f"max abs error too large: {max_err:.4f}"
        assert rel_err < 0.02,  f"mean relative error too large: {rel_err * 100:.3f}%"
        assert cos_sim > 0.9999, f"cosine similarity too low: {cos_sim:.6f}"

    def test_precision_batch4(self, quant_lm_head):
        """4-token fused decode path (NUM_BATCH=4, all 4 accumulators active).

        Observed on Thor/sm_110: max_abs~2.56, rel~0.85%, cos_sim~0.999964.
        Tolerances have ~40% headroom above observed values.
        """
        max_err, rel_err, cos_sim = self._precision_check(quant_lm_head, batch=4)
        assert max_err < 3.5,   f"max abs error too large: {max_err:.4f}"
        assert rel_err < 0.02,  f"mean relative error too large: {rel_err * 100:.3f}%"
        assert cos_sim > 0.9999, f"cosine similarity too low: {cos_sim:.6f}"


class TestMxfp8Precision:
    """Compare MXFP8 (e4m3 + per-32-element e8m0 scales) against fp16 cuBLAS reference.

    Only runs on Thor (sm_110+) — tl.dot_scaled maps to tcgen05.mma MX format.
    Uses the real Qwen3.5-122B lm_head shape (248320 × 3072).

    For random-Gaussian-distributed weights (this test), MXFP8 gives similar accuracy
    to plain FP8: the e8m0 group scale rounds up to the next power of 2, adding ~30%
    overhead per group that offsets the finer group granularity. The MX format wins
    on weight distributions with highly heterogeneous within-row dynamic range.
    """

    M = 248_320
    K = 3_072

    @pytest.fixture(scope="class")
    def quant_lm_head_mxfp8(self):
        import os
        from types import SimpleNamespace
        from vllm_quant_kernels._quant import QuantizedLogitsProcessor

        cap = torch.cuda.get_device_capability()
        if cap[0] < 10:
            pytest.skip("MXFP8 kernel requires Thor (sm_110+)")

        old = os.environ.get("VLLM_USE_MXFP8_LMHEAD")
        os.environ["VLLM_USE_MXFP8_LMHEAD"] = "1"

        w_fp16 = torch.randn(self.M, self.K, device="cuda", dtype=torch.float16)

        class _FakeWeight:
            def __init__(self, t):
                self.data = t
                self.dtype = t.dtype
                self.shape = t.shape
                self.device = t.device

        lm_head = SimpleNamespace(weight=_FakeWeight(w_fp16))

        processor = QuantizedLogitsProcessor(self.M)
        processor._init_int8(lm_head)

        if old is None:
            del os.environ["VLLM_USE_MXFP8_LMHEAD"]
        else:
            os.environ["VLLM_USE_MXFP8_LMHEAD"] = old

        assert hasattr(lm_head, "_mxfp8_w"), "mxfp8 quantization should have triggered"
        assert hasattr(lm_head, "_mxfp8_w_scales")
        return lm_head, w_fp16

    def _precision_check(self, quant_lm_head_mxfp8, batch):
        import torch.nn.functional as F
        from vllm_quant_kernels._quant import QuantizedLogitsProcessor

        lm_head, w_fp16 = quant_lm_head_mxfp8
        x = torch.randn(batch, self.K, device="cuda", dtype=torch.float16)
        ref = F.linear(x, w_fp16)

        processor = QuantizedLogitsProcessor(self.M)
        processor._int8_initialized = True
        out = processor._quantized_forward(x, lm_head, None)

        diff = (out.float() - ref.float()).abs()
        max_err  = diff.max().item()
        mean_err = diff.mean().item()
        ref_mag  = ref.float().abs().mean().item()
        rel_err  = mean_err / (ref_mag + 1e-8)
        cos_sim  = F.cosine_similarity(
            out.float().reshape(batch, -1),
            ref.float().reshape(batch, -1),
            dim=1,
        ).mean().item()
        print(
            f"\n[mxfp8] batch={batch}  max_abs={max_err:.4f}  mean_abs={mean_err:.4f}"
            f"  rel={rel_err * 100:.3f}%  cos_sim={cos_sim:.6f}"
        )
        return max_err, rel_err, cos_sim

    def test_precision_batch1(self, quant_lm_head_mxfp8):
        """MXFP8 single-token decode."""
        max_err, rel_err, cos_sim = self._precision_check(quant_lm_head_mxfp8, batch=1)
        assert max_err < 20.0,  f"max abs error too large: {max_err:.4f}"
        assert rel_err < 0.06,  f"mean relative error too large: {rel_err * 100:.3f}%"
        assert cos_sim > 0.998, f"cosine similarity too low: {cos_sim:.6f}"

    def test_precision_batch4(self, quant_lm_head_mxfp8):
        """MXFP8 4-token decode."""
        max_err, rel_err, cos_sim = self._precision_check(quant_lm_head_mxfp8, batch=4)
        assert max_err < 20.0,  f"max abs error too large: {max_err:.4f}"
        assert rel_err < 0.06,  f"mean relative error too large: {rel_err * 100:.3f}%"
        assert cos_sim > 0.998, f"cosine similarity too low: {cos_sim:.6f}"


class TestW8A8Precision:
    """Compare W8A8 INT8×INT8 quantized lm_head output against fp16 cuBLAS reference.

    Only runs on Thor (sm_110+) — tcgen05 INT8 tensor cores not available on earlier devices.
    Uses the real Qwen3.5-122B lm_head shape (248320 × 3072).

    W8A8 has two quantization layers (weights per-row, activations per-tensor) so
    expect slightly wider tolerances than W8A16 INT8 (rel ~0.85%).
    """

    M = 248_320
    K = 3_072

    @pytest.fixture(scope="class")
    def quant_lm_head_w8a8(self):
        """Quantize a random fp16 weight to W8A8 INT8 once; share across both tests."""
        import os
        from types import SimpleNamespace
        from vllm_quant_kernels._quant import QuantizedLogitsProcessor

        cap = torch.cuda.get_device_capability()
        if cap[0] < 10:
            pytest.skip("W8A8 kernel requires Thor (sm_110+)")

        old = os.environ.get("VLLM_USE_W8A8_LMHEAD")
        os.environ["VLLM_USE_W8A8_LMHEAD"] = "1"

        w_fp16 = torch.randn(self.M, self.K, device="cuda", dtype=torch.float16)

        class _FakeWeight:
            def __init__(self, t):
                self.data = t
                self.dtype = t.dtype
                self.shape = t.shape
                self.device = t.device

        lm_head = SimpleNamespace(weight=_FakeWeight(w_fp16))

        processor = QuantizedLogitsProcessor(self.M)
        processor._init_int8(lm_head)

        if old is None:
            del os.environ["VLLM_USE_W8A8_LMHEAD"]
        else:
            os.environ["VLLM_USE_W8A8_LMHEAD"] = old

        assert hasattr(lm_head, "_w8a8_w"), "w8a8 quantization should have triggered"
        assert not hasattr(lm_head, "_int8_w"), "should not have w8a16 weights"
        assert not hasattr(lm_head, "_fp8_w"), "should not have fp8 weights"
        return lm_head, w_fp16

    def _precision_check(self, quant_lm_head_w8a8, batch):
        import torch.nn.functional as F
        from vllm_quant_kernels._quant import QuantizedLogitsProcessor

        lm_head, w_fp16 = quant_lm_head_w8a8
        x = torch.randn(batch, self.K, device="cuda", dtype=torch.float16)

        # Reference: fp16 cuBLAS
        ref = F.linear(x, w_fp16)  # (batch, M)

        # W8A8 quantized path
        processor = QuantizedLogitsProcessor(self.M)
        processor._int8_initialized = True
        out = processor._quantized_forward(x, lm_head, None)  # (batch, M)

        diff = (out.float() - ref.float()).abs()
        max_err  = diff.max().item()
        mean_err = diff.mean().item()
        ref_mag  = ref.float().abs().mean().item()
        rel_err  = mean_err / (ref_mag + 1e-8)
        cos_sim  = F.cosine_similarity(
            out.float().reshape(batch, -1),
            ref.float().reshape(batch, -1),
            dim=1,
        ).mean().item()

        print(
            f"\n[w8a8] batch={batch}  max_abs={max_err:.4f}  mean_abs={mean_err:.4f}"
            f"  rel={rel_err * 100:.3f}%  cos_sim={cos_sim:.6f}"
        )
        return max_err, rel_err, cos_sim

    def test_precision_batch1(self, quant_lm_head_w8a8):
        """W8A8 single-token decode path (GEMM with BLOCK_N=1 tile)."""
        max_err, rel_err, cos_sim = self._precision_check(quant_lm_head_w8a8, batch=1)
        assert max_err < 5.0,    f"max abs error too large: {max_err:.4f}"
        assert rel_err < 0.025,  f"mean relative error too large: {rel_err * 100:.3f}%"
        assert cos_sim > 0.9998, f"cosine similarity too low: {cos_sim:.6f}"

    def test_precision_batch2(self, quant_lm_head_w8a8):
        """W8A8 minimum batch=2 (GEMM path, batch=1 falls through to int8_gmv)."""
        max_err, rel_err, cos_sim = self._precision_check(quant_lm_head_w8a8, batch=2)
        assert max_err < 5.0,   f"max abs error too large: {max_err:.4f}"
        assert rel_err < 0.025, f"mean relative error too large: {rel_err * 100:.3f}%"
        assert cos_sim > 0.9998, f"cosine similarity too low: {cos_sim:.6f}"

    def test_precision_batch4(self, quant_lm_head_w8a8):
        """W8A8 batch=4 decode path."""
        max_err, rel_err, cos_sim = self._precision_check(quant_lm_head_w8a8, batch=4)
        assert max_err < 5.0,   f"max abs error too large: {max_err:.4f}"
        assert rel_err < 0.025, f"mean relative error too large: {rel_err * 100:.3f}%"
        assert cos_sim > 0.9998, f"cosine similarity too low: {cos_sim:.6f}"


class TestFp8Precision:
    """Compare FP8 quantized lm_head output against fp16 cuBLAS reference.

    Only runs on Thor (sm_110+) — fp8 tensor cores not available on earlier devices.
    Uses the real Qwen3.5-122B lm_head shape (248320 × 3072).
    """

    M = 248_320
    K = 3_072

    @pytest.fixture(scope="class", autouse=False)
    def set_fp8_env(self):
        import os
        old = os.environ.get("VLLM_USE_FP8_LMHEAD")
        os.environ["VLLM_USE_FP8_LMHEAD"] = "1"
        yield
        if old is None:
            del os.environ["VLLM_USE_FP8_LMHEAD"]
        else:
            os.environ["VLLM_USE_FP8_LMHEAD"] = old

    @pytest.fixture(scope="class")
    def quant_lm_head_fp8(self, set_fp8_env):
        """Quantize a random fp16 weight to fp8 once; share across both tests."""
        from types import SimpleNamespace
        from vllm_quant_kernels._quant import QuantizedLogitsProcessor

        cap = torch.cuda.get_device_capability()
        if cap[0] < 10:
            pytest.skip("FP8 kernel requires Thor (sm_110+)")

        w_fp16 = torch.randn(self.M, self.K, device="cuda", dtype=torch.float16)

        class _FakeWeight:
            def __init__(self, t):
                self.data = t
                self.dtype = t.dtype
                self.shape = t.shape
                self.device = t.device

        lm_head = SimpleNamespace(weight=_FakeWeight(w_fp16))

        processor = QuantizedLogitsProcessor(self.M)
        processor._init_int8(lm_head)

        assert hasattr(lm_head, "_fp8_w"), "fp8 quantization should have triggered"
        assert not hasattr(lm_head, "_int8_w"), "should not have int8 weights"
        return lm_head, w_fp16

    def _precision_check(self, quant_lm_head_fp8, batch):
        import torch.nn.functional as F
        from vllm_quant_kernels._quant import QuantizedLogitsProcessor

        lm_head, w_fp16 = quant_lm_head_fp8
        x = torch.randn(batch, self.K, device="cuda", dtype=torch.float16)

        # Reference: fp16 cuBLAS
        ref = F.linear(x, w_fp16)  # (batch, M)

        # FP8 quantized path
        processor = QuantizedLogitsProcessor(self.M)
        processor._int8_initialized = True
        out = processor._quantized_forward(x, lm_head, None)  # (batch, M)

        diff = (out.float() - ref.float()).abs()
        max_err  = diff.max().item()
        mean_err = diff.mean().item()
        ref_mag  = ref.float().abs().mean().item()
        rel_err  = mean_err / (ref_mag + 1e-8)
        cos_sim  = F.cosine_similarity(
            out.float().reshape(batch, -1),
            ref.float().reshape(batch, -1),
            dim=1,
        ).mean().item()

        print(
            f"\n[fp8] batch={batch}  max_abs={max_err:.4f}  mean_abs={mean_err:.4f}"
            f"  rel={rel_err * 100:.3f}%  cos_sim={cos_sim:.6f}"
        )
        return max_err, rel_err, cos_sim

    def test_precision_batch1(self, quant_lm_head_fp8):
        """FP8 single-token decode path.

        Observed on Thor/sm_110: max_abs~9.8, rel~3.7%, cos_sim~0.9993.
        FP8 e4m3 has 3-bit mantissa so per-element error is ~12.5%;
        errors cancel over K=3072 but not as well as INT8.
        """
        max_err, rel_err, cos_sim = self._precision_check(quant_lm_head_fp8, batch=1)
        assert max_err < 20.0,  f"max abs error too large: {max_err:.4f}"
        assert rel_err < 0.06,  f"mean relative error too large: {rel_err * 100:.3f}%"
        assert cos_sim > 0.998, f"cosine similarity too low: {cos_sim:.6f}"

    def test_precision_batch4(self, quant_lm_head_fp8):
        """FP8 4-token decode path.

        Observed on Thor/sm_110: max_abs~10.8, rel~3.7%, cos_sim~0.9993.
        """
        max_err, rel_err, cos_sim = self._precision_check(quant_lm_head_fp8, batch=4)
        assert max_err < 20.0,  f"max abs error too large: {max_err:.4f}"
        assert rel_err < 0.06,  f"mean relative error too large: {rel_err * 100:.3f}%"
        assert cos_sim > 0.998, f"cosine similarity too low: {cos_sim:.6f}"
