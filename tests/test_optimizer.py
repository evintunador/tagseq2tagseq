"""
Tests for optimizers/muon.py and kernels/polar_express.py.

Coverage:
  - Triton kernels: XXT, XTX, ba_plus_cAA, polar_express
  - NorMuon variance reduction: second-moment update, Frobenius norm preservation
  - Muon cautious WD: sign-misaligned suppression (BF16 and FP32 paths)
  - BF16 mantissa tracking: sub-ULP accumulation
  - Adam cautious WD: sign-misaligned suppression
  - SingleDeviceMuonWithAuxAdam integration: params update, FP32 path
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from optimizers.muon import (
    _apply_normuon_variance_reduction,
    _muon_cautious_update_inplace,
    _muon_cautious_update_fp,
    _adam_update_step,
    SingleDeviceMuonWithAuxAdam,
)

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


# ===========================================================================
# TestTritonKernels
# ===========================================================================


@cuda_required
class TestTritonKernels:
    """Tests for XXT, XTX, ba_plus_cAA, and polar_express Triton kernels."""

    def test_xxt_matches_matmul_2d(self):
        from kernels.polar_express import XXT

        A = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")
        out = torch.empty(256, 256, dtype=torch.bfloat16, device="cuda")
        XXT(A, out)
        ref = A.float() @ A.float().mT
        torch.testing.assert_close(
            out.float(), ref, atol=1e-2, rtol=1e-1,
            msg="XXT 2D does not match A @ A.mT"
        )

    def test_xxt_matches_matmul_batched(self):
        from kernels.polar_express import XXT

        A = torch.randn(3, 256, 512, dtype=torch.bfloat16, device="cuda")
        out = torch.empty(3, 256, 256, dtype=torch.bfloat16, device="cuda")
        XXT(A, out)
        ref = A.float() @ A.float().mT
        torch.testing.assert_close(
            out.float(), ref, atol=1e-2, rtol=1e-1,
            msg="XXT batched does not match A @ A.mT"
        )

    def test_xtx_matches_matmul_2d(self):
        from kernels.polar_express import XTX

        A = torch.randn(512, 256, dtype=torch.bfloat16, device="cuda")
        out = torch.empty(256, 256, dtype=torch.bfloat16, device="cuda")
        XTX(A, out)
        ref = A.float().mT @ A.float()
        torch.testing.assert_close(
            out.float(), ref, atol=1e-2, rtol=1e-1,
            msg="XTX 2D does not match A.mT @ A"
        )

    def test_xtx_matches_matmul_batched(self):
        from kernels.polar_express import XTX

        A = torch.randn(3, 512, 256, dtype=torch.bfloat16, device="cuda")
        out = torch.empty(3, 256, 256, dtype=torch.bfloat16, device="cuda")
        XTX(A, out)
        ref = A.float().mT @ A.float()
        torch.testing.assert_close(
            out.float(), ref, atol=1e-2, rtol=1e-1,
            msg="XTX batched does not match A.mT @ A"
        )

    def test_ba_plus_caa_matches_formula_2d(self):
        from kernels.polar_express import ba_plus_cAA

        # ba_plus_cAA always receives a symmetric matrix in practice (output of XXT/XTX),
        # so we construct a symmetric input via B @ B.T.
        B = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")
        A = (B.float() @ B.float().mT).bfloat16()
        out = torch.empty(256, 256, dtype=torch.bfloat16, device="cuda")
        alpha, beta = 0.5, 2.0
        ba_plus_cAA(A, alpha=alpha, beta=beta, out=out)
        # For symmetric A: A @ A.T == A @ A == A^2
        A_f = A.float()
        ref = (alpha * (A_f @ A_f.mT) + beta * A_f).bfloat16()
        torch.testing.assert_close(
            out, ref, atol=1e-2, rtol=1e-1,
            msg="ba_plus_cAA 2D does not match alpha*(A@A.T) + beta*A"
        )

    def test_ba_plus_caa_matches_formula_batched(self):
        from kernels.polar_express import ba_plus_cAA

        # Symmetric input (as used in polar_express) for each batch element.
        B = torch.randn(3, 256, 512, dtype=torch.bfloat16, device="cuda")
        A = (B.float() @ B.float().mT).bfloat16()
        out = torch.empty(3, 256, 256, dtype=torch.bfloat16, device="cuda")
        alpha, beta = 0.5, 2.0
        ba_plus_cAA(A, alpha=alpha, beta=beta, out=out)
        A_f = A.float()
        ref = (alpha * (A_f @ A_f.mT) + beta * A_f).bfloat16()
        torch.testing.assert_close(
            out, ref, atol=1e-2, rtol=1e-1,
            msg="ba_plus_cAA batched does not match alpha*(A@A.T) + beta*A"
        )

    def test_polar_express_is_approximately_orthogonal(self):
        from kernels.polar_express import polar_express

        torch.manual_seed(42)
        grad = torch.randn(1024, 512, dtype=torch.float32, device="cuda")
        momentum_buf = torch.zeros_like(grad)
        momentum_t = torch.tensor(0.95)

        X = polar_express(grad, momentum_buf, momentum_t)

        # X should be BF16 and approximately have orthonormal rows scaled
        # uniformly: X @ X.T ≈ c * I  (c = spectral-norm scale)
        assert X.dtype == torch.bfloat16
        XtX = X.float() @ X.float().mT  # (1024, 1024)
        diag = torch.diagonal(XtX, dim1=-2, dim2=-1)

        # Off-diagonal entries should be near zero relative to diagonal scale
        mask = ~torch.eye(XtX.size(-1), dtype=torch.bool, device="cuda")
        off_diag_max = XtX[mask].abs().max().item()

        # Low variance in singular values: std/mean < 10%
        assert diag.std() / diag.mean() < 0.1, (
            f"Diagonal variance too high: std/mean = {diag.std() / diag.mean():.4f}"
        )
        assert off_diag_max < 0.1, (
            f"Off-diagonal max too large: {off_diag_max:.4f}"
        )


# ===========================================================================
# TestVarianceReduction
# ===========================================================================


class TestVarianceReduction:
    """Tests for _apply_normuon_variance_reduction."""

    def test_second_moment_buffer_updates(self):
        v_chunk = torch.randn(64, 128, dtype=torch.float32)
        second_momentum_buffer = torch.zeros(64, 1, dtype=torch.float32)
        buf_before = second_momentum_buffer.clone()

        _apply_normuon_variance_reduction(v_chunk.clone(), second_momentum_buffer, beta2=0.95, red_dim=-1)
        buf_after_1 = second_momentum_buffer.clone()

        _apply_normuon_variance_reduction(v_chunk.clone(), second_momentum_buffer, beta2=0.95, red_dim=-1)
        buf_after_2 = second_momentum_buffer.clone()

        # Buffer should change after first call
        assert not torch.allclose(buf_before, buf_after_1), \
            "second_momentum_buffer did not update after first call"
        # Buffer should change again after second call (EMA keeps moving)
        assert not torch.allclose(buf_after_1, buf_after_2), \
            "second_momentum_buffer did not update after second call"

    def test_frobenius_norm_preserved(self):
        torch.manual_seed(7)
        v_chunk = torch.randn(64, 128, dtype=torch.float32)
        second_momentum_buffer = torch.zeros(64, 1, dtype=torch.float32)

        norm_before = torch.linalg.norm(v_chunk).item()
        v_out = _apply_normuon_variance_reduction(v_chunk, second_momentum_buffer, beta2=0.95, red_dim=-1)
        norm_after = torch.linalg.norm(v_out).item()

        rel_err = abs(norm_before - norm_after) / max(norm_before, 1e-8)
        assert rel_err < 0.01, (
            f"Frobenius norm changed by {rel_err:.4%}: {norm_before:.4f} → {norm_after:.4f}"
        )


# ===========================================================================
# TestMuonCautiousUpdate
# ===========================================================================


class TestMuonCautiousUpdate:
    """Tests for _muon_cautious_update_inplace and _muon_cautious_update_fp."""

    def test_wd_suppressed_when_sign_misaligned_bf16(self):
        """When grad*p < 0 everywhere, WD must not fire (BF16 path)."""
        # p > 0, grad < 0  →  grad * p < 0  →  mask = False  →  no WD
        p_val = torch.full((4, 4), 2.0, dtype=torch.bfloat16)
        grad_val = torch.full((4, 4), -0.1, dtype=torch.float32)

        # Reference: no-WD update
        p_ref = p_val.clone()
        lr, wd = 0.01, 1.0  # large wd to make WD impact obvious

        # Manual no-WD update (expected result)
        p_ref_f32 = p_ref.float()
        p_ref_updated = p_ref_f32 - lr * grad_val.float()

        # Run the cautious update
        p_test = p_val.clone()
        mantissa = torch.zeros(p_test.shape, dtype=torch.uint16)
        lr_t = torch.tensor(lr)
        wd_t = torch.tensor(wd)
        _muon_cautious_update_inplace(
            p_test.view(torch.uint16), mantissa, grad_val, lr_t, wd_t
        )

        # Reconstruct full fp32 from the uint16 high bits + mantissa.
        # Uses numpy for the 32-bit shift since PyTorch CPU doesn't support << on uint32.
        hi = p_test.view(torch.uint16).numpy().astype(np.uint32) << 16
        lo = mantissa.numpy().astype(np.uint32)
        p_reconstructed = torch.from_numpy((hi | lo).view(np.float32))

        torch.testing.assert_close(
            p_reconstructed, p_ref_updated,
            atol=1e-3, rtol=1e-2,
            msg="WD was applied even though grad*p < 0 (BF16 path)"
        )

    def test_wd_suppressed_when_sign_misaligned_fp(self):
        """When grad*p < 0 everywhere, WD must not fire (FP32 path)."""
        p = torch.full((4, 4), 2.0, dtype=torch.float32)
        grad = torch.full((4, 4), -0.1, dtype=torch.float32)
        lr, wd = 0.01, 1.0

        p_ref = p.clone() - lr * grad  # no-WD reference

        _muon_cautious_update_fp(p, grad, lr, wd)

        torch.testing.assert_close(
            p, p_ref, atol=1e-5, rtol=1e-4,
            msg="WD was applied even though grad*p < 0 (FP32 path)"
        )

    def test_mantissa_accumulates_precision(self):
        """Sub-ULP updates should accumulate via the mantissa buffer."""
        # BF16 ULP at 1.0 is 1/128 ≈ 0.0078; use grad=0.0001 (well below ULP)
        p_val = torch.tensor([[1.0]], dtype=torch.bfloat16)
        mantissa = torch.zeros(p_val.shape, dtype=torch.uint16)
        lr_t = torch.tensor(1.0)
        wd_t = torch.tensor(0.0)
        grad = torch.tensor([[0.0001]], dtype=torch.float32)

        n_steps = 100
        for _ in range(n_steps):
            _muon_cautious_update_inplace(
                p_val.view(torch.uint16), mantissa, grad, lr_t, wd_t
            )

        # Reconstruct fp32 value via numpy (CPU uint32 shift not supported in torch).
        hi = p_val.view(torch.uint16).numpy().astype(np.uint32) << 16
        lo = mantissa.numpy().astype(np.uint32)
        p_reconstructed = torch.from_numpy((hi | lo).view(np.float32)).squeeze()

        expected = 1.0 - n_steps * 0.0001  # = 0.99
        # Allow one BF16 ULP tolerance (≈ 0.0078 at magnitude ~1)
        bf16_ulp = 1.0 / 128
        assert abs(p_reconstructed.item() - expected) < bf16_ulp, (
            f"Mantissa tracking failed: expected ≈{expected:.4f}, "
            f"got {p_reconstructed.item():.4f} "
            f"(diff={abs(p_reconstructed.item() - expected):.4f}, "
            f"BF16 ULP={bf16_ulp:.4f})"
        )


# ===========================================================================
# TestAdamCautiousUpdate
# ===========================================================================


class TestAdamCautiousUpdate:
    """Tests for _adam_update_step cautious WD."""

    def test_wd_suppressed_when_sign_misaligned(self):
        """When update*p < 0, WD term must not be added."""
        # We'll set up state so the Adam update is negative while p > 0,
        # making update * p < 0  →  WD should be suppressed.
        torch.manual_seed(0)
        size = (4, 4)
        p = torch.full(size, 2.0)
        # To guarantee a negative update from Adam: set exp_avg negative
        exp_avg = torch.full(size, -0.5)
        exp_avg_sq = torch.full(size, 0.25)  # sqrt = 0.5, so update = -1.0 * step_size
        beta1, beta2, eps = 0.9, 0.95, 1e-10

        # Bias-corrected step: use t=1
        t = 1
        step_size = 1.0 * ((1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t))
        eff_wd = 0.5  # large WD to make its effect obvious if applied

        step_size_t = torch.tensor(step_size)
        eff_wd_t = torch.tensor(eff_wd)

        # Run with WD
        p_wd = p.clone()
        exp_avg_wd = exp_avg.clone()
        exp_avg_sq_wd = exp_avg_sq.clone()
        # Provide a dummy grad that matches what Adam would expect
        g = torch.zeros(size)  # grad=0 so all update comes from exp_avg state
        _adam_update_step(p_wd, g, exp_avg_wd, exp_avg_sq_wd,
                          beta1, beta2, eps, step_size_t, eff_wd_t)

        # Run with eff_wd=0 (reference: no WD)
        eff_wd_zero_t = torch.tensor(0.0)
        p_no_wd = p.clone()
        exp_avg_no = exp_avg.clone()
        exp_avg_sq_no = exp_avg_sq.clone()
        _adam_update_step(p_no_wd, g, exp_avg_no, exp_avg_sq_no,
                          beta1, beta2, eps, step_size_t, eff_wd_zero_t)

        # If WD was suppressed (update*p < 0), both should be equal
        torch.testing.assert_close(
            p_wd, p_no_wd, atol=1e-5, rtol=1e-4,
            msg="Adam WD was applied even though update*p < 0"
        )


# ===========================================================================
# TestSingleDeviceOptimizer
# ===========================================================================


@cuda_required
class TestSingleDeviceOptimizer:
    """Integration tests for SingleDeviceMuonWithAuxAdam (requires CUDA for polar_express Triton kernels)."""

    def _make_model(self, device="cuda"):
        torch.manual_seed(1)
        layer = nn.Linear(64, 32, bias=True, device=device)
        return layer

    def test_muon_step_updates_params(self):
        layer = self._make_model()
        w0 = layer.weight.data.clone()
        b0 = layer.bias.data.clone()

        optimizer = SingleDeviceMuonWithAuxAdam([
            {"params": [layer.weight], "use_muon": True,
             "lr": 0.01, "momentum": 0.95, "weight_decay": 0.01, "beta2": 0.95},
            {"params": [layer.bias], "use_muon": False,
             "lr": 3e-4, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.0},
        ])

        for _ in range(3):
            optimizer.zero_grad()
            x = torch.randn(8, 64, device="cuda")
            loss = layer(x).pow(2).mean()
            loss.backward()
            optimizer.step()

        assert not torch.allclose(layer.weight.data, w0), \
            "Muon weight param did not update after 3 steps"
        assert not torch.allclose(layer.bias.data, b0), \
            "Adam bias param did not update after 3 steps"

    def test_muon_params_fp32_no_crash(self):
        """FP32 params should go through _muon_cautious_update_fp without error."""
        layer = self._make_model()
        assert layer.weight.dtype == torch.float32

        optimizer = SingleDeviceMuonWithAuxAdam([
            {"params": [layer.weight], "use_muon": True,
             "lr": 0.01, "momentum": 0.95, "weight_decay": 0.0, "beta2": 0.95},
        ])

        optimizer.zero_grad()
        x = torch.randn(8, 64, device="cuda")
        loss = layer(x).pow(2).mean()
        loss.backward()
        optimizer.step()  # must not raise
