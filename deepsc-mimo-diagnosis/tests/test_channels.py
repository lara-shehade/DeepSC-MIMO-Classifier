import math
import unittest

import torch

from channels import (
    MIMOChannel,
    MIMOConfig,
    complex_matrix_to_real_interleaved,
    real_interleaved_to_complex,
)


class ChannelMathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_real_interleaved_round_trip(self):
        original = torch.randn(3, 5, 16)
        complex_values = real_interleaved_to_complex(original, 2)
        restored = torch.view_as_real(complex_values).reshape_as(original)
        torch.testing.assert_close(restored, original)

    def test_real_matrix_matches_native_complex_multiplication(self):
        h = torch.randn(2, 2) + 1j * torch.randn(2, 2)
        x = torch.randn(11, 2) + 1j * torch.randn(11, 2)
        expected = torch.einsum("oi,ui->uo", h, x)
        matrix = complex_matrix_to_real_interleaved(h)
        x_real = torch.view_as_real(x).reshape(11, 4)
        actual_real = x_real @ matrix.T
        actual = torch.view_as_complex(actual_real.reshape(11, 2, 2).contiguous())
        torch.testing.assert_close(actual, expected)

    def test_zf_no_fading_no_noise_recovers_scaled_transmission(self):
        config = MIMOConfig(
            profile="no_fading", equalizer="zf", total_power_constraint=False
        )
        channel = MIMOChannel(config, seed=1)
        transmitted = torch.randn(4, 6, 16)
        received = channel(transmitted, 0.0)
        torch.testing.assert_close(received, transmitted, rtol=1e-5, atol=1e-5)

    def test_total_power_constraint_scales_by_sqrt_nt(self):
        transmitted = torch.randn(2, 3, 16)
        constrained = MIMOChannel(
            MIMOConfig(profile="no_fading", equalizer="zf"), seed=1
        )
        _, trace = constrained(transmitted, 0.0, return_trace=True)
        self.assertAlmostEqual(trace.power_scale, 1.0 / math.sqrt(2.0))
        ratio = trace.tx_scaled.abs().square().mean() / trace.tx_grouped.abs().square().mean()
        self.assertAlmostEqual(float(ratio), 0.5, places=6)

    def test_fading_scope_shapes_are_explicit(self):
        transmitted = torch.randn(3, 4, 16)
        expected = {
            "batch": (1, 1, 2, 2),
            "sample": (3, 1, 2, 2),
            "use": (3, 16, 2, 2),
        }
        for scope, shape in expected.items():
            channel = MIMOChannel(MIMOConfig(fading_scope=scope), seed=3)
            _, trace = channel(transmitted, 0.1, return_trace=True)
            self.assertEqual(tuple(trace.true_h.shape), shape)

    def test_lmmse_matches_direct_native_complex_formula(self):
        channel = MIMOChannel(MIMOConfig(covariance="system"), seed=9)
        transmitted = torch.randn(2, 7, 16)
        _, trace = channel(transmitted, 0.2, return_trace=True)
        h = trace.estimated_h.expand(2, trace.tx_scaled.shape[1], 2, 2)
        rx = trace.assumed_stream_covariance[:, None].expand(2, trace.tx_scaled.shape[1], 2, 2)
        rn = trace.noise_covariance
        expected_filter = rx @ h.mH @ torch.linalg.inv(h @ rx @ h.mH + rn)
        expected = torch.einsum("buni,bui->bun", expected_filter, trace.noisy_signal)
        torch.testing.assert_close(expected, trace.equalized_signal, rtol=2e-5, atol=2e-5)

    def test_channel_is_differentiable(self):
        transmitted = torch.randn(2, 3, 16, requires_grad=True)
        channel = MIMOChannel(MIMOConfig(), seed=5)
        received = channel(transmitted, 0.1)
        received.square().mean().backward()
        self.assertIsNotNone(transmitted.grad)
        self.assertTrue(torch.isfinite(transmitted.grad).all())

    def test_seed_reproducibility(self):
        transmitted = torch.randn(2, 3, 16)
        left = MIMOChannel(MIMOConfig(fading_scope="sample"), seed=123)
        right = MIMOChannel(MIMOConfig(fading_scope="sample"), seed=123)
        output_left, trace_left = left(transmitted, 0.1, return_trace=True)
        output_right, trace_right = right(transmitted, 0.1, return_trace=True)
        torch.testing.assert_close(trace_left.true_h, trace_right.true_h)
        torch.testing.assert_close(trace_left.noise, trace_right.noise)
        torch.testing.assert_close(output_left, output_right)

    def test_realization_sets_are_distinct_and_reproducible(self):
        transmitted = torch.randn(2, 3, 16)
        channel = MIMOChannel(MIMOConfig(fading_scope="sample"), seed=123)
        channel.reset(realization_set=0)
        _, zero = channel(transmitted, 0.1, return_trace=True)
        channel.reset(realization_set=1)
        _, one = channel(transmitted, 0.1, return_trace=True)
        self.assertFalse(torch.allclose(zero.true_h, one.true_h))
        self.assertFalse(torch.allclose(zero.noise, one.noise))
        channel.reset(realization_set=0)
        _, zero_again = channel(transmitted, 0.1, return_trace=True)
        torch.testing.assert_close(zero.true_h, zero_again.true_h)
        torch.testing.assert_close(zero.noise, zero_again.noise)

    def test_approved_imperfect_csi_is_power_normalized_and_paired(self):
        transmitted = torch.randn(64, 16, 16)
        baseline = MIMOChannel(MIMOConfig(fading_scope="use"), seed=911)
        impaired = MIMOChannel(
            MIMOConfig(fading_scope="use", profile="imperfect_csi", csi_rho=0.90),
            seed=911,
        )
        _, clean = baseline(transmitted, 0.1, return_trace=True)
        _, trace = impaired(transmitted, 0.1, return_trace=True)
        torch.testing.assert_close(clean.true_h, trace.true_h)
        torch.testing.assert_close(clean.noise, trace.noise)
        self.assertAlmostEqual(float(trace.estimated_h.abs().square().mean()), 1.0, delta=0.04)
        cross = (trace.estimated_h * trace.true_h.conj()).mean().real
        self.assertAlmostEqual(float(cross), 0.90, delta=0.04)

    def test_approved_spatial_correlation_matches_kronecker_target(self):
        rho = 0.6
        transmitted = torch.randn(64, 16, 16)
        channel = MIMOChannel(
            MIMOConfig(
                fading_scope="use", profile="spatial_correlation", spatial_rho=rho
            ),
            seed=912,
        )
        _, trace = channel(transmitted, 0.1, return_trace=True)
        h = trace.true_h.reshape(-1, 2, 2)
        receive_covariance = (h @ h.mH).mean(dim=0) / 2.0
        transmit_covariance = (h.mH @ h).mean(dim=0) / 2.0
        target = torch.tensor([[1.0, rho], [rho, 1.0]], dtype=h.dtype)
        torch.testing.assert_close(receive_covariance, target, rtol=0.05, atol=0.05)
        torch.testing.assert_close(transmit_covariance, target, rtol=0.05, atol=0.05)
        self.assertAlmostEqual(float(h.abs().square().mean()), 1.0, delta=0.04)


if __name__ == "__main__":
    unittest.main()
