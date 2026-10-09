"""Dependency-light executable checks for local and server smoke testing."""

import math
import tempfile

import torch
import torch.nn as nn

from channels import MIMOChannel, MIMOConfig, complex_matrix_to_real_interleaved
from experiment_data import ExperimentDataWriter
from models.transceiver import DeepSC
from utils import forward_pipeline, initNetParams, loss_function


def close(actual, expected, message):
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5, msg=message)


def main():
    torch.manual_seed(17)

    # Native-complex multiplication and the documented real equivalent agree.
    h = torch.randn(2, 2) + 1j * torch.randn(2, 2)
    x = torch.randn(19, 2) + 1j * torch.randn(19, 2)
    expected = torch.einsum("oi,ui->uo", h, x)
    matrix = complex_matrix_to_real_interleaved(h)
    real_result = torch.view_as_real(x).reshape(19, 4) @ matrix.T
    actual = torch.view_as_complex(real_result.reshape(19, 2, 2).contiguous())
    close(actual, expected, "real-equivalent matrix mismatch")

    # Power constraint and noiseless ZF recovery.
    transmitted = torch.randn(3, 5, 16, requires_grad=True)
    channel = MIMOChannel(MIMOConfig(profile="no_fading", equalizer="zf"), seed=2)
    output, trace = channel(transmitted, 0.0, return_trace=True)
    close(output, transmitted / math.sqrt(2.0), "noiseless identity channel mismatch")
    output.square().mean().backward()
    assert transmitted.grad is not None and torch.isfinite(transmitted.grad).all()

    # Covariance-aware LMMSE agrees with its direct complex expression.
    channel = MIMOChannel(MIMOConfig(fading_scope="sample"), seed=8)
    tx = torch.randn(2, 7, 16)
    _, trace = channel(tx, 0.2, return_trace=True)
    uses = trace.tx_scaled.shape[1]
    h = trace.estimated_h.expand(2, uses, 2, 2)
    rx = trace.assumed_stream_covariance[:, None].expand(2, uses, 2, 2)
    direct_filter = rx @ h.mH @ torch.linalg.inv(h @ rx @ h.mH + trace.noise_covariance)
    direct = torch.einsum("buni,bui->bun", direct_filter, trace.noisy_signal)
    close(direct, trace.equalized_signal, "LMMSE formula mismatch")

    # Every exposed modeling combination preserves shape and finite values.
    for scope in ("batch", "sample", "use"):
        for profile in ("baseline", "no_fading", "bad_channel", "noisy_csi", "stale_csi"):
            for equalizer in ("lmmse", "zf"):
                configured = MIMOChannel(
                    MIMOConfig(
                        fading_scope=scope,
                        profile=profile,
                        equalizer=equalizer,
                        channel_gain=0.5,
                        csi_noise_std=0.05,
                        allow_legacy_profiles=profile in {
                            "bad_channel", "noisy_csi", "stale_csi"
                        },
                    ),
                    seed=21,
                )
                candidate = configured(torch.randn(3, 4, 16), 0.1)
                assert candidate.shape == (3, 4, 16)
                assert torch.isfinite(candidate).all()
                if profile == "stale_csi":
                    # A smaller final batch must not reuse incompatible sample-shaped CSI.
                    final_batch = configured(torch.randn(1, 4, 16), 0.1)
                    assert final_batch.shape == (1, 4, 16)

    # The generated complex noise has two real components with the documented variance.
    probe = torch.zeros(64, 64, 16)
    _, noise_trace = MIMOChannel(MIMOConfig(fading_scope="sample"), seed=29)(
        probe, 0.2, return_trace=True
    )
    component_variance = torch.view_as_real(noise_trace.noise).var(unbiased=False)
    assert abs(float(component_variance) - 0.2 ** 2) < 0.004

    # Receiver covariance is system-defined and independent of the message.
    covariance_channel = MIMOChannel(MIMOConfig(), seed=31)
    _, first = covariance_channel(torch.randn(2, 4, 16), 0.1, return_trace=True)
    covariance_channel.reset()
    _, second = covariance_channel(torch.randn(2, 4, 16) * 0.01, 0.1, return_trace=True)
    close(
        first.assumed_stream_covariance,
        second.assumed_stream_covariance,
        "receiver covariance leaked current codeword information",
    )
    assert not torch.allclose(
        first.realized_stream_covariance, second.realized_stream_covariance
    )

    # Epoch-indexed realization sets are fresh and exactly reproducible.
    epoch_channel = MIMOChannel(MIMOConfig(), seed=37)
    epoch_tx = torch.randn(2, 4, 16)
    epoch_channel.reset(realization_set=0)
    _, epoch_zero = epoch_channel(epoch_tx, 0.1, return_trace=True)
    epoch_channel.reset(realization_set=1)
    _, epoch_one = epoch_channel(epoch_tx, 0.1, return_trace=True)
    assert not torch.allclose(epoch_zero.true_h, epoch_one.true_h)
    assert not torch.allclose(epoch_zero.noise, epoch_one.noise)
    epoch_channel.reset(realization_set=0)
    _, epoch_zero_again = epoch_channel(epoch_tx, 0.1, return_trace=True)
    close(epoch_zero.true_h, epoch_zero_again.true_h, "epoch H is not reproducible")
    close(epoch_zero.noise, epoch_zero_again.noise, "epoch noise is not reproducible")

    # CSI-error sampling cannot shift the physical H/AWGN realization stream.
    paired_tx = torch.randn(2, 4, 16)
    _, baseline_trace = MIMOChannel(MIMOConfig(), seed=41)(
        paired_tx, 0.1, return_trace=True
    )
    _, noisy_csi_trace = MIMOChannel(
        MIMOConfig(
            profile="noisy_csi", csi_noise_std=0.1,
            allow_legacy_profiles=True,
        ),
        seed=41,
    )(paired_tx, 0.1, return_trace=True)
    close(baseline_trace.true_h, noisy_csi_trace.true_h, "profile changed paired H")
    close(baseline_trace.noise, noisy_csi_trace.noise, "profile changed paired AWGN")

    stale = MIMOChannel(
        MIMOConfig(profile="stale_csi", allow_legacy_profiles=True), seed=43
    )
    _, first_stale = stale(paired_tx, 0.1, return_trace=True)
    assert not torch.allclose(first_stale.true_h, first_stale.estimated_h)

    for invalid in (
        {"profile": "noisy_csi", "allow_legacy_profiles": True},
        {"profile": "bad_channel", "allow_legacy_profiles": True},
        {"profile": "stale_csi", "stale_rho": 1.0, "allow_legacy_profiles": True},
    ):
        try:
            MIMOConfig(**invalid).validate()
        except ValueError:
            pass
        else:
            raise AssertionError("zero-effect legacy profile was accepted: {}".format(invalid))

    # Legacy loss divides by all positions after masking, exactly as original DeepSC.
    raw_losses = torch.tensor([2.0, 4.0, 6.0])
    targets = torch.tensor([1, 0, 2])
    criterion = lambda logits, target: logits
    assert abs(float(loss_function(raw_losses, targets, 0, criterion, "legacy")) - 8.0 / 3.0) < 1e-6
    assert abs(float(loss_function(raw_losses, targets, 0, criterion, "tokens")) - 4.0) < 1e-6

    # Full neural pipeline, gradients, and schema-v2 serialization.
    model = initNetParams(DeepSC(1, 20, 20, 20, 20, 8, 2, 16, 0.0))
    tokens = torch.tensor([[1, 4, 5, 2, 0], [1, 7, 2, 0, 0]])
    criterion = nn.CrossEntropyLoss(reduction="none")
    loss, pipeline_trace = forward_pipeline(
        model, tokens, tokens, 0.1, 0, criterion,
        channel_model=MIMOChannel(MIMOConfig(fading_scope="sample"), seed=3),
        collect_trace=True,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    with tempfile.TemporaryDirectory() as temporary:
        writer = ExperimentDataWriter(temporary, "baseline", run_id="smoke")
        artifact = writer.save_batch(
            pipeline_trace, "test", 0, 0, 0, sample_ids=torch.tensor([1, 2]),
            extra_metrics={"snr_db": 10.0},
        )
        payload = torch.load(artifact, map_location="cpu", weights_only=False)
        assert payload["schema_version"] == 2
        assert payload["channel"]["true_h"].shape == (2, 1, 2, 2)

    print("All DeepSC-MIMO runtime checks passed")


if __name__ == "__main__":
    main()
