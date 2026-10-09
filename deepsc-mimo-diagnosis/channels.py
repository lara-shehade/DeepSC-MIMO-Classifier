"""Differentiable 2x2 MIMO channel models used by DeepSC.

The neural channel encoder stores complex symbols as interleaved real values:
``[Re(x1), Im(x1), Re(x2), Im(x2), ...]``.  This module keeps that
representation at its boundary and performs the physical channel calculation
with native PyTorch complex tensors internally.  Using native complex tensors
removes ambiguity about the 4x4 real-equivalent matrix convention.
"""

from dataclasses import dataclass
import math
from typing import Optional

import torch


@dataclass(frozen=True)
class MIMOConfig:
    nt: int = 2
    nr: int = 2
    fading_scope: str = "batch"  # batch, sample, or use
    profile: str = "baseline"  # approved: baseline, imperfect_csi, spatial_correlation
    equalizer: str = "lmmse"  # lmmse or zf
    covariance: str = "system"  # receiver-known, message-independent covariance
    stream_variance: Optional[float] = None
    total_power_constraint: bool = True
    channel_gain: float = 1.0
    csi_noise_std: float = 0.0
    csi_rho: float = 1.0
    spatial_rho: float = 0.0
    stale_rho: float = 0.95
    covariance_floor: float = 1e-6
    allow_legacy_profiles: bool = False

    def validate(self) -> None:
        if self.nt != 2 or self.nr != 2:
            raise ValueError("this validated migration is intentionally limited to 2x2 MIMO")
        if self.fading_scope not in {"batch", "sample", "use"}:
            raise ValueError("fading_scope must be batch, sample, or use")
        if self.profile not in {
            "baseline", "imperfect_csi", "spatial_correlation", "no_fading",
            "bad_channel", "noisy_csi", "stale_csi"
        }:
            raise ValueError("unsupported channel profile: {}".format(self.profile))
        if self.equalizer not in {"lmmse", "zf"}:
            raise ValueError("equalizer must be lmmse or zf")
        if self.covariance != "system":
            raise ValueError("only receiver-known system covariance is supported")
        if self.stream_variance is not None and self.stream_variance <= 0:
            raise ValueError("stream_variance must be positive when provided")
        if self.channel_gain <= 0:
            raise ValueError("channel_gain must be positive")
        if self.csi_noise_std < 0:
            raise ValueError("csi_noise_std must be non-negative")
        if not 0 <= self.csi_rho <= 1:
            raise ValueError("csi_rho must be between zero and one")
        if not 0 <= self.spatial_rho < 1:
            raise ValueError("spatial_rho must be in [0, 1)")
        if not 0 <= self.stale_rho <= 1:
            raise ValueError("stale_rho must be between zero and one")
        if self.covariance_floor <= 0:
            raise ValueError("covariance_floor must be positive")
        legacy = {"bad_channel", "noisy_csi", "stale_csi"}
        if self.profile in legacy and not self.allow_legacy_profiles:
            raise ValueError(
                "{} is a legacy POC profile whose final scientific definition is pending; "
                "pass allow_legacy_profiles=True only for explicit legacy reproduction"
                .format(self.profile)
            )
        if self.profile == "bad_channel" and not 0 < self.channel_gain < 1.0:
            raise ValueError("legacy bad_channel requires 0 < channel_gain < 1")
        if self.profile == "noisy_csi" and self.csi_noise_std == 0.0:
            raise ValueError("noisy_csi requires csi_noise_std > 0")
        if self.profile == "stale_csi" and self.stale_rho == 1.0:
            raise ValueError("stale_csi with stale_rho=1 has no mismatch effect")
        if self.profile == "imperfect_csi" and not 0 <= self.csi_rho < 1.0:
            raise ValueError("imperfect_csi requires 0 <= csi_rho < 1")
        if self.profile == "spatial_correlation" and not 0 < self.spatial_rho < 1.0:
            raise ValueError("spatial_correlation requires 0 < spatial_rho < 1")


@dataclass
class ChannelTrace:
    underlying_iid_h: torch.Tensor
    true_h: torch.Tensor
    estimated_h: torch.Tensor
    tx_grouped: torch.Tensor
    tx_scaled: torch.Tensor
    faded_signal: torch.Tensor
    noise: torch.Tensor
    noisy_signal: torch.Tensor
    equalized_signal: torch.Tensor
    assumed_stream_covariance: torch.Tensor
    realized_stream_covariance: torch.Tensor
    noise_covariance: torch.Tensor
    equalized_real: torch.Tensor
    noise_std: float
    power_scale: float
    fading_scope: str
    profile: str
    equalizer: str


def real_interleaved_to_complex(x: torch.Tensor, streams: int) -> torch.Tensor:
    """Convert a real feature tensor to ``[..., uses, streams]`` complex form."""
    if not x.is_floating_point():
        raise TypeError("channel input must be floating point")
    if x.shape[-1] % (2 * streams) != 0:
        raise ValueError(
            "last dimension {} is not divisible by 2*streams ({})".format(
                x.shape[-1], 2 * streams
            )
        )
    paired = x.contiguous().view(x.shape[0], -1, streams, 2)
    return torch.complex(paired[..., 0], paired[..., 1])


def complex_to_real_interleaved(x: torch.Tensor, output_shape) -> torch.Tensor:
    """Convert ``[..., streams]`` complex values back to DeepSC real features."""
    real = torch.view_as_real(x).reshape(*output_shape)
    return real


def complex_matrix_to_real_interleaved(h: torch.Tensor) -> torch.Tensor:
    """Return the real matrix for interleaved row vectors.

    For ``y = H x`` in column-vector complex notation, this matrix ``A`` obeys
    ``y_real_row = x_real_row @ A.T``.
    """
    nr, nt = h.shape[-2:]
    out = torch.zeros(
        *h.shape[:-2], 2 * nr, 2 * nt, dtype=h.real.dtype, device=h.device
    )
    out[..., 0::2, 0::2] = h.real
    out[..., 0::2, 1::2] = -h.imag
    out[..., 1::2, 0::2] = h.imag
    out[..., 1::2, 1::2] = h.real
    return out


class MIMOChannel:
    """Flat-fading MIMO channel with perfect or impaired receiver CSI."""

    def __init__(self, config: Optional[MIMOConfig] = None, seed: Optional[int] = None):
        self.config = config or MIMOConfig()
        self.config.validate()
        self.seed = seed
        self._generators = {}
        self._previous_h = None
        self._realization_set = 0
        self._last_underlying_iid_h = None

    def reset(self, realization_set: int = 0) -> None:
        """Start a deterministic, independent realization set.

        Reusing an index exactly reproduces its H/noise sequence. Different
        indices produce fresh sequences while retaining base-seed reproducibility.
        """
        if not isinstance(realization_set, int) or realization_set < 0:
            raise ValueError("realization_set must be a non-negative integer")
        self._previous_h = None
        self._generators = {}
        self._realization_set = realization_set
        self._last_underlying_iid_h = None

    def _generator(self, device: torch.device, stream: str):
        if self.seed is None:
            return None
        key = (str(device), stream)
        if key not in self._generators:
            generator_device = device if device.type == "cuda" else torch.device("cpu")
            generator = torch.Generator(device=generator_device)
            offsets = {"channel": 0, "noise": 1, "csi": 2, "stale_initial": 3}
            generator.manual_seed(
                int(self.seed)
                + 1_000_003 * offsets[stream]
                + 10_000_019 * self._realization_set
            )
            self._generators[key] = generator
        return self._generators[key]

    def _randn(self, shape, reference: torch.Tensor, stream: str) -> torch.Tensor:
        return torch.randn(
            shape,
            dtype=reference.real.dtype,
            device=reference.device,
            generator=self._generator(reference.device, stream),
        )

    def _complex_normal(
        self, shape, reference: torch.Tensor, component_std: float, stream: str
    ):
        real = self._randn(shape, reference, stream) * component_std
        imag = self._randn(shape, reference, stream) * component_std
        return torch.complex(real, imag)

    def _channel_shape(self, batch: int, uses: int):
        cfg = self.config
        if cfg.fading_scope == "batch":
            return (1, 1, cfg.nr, cfg.nt)
        if cfg.fading_scope == "sample":
            return (batch, 1, cfg.nr, cfg.nt)
        return (batch, uses, cfg.nr, cfg.nt)

    def _sample_channel(self, x: torch.Tensor) -> torch.Tensor:
        batch, uses = x.shape[:2]
        cfg = self.config
        shape = self._channel_shape(batch, uses)
        if cfg.profile == "no_fading":
            eye = torch.eye(cfg.nt, dtype=x.dtype, device=x.device)
            return eye.reshape(1, 1, cfg.nr, cfg.nt).expand(shape)

        fresh = self._complex_normal(
            shape, x, 1.0 / math.sqrt(2.0), stream="channel"
        )
        self._last_underlying_iid_h = fresh
        if cfg.profile == "spatial_correlation":
            # Approved symmetric Kronecker model: Rr^(1/2) Hw Rt^(1/2).
            # eigh gives the principal Hermitian square root and the unit
            # diagonal of R preserves E[|H_ij|^2] = 1.
            rho = cfg.spatial_rho
            correlation = torch.tensor(
                [[1.0, rho], [rho, 1.0]],
                dtype=x.real.dtype,
                device=x.device,
            )
            eigenvalues, eigenvectors = torch.linalg.eigh(correlation)
            root = (eigenvectors * eigenvalues.clamp_min(0).sqrt()) @ eigenvectors.mT
            root = root.to(dtype=x.dtype)
            fresh = root @ fresh @ root
        if cfg.profile == "bad_channel":
            fresh = fresh * cfg.channel_gain
        has_compatible_history = (
            self._previous_h is not None and tuple(self._previous_h.shape) == tuple(fresh.shape)
        )
        if cfg.profile == "stale_csi" and not has_compatible_history:
            self._previous_h = self._complex_normal(
                shape, x, 1.0 / math.sqrt(2.0), stream="stale_initial"
            )
            has_compatible_history = True
        if cfg.profile == "stale_csi" and has_compatible_history:
            rho = cfg.stale_rho
            current = rho * self._previous_h + math.sqrt(max(0.0, 1.0 - rho * rho)) * fresh
        else:
            current = fresh
        return current

    def _estimated_channel(self, true_h: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if cfg.profile == "imperfect_csi":
            independent = self._complex_normal(
                true_h.shape, true_h, 1.0 / math.sqrt(2.0), stream="csi"
            )
            rho = cfg.csi_rho
            return rho * true_h + math.sqrt(max(0.0, 1.0 - rho * rho)) * independent
        if cfg.profile == "noisy_csi":
            error = self._complex_normal(
                true_h.shape, true_h, cfg.csi_noise_std, stream="csi"
            )
            return true_h + error
        if (
            cfg.profile == "stale_csi"
            and self._previous_h is not None
            and tuple(self._previous_h.shape) == tuple(true_h.shape)
        ):
            return self._previous_h
        return true_h

    def _stream_covariances(self, x: torch.Tensor):
        """Return receiver-assumed and trace-only realized covariances.

        The equalizer never observes the current hidden codeword. The realized
        second moment is recorded only for later diagnostics.
        """
        cfg = self.config
        realized = torch.einsum("bun,bum->bnm", x, x.conj()) / max(1, x.shape[1])
        realized = realized.detach()
        identity = torch.eye(cfg.nt, dtype=x.dtype, device=x.device).unsqueeze(0)
        realized = realized + cfg.covariance_floor * identity
        nominal = cfg.stream_variance
        if nominal is None:
            nominal = 2.0 / cfg.nt if cfg.total_power_constraint else 2.0
        assumed = float(nominal) * identity.expand(x.shape[0], cfg.nt, cfg.nt)
        return assumed, realized

    @staticmethod
    def _expand_channel(h: torch.Tensor, batch: int, uses: int) -> torch.Tensor:
        return h.expand(batch, uses, h.shape[-2], h.shape[-1])

    def _equalize(
        self,
        received: torch.Tensor,
        estimated_h: torch.Tensor,
        assumed_covariance: torch.Tensor,
        complex_noise_variance: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        batch, uses = received.shape[:2]
        h = self._expand_channel(estimated_h, batch, uses)
        h_h = h.mH
        identity_rx = torch.eye(cfg.nr, dtype=h.dtype, device=h.device)

        if cfg.equalizer == "zf":
            # pinv is stable for square and rank-deficient channel realizations.
            equalizer = torch.linalg.pinv(h)
        else:
            rx = assumed_covariance[:, None, :, :].expand(batch, uses, cfg.nt, cfg.nt)
            system = h @ rx @ h_h + complex_noise_variance * identity_rx
            # solve(system, h @ rx)^H == Rx H^H system^-1 for Hermitian system.
            equalizer = torch.linalg.solve(system, h @ rx).mH
        return torch.einsum("buni,bui->bun", equalizer, received)

    def __call__(
        self,
        tx_real: torch.Tensor,
        noise_std,
        return_trace: bool = False,
    ):
        cfg = self.config
        original_shape = tx_real.shape
        x = real_interleaved_to_complex(tx_real, cfg.nt)
        power_scale = 1.0 / math.sqrt(cfg.nt) if cfg.total_power_constraint else 1.0
        x_scaled = x * power_scale

        true_h = self._sample_channel(x_scaled)
        estimated_h = self._estimated_channel(true_h)
        expanded_h = self._expand_channel(true_h, x.shape[0], x.shape[1])
        faded = torch.einsum("buoi,bui->buo", expanded_h, x_scaled)

        noise_std_tensor = torch.as_tensor(
            noise_std, dtype=tx_real.dtype, device=tx_real.device
        )
        if noise_std_tensor.numel() != 1 or noise_std_tensor.item() < 0:
            raise ValueError("noise_std must be one non-negative scalar")
        noise = self._complex_normal(
            faded.shape, faded, float(noise_std_tensor.item()), stream="noise"
        )
        noisy = faded + noise

        assumed_covariance, realized_covariance = self._stream_covariances(x_scaled)
        # Real and imaginary noise components each have variance noise_std^2.
        complex_noise_variance = 2.0 * noise_std_tensor.square()
        equalized = self._equalize(
            noisy, estimated_h, assumed_covariance, complex_noise_variance
        )
        # Materialize real storage independently from the complex equalizer
        # result so both representations can coexist in a serialized trace.
        output = complex_to_real_interleaved(equalized, original_shape).clone()

        if cfg.profile == "stale_csi":
            self._previous_h = true_h.detach().clone()

        if not return_trace:
            return output

        noise_covariance = complex_noise_variance * torch.eye(
            cfg.nr, dtype=x.dtype, device=x.device
        )
        trace = ChannelTrace(
            underlying_iid_h=self._last_underlying_iid_h,
            true_h=true_h,
            estimated_h=estimated_h,
            tx_grouped=x.clone(),
            tx_scaled=x_scaled,
            faded_signal=faded,
            noise=noise,
            noisy_signal=noisy,
            equalized_signal=equalized,
            assumed_stream_covariance=assumed_covariance,
            realized_stream_covariance=realized_covariance,
            noise_covariance=noise_covariance,
            equalized_real=output.clone(),
            noise_std=float(noise_std_tensor.item()),
            power_scale=power_scale,
            fading_scope=cfg.fading_scope,
            profile=cfg.profile,
            equalizer=cfg.equalizer,
        )
        return output, trace


__all__ = [
    "ChannelTrace",
    "MIMOChannel",
    "MIMOConfig",
    "complex_matrix_to_real_interleaved",
    "complex_to_real_interleaved",
    "real_interleaved_to_complex",
]
