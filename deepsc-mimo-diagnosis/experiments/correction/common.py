"""Shared utilities for the diagnosis-gated restoration experiments (correction_v1).

Reads the immutable paired diagnostic collection under RUN/traces/development and the
frozen DeepSC checkpoints under RUN/checkpoints/{C,D}/last.pth. Never writes outside the
output directory passed by the caller.

Paired structure (verified in collect_profiles.py / analysis/first_divergence.py): for a
given physical seed and batch index, every profile uses the same sentences (same
selection), the same underlying channel draw H_w and the same standardized noise Z.
"""
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROFILE_DIRS = {
    "normal": [("normal", None)],
    "low_snr": [("low_snr", None)],
    "imperfect_csi": [("imperfect_csi", 0.99), ("imperfect_csi", 0.95), ("imperfect_csi", 0.90)],
    "spatial_correlation": [("spatial_correlation", 0.3), ("spatial_correlation", 0.6),
                            ("spatial_correlation", 0.9)],
    "literal": [("literal", 0.20)],
    "mismatch": [("mismatch", None)],
    "ood_hansard": [("ood_hansard", None)],
}
LABELS = ["normal", "low_snr", "imperfect_csi", "spatial_correlation", "literal", "mismatch", "ood"]
PROFILE_LABEL = {"normal": "normal", "low_snr": "low_snr", "imperfect_csi": "imperfect_csi",
                 "spatial_correlation": "spatial_correlation", "literal": "literal",
                 "mismatch": "mismatch", "ood_hansard": "ood"}
SEEDS = [3101, 3102, 3103, 3104, 3105]
SNR_NOMINAL_DB, SNR_LOW_DB = 10.0, 1.0


def snr_to_noise(snr_db):
    """Std of each real noise component, as utils.SNR_to_noise."""
    return 1.0 / math.sqrt(2.0 * 10.0 ** (snr_db / 10.0))


def sev_text(severity):
    return "none" if severity is None else "{:g}".format(severity).replace(".", "p")


def artifact_path(run_root, profile, severity, seed, batch):
    return (Path(run_root) / "traces" / "development" / profile /
            "seed{}_{}_{}".format(seed, profile, sev_text(severity)) / "artifacts" /
            "batch_{:03d}.pt".format(batch))


def load_artifact(run_root, profile, severity, seed, batch):
    return torch.load(artifact_path(run_root, profile, severity, seed, batch),
                      map_location="cpu", weights_only=False)


def setup_imports(code_root):
    """Make the frozen run snapshot importable (channels, collect_profiles, models, utils)."""
    code_root = str(Path(code_root).resolve())
    if code_root not in sys.path:
        sys.path.insert(0, code_root)


def load_vocab(shared_root):
    from main import resolve_vocab, validate_vocabulary
    path = resolve_vocab(None, shared_root)
    vocab = json.loads(Path(path).read_text(encoding="utf-8"))
    import os
    # synthetic tests only; real runs always validate against the approved shared vocabulary
    token_to_idx = validate_vocabulary(vocab, allow_nonshared=os.environ.get("CORRECTION_SYNTHETIC_TEST") == "1")
    return token_to_idx


def load_models(run_root, vocab_size, device="cpu"):
    from collect_profiles import checkpoint_model
    c, _ = checkpoint_model(Path(run_root) / "checkpoints" / "C" / "last.pth", vocab_size, device)
    d, _ = checkpoint_model(Path(run_root) / "checkpoints" / "D" / "last.pth", vocab_size, device)
    for model in (c, d):
        for p in model.parameters():
            p.requires_grad_(False)
    return {"C": c, "D": d}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ------------------------------------------------------------------ physical layer
def lmmse(received, h_est, noise_std):
    """LMMSE equalizer exactly as channels.MIMOChannel._equalize with Rx = I (nt = 2).

    received: complex [B, U, 2]; h_est: complex [..., 2, 2] (one realization per block);
    noise_std: std of each real noise component. Returns complex [B, U, 2].
    """
    h = h_est.reshape(2, 2).to(received.dtype)
    c = 2.0 * noise_std ** 2
    system = h @ h.mH + c * torch.eye(2, dtype=h.dtype)
    w = torch.linalg.solve(system, h).mH          # Rx H^H (H Rx H^H + cI)^-1 with Rx = I
    return torch.einsum("ni,bui->bun", w, received)


def lmmse_matrix(h_est, noise_std):
    h = h_est.reshape(2, 2)
    c = 2.0 * noise_std ** 2
    return torch.linalg.solve(h @ h.mH + c * torch.eye(2, dtype=h.dtype), h).mH


def to_real(eq_complex, length):
    """complex [B, U, 2] -> DeepSC real interleaved [B, L, 16] (channels.complex_to_real_interleaved)."""
    b = eq_complex.shape[0]
    return torch.view_as_real(eq_complex).reshape(b, length, 16).contiguous()


def to_complex(real):
    """real [B, L, 16] -> complex [B, 4L, 2] (channels.real_interleaved_to_complex)."""
    paired = real.contiguous().view(real.shape[0], -1, 2, 2)
    return torch.complex(paired[..., 0], paired[..., 1])


def transmitted_symbols(artifact):
    """Power-scaled transmitted symbols x (complex [B, U, 2]) recovered from faded = H x."""
    h = artifact["true_h"].reshape(2, 2).to(torch.complex128)
    faded = artifact["faded_signal"].to(torch.complex128)
    x = torch.einsum("ni,bui->bun", torch.linalg.inv(h), faded)
    return x.to(torch.complex64)


def token_mask(artifact, pad):
    return artifact["source_tokens"] != pad                # [B, L]


def use_mask(tok_mask):
    return tok_mask.repeat_interleave(4, dim=1)            # [B, 4L]


# ------------------------------------------------------------------ receiver-available context
def h_features(h):
    h = h.reshape(2, 2)
    return torch.stack([h.real, h.imag], -1).reshape(-1).float()          # 8


def error_cov_features(h_est, noise_std):
    """LMMSE error covariance (H^H H / c + I)^-1, c = complex noise variance, Rx = I (Kay 1993)."""
    h = h_est.reshape(2, 2)
    c = 2.0 * noise_std ** 2
    ce = torch.linalg.inv(h.mH @ h / c + torch.eye(2, dtype=h.dtype))
    return torch.stack([ce[0, 0].real, ce[1, 1].real, ce[0, 1].real, ce[0, 1].imag]).float()


def block_stats(r1_complex, umask):
    """Second moment of the equalized streams over the valid uses of the whole block."""
    v = r1_complex[umask]                                  # [N, 2]
    s = (v.unsqueeze(-1) @ v.conj().unsqueeze(-2)).mean(0)
    return torch.stack([s[0, 0].real, s[1, 1].real, s[0, 1].real, s[0, 1].imag]).float()


def context_vector(r1_complex, umask, h_est, noise_std):
    """kappa: H-hat (8) + LMMSE error covariance (4) + block second moments (4) + log noise var (1)."""
    return torch.cat([h_features(h_est), error_cov_features(h_est, noise_std),
                      block_stats(r1_complex, umask),
                      torch.tensor([math.log(2.0 * noise_std ** 2)])])


# ------------------------------------------------------------------ decoding and metrics
def decode(model, memory, source_tokens, pad, start, end, max_len=31):
    from collect_profiles import greedy_hidden_states
    source_mask = (source_tokens == pad).unsqueeze(-2)
    with torch.no_grad():
        _, tokens, lengths, _ = greedy_hidden_states(model, memory, source_mask, start, end, pad, max_len)
    return tokens, lengths


def strip(tokens, special, end):
    out = []
    for t in tokens:
        t = int(t)
        if t == end:
            break
        if t not in special:
            out.append(t)
    return out


def make_bleu():
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        smooth = SmoothingFunction().method1

        def bleu(ref, hyp, n):
            if not hyp:
                return 0.0
            return float(sentence_bleu([ref], hyp, weights=tuple([1.0 / n] * n),
                                       smoothing_function=smooth if n > 1 else None))
        return bleu
    except Exception:
        from collections import Counter

        def grams(seq, n):
            return Counter(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))

        def bleu(ref, hyp, n):
            if not hyp:
                return 0.0
            log_sum = 0.0
            for k in range(1, n + 1):
                h = grams(hyp, k)
                total = sum(h.values())
                if total == 0:
                    return 0.0
                r = grams(ref, k)
                overlap = sum(min(c, r[g]) for g, c in h.items())
                if overlap == 0 and n == 1:
                    return 0.0
                log_sum += math.log((overlap if overlap > 0 else 0.1) / total) / n
            bp = 1.0 if len(hyp) > len(ref) else math.exp(1 - len(ref) / len(hyp))
            return bp * math.exp(log_sum)
        return bleu


BLEU = make_bleu()


def text_metrics(ref_tokens, hyp_tokens, hyp_lengths, special, end):
    b1, b4, acc = [], [], []
    for i in range(ref_tokens.shape[0]):
        ref = strip(ref_tokens[i].tolist(), special, end)
        hyp = strip(hyp_tokens[i, :int(hyp_lengths[i])].tolist(), special, end)
        b1.append(BLEU(ref, hyp, 1))
        b4.append(BLEU(ref, hyp, 4))
        acc.append(sum(1 for j, t in enumerate(ref) if j < len(hyp) and hyp[j] == t) / max(len(ref), 1))
    return np.array(b1), np.array(b4), np.array(acc)


def rep_metrics(pred, target, mask):
    """Token-level NMSE and cosine over valid tokens. pred/target real [B, L, D], mask [B, L]."""
    p, t = pred[mask], target[mask]
    nmse = float(((p - t) ** 2).sum() / (t ** 2).sum().clamp_min(1e-12))
    cos = float(torch.nn.functional.cosine_similarity(p, t, dim=-1).mean())
    return nmse, cos
