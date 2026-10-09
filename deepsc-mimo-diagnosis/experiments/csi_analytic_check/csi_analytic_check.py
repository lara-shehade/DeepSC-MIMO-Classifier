"""Analytical check of receiver-side CSI restoration at R1 (2x2, 10 dB, Gaussian unit-power symbols).
Compares, by R1 NMSE against the perfect-CSI LMMSE output W(H)y:
  none         W(H_hat) y
  post_mean    E[W(H) | H_hat] y, with H | H_hat ~ CN(rho H_hat, (1-rho^2) I) (Monte Carlo)
  cov_procr    H_est = C^{1/2} U, C = YY^H/N - s2 I, U = polar(C^{1/2 H} H_hat) (second-order + Procrustes)
Gaussian symbols are an assumption: DeepSC symbols are not Gaussian (see paper)."""
import numpy as np

rng = np.random.default_rng(0)
s2 = 10 ** (-10 / 10)                     # complex noise variance at 10 dB, unit symbol power
N, TRIALS, MC = 12800, 400, 200


def cn(*shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def W(h):
    return h.conj().T @ np.linalg.inv(h @ h.conj().T + s2 * np.eye(2))


def psd_sqrt(c):
    w, v = np.linalg.eigh((c + c.conj().T) / 2)
    return (v * np.sqrt(np.clip(w, 1e-9, None))) @ v.conj().T


for rho in (0.99, 0.95, 0.90):
    err = {"none": [], "post_mean": [], "cov_procr": []}
    for _ in range(TRIALS):
        h = cn(2, 2)
        h_hat = rho * h + np.sqrt(1 - rho ** 2) * cn(2, 2)
        x = cn(2, N)
        y = h @ x + np.sqrt(s2) * cn(2, N)
        good = W(h) @ y
        den = np.sum(np.abs(good) ** 2)
        wm = np.mean([W(rho * h_hat + np.sqrt(1 - rho ** 2) * cn(2, 2)) for _ in range(MC)], axis=0)
        a = psd_sqrt(y @ y.conj().T / N - s2 * np.eye(2))
        p, _, qh = np.linalg.svd(a.conj().T @ h_hat)
        h_est = a @ (p @ qh)
        for k, est in (("none", W(h_hat) @ y), ("post_mean", wm @ y), ("cov_procr", W(h_est) @ y)):
            err[k].append(np.sum(np.abs(est - good) ** 2) / den)
    print("rho={:.2f}  ".format(rho) + "  ".join("{} {:.3f}".format(k, np.mean(v)) for k, v in err.items()), flush=True)
