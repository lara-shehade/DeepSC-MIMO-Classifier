"""Restoration maps G_c (correction_v1).

All maps minimize a squared error to the paired nominal target (conditional-mean estimator,
Kay 1993; Bishop 2006 sec. 1.5.5), with an identity term on nominal inputs (design choice of
this work). Variants:
  AffineMap      token-wise affine map, closed-form ridge least squares (linear semantic
                 equalizer form of Pannacci et al. 2025)
  TokenMLP       residual token-wise MLP conditioned on receiver-available context kappa
                 (channel-as-input design following DetNet / OAMP-Net2; noise-level input
                 following FFDNet), with an optional variance head (Nix & Weigend 1994)
  BlockMatrixNet context -> one 2x2 complex matrix applied to every symbol of the block
                 (structure of the CSI distortion: R1_good = W(H) W(H_hat)^-1 R1_bad)
"""
import torch
import torch.nn as nn


class AffineMap:
    def __init__(self, dim):
        self.dim = dim
        self.B = None

    def fit(self, X, T, N=None, gamma=0.1, lam=1e-3):
        """min ||[X 1]B - T||^2 + g||[N 1]B - N||^2 + l||B||^2, g = gamma * n_X / n_N."""
        X, T = X.double(), T.double()
        Xa = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], 1)
        A = Xa.T @ Xa
        rhs = Xa.T @ T
        if N is not None and gamma > 0:
            N = N.double()
            Na = torch.cat([N, torch.ones(len(N), 1, dtype=N.dtype)], 1)
            g = gamma * len(X) / len(N)
            A = A + g * Na.T @ Na
            rhs = rhs + g * Na.T @ N
        reg = lam * torch.trace(A) / A.shape[0]
        self.B = torch.linalg.solve(A + reg * torch.eye(A.shape[0], dtype=A.dtype), rhs).float()
        return self

    def __call__(self, x):
        shape = x.shape
        flat = x.reshape(-1, self.dim)
        out = torch.cat([flat, torch.ones(len(flat), 1)], 1) @ self.B
        return out.reshape(shape)


class TokenMLP(nn.Module):
    def __init__(self, dim, ctx_dim, hidden=256, variance=False):
        super().__init__()
        self.dim, self.ctx_dim, self.variance = dim, ctx_dim, variance
        self.body = nn.Sequential(nn.Linear(dim + ctx_dim, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU())
        self.mean = nn.Linear(hidden, dim)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)                 # starts as the identity map
        self.logvar = nn.Linear(hidden, dim) if variance else None

    def forward(self, x, ctx):
        """x [B, L, dim]; ctx [B, ctx_dim] (one context vector per block or per sentence)."""
        if self.ctx_dim:
            c = ctx[:, None, :].expand(x.shape[0], x.shape[1], self.ctx_dim)
            h = self.body(torch.cat([x, c], -1))
        else:
            h = self.body(x)
        mean = x + self.mean(h)
        if self.variance:
            return mean, self.logvar(h.detach())       # variance head trained on detached features
        return mean, None


class BlockMatrixNet(nn.Module):
    def __init__(self, ctx_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(ctx_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                 nn.SiLU(), nn.Linear(hidden, 8))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)              # starts as M = I

    def matrix(self, ctx):
        d = self.net(ctx).view(-1, 2, 2, 2)
        return torch.eye(2, dtype=torch.complex64) + torch.complex(d[..., 0], d[..., 1])

    def forward(self, x_complex, ctx):
        """x_complex [B, U, 2] symbols of one block; ctx [1, ctx_dim]."""
        m = self.matrix(ctx)[0]
        return torch.einsum("ni,bui->bun", m, x_complex)
