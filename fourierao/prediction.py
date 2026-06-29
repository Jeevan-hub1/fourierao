"""
Wavefront predictors:
  - PersistenceBaseline  (predict next = current)
  - LinearARBaseline     (least-squares autoregressive)
  - KoopmanPredictor     (DMD linear operator)
  - FNO2d                (residual Fourier Neural Operator)  <-- the novel core

Training uses EARLY STOPPING and RESIDUAL learning (verified essential).
"""
import math
import copy
import numpy as np
if not hasattr(np, "math"):
    np.math = math


class PersistenceBaseline:
    def predict(self, current):
        return current.copy()


class LinearARBaseline:
    """Global least-squares AR predictor on flattened modal/slope vectors."""
    def __init__(self, lags=4, horizon=1):
        self.lags = lags
        self.horizon = horizon
        self.W = None

    def fit(self, Z):
        lags, h = self.lags, self.horizon
        X = np.array([Z[t-lags+1:t+1].flatten() for t in range(lags, len(Z)-h)])
        Y = np.array([Z[t+h] for t in range(lags, len(Z)-h)])
        self.W, *_ = np.linalg.lstsq(X, Y, rcond=None)
        return self

    def predict_series(self, Z):
        lags, h = self.lags, self.horizon
        X = np.array([Z[t-lags+1:t+1].flatten() for t in range(lags, len(Z)-h)])
        return X @ self.W


class KoopmanPredictor:
    """Dynamic Mode Decomposition: learn linear operator A with Z[t+1]=A Z[t]."""
    def __init__(self, horizon=1):
        self.horizon = horizon
        self.A = None

    def fit(self, Z):
        X0, X1 = Z[:-1].T, Z[1:].T
        self.A = X1 @ np.linalg.pinv(X0)
        return self

    def predict(self, z):
        out = z.copy()
        for _ in range(self.horizon):
            out = self.A @ out
        return out



# ---------------------------------------------------------------------------
# Fourier Neural Operator (residual) — the novel predictor core.
# Only imported if torch is available.
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn

    class SpectralConv2d(nn.Module):
        """Spectral convolution: learn weights on the lowest Fourier modes."""
        def __init__(self, cin, cout, m1, m2):
            super().__init__()
            self.m1, self.m2 = m1, m2
            scale = 1.0 / (cin * cout)
            self.w1 = nn.Parameter(scale * torch.rand(cin, cout, m1, m2, dtype=torch.cfloat))
            self.w2 = nn.Parameter(scale * torch.rand(cin, cout, m1, m2, dtype=torch.cfloat))

        def forward(self, x):
            B, C, H, W = x.shape
            xft = torch.fft.rfft2(x)
            out = torch.zeros(B, self.w1.shape[1], H, W // 2 + 1, dtype=torch.cfloat)
            out[:, :, :self.m1, :self.m2] = torch.einsum(
                "bixy,ioxy->boxy", xft[:, :, :self.m1, :self.m2], self.w1)
            out[:, :, -self.m1:, :self.m2] = torch.einsum(
                "bixy,ioxy->boxy", xft[:, :, -self.m1:, :self.m2], self.w2)
            return torch.fft.irfft2(out, s=(H, W))

    class FNO2d(nn.Module):
        """Residual FNO: output = input + learned correction (starts at persistence)."""
        def __init__(self, width=20, modes=10):
            super().__init__()
            self.fc0 = nn.Conv2d(1, width, 1)
            self.s1 = SpectralConv2d(width, width, modes, modes); self.c1 = nn.Conv2d(width, width, 1)
            self.s2 = SpectralConv2d(width, width, modes, modes); self.c2 = nn.Conv2d(width, width, 1)
            self.f1 = nn.Conv2d(width, 32, 1); self.f2 = nn.Conv2d(32, 1, 1)
            nn.init.zeros_(self.f2.weight); nn.init.zeros_(self.f2.bias)  # start = identity

        def forward(self, x):
            h = self.fc0(x)
            h = torch.relu(self.s1(h) + self.c1(h))
            h = torch.relu(self.s2(h) + self.c2(h))
            return x + self.f2(torch.relu(self.f1(h)))   # residual / persistence-init

    def train_fno(frames_norm, horizon=1, width=20, modes=10, max_epochs=80,
                  patience=8, lr=1e-3, bs=32, verbose=True):
        """Train residual FNO with EARLY STOPPING (verified essential)."""
        F = frames_norm
        X = torch.tensor(F[:-horizon]).unsqueeze(1)
        Y = torch.tensor(F[horizon:]).unsqueeze(1)
        ntr = int(0.7 * len(X)); nval = int(0.85 * len(X))
        Xtr, Ytr = X[:ntr], Y[:ntr]
        Xv, Yv = X[ntr:nval], Y[ntr:nval]
        model = FNO2d(width, modes)
        opt = torch.optim.Adam(model.parameters(), lr)
        lossf = nn.MSELoss()
        best, best_state, wait = 1e9, None, 0
        for ep in range(max_epochs):
            model.train(); perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), bs):
                idx = perm[i:i+bs]; opt.zero_grad()
                lossf(model(Xtr[idx]), Ytr[idx]).backward(); opt.step()
            model.eval()
            with torch.no_grad():
                vl = lossf(model(Xv), Yv).item()
            if vl < best - 1e-6:
                best, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
                if wait >= patience:
                    break
        model.load_state_dict(best_state)
        model.eval()
        return model, best, ep

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
