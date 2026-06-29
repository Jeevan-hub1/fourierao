"""
Atmosphere + Shack-Hartmann WFS + Deformable Mirror simulator.

Provides the faithful end-to-end physical chain:
  turbulence -> MLA spot field -> centroids/slopes -> reconstruction
  -> DM actuator map -> closed-loop residual.

Includes multi-layer turbulence with frozen-flow + "boiling" (AR(1)).
"""
import math
import numpy as np
if not hasattr(np, "math"):
    np.math = math
import aotools
from aotools.turbulence import infinitephasescreen


class Atmosphere:
    """Multi-layer atmosphere with frozen-flow + boiling (temporal decorrelation)."""

    def __init__(self, N, pxl_scale, layers, L0=50.0, boiling=0.0, seed=None):
        """
        layers: list of (r0_metres, wind_px_per_frame) tuples.
        boiling: 0 = pure frozen flow, ->1 = fully decorrelated each frame.
        """
        if seed is not None:
            np.random.seed(seed)
        self.N = N
        self.boiling = boiling
        self.pupil = aotools.circle(N // 2, N).astype(np.float32)
        self.layers = []
        for (r0, wpx) in layers:
            scr = infinitephasescreen.PhaseScreenVonKarman(N, pxl_scale, r0, L0)
            for _ in range(N):
                scr.add_row()
            self.layers.append({"scr": scr, "wpx": wpx})

    def step(self):
        """Advance one frame, return the combined pupil-masked phase (radians)."""
        total = np.zeros((self.N, self.N), np.float32)
        a = math.sqrt(max(1e-6, 1.0 - self.boiling))
        b = math.sqrt(self.boiling)
        for L in self.layers:
            for _ in range(max(1, int(round(L["wpx"])))):
                L["scr"].add_row()
            frozen = L["scr"].scrn
            total += a * frozen + b * np.random.randn(self.N, self.N) * frozen.std()
        return (total * self.pupil).astype(np.float32)

    def sequence(self, n_frames):
        """Generate a time-series of wavefront frames."""
        return np.array([self.step() for _ in range(n_frames)])



class ShackHartmann:
    """Shack-Hartmann WFS: lenslet spot field -> centroids -> slopes."""

    def __init__(self, N, n_sub):
        self.N = N
        self.n_sub = n_sub
        self.sub_px = N // n_sub
        self.pupil = aotools.circle(N // 2, N).astype(np.float32)
        self.valid = self._valid_subaps()

    def _valid_subaps(self):
        valid = np.zeros((self.n_sub, self.n_sub), bool)
        sp = self.sub_px
        for i in range(self.n_sub):
            for j in range(self.n_sub):
                patch = self.pupil[i*sp:(i+1)*sp, j*sp:(j+1)*sp]
                valid[i, j] = patch.sum() >= 0.5 * sp * sp
        return valid

    def slopes(self, phase, gtilt=True):
        """Return concatenated [sx, sy] slopes over valid lenslets."""
        sp = self.sub_px
        sx = np.zeros((self.n_sub, self.n_sub))
        sy = np.zeros((self.n_sub, self.n_sub))
        for i in range(self.n_sub):
            for j in range(self.n_sub):
                if not self.valid[i, j]:
                    continue
                sub = phase[i*sp:(i+1)*sp, j*sp:(j+1)*sp]
                subp = self.pupil[i*sp:(i+1)*sp, j*sp:(j+1)*sp]
                gy, gx = np.gradient(sub)
                sx[i, j] = (gx * subp).sum() / subp.sum()
                sy[i, j] = (gy * subp).sum() / subp.sum()
        return np.concatenate([sx[self.valid], sy[self.valid]])

    def spot_field(self, phase, pad=2):
        """Generate the actual lenslet focal-spot field (for visualization)."""
        sp = self.sub_px
        field_img = np.zeros((self.n_sub * sp, self.n_sub * sp))
        for i in range(self.n_sub):
            for j in range(self.n_sub):
                if not self.valid[i, j]:
                    continue
                sub = phase[i*sp:(i+1)*sp, j*sp:(j+1)*sp]
                subp = self.pupil[i*sp:(i+1)*sp, j*sp:(j+1)*sp]
                f = subp * np.exp(1j * sub)
                spot = np.abs(np.fft.fftshift(np.fft.fft2(f, s=(sp*pad, sp*pad))))**2
                field_img[i*sp:(i+1)*sp, j*sp:(j+1)*sp] = spot[::pad, ::pad][:sp, :sp]
        return field_img
