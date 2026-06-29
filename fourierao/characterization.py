"""
Turbulence characterization from SH-WFS time-series:
  - r0 (Fried parameter) via calibrated Kolmogorov phase-variance law
  - wind speed/direction via sub-pixel Fourier cross-correlation
  - tau0 (Greenwood coherence time)
"""
import math
import numpy as np
if not hasattr(np, "math"):
    np.math = math
import aotools


class TurbulenceCharacterizer:
    def __init__(self, N, D, pxl_scale, loop_rate=1000.0):
        self.N = N
        self.D = D
        self.pxl_scale = pxl_scale
        self.dt = 1.0 / loop_rate
        self.pupil = aotools.circle(N // 2, N).astype(np.float32)
        self.pmask = self.pupil > 0
        self.C = 0.4863  # calibrated constant (finite-L0); recalibrate per system

    def estimate_r0(self, frames):
        """r0 from piston-removed phase variance: sigma^2 = C (D/r0)^(5/3)."""
        variances = []
        for ph in frames:
            v = ph[self.pmask]
            v = v - v.mean()
            variances.append(np.var(v))
        sigma2 = np.mean(variances)
        return self.D / (sigma2 / self.C) ** (3.0 / 5.0)

    def calibrate(self, frames, r0_known):
        """One-time calibration of constant C against a known-r0 dataset."""
        variances = [np.var(ph[self.pmask] - ph[self.pmask].mean()) for ph in frames]
        sigma2 = np.mean(variances)
        self.C = sigma2 / (self.D / r0_known) ** (5.0 / 3.0)
        return self.C

    def _subpixel_shift(self, a, b):
        """Sub-pixel translation between two frames via FFT phase-correlation."""
        N = self.N
        A, B = np.fft.fft2(a), np.fft.fft2(b)
        R = A * np.conj(B); R /= (np.abs(R) + 1e-9)
        cc = np.fft.ifft2(R).real
        py, px = np.unravel_index(np.argmax(cc), cc.shape)

        def refine(m, c, p):
            d = (m - 2*c + p)
            return 0.0 if d == 0 else 0.5 * (m - p) / d
        dy = refine(cc[(py-1) % N, px], cc[py, px], cc[(py+1) % N, px])
        dx = refine(cc[py, (px-1) % N], cc[py, px], cc[py, (px+1) % N])
        sy = py + dy; sx = px + dx
        if sy > N/2: sy -= N
        if sx > N/2: sx -= N
        return sy, sx

    def estimate_wind(self, frames):
        """Wind speed (m/s) and direction (deg) via averaged sub-pixel shifts."""
        shifts = np.array([self._subpixel_shift(frames[i], frames[i+1])
                           for i in range(len(frames) - 1)])
        # cross-correlation of (a,b) yields -shift
        mean_shift = -shifts.mean(axis=0)
        speed_px = np.sqrt((mean_shift**2).sum())
        speed = speed_px * self.pxl_scale / self.dt
        direction = math.degrees(math.atan2(mean_shift[0], mean_shift[1]))
        return speed, direction

    def estimate_tau0(self, r0, wind_speed):
        """Greenwood coherence time tau0 = 0.314 r0 / v (seconds)."""
        if wind_speed <= 0:
            return float("inf")
        return 0.314 * r0 / wind_speed
