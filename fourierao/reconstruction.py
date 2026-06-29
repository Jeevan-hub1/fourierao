"""
Wavefront reconstruction: MODAL (Zernike) and ZONAL (Southwell).
"""
import math
import numpy as np
if not hasattr(np, "math"):
    np.math = math
import aotools


class ModalReconstructor:
    """Reconstruct Zernike coefficients from slopes via an interaction matrix."""

    def __init__(self, shwfs, n_modes=15):
        self.shwfs = shwfs
        self.n_modes = n_modes
        N = shwfs.N
        self.pupil = aotools.circle(N // 2, N).astype(np.float32)
        self.zern = aotools.zernikeArray(n_modes, N)
        # Build interaction matrix M: slopes = M @ coeffs
        cols = [shwfs.slopes(self.zern[k] * self.pupil) for k in range(n_modes)]
        self.M = np.array(cols).T
        self.M_inv = np.linalg.pinv(self.M)   # command matrix

    def reconstruct(self, slopes):
        """slopes -> Zernike coefficients."""
        return self.M_inv @ slopes

    def to_wavefront(self, coeffs):
        """Zernike coefficients -> 2D wavefront."""
        return np.sum([coeffs[k] * self.zern[k] for k in range(self.n_modes)],
                      axis=0) * self.pupil


class ZonalReconstructor:
    """Southwell-geometry zonal reconstruction: slopes -> phase on grid."""

    def __init__(self, shwfs):
        self.shwfs = shwfs
        self.valid = shwfs.valid
        self.n_sub = shwfs.n_sub
        self._build_geometry()

    def _build_geometry(self):
        ns = self.n_sub
        self.idx = -np.ones((ns, ns), int)
        self.coords = np.argwhere(self.valid)
        for k, (i, j) in enumerate(self.coords):
            self.idx[i, j] = k
        self.nz = len(self.coords)

    def reconstruct(self, sx_grid, sy_grid):
        """Integrate slopes to phase grid (least squares)."""
        rows, rhs = [], []
        for (i, j) in self.coords:
            if j+1 < self.n_sub and self.valid[i, j+1]:
                r = np.zeros(self.nz); r[self.idx[i, j+1]] = 1; r[self.idx[i, j]] = -1
                rows.append(r); rhs.append(sx_grid[i, j])
            if i+1 < self.n_sub and self.valid[i+1, j]:
                r = np.zeros(self.nz); r[self.idx[i+1, j]] = 1; r[self.idx[i, j]] = -1
                rows.append(r); rhs.append(sy_grid[i, j])
        G = np.array(rows); rhs = np.array(rhs)
        phase_grid = np.linalg.lstsq(G, rhs, rcond=None)[0]
        out = np.zeros((self.n_sub, self.n_sub))
        for k, (i, j) in enumerate(self.coords):
            out[i, j] = phase_grid[k]
        return out
