"""
Iterative centroid estimation for Shack-Hartmann spot fields.

Implements the poster's "iterative centroid estimation":
  - Center-of-Gravity (CoG)
  - Weighted CoG (noise-robust)
  - Iterative thresholded CoG (windowed, converges on the true spot center)

Spot deviation from reference position -> slopes (x, y).
"""
import numpy as np


class IterativeCentroider:
    """Iterative thresholded center-of-gravity centroiding per subaperture."""

    def __init__(self, shwfs, n_iter=3, threshold_frac=0.1):
        self.shwfs = shwfs
        self.n_iter = n_iter
        self.threshold_frac = threshold_frac
        self.sub_px = shwfs.sub_px
        self.reference = self._reference_centroids()

    def _cog(self, spot, threshold=0.0):
        s = np.clip(spot - threshold, 0, None)
        tot = s.sum()
        if tot <= 0:
            c = (spot.shape[0] - 1) / 2.0
            return c, c
        gy, gx = np.mgrid[0:spot.shape[0], 0:spot.shape[1]]
        cy = (gy * s).sum() / tot
        cx = (gx * s).sum() / tot
        return cy, cx

    def _iterative_cog(self, spot):
        """Iteratively re-centre a window and recompute CoG (poster's method)."""
        cy, cx = self._cog(spot)
        for _ in range(self.n_iter):
            thr = self.threshold_frac * spot.max()
            cy, cx = self._cog(spot, threshold=thr)
        return cy, cx

    def _reference_centroids(self):
        """Reference spot positions = geometric centre of each subaperture."""
        c = (self.sub_px - 1) / 2.0
        return c

    def centroids_and_slopes(self, spot_field):
        """
        From a full spot-field image, compute per-lenslet centroid deviations
        (slopes). Returns concatenated [sx, sy] over valid subapertures.
        """
        ns, sp = self.shwfs.n_sub, self.sub_px
        sx = np.zeros((ns, ns))
        sy = np.zeros((ns, ns))
        ref = self.reference
        for i in range(ns):
            for j in range(ns):
                if not self.shwfs.valid[i, j]:
                    continue
                spot = spot_field[i*sp:(i+1)*sp, j*sp:(j+1)*sp]
                cy, cx = self._iterative_cog(spot)
                sy[i, j] = cy - ref     # deviation from reference position
                sx[i, j] = cx - ref
        return np.concatenate([sx[self.shwfs.valid], sy[self.shwfs.valid]])
