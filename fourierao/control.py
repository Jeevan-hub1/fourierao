"""
Deformable Mirror control: reconstructed wavefront -> conjugate ->
actuator map (in stroke-length units, nm) -> closed-loop correction.

Implements the problem-statement final stages:
  "The conjugate of this reconstructed wavefront is typically used to derive
   an actuator map in units of the actuator's stroke length, which is then
   fed to a deformable mirror (DM) to correct for this distortion."
"""
import numpy as np
import aotools


class DeformableMirror:
    """Maps a reconstructed wavefront to DM actuator commands (stroke, nm)."""

    def __init__(self, N, n_act=9, wavelength_nm=500.0):
        self.N = N
        self.n_act = n_act
        self.act_px = N // n_act
        self.wavelength_nm = wavelength_nm
        self.pupil = aotools.circle(N // 2, N).astype(np.float32)

    def actuator_map(self, reconstructed_wavefront):
        """
        Conjugate of the reconstructed wavefront -> actuator stroke map (nm).
        stroke = phase(rad) * lambda / (2*pi) / 2   (the /2 = reflection).
        """
        dm_shape = -reconstructed_wavefront            # conjugate
        amap = np.zeros((self.n_act, self.n_act))
        ap = self.act_px
        for i in range(self.n_act):
            for j in range(self.n_act):
                patch = dm_shape[i*ap:(i+1)*ap, j*ap:(j+1)*ap]
                phase_rad = patch.mean()
                amap[i, j] = phase_rad * self.wavelength_nm / (2 * np.pi) / 2.0
        return amap

    def apply(self, reconstructed_wavefront):
        """Return the DM correction surface (radians) = conjugate, full-res."""
        return -reconstructed_wavefront


class ClosedLoop:
    """
    Single-conjugate AO closed loop with optional prediction.
    Integrator control:  command += gain * residual_command.
    """

    def __init__(self, shwfs, reconstructor, dm, gain=0.4, predictor=None):
        self.shwfs = shwfs
        self.recon = reconstructor
        self.dm = dm
        self.gain = gain
        self.predictor = predictor
        self.pupil = shwfs.pupil
        self.pmask = self.pupil > 0
        self.dm_surface = np.zeros((shwfs.N, shwfs.N), np.float32)
        self.recon_scale = self._calibrate_scale()

    def _calibrate_scale(self):
        """Calibrate reconstruction-to-phase scale to guarantee loop stability."""
        import aotools
        N = self.shwfs.N
        test = aotools.zernikeArray(6, N)[3] * self.pupil   # a defocus probe
        c = self.recon.reconstruct(self.shwfs.slopes(test))
        rec = self.recon.to_wavefront(c)
        denom = np.sum(rec[self.pmask] ** 2)
        if denom <= 0:
            return 1.0
        return float(np.sum(rec[self.pmask] * test[self.pmask]) / denom)

    def step(self, incoming_phase, predicted_phase=None):
        """One closed-loop iteration. Returns (residual_phase, strehl).
        If predicted_phase is given, the loop pre-corrects it (servo-lag cancel)."""
        sense_phase = predicted_phase if predicted_phase is not None else incoming_phase
        residual = sense_phase + self.dm_surface          # what the WFS senses
        slopes = self.shwfs.slopes(residual)
        coeffs = self.recon.reconstruct(slopes)
        recon_wf = self.recon.to_wavefront(coeffs) / max(self.recon_scale, 1e-3)
        # integrator update: push DM to cancel the reconstructed residual
        self.dm_surface = self.dm_surface - self.gain * recon_wf
        res = (incoming_phase + self.dm_surface) * self.pupil
        strehl = float(np.exp(-np.var(res[self.pmask])))
        return res, strehl
