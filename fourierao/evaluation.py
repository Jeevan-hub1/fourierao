"""
Evaluation metrics for FourierAO.
"""
import math
import numpy as np
if not hasattr(np, "math"):
    np.math = math


def rms(a, b, mask=None):
    if mask is not None:
        return float(np.sqrt(np.mean((a[..., mask] - b[..., mask]) ** 2)))
    return float(np.sqrt(np.mean((a - b) ** 2)))


def strehl_from_residual(residual_phase, pupil):
    """Maréchal approximation: S ~ exp(-sigma^2) of residual phase variance."""
    var = np.var(residual_phase[pupil > 0])
    return float(np.exp(-var))


def psf(phase, pupil, oversample=2):
    """Point-spread function (the 'star image') from a pupil phase."""
    N = phase.shape[0]
    field = pupil * np.exp(1j * phase)
    p = np.abs(np.fft.fftshift(np.fft.fft2(field, s=(oversample*N, oversample*N))))**2
    return p / p.max()


def improvement_pct(baseline_err, model_err):
    """Percentage improvement of model over baseline."""
    if baseline_err == 0:
        return 0.0
    return 100.0 * (1.0 - model_err / baseline_err)


def prediction_efficiency(y_true, y_pred):
    """PE = 1 - MSE/var; 1=perfect, 0=no skill."""
    var = np.var(y_true)
    if var == 0:
        return 0.0
    return float(1.0 - np.mean((y_true - y_pred) ** 2) / var)
