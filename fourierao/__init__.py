"""
FourierAO — Predictive Shack-Hartmann Wavefront Reconstruction,
Turbulence Characterization, and Fourier-Neural-Operator Forecasting.

Bharatiya Antariksh Hackathon 2026.
"""
__version__ = "1.0.0"

import numpy as np
import math
# numpy 2.0 compatibility shim (aotools uses numpy.math)
if not hasattr(np, "math"):
    np.math = math
