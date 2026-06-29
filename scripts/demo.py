"""
FourierAO end-to-end demo:
  generate turbulence -> SH-WFS -> reconstruct -> characterize -> predict.
Run:  python scripts/demo.py
"""
import os, sys, math
import numpy as np
if not hasattr(np, "math"): np.math = math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fourierao.simulator import Atmosphere, ShackHartmann
from fourierao.reconstruction import ModalReconstructor
from fourierao.characterization import TurbulenceCharacterizer
from fourierao import evaluation as E
import aotools

N, n_sub, D, L0 = 48, 16, 1.0, 50.0
pxl_scale = D / N
pupil = aotools.circle(N//2, N).astype(np.float32)

print("="*56)
print("  FourierAO — end-to-end demo")
print("="*56)

atm = Atmosphere(N, pxl_scale, [(0.15, 1), (0.22, 2)], L0=L0, boiling=0.1, seed=1)
shwfs = ShackHartmann(N, n_sub)
modal = ModalReconstructor(shwfs, n_modes=15)
tc = TurbulenceCharacterizer(N, D, pxl_scale)

# reconstruct one frame
phase = atm.step()
coeffs = modal.reconstruct(shwfs.slopes(phase))
recon = modal.to_wavefront(coeffs)
print(f"\nReconstruction: residual Strehl "
      f"{E.strehl_from_residual(phase,pupil):.3f} -> "
      f"{E.strehl_from_residual(phase-recon,pupil):.3f}")

# characterize
frames = atm.sequence(60)
cal = Atmosphere(N, pxl_scale, [(0.15,1)], L0=L0, seed=2)
tc.calibrate(cal.sequence(40), 0.15)
r0 = tc.estimate_r0(frames)
wind, direction = tc.estimate_wind(frames)
tau0 = tc.estimate_tau0(r0, wind)
print(f"Turbulence: r0={r0:.3f} m | wind={wind:.1f} m/s @ {direction:.0f} deg "
      f"| tau0={tau0*1000:.2f} ms")
print("\nDemo complete. Run scripts/generate_results.py for full figures.")
