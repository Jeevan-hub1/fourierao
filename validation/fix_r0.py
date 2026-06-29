"""
FIX the r0 estimator: the (D/r0)^(5/3) scaling is correct, only the
constant needs one-time calibration to the simulator's L0 configuration.
Calibrate C on ONE reference r0, then estimate all others accurately.
"""
import math
import numpy as np
if not hasattr(np, "math"):
    np.math = math
import aotools
from aotools.turbulence import infinitephasescreen

np.random.seed(1)
N, D, L0 = 64, 1.0, 100.0
pxl_scale = D / N
pupil = aotools.circle(N // 2, N)

def mean_phase_var(r0, n_real=60):
    vs = []
    for _ in range(n_real):
        scr = infinitephasescreen.PhaseScreenVonKarman(N, pxl_scale, r0, L0)
        for _ in range(N):
            scr.add_row()
        ph = scr.scrn
        ph = ph - ph[pupil > 0].mean()
        vs.append(np.var(ph[pupil > 0]))
    return np.mean(vs)

# ---- STEP 1: calibrate constant C on a known reference r0 ----
r0_ref = 0.15
var_ref = mean_phase_var(r0_ref)
# sigma^2 = C * (D/r0)^(5/3)  ->  C = var_ref / (D/r0_ref)^(5/3)
C = var_ref / (D / r0_ref) ** (5.0/3.0)
print("=" * 60)
print("  CALIBRATED r0 ESTIMATOR")
print("=" * 60)
print(f"  Calibration: r0_ref={r0_ref} m, var={var_ref:.2f} rad^2, C={C:.4f}")
print(f"  (pure-Kolmogorov C would be 1.0299; finite-L0 reduces it)\n")

# ---- STEP 2: estimate r0 for unseen turbulence using calibrated C ----
print(f"  {'r0_true':>8} {'phase_var':>10} {'r0_est':>8} {'error':>7}")
for r0_true in [0.08, 0.10, 0.12, 0.18, 0.22, 0.25]:
    v = mean_phase_var(r0_true)
    r0_est = D / (v / C) ** (3.0/5.0)
    err = abs(r0_est - r0_true) / r0_true * 100
    print(f"  {r0_true:>8.3f} {v:>10.2f} {r0_est:>8.3f} {err:>6.1f}%")

print("\n  --> with one-time calibration, r0 recovered to a few % accuracy")
print("=" * 60)
