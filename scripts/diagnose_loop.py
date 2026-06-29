"""Diagnose why the closed loop diverges. Test on STATIC turbulence:
a working loop must drive Strehl -> ~1 on a fixed aberration."""
import os, sys, math
import numpy as np
if not hasattr(np, "math"): np.math = math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fourierao.simulator import Atmosphere, ShackHartmann
from fourierao.reconstruction import ModalReconstructor
import aotools

N, n_sub, D, L0 = 48, 16, 1.0, 50.0
pxl_scale=D/N; pupil=aotools.circle(N//2,N).astype(np.float32); pm=pupil>0
shwfs=ShackHartmann(N,n_sub); modal=ModalReconstructor(shwfs,n_modes=20)

# 1) Is reconstruction accurate? (recon_wf should ~ input phase)
atm=Atmosphere(N,pxl_scale,[(0.15,1)],L0=L0,seed=1)
ph=atm.step()
coeffs=modal.reconstruct(shwfs.slopes(ph))
recon=modal.to_wavefront(coeffs)
# compare on modes 2+ (piston/scale aside)
scale=np.sum(recon[pm]*ph[pm])/np.sum(recon[pm]**2)
print(f"[recon] reconstruction-to-truth scale factor = {scale:.3f}")
print(f"        (if far from 1.0 -> interaction matrix scale mismatch = divergence)")
print(f"[recon] corr(recon, truth) = {np.corrcoef(recon[pm],ph[pm])[0,1]:.3f}")

# 2) Static closed loop with the CORRECT scale + safe gain
print("\n[static loop] driving Strehl on a FIXED aberration:")
dm_surface=np.zeros((N,N),np.float32)
for gain in [0.3, 0.5]:
    dm_surface=np.zeros((N,N),np.float32)
    print(f"  gain={gain}:", end=" ")
    Ss=[]
    for it in range(15):
        residual=(ph+dm_surface)
        c=modal.reconstruct(shwfs.slopes(residual))
        rec=modal.to_wavefront(c)
        dm_surface = dm_surface - gain*rec/max(scale,1e-3)  # scale-corrected
        S=math.exp(-np.var((ph+dm_surface)[pm]))
        Ss.append(S)
    print(f"Strehl {Ss[0]:.3f} -> {Ss[-1]:.3f}  {'CONVERGES' if Ss[-1]>Ss[0] else 'DIVERGES'}")
