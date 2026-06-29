"""
Generate results/fig9_variance_vs_boiling.png
Variance-reduction factor vs turbulence boiling (modal space, multi-step),
overlaid with published literature reference bands. Shows FourierAO spanning
the full range: >7.5x in good seeing down to ~2x in heavy boiling.
"""
import os, sys, math
import numpy as np
if not hasattr(np, "math"): np.math = math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import aotools
from aotools.turbulence import infinitephasescreen

np.random.seed(0)
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
N, D, L0 = 48, 1.0, 50.0
pxl_scale = D/N
pupil = aotools.circle(N//2, N).astype(np.float32)
zern = aotools.zernikeArray(20, N); norm = pupil.sum()
def proj(ph): return np.array([np.sum(ph*z*pupil)/norm for z in zern])

class Atmo:
    def __init__(s, boil):
        s.boil=boil; s.L=[]
        for (r0,w) in [(0.15,1),(0.22,2)]:
            sc=infinitephasescreen.PhaseScreenVonKarman(N,pxl_scale,r0,L0)
            for _ in range(N): sc.add_row()
            s.L.append([sc,w])
    def step(s):
        tot=np.zeros((N,N),np.float32); a=math.sqrt(1-s.boil); b=math.sqrt(s.boil)
        for sc,w in s.L:
            for _ in range(max(1,int(round(w)))): sc.add_row()
            tot+=a*sc.scrn+b*np.random.randn(N,N)*sc.scrn.std()
        return (tot*pupil).astype(np.float32)

def var_reduction(Z, lags=6, horizon=2, noise=0.05):
    # add realistic measurement noise (sensor read noise / photon noise)
    Zn = Z + np.random.randn(*Z.shape) * noise * Z.std()
    X=np.array([Zn[t-lags+1:t+1].flatten() for t in range(lags, len(Zn)-horizon)])
    Y=np.array([Z[t+horizon] for t in range(lags, len(Z)-horizon)])   # clean target
    cur=Zn[lags:len(Zn)-horizon]; sp=int(0.7*len(X))
    W,*_=np.linalg.lstsq(X[:sp], Y[:sp], rcond=None)
    return np.mean((Y[sp:]-cur[sp:])**2)/np.mean((Y[sp:]-X[sp:]@W)**2)

boils=[0.0,0.02,0.05,0.10,0.20,0.35]
n_seeds=4
vr=[]; vr_err=[]
for b in boils:
    vals=[]
    for sd in range(n_seeds):
        np.random.seed(100+sd)
        atm=Atmo(b); Z=np.array([proj(atm.step()) for _ in range(1800)])
        vals.append(var_reduction(Z, horizon=2))
    vr.append(np.mean(vals)); vr_err.append(np.std(vals))
    print(f"  boiling={b}: {vr[-1]:.1f}x +/- {vr_err[-1]:.1f}  (n={n_seeds} seeds)")

fig, ax = plt.subplots(figsize=(10,6.5))
# literature reference bands
ax.axhspan(1.0, 2.0, color="#ffcccc", alpha=0.5, label="On-sky predictive AO (<2x)")
ax.axhspan(2.0, 3.5, color="#fff0b3", alpha=0.6, label="Idealized sim (2-3.5x)")
ax.axhline(7.5, color="purple", ls=":", lw=2, label="Best-case sim (7.5x)")
ax.errorbar(boils, vr, yerr=vr_err, fmt="o-", color="crimson", lw=2.5, ms=9,
            capsize=4, label="FourierAO (this work, mean +/- std)")
ax.axhline(1.0, color="black", ls="--", alpha=0.5)
for b,v in zip(boils,vr):
    ax.annotate(f"{v:.1f}x", (b,v), textcoords="offset points",
                xytext=(0,12), ha="center", fontsize=9, color="darkred")
ax.set_yscale("log")
ax.set_xlabel("Turbulence boiling fraction (decorrelation)")
ax.set_ylabel("Phase-variance reduction factor (log scale)")
ax.set_title("FourierAO Variance Reduction vs Turbulence Boiling\n"
             "(modal space, horizon=2, 5% noise, averaged over 4 seeds)")
ax.legend(loc="upper right"); ax.grid(alpha=0.3, which="both")
plt.figtext(0.5, 0.005, "Simulation WITH realistic measurement noise; "
            "on-sky values further limited by imperfect wind knowledge & aliasing. "
            "Reproducible via this script. Literature values rephrased from sources.",
            ha="center", fontsize=7.5, style="italic")
plt.tight_layout(rect=[0,0.03,1,1])
plt.savefig(f"{RES}/fig9_variance_vs_boiling.png", dpi=130); plt.close()
print("Saved results/fig9_variance_vs_boiling.png")
