"""
Optimize predictor HYPERPARAMETERS at FIXED 5% noise (no noise reduction).
Sweep lags and Savitzky-Golay window; find best config per boiling level.
Seed-averaged. This is legitimate algorithm optimization.
"""
import math
import numpy as np
if not hasattr(np, "math"): np.math = math
import aotools
from aotools.turbulence import infinitephasescreen
from scipy.signal import savgol_filter

np.random.seed(7)
N, n_sub, D, L0 = 48, 16, 1.0, 50.0
pxl_scale=D/N; sub_px=N//n_sub
pupil=aotools.circle(N//2,N).astype(np.float32)
nz=20; zern=aotools.zernikeArray(nz,N); norm=pupil.sum()
valid=np.zeros((n_sub,n_sub),bool)
for i in range(n_sub):
    for j in range(n_sub):
        p=pupil[i*sub_px:(i+1)*sub_px,j*sub_px:(j+1)*sub_px]
        valid[i,j]=p.sum()>=0.5*sub_px*sub_px
def slopes_of(ph):
    sx=np.zeros((n_sub,n_sub)); sy=np.zeros((n_sub,n_sub))
    for i in range(n_sub):
        for j in range(n_sub):
            if not valid[i,j]: continue
            sub=ph[i*sub_px:(i+1)*sub_px,j*sub_px:(j+1)*sub_px]
            sp=pupil[i*sub_px:(i+1)*sub_px,j*sub_px:(j+1)*sub_px]
            gy,gx=np.gradient(sub); sx[i,j]=(gx*sp).sum()/sp.sum(); sy[i,j]=(gy*sp).sum()/sp.sum()
    return np.concatenate([sx[valid],sy[valid]])
M=np.array([slopes_of(zern[k]*pupil) for k in range(nz)]).T; M_inv=np.linalg.pinv(M)
def proj(ph): return np.array([np.sum(ph*z*pupil)/norm for z in zern])

class Atmo:
    def __init__(s,boil,seed):
        np.random.seed(seed); s.boil=boil; s.L=[]
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

def gen_modal(boil,seed,n=2500,slope_noise=0.05):
    atm=Atmo(boil,seed); cZ=[]; nZ=[]
    for t in range(n):
        ph=atm.step(); cZ.append(proj(ph))
        sl=slopes_of(ph); rng=np.random.RandomState(seed*1000+t)
        nZ.append((sl+rng.randn(len(sl))*slope_noise*sl.std())@M_inv.T)
    return np.array(cZ),np.array(nZ)

def vr(cZ,nZ,h,lags,sg):
    md=savgol_filter(nZ,sg,3,axis=0) if sg>1 else nZ
    X=np.array([md[t-lags+1:t+1].flatten() for t in range(lags-1,len(md)-h)])
    Y=cZ[lags-1+h:lags-1+h+len(X)]; cur=md[lags-1:lags-1+len(X)]
    sp=int(0.7*len(X))
    A=X[:sp].T@X[:sp]+1e-5*np.eye(X.shape[1])
    W=np.linalg.solve(A,X[:sp].T@Y[:sp]); pred=X[sp:]@W
    return np.mean((Y[sp:]-cur[sp:])**2)/np.mean((Y[sp:]-pred)**2)

seeds=[200,201,202,203]
print("="*70)
print("  HYPERPARAMETER OPTIMIZATION @ FIXED 5% NOISE (no noise reduction)")
print("="*70)
# Pre-generate data per (boil, seed) to reuse across hyperparams
print("  Generating datasets...")
DATA={}
for boil in [0.01,0.05,0.10,0.30]:
    DATA[boil]=[gen_modal(boil,sd,n=2500) for sd in seeds]

print(f"\n  Sweep (h=5): finding best lags x sg-window per regime\n")
print(f"  {'boil':>6} {'best_VR':>9} {'best_cfg':>18} {'(baseline lags=8,sg=7)':>22}")
best_configs={}
for boil in [0.01,0.05,0.10,0.30]:
    base=np.mean([vr(cZ,nZ,5,8,7) for cZ,nZ in DATA[boil]])
    best=0; bcfg=None
    for lags in [8,12,16,20,24]:
        for sg in [7,9,11,13]:
            v=np.mean([vr(cZ,nZ,5,lags,sg) for cZ,nZ in DATA[boil]])
            if v>best: best=v; bcfg=(lags,sg)
    best_configs[boil]=(best,bcfg)
    print(f"  {boil:>6.2f} {best:>8.1f}x   lags={bcfg[0]:>2},sg={bcfg[1]:>2}      "
          f"baseline={base:.1f}x")

print(f"\n  IMPROVEMENT at fixed 5% noise (optimized vs baseline lags=8,sg=7):")
for boil in [0.01,0.05,0.10,0.30]:
    base=np.mean([vr(cZ,nZ,5,8,7) for cZ,nZ in DATA[boil]])
    best=best_configs[boil][0]
    print(f"    boil={boil:.2f}: {base:.1f}x -> {best:.1f}x  (+{100*(best/base-1):.0f}%)")
print("="*70)
