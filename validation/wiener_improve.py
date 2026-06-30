"""
IMPROVE at FIXED 5% noise via better RECONSTRUCTION (no noise reduction).

Idea: replace least-squares reconstructor (M^+) with Wiener/MMSE (Bayesian)
reconstructor that uses the turbulence statistical prior C_aa and noise
covariance C_n:
    W_wiener = C_aa M^T (M C_aa M^T + C_n)^-1
This is the minimum-variance estimator -> lower reconstruction error AND
cleaner inputs to the predictor, all at the SAME 5% slope noise.

Also test: per-mode Kalman smoother on the predicted modal series.
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
M=np.array([slopes_of(zern[k]*pupil) for k in range(nz)]).T
M_ls=np.linalg.pinv(M)   # least-squares reconstructor
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

def gen(boil,seed,n,slope_noise=0.05):
    atm=Atmo(boil,seed); cZ=[]; cS=[]; nS=[]
    for t in range(n):
        ph=atm.step(); cZ.append(proj(ph))
        sl=slopes_of(ph); cS.append(sl)
        rng=np.random.RandomState(seed*1000+t)
        nS.append(sl+rng.randn(len(sl))*slope_noise*sl.std())
    return np.array(cZ),np.array(cS),np.array(nS)

def build_wiener(cZ_train, cS_train, slope_noise):
    """Wiener reconstructor from empirical statistics (training only)."""
    C_aa = np.cov(cZ_train.T)                      # turbulence modal prior
    noise_var = (slope_noise*cS_train.std())**2
    C_n = noise_var*np.eye(M.shape[0])
    # W = C_aa M^T (M C_aa M^T + C_n)^-1
    MCaaMT = M @ C_aa @ M.T + C_n
    W = C_aa @ M.T @ np.linalg.inv(MCaaMT)
    return W

def ar_vr(modal_clean, modal_input, h, lags=12, sg=9):
    md = savgol_filter(modal_input, sg, 3, axis=0)
    X=np.array([md[t-lags+1:t+1].flatten() for t in range(lags-1,len(md)-h)])
    Y=modal_clean[lags-1+h:lags-1+h+len(X)]; cur=md[lags-1:lags-1+len(X)]
    sp=int(0.7*len(X))
    A=X[:sp].T@X[:sp]+1e-5*np.eye(X.shape[1])
    W=np.linalg.solve(A,X[:sp].T@Y[:sp]); pred=X[sp:]@W
    return np.mean((Y[sp:]-cur[sp:])**2)/np.mean((Y[sp:]-pred)**2)

print("="*70)
print("  WIENER (MMSE) RECONSTRUCTION vs LEAST-SQUARES  (FIXED 5% noise)")
print("="*70)
seeds=[200,201,202,203]; noise=0.05

print("\n[1] Reconstruction accuracy (modal RMSE vs truth):")
for boil in [0.01, 0.05, 0.30]:
    ls_err=[]; wn_err=[]
    for sd in seeds:
        cZ,cS,nS=gen(boil,sd,n=1500,slope_noise=noise)
        sp=int(0.7*len(cZ))
        W_w=build_wiener(cZ[:sp],cS[:sp],noise)
        a_ls=nS@M_ls.T          # least-squares recon
        a_wn=nS@W_w.T           # Wiener recon
        ls_err.append(np.sqrt(np.mean((a_ls[sp:]-cZ[sp:])**2)))
        wn_err.append(np.sqrt(np.mean((a_wn[sp:]-cZ[sp:])**2)))
    print(f"  boil={boil:.2f}: LeastSq RMSE={np.mean(ls_err):.4f} | "
          f"Wiener RMSE={np.mean(wn_err):.4f} | "
          f"improvement={100*(1-np.mean(wn_err)/np.mean(ls_err)):.1f}%")

print("\n[2] Prediction variance-reduction: LS-input vs Wiener-input (h=5):")
for boil in [0.01, 0.05, 0.10, 0.30]:
    ls_vr=[]; wn_vr=[]
    for sd in seeds:
        cZ,cS,nS=gen(boil,sd,n=2200,slope_noise=noise)
        sp=int(0.7*len(cZ))
        W_w=build_wiener(cZ[:sp],cS[:sp],noise)
        a_ls=nS@M_ls.T; a_wn=nS@W_w.T
        ls_vr.append(ar_vr(cZ,a_ls,h=5))
        wn_vr.append(ar_vr(cZ,a_wn,h=5))
    print(f"  boil={boil:.2f}: LS-input={np.mean(ls_vr):.1f}x | "
          f"Wiener-input={np.mean(wn_vr):.1f}x | "
          f"gain={100*(np.mean(wn_vr)/np.mean(ls_vr)-1):+.0f}%")
print("\n"+"="*70)
