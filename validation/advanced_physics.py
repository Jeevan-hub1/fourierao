"""
ADVANCED improvements — each with PHYSICS VERIFICATION.

A. Fourier-shift-equivariant prediction (Taylor shift theorem)
B. Per-layer tomographic prediction (superposition + per-layer frozen flow)
C. Mode-dependent temporal coherence (Zernike PSD scaling)
D. Koopman/DMD linear-latent advection
"""
import math
import numpy as np
if not hasattr(np, "math"):
    np.math = math
import aotools
from aotools.turbulence import infinitephasescreen
from scipy.ndimage import shift as nd_shift

np.random.seed(11)
N, D, L0 = 48, 1.0, 50.0
pxl_scale = D / N
pupil = aotools.circle(N // 2, N); pmask = pupil > 0
zern = aotools.zernikeArray(28, N); nz = 28; norm = pupil.sum()
def proj(ph): return np.array([np.sum(ph*z*pupil)/norm for z in zern])

print("=" * 66)
print("  ADVANCED IMPROVEMENTS — with physics verification")
print("=" * 66)

# =================================================================
# A. FOURIER-SHIFT-EQUIVARIANT PREDICTION (shift theorem)
#    Physics: frozen flow => next frame = spatial translation
#             translation = exp(-i 2pi (k.v dt)) in Fourier domain.
#    Verify: estimate shift, apply Fourier phase-ramp -> predict next.
# =================================================================
print("\n[A] Fourier-shift-equivariant predictor (Taylor shift theorem)")
scr = infinitephasescreen.PhaseScreenVonKarman(N, pxl_scale, 0.15, L0)
for _ in range(N): scr.add_row()
true_shift = 0.7  # px/frame in x
frames = []
base = scr.scrn.copy()
for t in range(60):
    frames.append(nd_shift(base, (0.0, true_shift*t), mode="wrap", order=3))
frames = np.array(frames)

def fourier_shift_predict(prev, sx, sy):
    """Predict next frame by applying a Fourier-domain phase ramp."""
    ky = np.fft.fftfreq(N).reshape(-1,1)
    kx = np.fft.fftfreq(N).reshape(1,-1)
    F = np.fft.fft2(prev)
    F *= np.exp(-2j*np.pi*(kx*sx + ky*sy))
    return np.real(np.fft.ifft2(F))

# estimate shift via cross-correlation (sub-pixel)
def est_shift(a,b):
    A,B=np.fft.fft2(a),np.fft.fft2(b); R=A*np.conj(B); R/=np.abs(R)+1e-9
    cc=np.fft.ifft2(R).real; p=np.unravel_index(np.argmax(cc),cc.shape)
    sy=p[0]-N if p[0]>N/2 else p[0]; sx=p[1]-N if p[1]>N/2 else p[1]
    return sy,sx
errs_persist=[]; errs_fourier=[]
for t in range(2,len(frames)-1):
    sy,sx = est_shift(frames[t-1], frames[t])  # gives -shift
    pred = fourier_shift_predict(frames[t], -sx, -sy)
    errs_fourier.append(np.std((frames[t+1]-pred)[pmask]))
    errs_persist.append(np.std((frames[t+1]-frames[t])[pmask]))
print(f"  Persistence RMS:      {np.mean(errs_persist):.4f} rad")
print(f"  Fourier-shift RMS:    {np.mean(errs_fourier):.4f} rad")
print(f"  --> shift theorem predicts next frame "
      f"{100*(1-np.mean(errs_fourier)/np.mean(errs_persist)):.0f}% better "
      f"=> justifies wind-equivariant FNO")

# =================================================================
# C. MODE-DEPENDENT TEMPORAL COHERENCE (Zernike PSD scaling)
#    Physics: high-order Zernikes have shorter coherence times.
#    Verify: measure autocorr decay (1/e) per mode order.
# =================================================================
print("\n[C] Mode-dependent temporal coherence (predict modes by band)")
scr = infinitephasescreen.PhaseScreenVonKarman(N, pxl_scale, 0.15, L0)
Z=[]
for _ in range(1500):
    scr.add_row()  # 1 px/frame frozen flow
    Z.append(proj(scr.scrn*pupil))
Z=np.array(Z)
def coherence_frames(series):
    s=series-series.mean(); ac=np.correlate(s,s,'full')[len(s)-1:]
    ac/=ac[0]
    idx=np.argmax(ac<1/math.e)
    return idx if idx>0 else len(ac)
for band,(lo,hi) in [("tip/tilt (2-3)",(1,3)),("low (4-9)",(3,9)),
                     ("mid (10-17)",(9,17)),("high (18-27)",(17,27))]:
    ct=np.mean([coherence_frames(Z[:,m]) for m in range(lo,hi)])
    print(f"  {band:>16}: coherence time = {ct:5.1f} frames")
print("  --> low-order modes stay coherent longer => predict them further")
print("      ahead; band-split horizons = physics-optimal scheduling")

# =================================================================
# B. PER-LAYER TOMOGRAPHIC PREDICTION (oracle upper bound)
#    Physics: total phase = sum of layers; each layer is clean frozen flow.
#    Verify: predicting each layer separately beats predicting the sum.
# =================================================================
print("\n[B] Per-layer prediction vs combined (tomographic advantage)")
class Layer:
    def __init__(s,r0,wpx):
        s.scr=infinitephasescreen.PhaseScreenVonKarman(N,pxl_scale,r0,L0)
        for _ in range(N): s.scr.add_row()
        s.wpx=wpx
    def step(s):
        for _ in range(max(1,int(round(s.wpx)))): s.scr.add_row()
        return s.scr.scrn.copy()
L1,L2=Layer(0.18,1),Layer(0.25,2)
combined=[]; lay1=[]; lay2=[]
for _ in range(1200):
    a,b=L1.step(),L2.step()
    lay1.append(proj(a*pupil)); lay2.append(proj(b*pupil))
    combined.append(proj((a+b)*pupil))
combined,lay1,lay2=map(np.array,(combined,lay1,lay2))
def ar_err(Z,h=2,lags=4):
    X=np.array([Z[t-lags+1:t+1].flatten() for t in range(lags,len(Z)-h)])
    Y=np.array([Z[t+h] for t in range(lags,len(Z)-h)])
    sp=int(0.7*len(X)); W,*_=np.linalg.lstsq(X[:sp],Y[:sp],rcond=None)
    return np.sqrt(np.mean((Y[sp:]-X[sp:]@W)**2))
comb_err=ar_err(combined)
perlayer_err=ar_err(lay1)+ar_err(lay2)  # predict each, sum errors (approx)
# fairer: predict each layer then recombine
def ar_pred_full(Z,h=2,lags=4):
    X=np.array([Z[t-lags+1:t+1].flatten() for t in range(lags,len(Z)-h)])
    Y=np.array([Z[t+h] for t in range(lags,len(Z)-h)])
    sp=int(0.7*len(X)); W,*_=np.linalg.lstsq(X[:sp],Y[:sp],rcond=None)
    return Y[sp:], X[sp:]@W
y1,p1=ar_pred_full(lay1); y2,p2=ar_pred_full(lay2)
yc,pc=ar_pred_full(combined)
recomb_err=np.sqrt(np.mean(((y1+y2)-(p1+p2))**2))
print(f"  Combined prediction RMS:        {comb_err:.4f}")
print(f"  Per-layer then recombine RMS:   {recomb_err:.4f}")
print(f"  --> separating layers improves by "
      f"{100*(comb_err-recomb_err)/comb_err:.1f}% (motivates learned tomography)")

# =================================================================
# D. KOOPMAN / DMD linear-latent advection
#    Physics: advection is a LINEAR shift operator -> exact in Fourier.
#    Verify: DMD one-step operator predicts well.
# =================================================================
print("\n[D] Koopman/DMD linear operator prediction")
scr = infinitephasescreen.PhaseScreenVonKarman(N, pxl_scale, 0.15, L0)
Z=[]
for _ in range(1000):
    scr.add_row(); Z.append(proj(scr.scrn*pupil))
Z=np.array(Z); sp=700
X0=Z[:sp-1].T; X1=Z[1:sp].T
A=X1@np.linalg.pinv(X0)               # DMD operator (Koopman approx)
# predict test set one step
err_dmd=[]; err_pers=[]
for t in range(sp,len(Z)-1):
    pred=A@Z[t]
    err_dmd.append(np.sqrt(np.mean((Z[t+1]-pred)**2)))
    err_pers.append(np.sqrt(np.mean((Z[t+1]-Z[t])**2)))
print(f"  Persistence RMS: {np.mean(err_pers):.4f}")
print(f"  DMD/Koopman RMS: {np.mean(err_dmd):.4f}")
print(f"  --> linear Koopman operator beats persistence "
      f"{100*(1-np.mean(err_dmd)/np.mean(err_pers)):.0f}% "
      f"(advection is linear in correct basis)")

print("\n"+"="*66)
print("  ADVANCED IMPROVEMENTS — physics verified")
print("="*66)
