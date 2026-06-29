"""
HONEST feasibility: test the FNO the RIGHT way.
  - early stopping (keep best-val model)
  - MULTI-STEP horizons (1,3,5 frames = real servo-lag), not just 1-step
  - heavier boiling (where nonlinearity lives)
  - compare FNO vs persistence vs linear-AR vs Koopman at each horizon
Find the regime (if any) where the FNO genuinely wins.
"""
import math, time, copy
import numpy as np
if not hasattr(np, "math"): np.math = math
import aotools
from aotools.turbulence import infinitephasescreen
import torch, torch.nn as nn

torch.manual_seed(0); np.random.seed(0)
N, D, L0 = 32, 1.0, 50.0
pxl_scale=D/N
pupil=aotools.circle(N//2,N).astype(np.float32); m=pupil>0
print("="*64); print("  HONEST FEASIBILITY — correct metric (multi-step)"); print("="*64)

class Atmo:
    def __init__(s,layers,boil):
        s.boil=boil; s.L=[]
        for (r0,w) in layers:
            sc=infinitephasescreen.PhaseScreenVonKarman(N,pxl_scale,r0,L0)
            for _ in range(N): sc.add_row()
            s.L.append([sc,w])
    def step(s):
        tot=np.zeros((N,N),np.float32); a=math.sqrt(1-s.boil); b=math.sqrt(s.boil)
        for sc,w in s.L:
            for _ in range(max(1,int(round(w)))): sc.add_row()
            tot+=a*sc.scrn+b*np.random.randn(N,N)*sc.scrn.std()
        return (tot*pupil).astype(np.float32)

BOIL=0.30   # heavier boiling = nonlinear regime
print(f"\n[1] Generating frames (boiling={BOIL})...")
atm=Atmo([(0.15,1),(0.22,2)],BOIL)
frames=np.array([atm.step() for _ in range(2500)])
mu,sd=frames.mean(),frames.std(); F=(frames-mu)/sd

class SpectralConv2d(nn.Module):
    def __init__(s,ci,co,m1,m2):
        super().__init__(); s.m1,s.m2=m1,m2; sc=1/(ci*co)
        s.w1=nn.Parameter(sc*torch.rand(ci,co,m1,m2,dtype=torch.cfloat))
        s.w2=nn.Parameter(sc*torch.rand(ci,co,m1,m2,dtype=torch.cfloat))
    def forward(s,x):
        B,C,H,W=x.shape; xf=torch.fft.rfft2(x)
        o=torch.zeros(B,s.w1.shape[1],H,W//2+1,dtype=torch.cfloat)
        o[:,:,:s.m1,:s.m2]=torch.einsum("bixy,ioxy->boxy",xf[:,:,:s.m1,:s.m2],s.w1)
        o[:,:,-s.m1:,:s.m2]=torch.einsum("bixy,ioxy->boxy",xf[:,:,-s.m1:,:s.m2],s.w2)
        return torch.fft.irfft2(o,s=(H,W))

class FNO(nn.Module):
    def __init__(s,w=20,md=10):
        super().__init__()
        s.fc0=nn.Conv2d(1,w,1)
        s.s1=SpectralConv2d(w,w,md,md); s.c1=nn.Conv2d(w,w,1)
        s.s2=SpectralConv2d(w,w,md,md); s.c2=nn.Conv2d(w,w,1)
        s.f1=nn.Conv2d(w,32,1); s.f2=nn.Conv2d(32,1,1)
        nn.init.zeros_(s.f2.weight); nn.init.zeros_(s.f2.bias)
    def forward(s,x):
        h=s.fc0(x); h=torch.relu(s.s1(h)+s.c1(h)); h=torch.relu(s.s2(h)+s.c2(h))
        return x+s.f2(torch.relu(s.f1(h)))   # residual

def train_es(H):
    """train FNO to predict H steps ahead, with EARLY STOPPING."""
    X=torch.tensor(F[:-H]).unsqueeze(1); Y=torch.tensor(F[H:]).unsqueeze(1)
    ntr=int(0.7*len(X)); nval=int(0.85*len(X))
    Xtr,Ytr=X[:ntr],Y[:ntr]; Xv,Yv=X[ntr:nval],Y[ntr:nval]; Xte,Yte=X[nval:],Y[nval:]
    model=FNO(); opt=torch.optim.Adam(model.parameters(),1e-3); lf=nn.MSELoss(); bs=32
    best=1e9; best_state=None; patience=0
    for ep in range(80):
        model.train(); perm=torch.randperm(len(Xtr))
        for i in range(0,len(Xtr),bs):
            idx=perm[i:i+bs]; opt.zero_grad()
            lf(model(Xtr[idx]),Ytr[idx]).backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl=lf(model(Xv),Yv).item()
        if vl<best-1e-6: best=vl; best_state=copy.deepcopy(model.state_dict()); patience=0
        else:
            patience+=1
            if patience>=8: break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(): pred=model(Xte).numpy().squeeze(1)
    yte=Yte.numpy().squeeze(1); xte=Xte.numpy().squeeze(1)
    # baselines
    per=np.sqrt(np.mean((yte[:,m]-xte[:,m])**2))
    # linear AR(H) global scalar
    Xa=X[:ntr].numpy().squeeze(1)[:,m]; Ya=Y[:ntr].numpy().squeeze(1)[:,m]
    phi=np.sum(Xa*Ya)/np.sum(Xa*Xa)
    lin=np.sqrt(np.mean((yte[:,m]-phi*xte[:,m])**2))
    fno=np.sqrt(np.mean((yte[:,m]-pred[:,m])**2))
    return per,lin,fno,ep

print("\n[2] Training FNO with early stopping at horizons 1,3,5...")
print(f"  {'horizon':>8} {'persist':>9} {'linAR':>9} {'FNO':>9} {'FNOvsPer':>9} {'epochs':>7}")
for H in [1,3,5]:
    t0=time.perf_counter()
    per,lin,fno,ep=train_es(H)
    dt=time.perf_counter()-t0
    imp=100*(1-fno/per)
    print(f"  {H:>8} {per:>9.4f} {lin:>9.4f} {fno:>9.4f} {imp:>+8.1f}% {ep:>7} ({dt:.0f}s)")

print("\n  INTERPRETATION:")
print("  - if FNO gain GROWS with horizon -> prediction value is real (servo-lag)")
print("  - if FNO ~ linear at all horizons -> linear/Koopman is enough; reframe")
print("="*64)
