# FourierAO

**Predictive Shack-Hartmann Wavefront Reconstruction, Turbulence Characterization, and Fourier-Neural-Operator Forecasting**

*Bharatiya Antariksh Hackathon 2026 — Challenge: "Developing and optimizing algorithms for Wavefront reconstruction and turbulence characterization using Shack-Hartmann Wavefront Sensor (SH-WFS) time-series data."*

---

## Overview

Atmospheric turbulence distorts wavefronts; a Shack-Hartmann sensor measures the distortion via a lenslet spot-field; adaptive optics corrects it with a deformable mirror. The dominant limit is **servo-lag**: by the time the correction is applied, the turbulence has already moved.

**FourierAO** is a complete, real-time SH-WFS engine that:

1. **Reconstructs** the wavefront — both **modal** (Zernike) and **zonal** (Southwell) — at **< 1 ms/frame**.
2. **Characterizes** the turbulence live — Fried parameter **r₀**, **wind** speed/direction (sub-pixel), and Greenwood time **τ₀** — directly from the spot time-series.
3. **Predicts** the wavefront forward with an **optimized predictor** that cancels servo-lag, plus a **residual Fourier Neural Operator (FNO)** that holds its advantage in the boiling regime where linear methods collapse.

---

## Final Results

### Optimized Predictor — all at a FIXED, realistic 5% measurement noise

FourierAO reduces the residual phase-variance by the factors below (validated on held-out seeds, 5% slope-level noise, no noise reduction applied):

![Peak performance](results/fig12_peak_performance.png)

| Regime | Conditions | Variance reduction |
|---|---|---|
| **Favorable** | frozen-flow, excellent seeing | **~16×** |
| **Good** | good seeing | **~9×** |
| **Moderate** | typical conditions | **~7×** |
| **Challenging** | heavy boiling | **~3×** |

**Up to ~16× variance reduction at 5% measurement noise — more than double the 7.5× published best-case benchmark.** The result was reached by optimizing the predictor's temporal history (16 lags) and denoising window (Savitzky-Golay, width 13), with the noise level held fixed at a realistic 5%.

### Benchmark vs Published Literature

![Literature benchmark](results/fig8_literature_benchmark.png)

| Work | Variance reduction | Regime |
|---|---|---|
| On-sky predictive AO (2023) | < 2× | real telescope |
| XAO prediction vs SPHERE (2020) | 2.0× | idealized sim |
| Spatiotemporal GP, SH-WFS (2024) | 3.5× | idealized (perfect wind) |
| Best-case predictive sim (2023) | 7.5× | noiseless upper bound |
| **FourierAO (this work)** | **3× → 16×** | **realistic sim, WITH 5% noise** |

### FNO Advantage in the Boiling Regime

Linear predictors (AR, Koopman) collapse to ~0% gain under realistic atmospheric boiling. The residual FNO uniquely maintains **27–33% improvement**, growing with prediction horizon.

![FNO advantage](results/fig6_key_result.png)

### Prediction Gain Grows with Servo-Lag Horizon

The longer the loop delay to be compensated, the more prediction helps — the physical signature of genuine servo-lag cancellation.

![Variance vs horizon](results/fig10_variance_vs_horizon.png)

### Operational Specifications — All Met

| Metric | Specification | Measured |
|---|---|---|
| Reconstruction latency | < 1 ms/frame | **0.007 ms** |
| Throughput | 500 fps | **>150,000 fps** |
| Temporal stability | σ < 0.05 λ | **0.009 λ** |
| Closed-loop Strehl | converges | **0.03 → 0.55** |

---

## Results Gallery

| Wavefront Reconstruction (spot field → 3D wavefront) | Star Image (PSF) before vs after correction |
|---|---|
| ![recon](results/fig1_reconstruction.png) | ![psf](results/fig4_psf.png) |

| Turbulence Characterization (r₀ accuracy) | Algorithm Optimization Console |
|---|---|
| ![char](results/fig5_characterization.png) | ![console](results/fig7_optimization_console.png) |

---

## The Optimizing Algorithm (how we reached ~16× at fixed noise)

Each stage independently verified to contribute:

1. **Iterative centroid estimation** — robust thresholded center-of-gravity over the lenslet spot field.
2. **Correct slope-level noise model** — noise enters at the lenslet level (physically faithful), not the modal level.
3. **Modal reconstruction denoising** — averaging 416 slope measurements into 20 modal coefficients provides ~4.6× intrinsic noise reduction.
4. **Temporal denoising** — Savitzky-Golay filter (window 13, order 3) on the modal time-series.
5. **History-based predictor** — 16-lag ridge-regularized autoregressive filter (Wiener-optimal for the linear dynamics).
6. **Residual FNO** — for the pixel-domain boiling regime where linear methods fail.

A Wiener/MMSE Bayesian reconstructor was also tested but gave negligible gain (the system is heavily overdetermined, so least-squares is already near-optimal) — documented honestly in `validation/`.

---

## Architecture

```
 SH-WFS spot time-series
        │
        ▼
 Iterative centroiding → slopes
        │
        ├──► Modal (Zernike) + Zonal (Southwell) reconstruction
        │
        ├──► Turbulence characterization (r₀, wind, τ₀)
        │
        └──► Optimized predictor (denoise → 16-lag AR + residual FNO)
                    │
                    ▼
             DM actuator map (stroke nm) → closed-loop correction
```

---

## Quick Start

```bash
pip install -r requirements.txt
# CPU torch: pip install torch --index-url https://download.pytorch.org/whl/cpu

python scripts/demo.py                    # end-to-end demo
python scripts/verify_requirements.py     # verify all poster specs
python scripts/peak_performance.py        # reproduce headline ~16x result
python scripts/generate_results.py        # regenerate base figures
```

---

## Project Layout

```
fourierao/
├── fourierao/               # working package (8 modules)
│   ├── simulator.py         # multi-layer turbulence + boiling, SH-WFS, DM
│   ├── centroiding.py       # iterative thresholded center-of-gravity
│   ├── reconstruction.py    # modal (Zernike) + zonal (Southwell)
│   ├── characterization.py  # r₀ (calibrated), wind (sub-pixel), τ₀
│   ├── prediction.py        # persistence, linear-AR, Koopman, residual FNO
│   ├── control.py           # DM actuator map + closed-loop integrator
│   └── evaluation.py        # RMS, Strehl, PSF, prediction efficiency
├── scripts/                 # runnable scripts + figure generators
├── results/                 # final figures (PNG)
├── validation/              # physics de-risking + ablation scripts
├── paper/                   # IEEE conference paper (LaTeX / Overleaf)
├── docs/                    # architecture, pitch script, poster mapping
└── requirements.txt
```

---

## Poster Requirement Fulfillment

Every element of the official poster is implemented and verified end-to-end
(`python scripts/verify_requirements.py`). Full mapping in
[docs/POSTER_MAPPING.md](docs/POSTER_MAPPING.md).

---

## Honest Notes

- **~16× is achieved in frozen-flow conditions** (valid on ~10–100 ms timescales at good sites); performance degrades gracefully to ~3× under heavy atmospheric boiling.
- **All results use a fixed, realistic 5% slope-level measurement noise** — the ~16× was reached by optimizing the predictor, *not* by reducing noise.
- **Numbers are seed-averaged and validated on held-out seeds** not used for tuning — reproducible every run.
- **Linear prediction is Wiener-optimal in modal space**; the FNO adds unique value only in the pixel-domain boiling regime, where it holds +27–33% while linear methods collapse to zero gain.

---

## License
MIT — for hackathon purposes.
