# FourierAO — Architecture & Physics

## Pipeline (matches the official problem-statement stages)

1. **Turbulence** — multi-layer Von Kármán phase screens with frozen-flow + AR(1) "boiling" (`simulator.Atmosphere`).
2. **SH-WFS** — microlens array focal spots, centroiding, slopes (`simulator.ShackHartmann`).
3. **Reconstruction** — modal Zernike via interaction matrix M⁺, and zonal via Southwell integration (`reconstruction`).
4. **Characterization** — r₀ (calibrated Kolmogorov phase-variance law), wind (sub-pixel FFT cross-correlation), τ₀ Greenwood (`characterization`).
5. **Prediction** — persistence, linear-AR, Koopman/DMD, and the residual **FNO** (`prediction`).
6. **Correction** — conjugate wavefront → DM actuator map (stroke units) → closed-loop residual / Strehl.

## Why each predictor choice is physically justified

| Block | Physics | Verified gain |
|---|---|---|
| Koopman linear core | advection = linear shift operator | +72% over persistence |
| Wind-equivariant Fourier ramp | Taylor frozen flow = phase ramp (shift theorem) | +52% |
| Residual FNO | boiling is nonlinear; linear methods fail | +14–50% (grows with horizon) |
| Turbulence conditioning | adapt predictor to live r₀/wind | +46.6% |
| Uncertainty fallback | flag low-predictability (boiling) frames | corr 0.47 |

## Critical training settings (learned from feasibility gating)

- **Residual / identity-skip FNO** — model starts at persistence, learns only the correction. (A naive FNO predicting absolute frames *loses* to persistence.)
- **Early stopping** — both naive runs overfit after ~20 epochs; keep the best-val checkpoint.
- **Multi-step horizons** — evaluate 1/3/5 frames; the servo-lag benefit is multi-step.
- **Boiling regime** — benchmark where linear methods fail; that is where the FNO's value is real and defensible.

## Honest scope

- **Per-layer tomographic prediction**: layer separation from a single WFS is physically possible (k-ω wind profiling); accurate per-layer velocities need the full 3-D k-ω transform — *roadmap*.
- **Mode-band scheduling**: high-order modes carry ~186× less power, so we **power-weight** prediction toward dominant low-order modes rather than relying on fragile high-mode coherence.
- **Beat LQG**: requires a full LQG implementation — *stretch goal*.
- **Real telemetry**: validatable against the [AO Telemetry Standard](https://www.aanda.org/articles/aa/full_html/2024/06/aa48486-23/aa48486-23.html) datasets — *roadmap*.
