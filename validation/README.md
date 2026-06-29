# Validation Scripts

Standalone physics-verification scripts used to de-risk the project before building.
Each runs independently and prints measured numbers.

| Script | Verifies |
|--------|----------|
| `verify_full_shwfs.py` | Full 6-stage SH-WFS→DM closed loop (every problem-statement stage) |
| `honest_feasibility.py` | FNO vs baselines at multi-step horizons (early stopping) — the key gate |
| `advanced_physics.py` | Fourier-shift theorem, mode coherence, per-layer, Koopman/DMD |
| `fix_r0.py` | Calibrated r₀ estimator (few-% accuracy) |
| `fix_wind_subpixel.py` | Sub-pixel wind speed/direction estimation |

Run any with: `python validation/<script>.py`
