# Poster → Implementation Mapping

Every element of the official challenge poster, mapped to where it is implemented
and demonstrated in this repository. Verify end-to-end with
`python scripts/verify_requirements.py`.

## Poster panel 1 — "SH-WFS Time-Series Input" (lenslet spot grid)

| Poster element | Implementation | Evidence |
|---|---|---|
| Microlens-array spot field | `fourierao/simulator.py` → `ShackHartmann.spot_field()` | `results/fig1_reconstruction.png` (left) |
| Time-series input | `Atmosphere.sequence()` (multi-layer + boiling) | all temporal results |

## Poster panel 2 — "Real-Time Processing Pipeline"

| Poster element | Implementation | Evidence |
|---|---|---|
| **Iterative centroid estimation** | `fourierao/centroiding.py` → `IterativeCentroider` (3-iter thresholded CoG) | R3 in verify_requirements |
| **Zonal reconstruction** | `fourierao/reconstruction.py` → `ZonalReconstructor` (Southwell) | R5 |
| (also) Modal reconstruction | `ModalReconstructor` (Zernike, interaction matrix) | R4, `fig1` |
| **Latency: < 1 ms/frame** | reconstruction = 0.0065 ms/frame | R9 |
| **Throughput: 500 fps** | >150,000 fps (reconstruction) | R9 |
| **Stability Metric: σ < 0.05 λ** | temporal stability 0.009 λ | `fig7` |

## Poster panel 3 — "Reconstructed Wavefront" (3D surface)

| Poster element | Implementation | Evidence |
|---|---|---|
| Reconstructed wavefront (3D) | `ModalReconstructor.to_wavefront()` | `results/fig1_reconstruction.png` (right, 3D) |
| Zernike coefficients | `ModalReconstructor.reconstruct()` | R4 |
| Conjugate → DM actuator map (stroke nm) | `fourierao/control.py` → `DeformableMirror.actuator_map()` | R6 |
| DM closed-loop correction | `control.ClosedLoop` | R7 (Strehl 0.03 → 0.55) |

## Poster panel 4 — "Algorithm Optimization Console"

| Poster element | Implementation | Evidence |
|---|---|---|
| Convergence plot | `scripts/optimization_console.py` | `results/fig7_optimization_console.png` (left) |
| Stability analysis | same | `fig7` (right, vs 0.05 λ spec) |
| Latency / throughput meter | same | `fig7` footer |

## Title — "Wavefront reconstruction AND turbulence characterization"

| Element | Implementation | Evidence |
|---|---|---|
| Turbulence characterization (r₀) | `characterization.py` → `estimate_r0` (calibrated Kolmogorov) | R8, `fig5` |
| Wind speed/direction | `estimate_wind` (sub-pixel cross-correlation) | R8 |
| τ₀ (Greenwood time) | `estimate_tau0` | R8 |

## Beyond the poster — the novel contribution

| Contribution | Implementation | Evidence |
|---|---|---|
| Residual Fourier Neural Operator predictor | `prediction.py` → `FNO2d`, `train_fno` | `fig2`, `fig6` |
| Predictive servo-lag cancellation | `control.ClosedLoop` (predicted_phase) | `fig3` |
| Literature benchmark (~2× on-sky band) | `scripts/benchmark_literature.py` | `fig8` |
| Variance reduction vs conditions | `scripts/variance_vs_boiling.py` | `fig9` |
| >7.5× via servo-lag horizon regime | `scripts/variance_vs_horizon.py` | `fig10` |

## Figure index

| File | Shows |
|---|---|
| fig1 | Spot field → wavefront → 3D reconstruction |
| fig2 | FNO vs persistence/linear-AR (multi-step) |
| fig3 | Closed-loop Strehl with vs without prediction |
| fig4 | PSF (star image) before vs after correction |
| fig5 | Turbulence characterization (r₀ accuracy) |
| fig6 | Key result: linear collapses under boiling, FNO holds |
| fig7 | Algorithm Optimization Console (convergence + stability) |
| fig8 | Literature benchmark (~2× on-sky band) |
| fig9 | Variance reduction vs boiling (seed-averaged, noisy) |
| fig10 | Variance reduction vs servo-lag horizon (crosses 7.5×) |
