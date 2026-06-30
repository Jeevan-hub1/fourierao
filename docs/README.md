# FourierAO — Documentation Index

Complete documentation for the FourierAO predictive Shack-Hartmann wavefront
reconstruction and turbulence-characterization framework.

| Document | Contents |
|----------|----------|
| [RESULTS.md](RESULTS.md) | **Final validated results + ablation study.** Headline ~16× variance reduction, the optimization journey (7.7× → 16×), the full ablation establishing the optimal configuration, literature benchmark, reproducibility, and honest limitations. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture, the processing pipeline, and the physics behind each design choice. |
| [POSTER_MAPPING.md](POSTER_MAPPING.md) | Line-by-line mapping of every official poster requirement to its implementation, figure, and verification ID. |
| [PITCH_SCRIPT.md](PITCH_SCRIPT.md) | 5-minute presentation script (slide-by-slide with timing) and judge Q&A preparation. |

## Quick reference — final numbers

| Metric | Value |
|--------|-------|
| Variance reduction (frozen-flow, 5% noise) | ~16× |
| Variance reduction (good seeing) | ~9× |
| Variance reduction (heavy boiling) | ~3× |
| Reconstruction latency | 0.007 ms |
| Throughput | >150,000 fps |
| Temporal stability | 0.009 λ |
| Closed-loop Strehl | 0.03 → 0.55 |

## Paper

A full IEEE-format conference paper is in [`../paper/main.tex`](../paper/main.tex)
(Overleaf-ready).

## Reproduce everything

```bash
pip install -r ../requirements.txt
python ../scripts/peak_performance.py        # ~16x headline (fig12)
python ../scripts/verify_requirements.py     # operational specs
python ../scripts/benchmark_literature.py    # literature comparison
```
