# Recovering Latent Neural-Population Dynamics with a GRU Sequence Autoencoder

Simulated benchmark in the style of LFADS: a 3-D Lorenz system drives 60 neurons through a log-linear map with
Poisson spiking (600 trials × 100 bins; 450 train / 150 held-out). A bidirectional-GRU encoder produces a 3-D latent
trajectory, decoded to firing rates by a small MLP with exponential output; trained with Poisson negative
log-likelihood. Baselines: PCA on raw counts and on Gaussian-smoothed counts.

```
pip install torch numpy scipy scikit-learn matplotlib
python latent_dynamics.py     # ~2 min; writes results.json and latents.png
```

## Results (held-out trials, `results.json`)
| Method | R² (latents → true Lorenz state, linear map) | R² (rates) |
|---|---|---|
| GRU autoencoder | 0.77 | 0.54 |
| PCA (raw counts) | 0.70 | 0.54 |
| PCA (smoothed) | 0.48 | 0.30 |

## Caveats
Single training seed and single simulated dataset, so the GRU's gain over raw-count PCA is modest and not shown to
be significant. Signal strength and baseline rate were adjusted during development (first settings were too weak /
unrealistically strong). No real neural recordings; no hyper-parameter search; the rate R² uses a linear decoder for PCA.
