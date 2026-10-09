# %% [markdown]
# # 05 · Ensembles and forecast uncertainty
#
# Chaotic forecasts diverge, so a single trajectory says little about how far you can trust it. An
# ensemble of independently initialized HCNNs gives a spread you can compare against the actual
# error. Every architecture has an ensemble (`hcnn.build_ensemble(name, ...)`), backed by the same
# `HCNNEnsemble` (mean / median / weighted aggregation, `predict_with_uncertainty`).
#
# Change `ARCH` to run this tutorial with any registered architecture.
#
# Run: `python tutorials/05_ensembles_and_uncertainty.py`

# %%
import os
from pathlib import Path

import matplotlib
FAST = os.environ.get("HCNN_TUTORIAL_FAST") == "1"  # tiny run used by tests/test_tutorials.py
if FAST:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import hcnn
from hcnn.core.training import EnsembleTrainer, HCNNTrainer
from hcnn.utils.data_generation import LorenzSolver
from hcnn.utils.data_preprocessing import SlidingWindowDataset, train_val_test_split
from hcnn.utils.plotting import EnsemblePlotter

OUT = Path(os.environ.get("HCNN_TUTORIAL_OUT", "tutorial_outputs")) / "05_ensembles"
OUT.mkdir(parents=True, exist_ok=True)

ARCH = "vanilla"  # any of hcnn.list_architectures()
N_MEMBERS = 2 if FAST else 5
T_END, EPOCHS = (12.0, 1) if FAST else (60.0, 100)
CONTEXT, HORIZON, WINDOW = (100, 50, 50) if FAST else (200, 400, 100)

# %%
lorenz, _ = LorenzSolver(start=0.0, stop=T_END, ics=(1.0, 0.0, 1.25), time_grid=0.01, burn_in=20.0)
train, val, test, norm = train_val_test_split(lorenz, 0.6, 0.2, scaling_factor=0.02)
loader = DataLoader(SlidingWindowDataset(train, window_size=WINDOW), batch_size=32, shuffle=False)

# %% [markdown]
# ## Build and train
#
# `seed` makes member `i` start from `seed + i`: members differ, but the ensemble is reproducible.
# `EnsembleTrainer` trains each member with its own `HCNNTrainer` and validation-based selection.

# %%
# Note: ensemble constructors default to init_range=(-0.75, 0.75), wider than the single models'
# (-0.5, 0.5); with the wider range most Lorenz members diverge, so pass it explicitly.
ensemble = hcnn.build_ensemble(ARCH, n_ensemble=N_MEMBERS, n_obs_vars=3, n_hid_vars=20,
                               init_range=(-0.5, 0.5), seed=0)
ens_trainer = EnsembleTrainer(ensemble, HCNNTrainer, save_dir=str(OUT / "checkpoints"),
                              learning_rate=1e-2)
best_vals = ens_trainer.train_and_validate(loader, EPOCHS, calibration_window=val[:CONTEXT],
                                           val_data=val[CONTEXT:CONTEXT + HORIZON])
for t in ens_trainer.trainers:
    t.restore_best()
print("per-member best validation MSE:", [round(v, 4) for v in best_vals])

# %% [markdown]
# ## Forecast with uncertainty

# %%
ensemble.eval()
with torch.no_grad():
    u = ensemble.predict_with_uncertainty(test[:CONTEXT].unsqueeze(0), forecast_horizon=HORIZON,
                                          return_individual=True)
truth = test[CONTEXT:CONTEXT + HORIZON]
mean, std = u["forecasts_mean"][0], u["forecasts_var"][0].sqrt()
print(f"model agreement: {u['model_agreement']:.3f} (1 = identical members)")
print(f"ensemble-mean RMSE: {((mean - truth) ** 2).mean().sqrt():.4f}   "
      f"mean spread (std): {std.mean():.4f}")

# Is the spread an honest error bar? For a calibrated ensemble, spread ≈ error. Members that differ
# only in their seed tend to be *under-dispersed* (spread well below the error): they share data,
# architecture and training, so they make the same mistakes. Calibrating HCNN ensembles is an open
# research question in this project.
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(std.mean(dim=1).numpy(), label="mean member std")
ax.plot((mean - truth).abs().mean(dim=1).numpy(), label="|ensemble mean − truth|", alpha=0.7)
ax.set_xlabel("forecast step"); ax.legend()
fig.savefig(OUT / "spread_vs_error.png", dpi=120, bbox_inches="tight")

members = u["individual_forecasts"][:, 0]                    # (n_members, HORIZON, 3)
EnsemblePlotter.plot_ensemble_predictions(
    torch.stack([norm.inverse_transform(m) for m in members]), norm.inverse_transform(truth),
    add_statistics="both", save_path=OUT / "ensemble_forecast.png",
)
