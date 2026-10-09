# %% [markdown]
# # 02 · Partial Teacher Forcing (PTF) HCNN
#
# `hcnn/architectures/ptf/`. Full teacher forcing hides the model's own errors from it during
# training. PTF applies **dropout to the correction** `δ_t`, so on dropped coordinates the model
# must continue from its own prediction:
#
# ```
# m_t ~ Bernoulli(1 − p(epoch)),   s_{t+1} = A tanh(s_t − Cᵀ(m_t ⊙ (ŷ_t − y_t)))
# ```
#
# A schedule raises `p` over training. This tutorial plots the six schedules, then trains one.
#
# Run: `python tutorials/02_ptf_hcnn.py`

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
from hcnn.architectures.ptf import PTFHCNNCell
from hcnn.core.training import HCNNTrainer
from hcnn.utils.data_generation import LorenzSolver
from hcnn.utils.data_preprocessing import SlidingWindowDataset, forecast_windows, train_val_test_split
from hcnn.utils.plotting import ChaoticSystemPlotter

OUT = Path(os.environ.get("HCNN_TUTORIAL_OUT", "tutorial_outputs")) / "02_ptf"
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)

DT = 0.05  # sampling step - the most important setting (docs/vanilla_hcnn_review.md)
T_END, EPOCHS = (60.0, 2) if FAST else (300.0, 150)
CONTEXT, HORIZON, VAL_H, WINDOW = (50, 20, 10, 50) if FAST else (200, 100, 20, 100)

# %% [markdown]
# ## The six dropout schedules
#
# `dropout_params` override the defaults listed in `hcnn/architectures/ptf/README.md`. The trainer
# calls `model.update_dropout_epoch(epoch)` at the start of every epoch.

# %%
N_PLOT = 40
schedules = {
    "constant": {"p": 0.2},
    "linear": {"start_p": 0.0, "end_p": 0.4, "total_epochs": N_PLOT},
    "exponential": {"start_p": 0.0, "end_p": 0.4, "total_epochs": N_PLOT},
    "cosine": {"start_p": 0.0, "end_p": 0.4, "total_epochs": N_PLOT},
    "step": {"schedule": [(10, 0.1), (20, 0.25), (30, 0.4)]},
    "adaptive": {"target_p": 0.4, "total_epochs": N_PLOT},  # 0 for the first half, then a ramp
}
fig, ax = plt.subplots(figsize=(8, 4))
for strategy, params in schedules.items():
    cell = PTFHCNNCell(n_obs_vars=3, n_hid_vars=5, dropout_strategy=strategy, dropout_params=params)
    ps = []
    for epoch in range(1, N_PLOT + 1):
        cell.update_dropout_epoch(epoch)
        ps.append(cell.dropout_module.p)
    ax.plot(range(1, N_PLOT + 1), ps, label=strategy)
ax.set_xlabel("epoch"); ax.set_ylabel("dropout probability p on δ_t"); ax.legend()
fig.savefig(OUT / "schedules.png", dpi=120, bbox_inches="tight")

# %% [markdown]
# ## Train a PTF HCNN with the adaptive schedule

# %%
lorenz, _ = LorenzSolver(start=0.0, stop=T_END, ics=(1.0, 0.0, 1.25), time_grid=DT, burn_in=20.0)
train, val, test, norm = train_val_test_split(lorenz, 0.6, 0.2, scaling_factor=0.02)
# Validate on 6 windows, VAL_H steps ahead (~1 Lyapunov time): longer horizons reward
# predicting the mean, a single window makes epoch selection noisy.
val_cal, val_target = forecast_windows(val, CONTEXT, VAL_H, n_windows=6)
loader = DataLoader(SlidingWindowDataset(train, window_size=WINDOW), batch_size=32, shuffle=True)

model = hcnn.PTF_Model(n_obs_vars=3, n_hid_vars=20, dropout_strategy="adaptive",
                       dropout_params={"target_p": 0.3, "total_epochs": EPOCHS})
trainer = HCNNTrainer(model, learning_rate=1e-2, lr_schedule="cosine", save_dir=str(OUT / "checkpoints"))
best_val = trainer.train_and_validate(loader, EPOCHS, calibration_window=val_cal,
                                      val_data=val_target, verbose=False)
print(f"final dropout p: {model.cell.dropout_module.p:.3f}   best val MSE: {best_val:.5f}")
trainer.restore_best()

# %% [markdown]
# ## What PTF returns
#
# Besides the usual fields, the output carries `partial_delta_terms`: the dropout-masked
# corrections. In `eval()` mode dropout is off, so they equal `delta_terms`.

# %%
batch = next(iter(loader))
model.train()
out = model(batch[:1])
kept = (out.partial_delta_terms != 0).float().mean().item()
print(f"train mode: {kept:.0%} of the correction entries were kept")

model.eval()
with torch.no_grad():
    out = model(test[:CONTEXT].unsqueeze(0), forecast_horizon=HORIZON)
truth = test[CONTEXT:CONTEXT + HORIZON]
print(f"test forecast RMSE (normalized): {((out.forecasts[0] - truth) ** 2).mean().sqrt():.4f}")
ChaoticSystemPlotter(truth, "Lorenz").plot_true_vs_predicted(
    norm.inverse_transform(out.forecasts[0]), norm.inverse_transform(truth),
    save_path=OUT / "test_forecast.png",
)
