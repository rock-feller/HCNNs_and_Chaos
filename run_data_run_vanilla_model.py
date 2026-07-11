"""
End-to-end demo: train a Vanilla HCNN to forecast the Lorenz system.

Run from the repo root:  python run_data_run_vanilla_model.py

This exercises the consolidated `hcnns_chaos` stack end to end, following the
leakage-safe protocol:
  1. generate a Lorenz trajectory with a burn-in transient discarded,
  2. split into train / test BEFORE fitting any statistics,
  3. fit normalization on the TRAIN split only and apply it to both,
  4. train with HCNNTrainer, then forecast autonomously and plot.
"""

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from hcnns_chaos.utils.data_generation import LorenzSolver
from hcnns_chaos.utils.data_preprocessing import NormalizationStrategy, SlidingWindowDataset
from hcnns_chaos.core.models.hcnn_models import Vanilla_Model
from hcnns_chaos.core.training.hcnn_trainer import HCNNTrainer

torch.manual_seed(0)

# --- 1. Data: Lorenz with burn-in so we start on the attractor ---------------
SCALING = 0.02
CONTEXT, HORIZON = 500, 500
lorenz, _ = LorenzSolver(start=0.0, stop=40.0, ics=(1.0, 0.0, 1.25),
                         time_grid=0.01, burn_in=20.0)   # ~4000 points on-attractor

# --- 2. Split FIRST, then 3. fit normalization on TRAIN only (no leakage) ----
train_size = int(0.7 * len(lorenz))
norm = NormalizationStrategy().fit(lorenz[:train_size], SCALING)
train = norm.transform(lorenz[:train_size])
test = norm.transform(lorenz[train_size:])

# Sliding-window training batches over the training split only.
window = 200
dataset = SlidingWindowDataset(train, window_size=window)
loader = DataLoader(dataset, batch_size=20, shuffle=False)

# Calibration + forecast target taken from the (held-out) test split.
calibration = test[:CONTEXT]
forecast_target = test[CONTEXT:CONTEXT + HORIZON]

# --- 4. Model + training -----------------------------------------------------
model = Vanilla_Model(n_obs_vars=3, n_hid_vars=10, s0_nature="random_", train_s0=True)
trainer = HCNNTrainer(model, loss_fn="mse", backprop_mode="per_batch",
                      learning_rate=1e-2, save_dir="./demo_checkpoints")
trainer.train_and_validate(
    data_loader=loader,
    num_epochs=30,
    calibration_window=calibration,
    val_data=forecast_target,
    verbose=True,
)

# --- Autonomous forecast + plot ---------------------------------------------
model.eval()
with torch.no_grad():
    out = model(calibration.unsqueeze(0), forecast_horizon=HORIZON)
forecast = out.forecasts.squeeze(0)  # (HORIZON, 3)

fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
for i, (ax, name) in enumerate(zip(axes, ["x", "y", "z"])):
    ax.plot(forecast_target[:, i].cpu(), "b", label="true")
    ax.plot(forecast[:, i].cpu(), "r", label="HCNN forecast")
    ax.set_ylabel(name)
    ax.legend(loc="upper right")
axes[-1].set_xlabel("forecast step")
fig.suptitle("Vanilla HCNN - autonomous Lorenz forecast")
plt.tight_layout()
plt.savefig("demo_lorenz_forecast.png", dpi=120)
print("Saved forecast plot to demo_lorenz_forecast.png")
