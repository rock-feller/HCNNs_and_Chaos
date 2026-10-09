# %% [markdown]
# # 01 · Vanilla HCNN on the Lorenz system
#
# The reference architecture (`hcnn/architectures/vanilla/`). With `C = [I | 0]` and `ŷ_t = C s_t`:
#
# ```
# s_{t+1} = A tanh(s_t − Cᵀ(ŷ_t − y_t))  (teacher forcing: the observed part is replaced by y_t)
# s_{t+1} = A tanh(s_t)                 (autonomous forecast)
# ```
#
# You will train with validation-based model selection, restore the selected epoch, and forecast the
# held-out test segment once.
#
# Run: `python tutorials/01_vanilla_hcnn.py`

# %%
import os
from pathlib import Path

import matplotlib
FAST = os.environ.get("HCNN_TUTORIAL_FAST") == "1"  # tiny run used by tests/test_tutorials.py
if FAST:
    matplotlib.use("Agg")
import torch
from torch.utils.data import DataLoader

import hcnn
from hcnn.core.training import HCNNTrainer
from hcnn.utils.data_generation import LorenzSolver
from hcnn.utils.data_preprocessing import SlidingWindowDataset, forecast_windows, train_val_test_split
from hcnn.utils.plotting import ChaoticSystemPlotter

OUT = Path(os.environ.get("HCNN_TUTORIAL_OUT", "tutorial_outputs")) / "01_vanilla"
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)

DT = 0.05  # sampling step - the most important setting (docs/vanilla_hcnn_review.md)
T_END, EPOCHS = (60.0, 1) if FAST else (300.0, 150)
CONTEXT, HORIZON, VAL_H, WINDOW = (50, 20, 10, 50) if FAST else (200, 100, 20, 100)

# %% [markdown]
# ## Data (see tutorial 00)

# %%
lorenz, _ = LorenzSolver(start=0.0, stop=T_END, ics=(1.0, 0.0, 1.25), time_grid=DT, burn_in=20.0)
train, val, test, norm = train_val_test_split(lorenz, train_ratio=0.6, val_ratio=0.2,
                                              scaling_factor=0.02)
# Validate on 6 windows, VAL_H steps ahead (~1 Lyapunov time): longer horizons reward
# predicting the mean, a single window makes epoch selection noisy.
val_cal, val_target = forecast_windows(val, CONTEXT, VAL_H, n_windows=6)
loader = DataLoader(SlidingWindowDataset(train, window_size=WINDOW), batch_size=32, shuffle=True)

# %% [markdown]
# ## Model
#
# `n_hid_vars` hidden coordinates extend the 3 observed ones into a 23-dimensional state.

# %%
model = hcnn.Vanilla_Model(n_obs_vars=3, n_hid_vars=20, s0_nature="random_", train_s0=True)
# (equivalently: hcnn.build_model("vanilla", n_obs_vars=3, n_hid_vars=20))
print(model)

# %% [markdown]
# ## Train, selecting the epoch on the validation segment
#
# Each epoch, the trainer forecasts `VAL_H` steps (≈ 1 Lyapunov time) after each of 6 validation
# windows and keeps the epoch with the lowest mean forecast error. The learning rate follows a
# cosine schedule. These choices, and `DT = 0.05`, come from the ablation in
# `docs/vanilla_hcnn_review.md`: together they made forecasts valid about 10× longer than the old
# defaults.

# %%
trainer = HCNNTrainer(model, loss_fn="mse", learning_rate=1e-2, lr_schedule="cosine",
                      save_dir=str(OUT / "checkpoints"))
best_val = trainer.train_and_validate(
    loader,
    num_epochs=EPOCHS,
    calibration_window=val_cal,
    val_data=val_target,
    verbose=False,
)
trainer.restore_best()  # the model now holds the selected epoch, not the last one
print(f"best validation forecast MSE: {best_val:.5f}")

# %% [markdown]
# ## Forecast the test segment, once

# %%
model.eval()
with torch.no_grad():
    out = model(test[:CONTEXT].unsqueeze(0), forecast_horizon=HORIZON)
forecast = out.forecasts[0]                      # (HORIZON, 3), normalized units
truth = test[CONTEXT:CONTEXT + HORIZON]
print(f"test forecast RMSE (normalized): {((forecast - truth) ** 2).mean().sqrt():.4f}")

ChaoticSystemPlotter(truth, "Lorenz").plot_true_vs_predicted(
    norm.inverse_transform(forecast), norm.inverse_transform(truth),
    save_path=OUT / "test_forecast.png",
)
print(f"figure saved to {OUT}/test_forecast.png")
