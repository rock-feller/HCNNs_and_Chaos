# %% [markdown]
# # 03 · LSTM-formulation (LForm) HCNN
#
# `hcnn/architectures/lform/`. A learnable **diagonal gate** `D ∈ (0, 1)` mixes "keep the corrected
# state" (memory) with "take the vanilla update":
#
# ```
# r_t     = s_t − Cᵀ δ_t
# s_{t+1} = r_t + D (A tanh(r_t) − r_t)        D → 0: memory,  D → 1: vanilla
# ```
#
# This tutorial trains an LForm HCNN and looks at what the gate learned for the observed and the
# hidden coordinates.
#
# Run: `python tutorials/03_lform_hcnn.py`

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
from hcnn.core.training import HCNNTrainer
from hcnn.utils.data_generation import LorenzSolver
from hcnn.utils.data_preprocessing import SlidingWindowDataset, train_val_test_split
from hcnn.utils.plotting import ChaoticSystemPlotter

OUT = Path(os.environ.get("HCNN_TUTORIAL_OUT", "tutorial_outputs")) / "03_lform"
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)

T_END, EPOCHS = (12.0, 1) if FAST else (60.0, 150)
CONTEXT, HORIZON, WINDOW = (100, 50, 50) if FAST else (200, 300, 100)
N_OBS, N_HID = 3, 20

# %%
lorenz, _ = LorenzSolver(start=0.0, stop=T_END, ics=(1.0, 0.0, 1.25), time_grid=0.01, burn_in=20.0)
train, val, test, norm = train_val_test_split(lorenz, 0.6, 0.2, scaling_factor=0.02)
loader = DataLoader(SlidingWindowDataset(train, window_size=WINDOW), batch_size=32, shuffle=False)

# %% [markdown]
# ## Model
#
# `init_diag` sets the starting gate. Starting half-open (0.5) lets training push each coordinate
# toward memory or toward dynamics.

# %%
model = hcnn.LForm_Model(n_obs_vars=N_OBS, n_hid_vars=N_HID, init_diag=0.5)
d_before = model.get_diagonal_values().detach().clone()

trainer = HCNNTrainer(model, learning_rate=1e-2, save_dir=str(OUT / "checkpoints"))
best_val = trainer.train_and_validate(loader, EPOCHS, calibration_window=val[:CONTEXT],
                                      val_data=val[CONTEXT:CONTEXT + HORIZON], verbose=False)
trainer.restore_best()
print(f"best validation forecast MSE: {best_val:.5f}")

# %% [markdown]
# ## What did the gate learn?

# %%
d_after = model.get_diagonal_values().detach()
print("gate on observed coordinates:", d_after[:N_OBS].numpy().round(3))
print(f"gate on hidden coordinates: mean {d_after[N_OBS:].mean():.3f}, "
      f"min {d_after[N_OBS:].min():.3f}, max {d_after[N_OBS:].max():.3f}")

fig, ax = plt.subplots(figsize=(9, 3))
ax.bar(range(N_OBS + N_HID), d_before.numpy(), color="lightgrey", label="init")
ax.bar(range(N_OBS + N_HID), d_after.numpy(), width=0.5, label="trained")
ax.axvline(N_OBS - 0.5, color="k", lw=0.8, ls="--")
ax.set_xlabel("state coordinate (left of dashed line: observed)"); ax.set_ylabel("D_ii")
ax.legend(); fig.savefig(OUT / "gate.png", dpi=120, bbox_inches="tight")

# %% [markdown]
# ## Forecast the test segment

# %%
model.eval()
with torch.no_grad():
    out = model(test[:CONTEXT].unsqueeze(0), forecast_horizon=HORIZON)
truth = test[CONTEXT:CONTEXT + HORIZON]
print(f"test forecast RMSE (normalized): {((out.forecasts[0] - truth) ** 2).mean().sqrt():.4f}")
ChaoticSystemPlotter(truth, "Lorenz").plot_true_vs_predicted(
    norm.inverse_transform(out.forecasts[0]), norm.inverse_transform(truth),
    save_path=OUT / "test_forecast.png",
)
