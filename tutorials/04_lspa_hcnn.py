# %% [markdown]
# # 04 · Large-Sparse (LSpa) HCNN
#
# `hcnn/architectures/lspa/`. For large state spaces the dense transition matrix becomes the
# bottleneck. LSpa fixes a sparsity mask `M` at initialization:
#
# ```
# s_{t+1} = (M ⊙ A) tanh(r_t)
# ```
#
# `mask_type="random_block"` sparsifies the whole matrix. `"non_obs_block"` sparsifies only the
# columns of the hidden variables, so every coordinate still reads all observables. This tutorial
# looks at both masks, then trains a 90 %-sparse model with a much larger hidden state.
#
# Run: `python tutorials/04_lspa_hcnn.py`

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

OUT = Path(os.environ.get("HCNN_TUTORIAL_OUT", "tutorial_outputs")) / "04_lspa"
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)

T_END, EPOCHS = (12.0, 1) if FAST else (60.0, 150)
CONTEXT, HORIZON, WINDOW = (100, 50, 50) if FAST else (200, 300, 100)
N_OBS, N_HID = 3, 60

# %% [markdown]
# ## The two mask types

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
for ax, mask_type in zip(axes, ["random_block", "non_obs_block"]):
    m = hcnn.LSpa_Model(n_obs_vars=N_OBS, n_hid_vars=N_HID, sparsity_ratio=0.9, mask_type=mask_type)
    info = m.get_sparsity_info()
    ax.imshow(m.visualize_sparsity_pattern().numpy(), cmap="Greys", interpolation="nearest")
    ax.set_title(f"{mask_type}\n{info['non_zero_parameters']} / {info['total_parameters']} non-zero")
fig.savefig(OUT / "masks.png", dpi=120, bbox_inches="tight")

# %% [markdown]
# ## Train a 90 %-sparse LSpa HCNN

# %%
lorenz, _ = LorenzSolver(start=0.0, stop=T_END, ics=(1.0, 0.0, 1.25), time_grid=0.01, burn_in=20.0)
train, val, test, norm = train_val_test_split(lorenz, 0.6, 0.2, scaling_factor=0.02)
loader = DataLoader(SlidingWindowDataset(train, window_size=WINDOW), batch_size=32, shuffle=False)

model = hcnn.LSpa_Model(n_obs_vars=N_OBS, n_hid_vars=N_HID, sparsity_ratio=0.9,
                        mask_type="non_obs_block")
trainer = HCNNTrainer(model, learning_rate=1e-2, save_dir=str(OUT / "checkpoints"))
best_val = trainer.train_and_validate(loader, EPOCHS, calibration_window=val[:CONTEXT],
                                      val_data=val[CONTEXT:CONTEXT + HORIZON], verbose=False)
trainer.restore_best()
print(f"best validation forecast MSE: {best_val:.5f}")

# The mask is re-applied on every forward pass, so training never fills in pruned weights:
model(val[:CONTEXT].unsqueeze(0))
print(f"actual sparsity after training: {model.get_sparsity_info()['actual_sparsity']:.3f}")

# %% [markdown]
# ## Forecast the test segment
#
# Note: keep evaluating the *trained object*. Reloading an LSpa `state_dict` into a fresh model
# currently re-draws the mask (a known issue, tracked in `tests/test_architecture_contract.py`).

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
