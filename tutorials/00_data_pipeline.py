# %% [markdown]
# # 00 · The leakage-safe data pipeline
#
# Every tutorial and experiment in this repository prepares data the same way:
#
# 1. **Generate** a trajectory with a **burn-in**. The transient before the trajectory reaches the
#    attractor is integrated and discarded.
# 2. **Split chronologically** into train / validation / test **before** computing any statistic.
# 3. **Fit the normalization on the training part only**, then apply it to all three.
# 4. **Cut the training part into sliding windows**: each window is one teacher-forced HCNN sample.
#
# Validation is for model selection; the test part is touched exactly once, at the end.
#
# Run: `python tutorials/00_data_pipeline.py` (or open it as a notebook with jupytext).

# %%
import os
from pathlib import Path

import matplotlib
FAST = os.environ.get("HCNN_TUTORIAL_FAST") == "1"  # tiny run used by tests/test_tutorials.py
if FAST:
    matplotlib.use("Agg")
import torch
from torch.utils.data import DataLoader

from hcnn.utils.data_generation import ChaoticSystemGenerator
from hcnn.utils.data_preprocessing import (
    NormalizationStrategy,
    SlidingWindowDataset,
    forecast_windows,
    train_val_test_split,
)
from hcnn.utils.plotting import ChaoticSystemPlotter

OUT = Path(os.environ.get("HCNN_TUTORIAL_OUT", "tutorial_outputs")) / "00_data"
OUT.mkdir(parents=True, exist_ok=True)
DT = 0.05  # sampling step, see section 1b
T_END = 60.0 if FAST else 300.0

# %% [markdown]
# ## 1. Generate trajectories (with burn-in)
#
# `ChaoticSystemGenerator` integrates Lorenz, Rössler and Chua. `burn_in` is in time units.
# The returned trajectory has the same length with or without it, but starts on the attractor.

# %%
gen = ChaoticSystemGenerator()
lorenz, t = gen.generate_lorenz(stop=T_END, time_grid=DT, burn_in=20.0)
rossler, _ = gen.generate_rossler(stop=T_END, time_grid=DT, burn_in=50.0)
chua, _ = gen.generate_chua(stop=T_END, time_grid=DT, burn_in=20.0)
for name, traj in [("lorenz", lorenz), ("rossler", rossler), ("chua", chua)]:
    print(f"{name:8s} shape={tuple(traj.shape)}  first point={traj[0].numpy().round(2)}")

no_burn, _ = gen.generate_lorenz(stop=T_END, time_grid=DT, burn_in=0.0)
print("without burn-in the trajectory starts at the initial condition:", no_burn[0].numpy())

ChaoticSystemPlotter(lorenz, "Lorenz").plot_phase_space(save_path=OUT / "lorenz_phase_space.png")

# %% [markdown]
# ## 1b. Choose the sampling step deliberately
#
# One HCNN step must map `s_t` to `s_{t+Δt}`. If `Δt` is tiny, consecutive states are almost
# identical and the vanilla map `A tanh(·)` has to learn a near-identity, which is badly conditioned.
# On Lorenz, `Δt = 0.05` gave a 6× longer valid forecast than `Δt = 0.01` (same number of samples;
# `docs/vanilla_hcnn_review.md`). Report `Δt` with every result, and measure forecast horizons in
# Lyapunov times (Lorenz: λ ≈ 0.906, so one Lyapunov time is ≈ 1.1 time units = 22 steps here).

# %%
fine, _ = gen.generate_lorenz(stop=10.0, time_grid=0.01, burn_in=20.0)
for dt, traj in [(0.01, fine), (DT, lorenz)]:
    step = (traj[1:] - traj[:-1]).norm(dim=1).mean() / traj.std(0).norm()
    print(f"dt={dt}: mean change per step = {step:.1%} of the attractor size")

# %% [markdown]
# ## 2–3. Split first, then normalize with training statistics
#
# Written out by hand, so you can see what happens:

# %%
n = len(lorenz)
i_train, i_val = int(0.6 * n), int(0.8 * n)
norm = NormalizationStrategy().fit(lorenz[:i_train], scaling_factor=0.02)  # TRAIN ONLY
train = norm.transform(lorenz[:i_train])
val = norm.transform(lorenz[i_train:i_val])
test = norm.transform(lorenz[i_val:])
print("train mean (≈0 by construction):", train.mean(0).numpy().round(4))
print("test mean (not exactly 0, and that is correct):", test.mean(0).numpy().round(4))

# %% [markdown]
# The same in one call. The other tutorials use this:

# %%
train2, val2, test2, norm2 = train_val_test_split(lorenz, train_ratio=0.6, val_ratio=0.2,
                                                  scaling_factor=0.02)
assert torch.allclose(train, train2) and torch.allclose(test, test2)
# Map predictions back to physical units with the same fitted statistics:
assert torch.allclose(norm2.inverse_transform(test2), lorenz[i_val:], atol=1e-4)

# %% [markdown]
# ## 4. Sliding windows over the training part only

# %%
WINDOW = 50 if FAST else 100
dataset = SlidingWindowDataset(train, window_size=WINDOW)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
batch = next(iter(loader))
print(f"{len(dataset)} windows; one batch has shape {tuple(batch.shape)} = (batch, window, n_obs)")

# %% [markdown]
# ## 5. Validation / test windows
#
# A chaotic forecast's error depends strongly on where it starts, so judge models on several
# evenly spaced windows: each pairs `context` observed steps with the `horizon` steps that follow.

# %%
cal, target = forecast_windows(val, context=100, horizon=20, n_windows=6)
print(f"validation: calibration {tuple(cal.shape)}, targets {tuple(target.shape)}")
print(f"figures saved to {OUT}/")
