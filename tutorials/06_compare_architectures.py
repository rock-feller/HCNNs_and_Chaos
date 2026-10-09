# %% [markdown]
# # 06 · Compare all registered architectures
#
# The registry lets you treat architectures generically. This loop trains **every** registered
# architecture under the same data, budget and protocol, then reports the test error. A new
# architecture added under `hcnn/architectures/` shows up here automatically.
#
# A fair benchmark needs several seeds and a tuned budget per model. Treat this as a template for
# such a study, not as a result.
#
# Run: `python tutorials/06_compare_architectures.py`

# %%
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import hcnn
from hcnn.core.training import HCNNTrainer
from hcnn.utils.data_generation import LorenzSolver
from hcnn.utils.data_preprocessing import SlidingWindowDataset, forecast_windows, train_val_test_split

FAST = os.environ.get("HCNN_TUTORIAL_FAST") == "1"  # tiny run used by tests/test_tutorials.py
OUT = Path(os.environ.get("HCNN_TUTORIAL_OUT", "tutorial_outputs")) / "06_compare"
OUT.mkdir(parents=True, exist_ok=True)

DT = 0.05  # sampling step - the most important setting (docs/vanilla_hcnn_review.md)
T_END, EPOCHS = (60.0, 1) if FAST else (300.0, 150)
CONTEXT, HORIZON, VAL_H, WINDOW = (50, 20, 10, 50) if FAST else (200, 100, 20, 100)
SEED = 0

# Architecture-specific keyword arguments; everything else uses the defaults.
EXTRA_KWARGS = {
    "ptf": {"dropout_strategy": "adaptive", "dropout_params": {"target_p": 0.3, "total_epochs": EPOCHS}},
    "lform": {"init_diag": 0.5},
    "lspa": {"sparsity_ratio": 0.5},
}

print(hcnn.describe_architectures())

# %%
lorenz, _ = LorenzSolver(start=0.0, stop=T_END, ics=(1.0, 0.0, 1.25), time_grid=DT, burn_in=20.0)
train, val, test, norm = train_val_test_split(lorenz, 0.6, 0.2, scaling_factor=0.02)
# Validate on 6 windows, VAL_H steps ahead (~1 Lyapunov time): longer horizons reward
# predicting the mean, a single window makes epoch selection noisy.
val_cal, val_target = forecast_windows(val, CONTEXT, VAL_H, n_windows=6)
loader = DataLoader(SlidingWindowDataset(train, window_size=WINDOW), batch_size=32, shuffle=True)
truth = test[CONTEXT:CONTEXT + HORIZON]

# %%
results = {}
for name in hcnn.list_architectures():
    torch.manual_seed(SEED)
    model = hcnn.build_model(name, n_obs_vars=3, n_hid_vars=20, **EXTRA_KWARGS.get(name, {}))
    trainer = HCNNTrainer(model, learning_rate=1e-2, lr_schedule="cosine", save_dir=str(OUT / name))
    t0 = time.perf_counter()
    best_val = trainer.train_and_validate(loader, EPOCHS, calibration_window=val_cal,
                                          val_data=val_target, verbose=False)
    elapsed = time.perf_counter() - t0
    trainer.restore_best()
    model.eval()
    with torch.no_grad():
        forecast = model(test[:CONTEXT].unsqueeze(0), forecast_horizon=HORIZON).forecasts[0]
    # Count non-zero trainable entries: LSpa's masked weights and LForm's off-diagonal D are zeros.
    n_params = sum(int((p != 0).sum()) for p in model.parameters() if p.requires_grad)
    results[name] = {
        "params": n_params,
        "val_mse": best_val,
        "test_rmse": ((forecast - truth) ** 2).mean().sqrt().item(),
        "train_s": elapsed,
    }

# %%
print(f"\n{'architecture':<12} {'nnz params':>10} {'val MSE':>9} {'test RMSE':>10} {'train s':>8}")
for name, r in sorted(results.items(), key=lambda kv: kv[1]["test_rmse"]):
    print(f"{name:<12} {r['params']:>10} {r['val_mse']:>9.4f} {r['test_rmse']:>10.4f} {r['train_s']:>8.1f}")
