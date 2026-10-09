# Tutorials

Small, self-contained scripts. Each one runs on its own and shows one thing. Every tutorial follows
the same leakage-safe protocol: burn-in, then a chronological train / validation / test split,
normalization fitted on train only, epoch selection on validation, and a single evaluation on test.

| # | Script | What you learn | Full run* |
|---|---|---|---|
| 00 | [`00_data_pipeline.py`](00_data_pipeline.py) | generating Lorenz / Rössler / Chua with burn-in; split, normalize, window | ~10 s |
| 01 | [`01_vanilla_hcnn.py`](01_vanilla_hcnn.py) | the reference HCNN: train, select on validation, `restore_best`, forecast | ~2 min |
| 02 | [`02_ptf_hcnn.py`](02_ptf_hcnn.py) | Partial Teacher Forcing: the six dropout schedules, `partial_delta_terms` | ~2–3 min |
| 03 | [`03_lform_hcnn.py`](03_lform_hcnn.py) | LSTM formulation: the diagonal memory gate `D` and what it learns | ~3 min |
| 04 | [`04_lspa_hcnn.py`](04_lspa_hcnn.py) | Large-Sparse: mask types, a 90 %-sparse model with a large hidden state | ~3 min |
| 05 | [`05_ensembles_and_uncertainty.py`](05_ensembles_and_uncertainty.py) | ensembles of any architecture, forecast spread vs error, calibration | ~6 min |
| 06 | [`06_compare_architectures.py`](06_compare_architectures.py) | looping over the registry to compare every architecture | ~9 min |

\*Measured on an Apple-silicon laptop CPU, 2 threads per run (150 epochs; 05: 5 members × 100 epochs). Fast mode takes a few seconds each.

## Running

```bash
pip install -e ".[dev]"              # once, from the repo root
python tutorials/01_vanilla_hcnn.py  # figures/checkpoints go to ./tutorial_outputs/
```

The scripts use `# %%` cells, so VS Code / PyCharm run them cell by cell. To get a notebook, run
`pip install -e ".[notebooks]"` and then `jupytext --to notebook tutorials/01_vanilla_hcnn.py`
(generated notebooks are git-ignored; edit the `.py`).

Environment variables:

| Variable | Effect |
|---|---|
| `HCNN_TUTORIAL_FAST=1` | tiny data and 1–2 epochs; this is how `tests/test_tutorials.py` runs every tutorial in CI |
| `HCNN_TUTORIAL_OUT=<dir>` | where figures and checkpoints are written (default `./tutorial_outputs`) |

## Adding a tutorial

For a new architecture, copy `01_vanilla_hcnn.py` to `NN_<name>_hcnn.py`. Keep the `FAST` /
`HCNN_TUTORIAL_OUT` lines at the top, so the test suite picks it up and runs it automatically. Show
what is specific to your architecture (as 02–04 do), not only the training loop.
