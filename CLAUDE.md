# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HCNNs and Chaos** is a PhD research project modeling chaotic dynamical systems (Lorenz, Rössler, Rabinovich–Fabrikant) with:
- **HCNNs** (Historical Consistent Neural Networks) — Vanilla, PTF (Partial Teacher Forcing), LForm (LSTM Formulation), LSpa (Large Sparse)
- **Classic RNNs and LSTMs** (baselines)

HCNNs reconstruct the observed past with teacher forcing, then roll out autonomously to forecast. Implemented in PyTorch with CPU/CUDA/MPS support.

## Single source of truth: `hcnn/`

**`hcnn/` is the one canonical package.** The former parallel `src/` tree has been removed (it was older, partially-migrated, and buggy). All new work goes in `hcnn/`. If you find references to `src.*` (e.g. in older notebook cells), they are stale — map them to `hcnn` per the table below.

```
hcnn/
├── core/
│   ├── base.py                 # BaseHCNNCell, BaseHCNNModel (shared rollout), BaseEnsemble, CellOutput + output namedtuples
│   ├── cells/                  # vanilla.py, ptf.py, lform.py, lspa.py — the recurrence per variant
│   ├── layers/                 # linear.py (CustomLinear, DiagonalMatrix), sparse.py (CustomSparseLinear), dropout.py (PTF schedules)
│   ├── models/                 # hcnn_models.py (Vanilla_Model, PTF_Model, LForm_Model, LSpa_Model), classic_models.py (RNNModel, LSTMModel)
│   ├── ensembles/              # hcnn_ensembles.py (generic _HCNNEnsembleBase + 4 variants), classic_ensembles.py
│   └── training/               # hcnn_trainer.py (HCNNTrainer)
├── utils/
│   ├── data_generation.py      # chaotic-system generators + burn-in (canonical)
│   ├── data_preprocessing.py   # NormalizationStrategy (fit/transform), SlidingWindowDataset, prepare_chaotic_data
│   ├── ensemble_trainer.py     # HCNNEnsembleTrainer
│   ├── device.py, checkpoints.py
│   └── fully_unfolded_mode.py, abridged_mode.py  # alternative training modes (secondary)
└── config/base.py              # YAML config scaffolding (skeleton)

chaotic_data/systems.py         # DEPRECATED (notebook-facing) — thin/older solvers, now with burn_in; prefer hcnn.utils.data_generation
data_utils/preprocess.py        # DEPRECATED (notebook-facing) — prefer hcnn.utils.data_preprocessing
test/                           # legacy tests (define private copies — do NOT test real code); real suite lives in tests/
tests/                          # canonical pytest suite that imports hcnn
```

## Key architecture (read `base.py` first)

### The rollout lives once in `BaseHCNNModel.forward`
Every variant shares the same two-phase rollout — teacher-forced calibration over the observed window, then autonomous forecast. It is implemented **once** in `BaseHCNNModel.forward` ([core/base.py]). Each cell returns a unified `CellOutput(expectation, next_state, delta_term, extras)`; each model is a thin wrapper that (a) builds its cell and (b) optionally overrides `_pack_output` (PTF adds `partial_delta_terms` via `extras`). Do **not** reintroduce a per-model `forward`.

### The recurrence lives in the cells
| Variant | Transition | Idea |
|---|---|---|
| Vanilla | `s_{t+1} = A·tanh(s_t − Cᵀδ_t)` | baseline |
| PTF | dropout on `δ_t` before the correction | partial teacher forcing (6 schedules in `layers/dropout.py`) |
| LForm | `s_{t+1} = r_t + D·(A·tanh(r_t) − r_t)`, `D` diagonal ∈(0,1) | gated residual / memory |
| LSpa | `A_sparse·tanh(r_t)` | masked transition for scale |

`δ_t = y_t − C·s_t` is the teacher-forcing correction; observation matrix `C = [I | 0]`.

### Device semantics (standard PyTorch)
Layers build on the default device (CPU); call `model.to(device)` to move a whole model. Do **not** reintroduce auto-device detection inside layers (it previously split models across CPU/MPS).

### Ensembles
`_HCNNEnsembleBase` wraps N members in an `nn.ModuleDict`. Aggregation (`mean`/`median`/`weighted_mean`, weights normalized) reduces over the member axis; `predict_with_uncertainty` returns mean/var + `model_agreement` for every variant. Pass `seed=` for reproducible member diversity.

## Import map (old `src.*` → canonical)

| Old (removed) | New |
|---|---|
| `from src.models.HCNN.hcnn_models import Vanilla_Model, PTF_Model, LForm_Model, LSpa_Model` | `from hcnn.core.models.hcnn_models import ...` |
| `from src.models.classic import RNN_Model, LSTM_Model` | `from hcnn.core.models.classic_models import RNNModel as RNN_Model, LSTMModel as LSTM_Model` |
| `from src.single_trainers.hcnn_training import HCNNTrainer` | `from hcnn.core.training import HCNNTrainer` |
| `from src.ensembles.hcnns import VanillaHCNNEnsemble, HCNNpTFEnsemble, HCNNLFormEnsemble, LSpaEnsemble` | `from hcnn.core.ensembles.hcnn_ensembles import VanillaHCNNEnsemble, PTFHCNNEnsemble, LFormHCNNEnsemble, LSpaHCNNEnsemble` |
| `from src.ensembles.classics import RNNEnsemble, LSTMEnsemble` | `from hcnn.core.ensembles.classic_ensembles import RNNEnsemble, LSTMEnsemble` |
| `from src.ensemble_trainers.hcnn_training import HCNNEnsembleTrainer` | `from hcnn.utils.ensemble_trainer import HCNNEnsembleTrainer` |
| classic single/ensemble trainers (`RNNTrainer`, `EnsembleRNNTrainer`, `EnsembleLSTMTrainer`) | **no equivalent yet** — flagged in notebooks; port to `hcnn` when needed |

## Environment & common tasks

Conda env `hcnn_env` (Python 3.11, torch 2.x via pip, numpy/scipy/pandas/matplotlib/scikit-learn/pyyaml/tqdm/pytest). Note: `requirements.txt` is a full-system `pip freeze` (Linux/CUDA-specific) — do **not** `pip install -r` it on macOS; install the curated deps above.

```bash
# run the canonical test suite (import the real package)
PYTHONPATH=. conda run -n hcnn_env python -m pytest tests/ -q

# end-to-end demo (leak-free Lorenz forecast)
PYTHONPATH=. conda run -n hcnn_env python run_data_run_vanilla_model.py
```

### Instantiation
```python
from hcnn.core.models.hcnn_models import Vanilla_Model
m = Vanilla_Model(n_obs_vars=3, n_hid_vars=10, s0_nature="random_", train_s0=True)
out = m(data_window, forecast_horizon=500)          # out.expectations, out.forecasts, ...
m = m.to("mps")                                      # standard device placement
```

### Leakage-safe data protocol (always)
1. generate with `burn_in` to discard the transient;
2. **split train/test first**;
3. `NormalizationStrategy().fit(train, scale)` then `.transform(train)` / `.transform(test)` — never fit on the full series;
4. select models on a validation window that is **not** the final test set.

## Notes for future work
- Keep the rollout in `BaseHCNNModel`; keep device semantics standard; keep normalization fit-on-train.
- `LSpa` sparsity is currently a masked *dense* matrix (correctness, not efficiency) — true sparse ops are a research direction.
- Legacy `test/` tests private copies, not the real code; add new tests under `tests/`.
- `config/` is a scaffold (no concrete configs yet).
