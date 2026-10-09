# HCNNs and Chaos

`hcnn` is a PyTorch library for modeling and forecasting chaotic dynamical systems (Lorenz, Rössler,
Chua, …) with **Historical Consistent Neural Networks (HCNNs)** and their variants, with RNN/LSTM
baselines for comparison. It grew out of PhD research and is organised so that several people can
develop different HCNN architectures in parallel.

An HCNN keeps one state vector `s_t` holding the observed variables (first coordinates) and hidden
variables. Over the observed past it is **teacher-forced**: `r_t = s_t − Cᵀ(C s_t − y_t)` replaces
the observed part of the state with the data. Beyond the past it runs **autonomously** to forecast.

## Install

```bash
# 1. PyTorch for your platform (CUDA / CPU / Apple MPS), see pytorch.org
pip3 install torch
# 2. hcnn, editable, with the test dependencies
git clone https://github.com/rock-feller/HCNNs_and_Chaos.git && cd HCNNs_and_Chaos
pip install -e ".[dev]"            # or ".[dev,notebooks]" for Jupyter + jupytext
```

Python ≥ 3.10. Do not use `requirements.txt`; it is a machine-specific `pip freeze`.

## Quick start

```python
import torch, hcnn
from hcnn.core.training import HCNNTrainer
from hcnn.utils.data_generation import LorenzSolver
from hcnn.utils.data_preprocessing import SlidingWindowDataset, train_val_test_split

lorenz, _ = LorenzSolver(0.0, 60.0, (1.0, 0.0, 1.25), time_grid=0.01, burn_in=20.0)
train, val, test, norm = train_val_test_split(lorenz, 0.6, 0.2, scaling_factor=0.02)
loader = torch.utils.data.DataLoader(SlidingWindowDataset(train, window_size=100), batch_size=32)

model = hcnn.build_model("lform", n_obs_vars=3, n_hid_vars=20)        # any registered architecture
trainer = HCNNTrainer(model, learning_rate=1e-2)
trainer.train_and_validate(loader, num_epochs=150, calibration_window=val[:200], val_data=val[200:500])
trainer.restore_best()

forecast = model.eval()(test[:200].unsqueeze(0), forecast_horizon=300).forecasts   # (1, 300, 3)
```

## Architectures

Each lives in its own package under [`hcnn/architectures/`](hcnn/architectures/README.md), with its
own README, tests and tutorial.

| Name | Transition | Idea |
|---|---|---|
| [`vanilla`](hcnn/architectures/vanilla/README.md) | `s_{t+1} = A tanh(s_t − Cᵀ(ŷ_t − y_t))` | the reference HCNN |
| [`ptf`](hcnn/architectures/ptf/README.md) | dropout on `δ_t`, scheduled over epochs | partial teacher forcing |
| [`lform`](hcnn/architectures/lform/README.md) | `s_{t+1} = r_t + D(A tanh(r_t) − r_t)` | diagonal memory gate |
| [`lspa`](hcnn/architectures/lspa/README.md) | `s_{t+1} = (M⊙A) tanh(r_t)` | sparse transition for large states |

```python
hcnn.list_architectures()                    # ['lform', 'lspa', 'ptf', 'vanilla']
ens = hcnn.build_ensemble("ptf", n_ensemble=10, n_obs_vars=3, n_hid_vars=20, seed=0)
ens.predict_with_uncertainty(window, forecast_horizon=300)   # mean, variance, agreement
```

## Tutorials

Short, runnable scripts in [`tutorials/`](tutorials/README.md): the data pipeline (00), one per
architecture (01–04), ensembles and uncertainty (05), and comparing all architectures (06).

```bash
python tutorials/01_vanilla_hcnn.py
```

## Repository layout

```text
hcnn/
├── core/            shared framework: base classes + the one rollout, registry, layers,
│                    generic ensemble, trainers
├── architectures/   one self-contained package per HCNN variant (+ _template/ for new ones)
├── baselines/       RNN / LSTM comparison models
└── utils/           data generation, preprocessing, plotting, checkpoints
tests/               pytest suite; test_architecture_contract.py runs on every architecture
tutorials/           runnable examples (smoke-tested in CI)
docs/                research notes and topic proposals
*.ipynb              research notebooks (older workflow; import from chaotic_data/ and data_utils/)
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, the invariants every change must keep, and the PR
workflow, and [`COMMIT_POLICY.md`](COMMIT_POLICY.md) for commit messages. To add an architecture,
follow [`hcnn/architectures/README.md`](hcnn/architectures/README.md).

```bash
pytest -q          # full suite, including the tutorials in fast mode
```

## License

MIT, see [`LICENSE`](LICENSE).
