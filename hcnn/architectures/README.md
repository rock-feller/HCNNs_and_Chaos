# HCNN architectures

Each HCNN variant is a self-contained package in this folder. A collaborator who works on one
architecture should only need to touch **its folder, its test file and its tutorial**. The shared
framework in `hcnn/core/` stays out of the way.

| Architecture | Package | Transition | Tutorial |
|---|---|---|---|
| Vanilla | [`vanilla/`](vanilla/README.md) | `s_{t+1} = A tanh(s_t − Cᵀ(ŷ_t − y_t))` | `tutorials/01_vanilla_hcnn.py` |
| Partial Teacher Forcing | [`ptf/`](ptf/README.md) | vanilla with scheduled dropout on `δ_t` | `tutorials/02_ptf_hcnn.py` |
| LSTM formulation | [`lform/`](lform/README.md) | `s_{t+1} = r_t + D(A tanh(r_t) − r_t)` | `tutorials/03_lform_hcnn.py` |
| Large-Sparse | [`lspa/`](lspa/README.md) | `s_{t+1} = (M⊙A) tanh(r_t)` | `tutorials/04_lspa_hcnn.py` |

```python
import hcnn
hcnn.list_architectures()                     # ['lform', 'lspa', 'ptf', 'vanilla']
print(hcnn.describe_architectures())
model = hcnn.build_model("lspa", n_obs_vars=3, n_hid_vars=20, sparsity_ratio=0.8)
ens = hcnn.build_ensemble("ptf", n_ensemble=5, n_obs_vars=3, n_hid_vars=20, seed=0)
```

## Anatomy of an architecture package

```
hcnn/architectures/<name>/
├── __init__.py   # exports + register_architecture(ArchitectureSpec(name="<name>", ...))
├── cell.py       # the recurrence: BaseHCNNCell subclass returning CellOutput
├── model.py      # BaseHCNNModel subclass: builds the cell; NO forward()
├── ensemble.py   # HCNNEnsemble subclass (sets model_cls, spells out __init__)
└── README.md     # idea, equations, hyper-parameters, maintainers, references
```

Packages are **discovered automatically** on `import hcnn`, and any folder whose name does not
start with `_` is imported. You never edit a shared list, so two collaborators adding
architectures in parallel don't conflict.

## What the framework gives you, and what you must respect

The framework gives you:

- **The rollout.** `BaseHCNNModel.forward` runs teacher-forced calibration over the observed window
  and then the autonomous forecast, for every architecture. You write one step of the recurrence;
  you never write the sequence loop.
- **Training.** `HCNNTrainer` (`train_only`, `train_and_validate`, `restore_best`) works on any
  `BaseHCNNModel`. If your model has a per-epoch schedule, expose `update_dropout_epoch(epoch)`
  as PTF does, and the trainer calls it.
- **Ensembles and uncertainty.** `HCNNEnsemble` gives mean / median / weighted aggregation and
  `predict_with_uncertainty`.
- **Reusable layers** in `hcnn/core/layers/`: `CustomLinear`, `DiagonalMatrix`,
  `CustomSparseLinear`, and the PTF dropout schedules.

The contract, checked automatically for every registered architecture by
`tests/test_architecture_contract.py`:

1. `cell.forward(state, teacher_forcing, observation, externals)` returns
   `CellOutput(expectation, next_state, delta_term, extras)` with the documented shapes.
2. `expectation = C s_t` with `C = [I | 0]`: the first `n_obs_vars` state coordinates are the
   observables.
3. Forecasting (`teacher_forcing=False`) uses no ground truth.
4. The model is constructible as `Model(n_obs_vars=..., n_hid_vars=...)`, with defaults for
   everything else.
5. Parameters are created on the default device. Never call `.to("mps")` / `.cuda()` inside a
   layer; the user calls `model.to(device)`.
6. Gradients reach every trainable parameter, and one epoch of `HCNNTrainer` gives a finite loss.

## Adding a new architecture

1. **Branch:** `git checkout -b arch/<name>`.
2. **Copy the template:** `cp -r hcnn/architectures/_template hcnn/architectures/<name>`.
3. **Rename** every `Template`/`template` and resolve every `TODO`. The spec `name` must equal the
   folder name.
4. **Implement the idea** in `cell.py` (the template ships the vanilla recurrence in
   `_transition`, so it passes the contract tests before you change anything).
5. **Run the contract tests:** `pytest tests/test_architecture_contract.py -k <name>`.
6. **Add architecture-specific tests** in `tests/architectures/test_<name>.py`. Re-derive one step
   of your recurrence by hand from the cell's own weights and assert that the cell matches (see
   `test_vanilla.py`).
7. **Add a tutorial** `tutorials/NN_<name>_hcnn.py` (copy `01_vanilla_hcnn.py`). It runs in the test
   suite in fast mode automatically.
8. **Claim ownership:** add a line for your folder in `.github/CODEOWNERS`.
9. **Commit** with your architecture as the scope, e.g. `arch(<name>): add gated sparse transition`
   (see [`COMMIT_POLICY.md`](../../COMMIT_POLICY.md)), and open a PR.

Changing something in `hcnn/core/` (the rollout, the base classes, the trainer) affects every
architecture. Do it in a separate PR, explain why, and expect a maintainer review.
