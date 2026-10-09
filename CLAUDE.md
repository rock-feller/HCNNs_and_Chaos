# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HCNNs and Chaos** is a PhD research library modeling chaotic dynamical systems (Lorenz, Rössler, Chua) with:
- **HCNNs** (Historical Consistent Neural Networks): Vanilla, PTF (Partial Teacher Forcing), LForm (LSTM Formulation), LSpa (Large Sparse)
- **Classic RNNs and LSTMs** (baselines)

HCNNs reconstruct the observed past with teacher forcing, then roll out autonomously to forecast. PyTorch, CPU/CUDA/MPS. Several collaborators develop different architectures in parallel, and the layout is built for that.

## Layout: shared framework vs. one package per architecture

`hcnn/` is the single canonical package (the old `src/` tree is gone; map stale `src.*` imports with the table at the bottom).

```
hcnn/
├── core/                      # SHARED framework; changes affect every architecture
│   ├── base.py                # BaseHCNNCell, BaseHCNNModel (THE rollout), CellOutput, output namedtuples
│   ├── registry.py            # ArchitectureSpec, register_architecture, build_model/build_ensemble, list_architectures
│   ├── layers/                # CustomLinear, DiagonalMatrix, CustomSparseLinear, PTF dropout schedules (reusable)
│   ├── ensembles/base.py      # HCNNEnsemble (generic aggregation + predict_with_uncertainty)
│   ├── training/              # BaseTrainer (+restore_best), HCNNTrainer, EnsembleTrainer
│   └── cells/, models/, ensembles/hcnn_ensembles.py   # LEGACY import paths only (lazy re-exports)
├── architectures/             # one self-contained package per variant
│   ├── vanilla/ ptf/ lform/ lspa/   # each: __init__.py (registers), cell.py, model.py, ensemble.py, README.md
│   └── _template/             # copy-me skeleton (skipped by discovery; passes the contract as shipped)
├── baselines/                 # RNN/LSTM models, ensembles, SequenceModelTrainer: NOT part of the HCNN method
└── utils/                     # data_generation (burn-in), data_preprocessing (NormalizationStrategy,
                               # train_val_test_split, SlidingWindowDataset), plotting, checkpoints, ...
tests/
├── test_architecture_contract.py   # parametrized over EVERY registered architecture + _template
├── architectures/test_<name>.py    # architecture-specific (re-derive one step of the formula by hand)
└── test_tutorials.py               # runs every tutorials/NN_*.py with HCNN_TUTORIAL_FAST=1
tutorials/                     # 00 data, 01–04 one per architecture, 05 ensembles, 06 compare-all
tools/check_commit_msg.py      # commit-message linter (hook + CI)

chaotic_data/, data_utils/     # DEPRECATED, used only by the root *.ipynb notebooks
test/                          # legacy tests of private copies; ignored (pytest testpaths = tests)
```

## Key architecture

- **Registry and auto-discovery.** `hcnn/architectures/__init__.py` imports every sub-package whose name does not start with `_`. Each package's `__init__.py` calls `register_architecture(ArchitectureSpec(name=<folder name>, model_cls, cell_cls, ensemble_cls, summary, maintainers))`. Adding an architecture therefore needs no edit to any shared file. `hcnn.build_model("lform", n_obs_vars=3, n_hid_vars=20)`.
- **The rollout lives once** in `BaseHCNNModel.forward`: teacher-forced calibration over the window, then an autonomous forecast. Cells return `CellOutput(expectation, next_state, delta_term, extras)`; models only build their cell and may override `_pack_output` (PTF adds `partial_delta_terms` from `extras`). Never add a per-architecture `forward`.
- **Contract** (enforced by `tests/test_architecture_contract.py`): `expectation = C s_t` with `C = [I | 0]`; `delta_term = y − expectation` under teacher forcing, `None` otherwise; an autonomous step ignores observations; the model is constructible as `Model(n_obs_vars, n_hid_vars)`; it builds on CPU; gradients reach all parameters; it trains with `HCNNTrainer`; its ensemble works.
- **Avoid import cycles:** architectures import from `hcnn.core`, so `hcnn.core.cells` / `hcnn.core.models` / `hcnn.core.ensembles` resolve moved names lazily via module `__getattr__`. Don't make `hcnn.core` eagerly import `hcnn.architectures`.
- **Device semantics:** layers build on the default device; callers use `model.to(device)`. No auto-detection inside layers.
- **Trainers:** `BaseTrainer` owns the loop; subclasses implement `_batch_loss` / `_forecast`. `train_and_validate` selects the epoch on `val_data` (never the test set). `restore_best()` reloads the selected epoch's in-memory weights. `_on_epoch_start` calls `model.update_dropout_epoch` if present (PTF schedules).
- **Leakage-safe data:** burn-in → chronological split → `NormalizationStrategy` fitted on train only (`train_val_test_split` does both) → select on validation → evaluate test once.

### Known open bugs (tracked as strict xfails in the contract test; fix in separate `math(core)` / `fix(lspa)` PRs)
- **Rollout off-by-one:** the forecast restarts from the uncorrected `s_{T-1}`, so `forecasts[0] == expectations[-1]` and the last observation is ignored.
- **LSpa reload:** `sparsity_mask` is a non-persistent buffer re-drawn at init and applied in place in `forward`, so `load_state_dict` into a fresh `LSpa_Model` corrupts the weights.

Training gotchas (measured on Lorenz, dt=0.01, scaling 0.02):
- **Budget:** Vanilla with `n_hid_vars=20`, Adam lr 1e-2 needs about 150 epochs (test RMSE@300 ≈ 0.09 normalized). After 30 epochs it is still underfit and the selected forecast collapses toward the mean (≈0.17–0.18).
- **Ensemble init:** the `*HCNNEnsemble` constructors default to `init_range=(-0.75, 0.75)`, while the models default to `(-0.5, 0.5)`. With the wider range most members diverge, so pass `init_range` explicitly.

## Commands

Conda env `hcnn_env` (Python 3.11). Setup: install torch for the platform, then `pip install -e ".[dev]"` (pyproject.toml; `requirements-dev.txt` is just `-e .[dev]`). Never `pip install -r requirements.txt` (machine-specific freeze). No linter/formatter is configured.

```bash
pytest -q                                            # full suite (testpaths = tests; includes tutorials in fast mode)
pytest tests/test_architecture_contract.py -k lspa   # one architecture's contract
pytest tests/architectures/test_ptf.py -k schedule   # single test
python tutorials/01_vanilla_hcnn.py                  # full end-to-end run (outputs -> ./tutorial_outputs/)
HCNN_TUTORIAL_FAST=1 python tutorials/02_ptf_hcnn.py # fast smoke run
echo "fix(ptf): clamp p" | python tools/check_commit_msg.py -   # lint a commit message
```

If `hcnn` is not installed in the env, prefix commands with `PYTHONPATH=.`. `conda run` can fail in sandboxed shells; calling `~/miniconda3/envs/hcnn_env/bin/python -m pytest` directly works.

## Commits, CI, PRs

- `COMMIT_POLICY.md` is authoritative (the `commit-message` skill in `.claude/skills/` summarises it). Format: `<type>(<scope>): <imperative lowercase summary>` (≤50 chars ideal, 72 max, no period), a blank line, past-tense bullet body, then a *why* paragraph (required for research types), then footers. Research types: `math arch data hparam exp`. Engineering types: `feat fix perf refactor docs test build ci style chore`. Scope = the architecture package name when the change is confined to it. `!` or `BREAKING CHANGE:` marks breaking changes. No ticket IDs.
- Local hook: `git config core.hooksPath .githooks`. CI (`.github/workflows/ci.yml`): `pytest` on 3.10/3.11/3.12, a package build + wheel-import check, and on PRs a lint of every commit plus the PR title (`--header-only`). The `CI gate` job aggregates them; it is the single required check on protected `master` (`tools/protect_branch.sh`). Never require the individual job names in branch protection.
- CD (`.github/workflows/release.yml`): pushing tag `vX.Y.Z` (must equal `hcnn.__version__` and be on `master`) builds, tests the installed wheel from a source-free directory, and creates a GitHub Release. PyPI publishing via trusted publishing is opt-in (`vars.PUBLISH_TO_PYPI == 'true'`, environment `pypi`).
- Branch `<type>/<topic>`; keep research changes and refactors in separate PRs. `.github/CODEOWNERS` maps architecture folders to owners.
- PR descriptions: What / Why / Changes / Results impact / How it was checked, short and plain (`COMMIT_POLICY.md` §8, `pr-description` skill). Merge with merge commits, not squash.

## Import map (old `src.*` → canonical)

| Old (removed) | New |
|---|---|
| `src.models.HCNN.hcnn_models` (`Vanilla_Model`, …) | `from hcnn import Vanilla_Model, PTF_Model, LForm_Model, LSpa_Model` |
| `src.ensembles.hcnns` (`HCNNpTFEnsemble`, `HCNNLFormEnsemble`, `LSpaEnsemble`) | `from hcnn import VanillaHCNNEnsemble, PTFHCNNEnsemble, LFormHCNNEnsemble, LSpaHCNNEnsemble` |
| `src.single_trainers.hcnn_training.HCNNTrainer` | `from hcnn.core.training import HCNNTrainer` |
| `src.ensemble_trainers.hcnn_training.HCNNEnsembleTrainer` | `from hcnn.utils.ensemble_trainer import HCNNEnsembleTrainer` (standalone; prefer `EnsembleTrainer(ens, HCNNTrainer)`) |
| `src.models.classic` / `src.ensembles.classics` | `from hcnn.baselines import RNNModel, LSTMModel, RNNEnsemble, LSTMEnsemble` |
| `RNNTrainer` / `EnsembleRNNTrainer` / `EnsembleLSTMTrainer` | `from hcnn.baselines import SequenceModelTrainer, SequenceEnsembleTrainer` |

`hcnn.core.models.hcnn_models` and `hcnn.core.ensembles.hcnn_ensembles` still work (the notebooks use them).
