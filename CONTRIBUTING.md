# Contributing to `hcnn`

Thanks for contributing! This guide covers how to set up your environment, run the
regression suite, follow our commit conventions, and open a clean pull request.

`hcnn` is a research library for modeling dynamical systems with Historical
Consistent Neural Networks (HCNNs) and baseline sequence models. Correctness and
reproducibility matter more than speed of merging — a PR that adds a feature but
weakens the leakage-safety or the recurrence contract will be sent back.

---

## 1. Environment setup

We develop against a dedicated conda environment. **Do not** `pip install -r
requirements.txt` — that file is a full-system `pip freeze` and is not installable
on macOS or in CI. Use `requirements-dev.txt`.

```bash
conda create -y -n hcnn_env python=3.11
conda activate hcnn_env

# Install a stable PyTorch first (platform-specific):
pip3 install torch                                            # local: CUDA / Apple MPS
# or, for a CPU-only box / CI:
# pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements-dev.txt
```

## 2. Running the tests (do this before every PR)

The canonical suite lives in `tests/` and imports the **real** package (the legacy
`test/` directory tests private copies and is not maintained — ignore it).

```bash
PYTHONPATH=. pytest tests/ -q
```

The end-to-end demo is a quick sanity check that the full pipeline runs:

```bash
PYTHONPATH=. python run_data_run_vanilla_model.py
```

CI (GitHub Actions, `.github/workflows/ci.yml`) runs `pytest tests/` on Python 3.11
and 3.12 for every push to `master` and every pull request. **A PR cannot merge
with red CI.**

## 3. Architecture invariants (please preserve these)

These are the load-bearing design decisions. If your change needs to break one,
call it out explicitly in the PR and explain why.

- **The rollout lives once** in `BaseHCNNModel.forward`. Cells return a unified
  `CellOutput`; models are thin cell-wrappers. Do not add a per-variant `forward`.
- **Standard device semantics.** Layers build on the default device; the user calls
  `model.to(device)`. Do not auto-detect/grab MPS/CUDA inside a layer.
- **Leakage-safe data.** Split train/test *before* fitting normalization
  (`NormalizationStrategy.fit` on train only); generate trajectories with a
  `burn_in`; never select models on the final test set.
- **Trainers share `BaseTrainer`.** New model families add a trainer by subclassing
  it and implementing `_batch_loss` / `_forecast` — not by copying the loop.
- **Baselines are separate.** RNN/LSTM comparison code lives in `hcnn.baselines`,
  not `hcnn.core`.

New behavior should come with a test in `tests/` that would fail without it.

## 4. Commit messages

We follow the Conventional-Commits-based rules in
[`rules_for_commit.md`](rules_for_commit.md). In short:

```
<type>(<scope>): <imperative summary, <=50 chars, no trailing period>

<body: explain the WHY — math/hyperparameter/architecture rationale>

<footer: Fixes #123 / paper section refs>
```

Allowed types: `math`, `arch`, `data`, `hparam`, `feat`, `fix`, `docs`, `perf`,
`refactor`, `test`. Common scopes: `forecaster`, `trainer`, `priors`, `ensembles`,
`layers`, `data`, `pkg`, a specific system (`lorenz`, …). Keep the subject under 72
characters, imperative mood, no trailing period. See `rules_for_commit.md` for the
full type table and examples.

## 5. Branch & PR workflow

1. Branch from `master`: `git checkout -b <type>/<short-topic>`
   (e.g. `feat/mamba-cell`, `fix/sparse-mask-device`).
2. Make focused commits following the rules above.
3. Run `pytest tests/ -q` locally — it must be green.
4. Push and open a PR. The description is auto-populated from
   [`.github/pull_request_template.md`](.github/pull_request_template.md) — fill in
   every section (summary, type, testing evidence, checklist).
5. Keep PRs focused. A PR that mixes a math change with a large refactor is hard to
   review and hard to bisect later; split them.

## 6. What a reviewer looks for

- Tests added/updated and CI green.
- Commit messages follow the convention (types distinguish math changes from
  engineering).
- No regression of the invariants in §3 (leakage, device, single rollout).
- The "why" is explained — especially for changed constants, learning rates, or
  architectural blocks.

Thank you for keeping the research codebase clean, reproducible, and structured!
