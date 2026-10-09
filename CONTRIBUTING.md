# Contributing to `hcnn`

`hcnn` is a research library for modeling chaotic dynamical systems with Historical Consistent
Neural Networks (HCNNs). Several people work on different HCNN architectures in parallel.
Correctness and reproducibility matter more than speed of merging. A PR that adds a feature but
weakens leakage-safety or the shared recurrence contract will be sent back.

---

## 1. Set up once

```bash
git clone https://github.com/rock-feller/HCNNs_and_Chaos.git && cd HCNNs_and_Chaos
conda create -y -n hcnn_env python=3.11 && conda activate hcnn_env

# PyTorch first, for your platform:
pip3 install torch                                                   # CUDA / Apple MPS
# pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU only

pip install -e ".[dev]"                     # hcnn (editable) + pytest
git config core.hooksPath .githooks         # commit-message check on every commit
git config commit.template .gitmessage      # commit-message skeleton in your editor
```

Do **not** `pip install -r requirements.txt`. It is a full-system `pip freeze` and is not installable
on macOS or in CI.

## 2. Where your work goes

```text
hcnn/
├── core/            shared framework: base classes + THE rollout, registry, layers,
│                    generic ensemble, trainers. Owned by the maintainer; changes
│                    affect every architecture.
├── architectures/   one package per HCNN variant: vanilla/ ptf/ lform/ lspa/ ...
│   └── _template/   copy this to start a new architecture
├── baselines/       RNN / LSTM comparison models (not part of the HCNN method)
└── utils/           data generation, preprocessing, plotting, checkpoints
tests/
├── test_architecture_contract.py   runs against EVERY registered architecture
├── architectures/test_<name>.py    architecture-specific tests
└── ...                             data, layers, ensembles, trainer, registry, tutorials
tutorials/           one runnable script per architecture/topic (smoke-tested in CI)
```

**Working on one architecture** means touching only `hcnn/architectures/<name>/`,
`tests/architectures/test_<name>.py` and `tutorials/NN_<name>_hcnn.py`. You don't need to edit
any shared file, so parallel work doesn't conflict.

**Adding a new architecture:** follow the step-by-step guide in
[`hcnn/architectures/README.md`](hcnn/architectures/README.md). In short: copy `_template/`,
implement the cell, add tests and a tutorial, and add yourself to `.github/CODEOWNERS`.

**Changing `hcnn/core/`** (the rollout, base classes, trainer, registry) affects everyone. Open a
separate PR for it, explain why in the description, and expect a maintainer review.

## 3. Architecture invariants (please preserve these)

These are the load-bearing design decisions. Most are checked automatically by
`tests/test_architecture_contract.py`. If your change needs to break one, call it out explicitly in
the PR.

- **The rollout lives once** in `BaseHCNNModel.forward`. Cells return a `CellOutput`; models are
  thin wrappers that build their cell. Never add a per-architecture `forward`.
- **Observation matrix `C = [I | 0]`:** `expectation = C s_t`, and teacher forcing sets
  `r_t = s_t − Cᵀ(ŷ_t − y_t)`, so the observed part of `r_t` *is* the data. Forecasting uses no
  ground truth.
- **Standard device semantics.** Layers build on the default device; the user calls
  `model.to(device)`. Never auto-detect or grab MPS/CUDA inside a layer.
- **Leakage-safe data.** Use a burn-in, split chronologically *before* fitting normalization
  (`train_val_test_split` or `NormalizationStrategy.fit` on train only), select models on
  validation, and touch the test set once.
- **Trainers share `BaseTrainer`.** New model families implement `_batch_loss` / `_forecast`; they
  do not copy the loop.
- **Baselines stay in `hcnn.baselines`,** not in `hcnn.core` or `hcnn.architectures`.

## 4. Before every PR

```bash
pytest -q                                              # full suite, incl. tutorials in fast mode
pytest tests/test_architecture_contract.py -k lform    # just one architecture's contract
pytest tests/architectures/test_lform.py               # just one architecture's own tests
python tutorials/03_lform_hcnn.py                      # a real (non-fast) end-to-end run
```

CI checks every PR before it can merge; see §8 for what the gate contains.

New behaviour comes with a test in `tests/` that would fail without it. For an architecture,
re-derive one step of the recurrence by hand from the cell's own weights (see
`tests/architectures/test_vanilla.py`).

## 5. Commits

Follow [`COMMIT_POLICY.md`](COMMIT_POLICY.md). The hook from §1 checks every message locally. In
short:

```text
<type>(<scope>): <imperative summary, <=50 chars, no period>

- <Past-tense verb> <what changed>
- <Past-tense verb> <what changed>

<Why: required for research types (math/arch/data/hparam/exp)>
```

- **Research types** (`math`, `arch`, `data`, `hparam`, `exp`) mark commits that may change
  results. **Engineering types** (`feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `build`, `ci`,
  `style`, `chore`) must not.
- **Scope** = your architecture's package name when the change is confined to it (`arch(lform): …`),
  so `git log --grep '(lform)'` shows your history.
- The type also sets the version bump (`COMMIT_POLICY.md` §6).

If you use Claude Code, the `commit-message` skill in `.claude/skills/` drafts messages that follow
this policy.

## 6. Branch and PR workflow

1. Branch from `master`: `git checkout -b <type>/<short-topic>`, e.g. `arch/gated-sparse`,
   `fix/lspa-mask-reload`.
2. Make focused commits following §5.
3. Run `pytest -q` locally. It must be green.
4. Push and open a PR. Its title follows the commit format. The description template
   (`.github/pull_request_template.md`) asks for What / Why / Changes / Results impact / How it was
   checked; keep it short and plain (`COMMIT_POLICY.md` §8).
5. Keep PRs focused. A PR that mixes a math change with a large refactor is hard to review and hard
   to bisect later, so split them.

## 7. What a reviewer looks for

- Tests added or updated, and CI green (including the architecture contract and tutorials).
- Commit messages and PR title follow the policy. The type tells research and engineering changes
  apart.
- No regression of the invariants in §3.
- The "why" is explained, especially for changed constants, learning rates, or architectural blocks.
- For a new architecture: README filled in, tutorial added, CODEOWNERS line added.

## 8. CI/CD

**The CI gate** (`.github/workflows/ci.yml`) runs on every pull request, every push to `master`
and every merge-queue run:

| Job | Checks |
|---|---|
| `test (Python 3.10 / 3.11 / 3.12)` | `pytest`: unit tests, the architecture contract, all tutorials in fast mode |
| `build package` | sdist + wheel build, `twine check --strict`, the wheel installs and imports cleanly |
| `commit messages` | every commit in the PR and the PR title follow `COMMIT_POLICY.md` |
| **`CI gate`** | passes only if all of the above passed; the one check branch protection requires |

If only `commit messages` fails because of the **PR title**, edit the title: the gate re-runs by
itself. (A failing *commit* message needs a new commit or a reworded branch.)

`master` is protected (`tools/protect_branch.sh`): changes land only through pull requests, the
`CI gate` must pass on a branch that is up to date with `master`, conversations must be resolved,
and this applies to admins too. Merge with **"Create a merge commit"** to keep the individually
tested commits.

**Releasing** (`.github/workflows/release.yml`):

1. Open a PR that bumps `__version__` in `hcnn/__init__.py` following SemVer (`COMMIT_POLICY.md`
   §6), e.g. `build(pkg): release 0.3.0`, and merge it through the gate.
2. Tag the merge commit on `master` and push the tag:
   ```bash
   git switch master && git pull
   git tag -a v0.3.0 -m "hcnn 0.3.0" && git push origin v0.3.0
   ```
3. The workflow checks the tag equals `__version__` and is on `master`, builds the package, runs
   the full suite against the *installed wheel*, and publishes a GitHub Release with the wheel,
   sdist and notes generated from the merged PR titles.
4. PyPI (optional, off by default): add a trusted publisher on pypi.org (repository
   `rock-feller/HCNNs_and_Chaos`, workflow `release.yml`, environment `pypi`), create the `pypi`
   environment in the repository settings, then set the repository variable `PUBLISH_TO_PYPI=true`.
   No API token is stored.

Thank you for keeping the research codebase clean, reproducible, and structured!
