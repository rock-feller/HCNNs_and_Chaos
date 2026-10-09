<!--
PR guide: fill in every section. See CONTRIBUTING.md for the workflow and
COMMIT_POLICY.md for the commit / PR-title convention. Keep PRs focused.
The PR TITLE must follow the commit format, e.g. `arch(lform): gate externals with D`.
-->

## Summary

<!-- What does this PR do, and WHY? One or two paragraphs. For research changes,
state the mathematical / architectural motivation and reference the paper section
or issue. -->

## Architecture(s) / area touched

<!-- e.g. `lform` only · new architecture `<name>` · shared `core` (affects all) -->

## Type of change

<!-- Tick all that apply. These mirror the commit types in COMMIT_POLICY.md. -->

- [ ] `math` — core equations / loss / constraints
- [ ] `arch` — architecture (new variant, cells, layers, gates)
- [ ] `data` — data pipeline / generators / normalization
- [ ] `hparam` — hyperparameters / seeds / run configs
- [ ] `exp` — experiments / notebooks / results
- [ ] `feat` — new software feature
- [ ] `fix` — bug fix
- [ ] `perf` — performance improvement
- [ ] `refactor` — structure only, no behavior change
- [ ] `docs` / `test` / `build` / `ci` / `chore`
- [ ] **Breaking change** (public API or checkpoint compatibility) — described below

## Related issues / references

<!-- e.g. Fixes #42 ; paper draft Eq. 4.2 -->

## How was this tested?

- [ ] `pytest -q` passes locally (includes the architecture contract and tutorials in fast mode)
- [ ] Added/updated tests in `tests/` that cover this change
- [ ] (if runtime behavior) ran the relevant tutorial in full, e.g. `python tutorials/01_vanilla_hcnn.py`

<!-- Paste key output (loss curves, test summary, before/after numbers). -->

## New architecture checklist (skip otherwise)

- [ ] Package in `hcnn/architectures/<name>/` with a filled-in `README.md`
- [ ] Architecture-specific tests in `tests/architectures/test_<name>.py`
- [ ] Tutorial `tutorials/NN_<name>_hcnn.py`
- [ ] `.github/CODEOWNERS` line for the folder, test and tutorial

## Architecture-invariant checklist

<!-- See CONTRIBUTING.md §3. Confirm your change preserves them, or explain below
why it must break one. -->

- [ ] Rollout still lives once in `BaseHCNNModel.forward` (no per-architecture forward)
- [ ] Standard device semantics (no auto-grab of MPS/CUDA in layers)
- [ ] Data flow is leakage-safe (normalization fit on train only; val ≠ test)
- [ ] New trainers subclass `BaseTrainer` (no copied training loop)
- [ ] Baselines stay in `hcnn.baselines`

## Commit hygiene

- [ ] Commits and PR title follow `COMMIT_POLICY.md` (`type(scope): imperative summary`)
- [ ] PR is focused (not a mix of unrelated math + refactor changes)
