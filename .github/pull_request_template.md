<!--
PR guide: fill in every section. See CONTRIBUTING.md for the full workflow and
rules_for_commit.md for the commit-message convention. Keep PRs focused.
-->

## Summary

<!-- What does this PR do, and WHY? One or two paragraphs. For research changes,
state the mathematical / architectural motivation and reference the paper section
or issue. -->

## Type of change

<!-- Tick all that apply. These mirror the commit types in rules_for_commit.md. -->

- [ ] `math` — core equations / loss / constraints
- [ ] `arch` — network architecture (layers, cells, activations)
- [ ] `data` — data pipeline / generators / normalization
- [ ] `hparam` — hyperparameters / seeds / run configs
- [ ] `feat` — new software feature
- [ ] `fix` — bug fix
- [ ] `perf` — performance improvement
- [ ] `refactor` — structure only, no behavior change
- [ ] `docs` — documentation only
- [ ] `test` — tests only

## Related issues / references

<!-- e.g. Fixes #42 ; paper draft Eq. 4.2 -->

## How was this tested?

<!-- Commands run and their result. At minimum: -->

- [ ] `PYTHONPATH=. pytest tests/ -q` passes locally
- [ ] Added/updated tests in `tests/` that cover this change
- [ ] (if runtime behavior) ran `python run_data_run_vanilla_model.py` or an
      equivalent end-to-end check

<!-- Paste key output (loss curves, test summary, before/after numbers). -->

## Architecture-invariant checklist

<!-- These are the load-bearing invariants (see CONTRIBUTING.md §3). Confirm your
change preserves them, or explain below why it must break one. -->

- [ ] Rollout still lives once in `BaseHCNNModel.forward` (no per-variant forward)
- [ ] Standard device semantics (no auto-grab of MPS/CUDA in layers)
- [ ] Data flow is leakage-safe (normalization fit on train only; val ≠ test)
- [ ] New trainers subclass `BaseTrainer` (no copied training loop)
- [ ] Baselines stay in `hcnn.baselines`, not `hcnn.core`

## Commit hygiene

- [ ] Commits follow `rules_for_commit.md` (`type(scope): imperative summary`)
- [ ] PR is focused (not a mix of unrelated math + refactor changes)
