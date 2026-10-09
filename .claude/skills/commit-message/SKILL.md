---
name: commit-message
description: Writes git commit messages for this repo following COMMIT_POLICY.md - research/engineering type prefixes, architecture scopes, version-bump implications, and a past-tense bullet body with a "why". Use when the user asks to write, generate, or draft a commit message, asks "what should I commit this as", or asks to commit staged changes.
---

# Commit message (HCNNs and Chaos)

The full policy is in `COMMIT_POLICY.md` at the repo root; this skill is the working summary.
The message must pass `python tools/check_commit_msg.py`.

## Template

```
<type>(<scope>): <imperative summary>

- <Past-tense verb> <what changed>
- <Past-tense verb> <what changed, and why if not obvious>

<Why paragraph - REQUIRED for math/arch/data/hparam/exp>

<Footers: Fixes #12 | Refs: #34, draft Eq. 4.2 | BREAKING CHANGE: ...>
```

## Rules

1. **Header**: lowercase type, optional lowercase scope in parentheses, `: `, then an imperative
   lowercase summary with no trailing period. ≤ 50 chars if possible, never > 72.
2. **Body**: 2–4 bullets, each starting with a past-tense verb (Added, Fixed, Updated, Removed,
   Refactored, Moved, Validated, …), describing what changed. Research types also get a short
   paragraph explaining *why* (motivation, equation / paper section, expected effect on results).
   Trivial `docs`/`style`/`chore` commits may omit the body.
3. **One logical change per commit.** If the staged diff mixes a research change with a refactor,
   say so and propose splitting it (`git add -p`) instead of writing one message.
4. **Breaking changes** (public API renamed/removed, checkpoints no longer load): add `!` after the
   type/scope and a `BREAKING CHANGE: <what breaks, how to migrate>` footer.
5. Never invent issue numbers or paper references; only include footers the user gave you or that
   appear in the branch name / diff.

## Choosing the type

Check the research types first: they flag commits that may change numerical results.

| Type | Bump | Use for |
|------|------|---------|
| `math` | MINOR | equations, losses, constraints, teacher-forcing correction |
| `arch` | MINOR | new architecture; cell / layer / gate / activation changes |
| `data` | MINOR | generators, burn-in, normalization, splits, windowing |
| `hparam` | MINOR | default hyperparameters, seeds, schedules, run configs |
| `exp` | none | experiment scripts / notebooks, result tables, figures |
| `feat` | MINOR | new user-facing capability (API, helper, tool) |
| `fix` | PATCH | bug fix |
| `perf` | PATCH | faster/lighter, identical results |
| `refactor` | PATCH | restructure, no behaviour change |
| `docs` | none | READMEs, docstrings, tutorial prose, policy |
| `test` | none | tests only |
| `build` | none | `pyproject.toml`, dependencies |
| `ci` | none | GitHub Actions, hooks, commit checker |
| `style` | none | formatting only |
| `chore` | none | housekeeping (gitignore, …) |
| any `!` / `BREAKING CHANGE:` | MAJOR (MINOR while 0.x) | |

## Choosing the scope from the diff

| Paths touched | Scope |
|---|---|
| only `hcnn/architectures/<name>/` (+ `tests/architectures/test_<name>.py`, its tutorial) | `<name>` |
| `hcnn/core/base.py`, the rollout | `core` |
| `hcnn/core/registry.py` | `registry` |
| `hcnn/core/layers/` | `layers` |
| `hcnn/core/ensembles/` | `ensembles` |
| `hcnn/core/training/` | `trainer` |
| `hcnn/utils/data_*.py` | `data`, or the system (`lorenz`, `rossler`, `chua`) |
| `hcnn/baselines/` | `baselines` |
| `tutorials/` | `tutorials` |
| `*.ipynb` | `notebooks` (usually with `exp`) |
| `pyproject.toml`, `requirements*.txt` | `pkg` / `deps` |
| `.github/`, `.githooks/`, `tools/` | `ci` |

Several areas → use the dominant one, or omit the scope for genuinely repo-wide changes.

## Examples

```
arch(lspa): add banded sparsity mask

- Added a "banded" mask_type keeping |i - j| <= bandwidth
- Exposed bandwidth on LSpa_Model and LSpaHCNNEnsemble
- Added mask-shape tests and documented the option in the README

Random masks scatter couplings across the state; a band keeps each
hidden unit coupled to its neighbours, which matches the local structure
of discretised PDE states we want to scale to.

Refs: #41
```

```
fix(trainer): restore best weights for test eval

- Kept the selected epoch's state_dict in memory during training
- Added BaseTrainer.restore_best() and a regression test
```

```
refactor(core)!: rename delta_term to correction

- Renamed the field in CellOutput and every cell
- Updated the contract tests and architecture READMEs

BREAKING CHANGE: code reading CellOutput.delta_term must use .correction
```

## Workflow

1. Run `git diff --staged --stat`, then `git diff --staged`. If nothing is staged, ask what to include
   (or look at `git status`).
2. Pick the **type** from the table, research types first.
3. Pick the **scope** from the paths.
4. Write the header, the bullets, and for research types the *why* paragraph.
5. Validate it: `printf '%s' "<message>" | python tools/check_commit_msg.py -`, and fix any error.
6. Show the message for review. Commit locally only if the user asked you to commit
   (`git commit -F <file>`). Never push; the user reviews the commits and pushes. For the PR, use
   the `pr-description` skill.
