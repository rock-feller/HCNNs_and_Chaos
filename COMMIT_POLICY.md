# Commit Policy

Several people work on different HCNN architectures in this repository at the same time. The
history has to answer three questions at a glance:

1. **Did the math change, or only the engineering?** A `math` or `arch` commit can change published
   numbers; a `refactor` must not.
2. **Which architecture or module does it touch?** The scope lets each collaborator filter the log to
   their own work: `git log --oneline --grep '(lform)'`.
3. **What does it do to the version?** The type determines the version bump.

The format follows [Conventional Commits](https://www.conventionalcommits.org), extended for ML
research. A local hook and CI check every commit (see [Enforcement](#7-enforcement)).

---

## 1. Format

```text
<type>(<scope>): <imperative summary>

- <Past-tense verb> <what changed>
- <Past-tense verb> <what changed, and why if it is not obvious>

<Why: the motivation. Required for research types.>

<Footers: Fixes #12 / Refs: draft Eq. 4.2 / BREAKING CHANGE: ...>
```

**Example: a research change**

```text
arch(lform): gate the external input with D

- Routed B·u_t through the diagonal gate D instead of adding it after
- Added a regression test re-deriving one step with externals
- Updated the LForm README equation

Exogenous drivers bypassed the memory gate, so a closed gate (D -> 0)
still let u_t overwrite the state. Gating them keeps "D -> 0 = memory"
true with and without externals.

Refs: #31, thesis ch. 4, Eq. 4.7
```

**Example: an engineering change**

```text
fix(trainer): restore best weights for test eval

- Kept the selected epoch's state_dict in memory during training
- Added BaseTrainer.restore_best() and a test for it
```

## 2. Header rules

1. **Imperative mood, as if giving the codebase a command:** `fix shape mismatch`, not `fixed …` or
   `fixes …`.
2. **Lowercase** type, scope and first word of the summary. Use acronyms as written (`HCNN`, `PTF`).
3. **No trailing period.**
4. **50 characters or fewer** if possible, and **never more than 72**.
5. **One logical change per commit.** If the summary needs "and", consider splitting.

## 3. Types (`<type>`)

Pick the type that matches the *primary intent* of the diff. Don't default to `feat` or `fix` without
checking whether a research type fits better.

### 🔬 Research types: may change numerical results

| Type | Use for | Example |
|---|---|---|
| `math` | equations, loss functions, constraints, the teacher-forcing correction | `math(core): scale correction by observation noise` |
| `arch` | a new architecture, or changes to cells, layers, gates or activations | `arch(lspa): add banded sparsity mask` |
| `data` | generators, burn-in, normalization, splits, noise, windowing | `data(lorenz): discard 20 time units of transient` |
| `hparam` | default hyperparameters, seeds, schedules, run configs | `hparam(ptf): raise default target_p to 0.4` |
| `exp` | experiment scripts and notebooks, result tables, figures | `exp(lorenz): add 50-member PTF horizon sweep` |

Research commits **must have a body** that says *why*: the motivation, the equation or paper
section, and the expected effect (e.g. "prevents divergence after ~500 steps").

### 🛠️ Engineering types: must not change numerical results

| Type | Use for | Example |
|---|---|---|
| `feat` | new user-facing capability (API, helper, CLI, checkpoint tool) | `feat(registry): build ensembles by name` |
| `fix` | bugs: shape, device, NaN, wrong behaviour | `fix(trainer): cast targets to float32 on MPS` |
| `perf` | faster or lighter, with the same results | `perf(lspa): skip masked rows in transition` |
| `refactor` | restructuring with no behaviour change | `refactor(core): move ensembles next to cells` |
| `docs` | READMEs, docstrings, tutorials' prose, this policy | `docs(ptf): document dropout schedule defaults` |
| `test` | adding or fixing tests only | `test(lform): check gate stays in (0, 1)` |
| `build` | packaging and dependencies (`pyproject.toml`) | `build(pkg): add notebooks extra` |
| `ci` | GitHub Actions, hooks, the commit checker | `ci: lint commit messages on pull requests` |
| `style` | formatting only | `style(vanilla): wrap long lines` |
| `chore` | anything else that ships nothing (gitignore, housekeeping) | `chore: ignore tutorial outputs` |

> If an "engineering" commit turns out to change results (e.g. a `fix` in the rollout), say so in
> the body. Reviewers then know old numbers need re-running.

## 4. Scopes (`<scope>`)

The scope says *where* the change happened. Use the most specific one that fits:

| Area | Scopes |
|---|---|
| One architecture | its package name: `vanilla`, `ptf`, `lform`, `lspa`, or your new `<name>` |
| Shared framework | `core` (base classes, rollout), `registry`, `layers`, `ensembles`, `trainer` |
| Data | `data` (pipeline), or a system: `lorenz`, `rossler`, `chua`, … |
| Everything else | `baselines`, `tutorials`, `notebooks`, `utils`, `config`, `pkg`, `ci`, `deps` |

- A change confined to `hcnn/architectures/<name>/` (plus its test and tutorial) uses `<name>` as the
  scope. This is how a collaborator finds their own history.
- Omit the scope only when a change is genuinely repository-wide (`ci: …`, `docs: …`).
- Research types (`math`, `arch`, `data`, `hparam`, `exp`) should always have a scope.

## 5. Body and footers

**Body:** 2–4 bullets, each starting with a **past-tense verb** (Added, Fixed, Updated, Removed,
Refactored, Moved, Validated, …). They describe *what changed*. Add a short paragraph for the *why*
when the bullets don't make it obvious; for research types this paragraph is mandatory. Wrap lines
at about 72 characters. A trivial `docs`/`style`/`chore` commit may skip the body.

**Footers** (one per line, after a blank line):

| Footer | When |
|---|---|
| `Fixes #12` / `Closes #12` | the commit resolves an issue |
| `Refs: #34, draft Eq. 4.2` | issues, paper or thesis sections it relates to |
| `BREAKING CHANGE: <what breaks and how to migrate>` | public API or checkpoint compatibility breaks |
| `Co-authored-by: Name <email>` | pair work |

A breaking change can also be flagged in the header with `!`: `refactor(core)!: rename CellOutput fields`.

## 6. Versioning

`hcnn` follows [SemVer](https://semver.org). The commit types since the last release decide the next
version:

| Commit | Bump | Example |
|---|---|---|
| any `BREAKING CHANGE:` footer or `!` | **MAJOR** (1.4.2 → 2.0.0) | renamed a public class, checkpoints no longer load |
| `feat`, `arch`, `math`, `data`, `hparam` | **MINOR** (1.4.2 → 1.5.0) | new architecture, new loss, changed defaults |
| `fix`, `perf`, `refactor` | **PATCH** (1.4.2 → 1.4.3) | bug fix with no new API |
| `docs`, `test`, `build`, `ci`, `style`, `chore`, `exp` | none | |

While the version is `0.x`, breaking changes bump MINOR instead of MAJOR. The version lives in
`hcnn/__init__.py` (`__version__`), and `pyproject.toml` reads it from there.

## 7. Enforcement

Run once per clone:

```bash
git config core.hooksPath .githooks       # runs tools/check_commit_msg.py on every commit
git config commit.template .gitmessage    # pre-fills the editor with the format
```

- **Locally**, the `commit-msg` hook rejects a malformed header, an unknown type, a header over
  72 characters, a trailing period, a missing blank line after the header, and a research commit
  without a body. It also warns about non-imperative summaries and headers over 50 characters.
- **In CI**, the `commit-messages` job runs the same checker on every commit of a pull request.
- **PR titles** are checked the same way (header only): they appear in the merge commit and in the
  generated release notes.

Check a message by hand: `echo "fix(ptf): clamp p" | python tools/check_commit_msg.py -`.

## 8. Pull requests

**Title:** the commit header format (`<type>(<scope>): <imperative summary>`), using the type of
the PR's main intent.

**Description:** simple and descriptive. A reader should understand in under a minute what changed,
why, and whether any research result is affected. Use these sections (the PR template pre-fills
them; the `pr-description` skill in `.claude/skills/` drafts them):

| Section | Content |
|---|---|
| **What** | One or two plain sentences: what the PR changes |
| **Why** | The motivation: research question, bug and symptom, or paper / thesis section |
| **Changes** | One bullet per logical change (usually one per commit) |
| **Results impact** | `None.`, or which numerical results change, by how much, and what must be re-run. Mandatory for research types |
| **How it was checked** | Evidence beyond CI: before/after numbers, seeds, data, figures |
| **Notes for reviewers** | Optional: open questions, follow-ups, stacked-PR base |

Keep it to about 150–350 words. Prefer numbers to adjectives, never invent numbers or references,
and link a `docs/` write-up for long evidence instead of pasting it.

**Merging:** use **"Create a merge commit"**, not squash. The individually checked commits stay in
the history, so `git bisect` can pinpoint which change moved a result.

## 9. Cheat-sheet

```text
arch(<name>): add <idea>                      new architecture or cell change   (body required)
math(core): <change to the shared equations>  touches every architecture        (body required)
data(lorenz): <pipeline change>               may change results                (body required)
feat(registry): <new capability>
fix(<name>): <bug>
refactor(<scope>): <restructure, same results>
test(<name>): <tests only>
docs(tutorials): <prose / tutorials>
```

Thank you for keeping our research codebase clean, reproducible, and structured!

Rockefeller, PhD
