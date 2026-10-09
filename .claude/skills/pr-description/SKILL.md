---
name: pr-description
description: Drafts pull-request titles and descriptions for this research repo following COMMIT_POLICY.md §8 - short, plain-language What / Why / Changes / Results impact / How it was checked. Use when the user asks to write, draft or review a PR description or PR title. Drafts only; the user reviews and opens or edits the PR.
---

# PR description (HCNNs and Chaos)

The rules are in `COMMIT_POLICY.md` §8; this skill is the working summary. A reader (a student, a
co-author, you in six months) should understand in under a minute **what changed, why, and whether
any research result is affected**.

## Title

Same format as a commit header, because it appears in the release notes:
`<type>(<scope>): <imperative summary>`, ≤ 50 characters if possible, never > 72. Pick the type of
the PR's main intent (a research type if any commit changes results). Check it with
`printf '%s\n' "<title>" | python tools/check_commit_msg.py --header-only -`.

## Description template

```markdown
## What
One or two sentences, in plain words: what this PR changes.

## Why
The motivation: the research question, the bug and its symptom, or the paper / thesis section.

## Changes
- One bullet per logical change (usually one per commit), past tense.

## Results impact
"None." - or which numerical results change, by how much, and which earlier results must be re-run.

## How it was checked
Tests and evidence: before/after numbers (a small table is fine), seeds, dataset, figures.

## Notes for reviewers
Optional: open questions, follow-ups, what was deliberately left out, stacked-PR base.
```

## Rules

1. **Short.** Aim for 150–350 words. Put long evidence in a doc under `docs/` and link it.
2. **Plain and specific.** Numbers over adjectives ("valid forecast 0.39 → 1.27 Lyapunov times, 5
   seeds", not "much better"). No marketing words, no emoji headings.
3. **Results impact is mandatory.** Write "None." for pure engineering PRs. For research types
   (`math`, `arch`, `data`, `hparam`, `exp`) state which results change and whether earlier runs must
   be repeated.
4. **Don't restate what CI enforces** (tests pass, commit format). Say what *you* checked beyond it.
5. **Never invent** numbers, issue links or paper references. Use only what the commits, the diff,
   the docs or the user provide.
6. Mention breaking changes (API, checkpoints) explicitly, with the migration.

## Workflow

1. Collect the facts: `git log --format='%h %s%n%b' <base>..<head>`, `git diff --stat <base>..<head>`,
   and any review or results doc the branch adds.
2. Choose the title and validate it with `--header-only` (above).
3. Fill in the template. One bullet per commit is the default for **Changes**; merge trivial ones.
4. **Present the title and description to the user for review.** Do not run `gh pr create`,
   `gh pr edit` or `git push` - the user reviews, then opens or updates the PR themselves.

## Example

```markdown
fix(lspa): persist the sparsity mask

## What
LSpa models now save their sparsity mask with the weights, so a reloaded model is the same model.

## Why
The mask was drawn at random at construction and not saved. Reloading a checkpoint drew a new
mask and silently zeroed different weights.

## Changes
- Registered the mask as a persistent buffer; old checkpoints rebuild it from the zero pattern
- Applied the mask in `forward` instead of overwriting the weights
- Removed a hidden gradient hook (clip to 1.0, NaN → 0)

## Results impact
LSpa results computed from reloaded checkpoints were wrong and should be re-run. Training
without reloading is affected only through the removed gradient clipping.

## How it was checked
Round-trip test: forecasts identical after save/load (previously differed by up to 0.21);
old-format checkpoints load and match.
```
