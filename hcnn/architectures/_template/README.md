# <Name> HCNN

> **Maintainer(s):** @your-github-handle · **Status:** experimental · **Commit scope:** `<name>`

## Idea

TODO: two or three sentences — what problem does this variant address, and how?

## Recurrence

With `ŷ_t = C s_t` and, under teacher forcing, `r_t = s_t − Cᵀ (ŷ_t − y_t)` (the observed part of `r_t` is the data):

```
s_{t+1} = TODO
```

## Hyper-parameters

| Argument | Default | Meaning |
|---|---|---|
| `n_obs_vars`, `n_hid_vars` | — | observed / hidden state sizes |
| TODO | | |

## Files

| File | Contents |
|---|---|
| `cell.py` | the recurrence (`CellOutput` per step) |
| `model.py` | wrapper around the shared rollout |
| `ensemble.py` | ensemble wrapper |
| `tests/architectures/test_<name>.py` | architecture-specific tests |
| `tutorials/NN_<name>_hcnn.py` | runnable tutorial |

## References

TODO: paper / thesis section.
