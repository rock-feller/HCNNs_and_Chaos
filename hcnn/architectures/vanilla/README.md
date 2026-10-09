# Vanilla HCNN

> **Maintainer(s):** @rock-feller · **Status:** stable (reference architecture) · **Commit scope:** `vanilla`

## Idea

The baseline Historical Consistent Neural Network: one state vector `s_t` holds both the observed
and the hidden variables, and a single shared transition matrix `A` evolves it. Over the observed
window the model is *teacher-forced* (the observed coordinates are replaced by the truth); beyond it,
it runs autonomously.

## Recurrence

With `C = [I | 0]`, `ŷ_t = C s_t` and `δ_t = y_t − ŷ_t`:

```
r_t     = s_t − Cᵀ δ_t          (teacher forcing; r_t = s_t when forecasting)
s_{t+1} = A tanh(r_t) [+ B u_t]
```

## Hyper-parameters

| Argument | Default | Meaning |
|---|---|---|
| `n_obs_vars`, `n_hid_vars` | — | observed / hidden state sizes |
| `s0_nature` | `"random_"` | initial state: `"random_"` (uniform in `init_range`) or `"zeros_"` |
| `train_s0` | `True` | learn the initial state |
| `init_range` | `(-0.5, 0.5)` | uniform init range for `A` (and `B`) |
| `n_ext_vars` | `None` | number of exogenous inputs `u_t` |

## Files

`cell.py` (recurrence) · `model.py` (wrapper on the shared rollout) · `ensemble.py` ·
tests: `tests/architectures/test_vanilla.py` · tutorial: `tutorials/01_vanilla_hcnn.py`
