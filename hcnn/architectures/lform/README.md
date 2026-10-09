# LSTM-formulation (LForm) HCNN

> **Maintainer(s):** @rock-feller · **Status:** stable · **Commit scope:** `lform`

## Idea

A gated residual update inspired by the LSTM cell state. A learnable diagonal gate `D ∈ (0, 1)`
decides, per state coordinate, how much of the vanilla update to take versus how much of the
(corrected) state to keep. `D → 0` gives pure memory and `D → 1` gives vanilla dynamics. This helps
gradients flow over long windows.

## Recurrence

```
r_t     = s_t − Cᵀ δ_t                   (teacher forcing; r_t = s_t when forecasting)
s_{t+1} = r_t + D (A tanh(r_t) − r_t) [+ B u_t]
```

`D` is a `DiagonalMatrix` (`hcnn/core/layers/linear.py`): off-diagonals are masked to zero and the
diagonal is clamped to `[ε, 1 − ε]`.

## Hyper-parameters

| Argument | Default | Meaning |
|---|---|---|
| `init_diag` | `1.0` | initial value of the diagonal of `D` |
| `init_range` | `(-0.75, 0.75)` | uniform init range for `A` (and `B`) |
| `n_obs_vars`, `n_hid_vars`, `s0_nature`, `train_s0`, `n_ext_vars` | | as for Vanilla |

Inspect the gate with `model.get_diagonal_values()`.

## Files

`cell.py` · `model.py` · `ensemble.py` · tests: `tests/architectures/test_lform.py` ·
tutorial: `tutorials/03_lform_hcnn.py`
