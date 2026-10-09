# Large-Sparse (LSpa) HCNN

> **Maintainer(s):** @rock-feller · **Status:** stable (dense-masked implementation) · **Commit scope:** `lspa`

## Idea

Large state spaces make the dense `A` (`n_state²` parameters) expensive and prone to overfitting.
LSpa uses a fixed sparsity mask on the transition matrix, so most state coordinates interact only
with a few others.

## Recurrence

```
r_t     = s_t − Cᵀ δ_t
s_{t+1} = (M ⊙ A) tanh(r_t) [+ B u_t]       M ∈ {0,1}^{n_state × n_state} fixed at init
```

Mask types (`hcnn/core/layers/sparse.py`):
- `random_block`: uniform random sparsity over the whole matrix;
- `non_obs_block`: only the columns of the hidden variables (the obs→hid and hid→hid blocks) are
  sparsified; the obs→obs and hid→obs blocks stay dense.

> The mask is currently applied to a **dense** matrix, which is correct but not efficient. True sparse
> kernels are an open research direction for this architecture.

## Hyper-parameters

| Argument | Default | Meaning |
|---|---|---|
| `sparsity_ratio` | `0.5` | fraction of zeroed weights, in `[0, 1)` |
| `mask_type` | `"random_block"` | `random_block` or `non_obs_block` |
| `bias` | `False` | bias in the sparse transition |
| `init_range` | `(-0.75, 0.75)` | uniform init range |
| `n_obs_vars`, `n_hid_vars`, `s0_nature`, `train_s0`, `n_ext_vars` | | as for Vanilla |

Inspect with `model.get_sparsity_info()` and `model.visualize_sparsity_pattern()`.

## Files

`cell.py` · `model.py` · `ensemble.py` · tests: `tests/architectures/test_lspa.py` ·
tutorial: `tutorials/04_lspa_hcnn.py`
