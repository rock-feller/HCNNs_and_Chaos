# Partial Teacher Forcing (PTF) HCNN

> **Maintainer(s):** @rock-feller · **Status:** stable · **Commit scope:** `ptf`

## Idea

Full teacher forcing hides the model's own errors from it during training, so it never learns to
recover from them in the autonomous forecast. PTF applies dropout to the correction `δ_t`: on dropped
coordinates the model keeps its own prediction. This is standard inverted dropout: surviving entries
are scaled by `1/(1−p)`, and it is inactive in `eval()` mode. A schedule raises the dropout rate during training,
moving the model gradually from fully teacher-forced to closer-to-autonomous behaviour.

## Recurrence

```
δ̃_t     = dropout_p(epoch)(δ_t)        (returned as `partial_delta_terms`)
r_t     = s_t − Cᵀ δ̃_t
s_{t+1} = A tanh(r_t) [+ B u_t]
```

The output is a `PTFHCNNOutput` (an `HCNNOutput` plus `partial_delta_terms`). `HCNNTrainer` advances
the schedule automatically at each epoch via `model.update_dropout_epoch(epoch)`.

## Hyper-parameters

| Argument | Default | Meaning |
|---|---|---|
| `dropout_strategy` | `"adaptive"` | `adaptive`, `linear`, `exponential`, `cosine`, `step` or `constant` |
| `dropout_params` | `None` | schedule parameters, merged over the defaults below |
| `n_obs_vars`, `n_hid_vars`, `s0_nature`, `train_s0`, `init_range`, `n_ext_vars` | | as for Vanilla |
| `target_prob`, `drop_output` | `0.5`, `False` | stored on the model but **not used** by the cell (kept for API compatibility) |

Schedule defaults (`ptf/cell.py`): `adaptive {target_p: 0.3, total_epochs: 100}`,
`linear/cosine {start_p: 0.0, end_p: 0.3, total_epochs: 100}`, `exponential` (as linear, plus
`decay_rate: 0.1`), `step {schedule: [(50, 0.1), (75, 0.2), (90, 0.3)]}`, `constant {p: 0.1}`.
`adaptive` keeps `p = 0` for the first half of training and then ramps linearly to `target_p`.
The schedule classes are in `hcnn/core/layers/dropout.py`, which other architectures can reuse.

## Files

`cell.py` · `model.py` · `ensemble.py` · tests: `tests/architectures/test_ptf.py` ·
tutorial: `tutorials/02_ptf_hcnn.py`
