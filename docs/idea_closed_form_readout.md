# Research idea: a closed-form readout for the observed rows of A

*October 2026. Status: idea, not yet tested. The two structural facts below are checked numerically;
the training benefit is a hypothesis.*

The Vanilla HCNN under teacher forcing, with `C = [I | 0]`, `s_t = [o_t ; h_t]` and no external
inputs:

```text
ŷ_t     = C s_t = o_t                 δ_t = y_t − ŷ_t
r_t     = [y_t ; h_t]                 (teacher forcing: the observed part is the data)
z_t     = tanh(r_t)
s_{t+1} = A z_t
```

Split `A` by output rows (observed `o`, hidden `h`) and input columns (`A` is stored as `(out, in)`):

```text
A = [ A_oo  A_oh ]   = [ A_o ]        ŷ_{t+1} = A_o z_t = A_oo tanh(y_t) + A_oh tanh(h_t)
    [ A_ho  A_hh ]     [ A_h ]        h_{t+1} = A_h z_t = A_ho tanh(y_t) + A_hh tanh(h_t)
```

## The two facts the idea rests on

1. **The hidden trajectory does not depend on the observed rows `A_o`.** `A_o` produces `o_{t+1}`,
   and the data replaces it at the next step (`r_{t+1} = [y_{t+1} ; h_{t+1}]`). So `h_1, …, h_T`
   depend only on `A_h`, `h_0` and the data.
   *Check:* adding a random matrix to `A_o` changed the hidden states by at most 1e-14.
2. **For fixed `A_h`, the calibration loss is quadratic in `A_o`.** With `z_t` fixed by fact 1:

   ```text
   L(A_o, A_h) = ½ Σ_{t=1}^{T−1} ‖y_t − A_o z_{t−1}(A_h)‖²
   ```

   This is linear least squares, so the optimal readout has a closed form. Stack every `z_t`
   (t = 0 … T−2, over all windows) as the rows of `Z` and the matching `y_{t+1}` as the rows of `Y`:

   ```text
   A_o*(A_h) = Yᵀ Z (Zᵀ Z + β I)⁻¹          (ridge β > 0 for conditioning)
   ```

   The gradient of `A_o` is accordingly local, `−Σ_t δ_{t+1} z_tᵀ`, and was checked against
   autograd to 2e-15. *Check:* with random data and random `A_h`, one solve lowered the calibration
   MSE from 6.1 to 0.24.

The t = 0 term (`ŷ_0 = C s_0`) involves no row of `A`, so it plays no part here.

The calibration loss is a *separable nonlinear least-squares* problem: linear in `A_o`, nonlinear
in `A_h`. This is the setting of **variable projection** (Golub & Pereyra, 1973): eliminate the
linear parameters analytically, then optimise the reduced loss `L(A_o*(A_h), A_h)` over the
nonlinear ones only. At the exact minimiser, the envelope theorem gives the reduced loss's gradient
as `∂L/∂A_h` evaluated at `A_o*`. So the existing backprop code can be reused unchanged, with
`A_o` set to `A_o*` and excluded from the optimizer.

It is also the reservoir-computing picture (echo state networks, Jaeger 2001; chaotic forecasting,
Pathak et al. 2018; next-generation reservoir computing, Gauthier et al. 2021), with one difference:
the "reservoir" `A_h` is trained, not random. The hidden-state synchronisation measured for the
initial state (a random `h_0` is forgotten within 20–50 steps at dt 0.05) is the echo state
property of that reservoir.

## Hypotheses

- **H1 (speed).** Fewer epochs to a given valid prediction time, because 69 of the 529 parameters
  (3 + 20 state) are solved exactly every step and never lag behind `A_h`.
- **H2 (reliability).** Fewer failed seeds. In a 40-epoch pilot, 2 of 3 seeds failed under every
  loss variant; a readout that is always optimal for the current hidden dynamics might remove one
  failure mode.
- **H3 (the risk).** The forecast feeds `ŷ` back as input, so `A_o` becomes recurrent and the
  forecast loss is *not* quadratic in it. A calibration-optimal `A_o*` may forecast worse than one
  trained jointly. If so, a closed-form start followed by joint fine-tuning should recover it.

## Variants to compare

| Variant | `A_o` | `A_h`, `h_0` |
|---|---|---|
| V0 baseline | Adam, jointly | Adam |
| V1 alternating | closed form on all training windows, once per epoch | Adam, `A_o` frozen within the epoch |
| V2 variable projection | closed form every batch, from running sufficient statistics `Σ zzᵀ`, `Σ y zᵀ` | Adam |
| V3 warm start | closed form once at initialisation (ESN-style), then Adam | Adam |
| V1/V2 + fine-tune | as V1/V2, then joint Adam for the last epochs | Adam |

**Cost.** The sufficient statistics are `n × n` and `n_obs × n` (23 × 23 and 3 × 23), accumulated in
`O(N n²)`; the solve is `O(n³)`. Both are negligible next to the rollout.

## Protocol

- **Protocol:** the one in `docs/vanilla_hcnn_review.md` (Lorenz, `dt = 0.05`, cosine LR, shuffled
  windows, selection on several validation windows at about 1 Lyapunov time, test evaluated once).
- **Seeds:** 5 per variant, about 150 epochs.
- **Metrics:** median valid prediction time in Lyapunov times; number of failed seeds; epochs and
  wall-clock time to reach a fixed valid prediction time.
- **Choosing β:** on validation, over a small grid (e.g. 1e-6 … 1e-2).

## Scope across architectures

| Architecture | Does fact 1 hold? | Readout problem |
|---|---|---|
| vanilla | yes | ridge regression |
| lspa | yes | ridge per row, restricted to that row's unmasked columns |
| lform | yes, for fixed `D` | `o_{t+1} = y_t + d_o ⊙ (A_o z_t − y_t)`: still linear in `A_o`, a row-scaled least squares |
| ptf, asym_ptf | **no**, in the training forward pass: dropped observed coordinates run free, so `h_t` depends on `A_o` | does not apply as stated |
| with externals `B u_t` | yes | add `u_t` to the regressors (solve `[A_o B_o]` jointly) |

## Implementation notes

- **Where:** start in `experiments/`, as a `HCNNTrainer` subclass that overrides `_run_train_epoch`.
  Nothing in `hcnn/core` needs to change for a first test.
- **Freezing `A_o`:** after each backward pass, zero `A.weight.grad[:n_obs]`; after each optimizer
  step, write `A_o*` into `A.weight[:n_obs]` under `no_grad`. Adam's moments for those rows then stay
  zero; any weight decay on them is overwritten by the solve.
- **Building `Z`:** `z_t = tanh([y_t ; h_t])` comes straight from `model(window).states` (replace the
  observed part with the data). Detach it; the solve takes no part in backprop.
- **First check:** a unit test that perturbing `A_o` leaves `states[..., n_obs:]` unchanged under
  teacher forcing, so a future cell change that breaks fact 1 is caught.
