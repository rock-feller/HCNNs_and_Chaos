# Vanilla HCNN: critical review and measured improvements

*October 2026. Branch `fix/vanilla-hcnn-review`. Every claim below comes with the measurement
behind it.*

The equations are unchanged and are the ones the code must implement:

```text
ŷ_t     = C s_t                      C = [I | 0]
r_t     = s_t − Cᵀ (ŷ_t − y_t)       teacher forcing (observed part of r_t := y_t)
r_t     = s_t                        autonomous forecast
s_{t+1} = A tanh(r_t) [+ B u_t]
```

## TL;DR

1. **The code did not implement these equations.** Three regressions crept in after the original
   2024 implementation, which was correct: a sign flip in teacher forcing, rescaled PTF dropout, and
   a forecast that ignored the last observation. They are fixed here. The sign flip was the main
   source of the "exploding gradients" problem.
2. **The biggest measured lever is not on the optimisation list. It is the sampling step.**
   Training on Lorenz sampled at `dt = 0.05` instead of `0.01` gave a valid prediction time **6×
   longer** (median 1.52 vs 0.25 Lyapunov times; all 5 seeds better).
3. **Most generic RNN tricks do little or harm.** Clipping, lower learning rates, spectral-radius-1
   initialisation and washout gave no gain, and LayerNorm would break the equations. What does help:
   a cosine LR schedule (+40 % at `dt = 0.05`), validating on several windows at about one Lyapunov
   time, and shuffling windows. Together with the fixes and `dt = 0.05`, median valid prediction time
   went from **0.21 to 2.15 Lyapunov times (≈ 10×)**.

---

## 1. Correctness bugs (all regressions; the 2024 code was right)

| # | Bug | Evidence | Fix |
|---|---|---|---|
| 1 | **Teacher-forcing sign.** Code computed `δ = y − ŷ` and then `r = s − Cᵀδ`, i.e. `r_obs = 2ŷ − y` instead of `y`. Same bug in all 4 cells and the template. | The corrected observed state missed the data by 2.9–5.0 (data scale ≈ 0.4). `∂r/∂s` was 2 instead of 0 on the observed coordinates, so ‖∂L/∂A‖ grew from 1.3 to 19.8 as the window grew from 5 to 80 steps (correct equation: 0.8 → 1.7). Training loss spiked up to 1.6 mid-training (fixed code: ≤ 3e-3). | `r = s + Cᵀδ`. New contract test: under teacher forcing the next state must not depend on the model's own observed coordinates. |
| 2 | **PTF dropout rescaled kept corrections by 1/(1−p)** (plain `nn.Dropout`). A kept correction then overshoots: `r_obs = y + p/(1−p)·(y − ŷ)`. The 2024 code cancelled this with a `(1 − p)` factor; the refactor dropped it. | Read the code; test `test_kept_corrections_are_not_rescaled`. | Unscaled Bernoulli mask. |
| 3 | **Forecast started one step too early.** The rollout restarted from the *uncorrected* `s_{T−1}`, so `forecasts[0] == expectations[−1]`, the last observation was ignored, and every forecast was shifted one step against its target. | `forecasts[0] == expectations[-1]` held exactly; changing the last observation by 100 did not change the forecast. | Forecast from `s_T = f(r_{T−1})`, as in the 2024 `forecast()` method. |
| 4 | **LSpa: reloaded models had a different random mask.** The mask was a non-persistent buffer, applied by overwriting `weight.data` in `forward`. | Forecast differed by up to 0.21 after `load_state_dict`. | Persistent mask, `F.linear(x, W ⊙ M)`; old checkpoints still load (mask rebuilt from the zero pattern). |

The tests did not catch #1 because they re-derived the formula with the same wrong sign. The new
contract tests check the *property* (`r_obs = y`), not a transcription of the code. Run against the
old code, they fail 19 times.

## 2. Naive implementations found in the code base

| Where | Issue | Status |
|---|---|---|
| `CustomSparseLinear` (LSpa) gradient hook | Silently **clipped the gradient to norm 1.0** and **replaced NaN/inf with 0** on every backward. This hides divergence, and LSpa trained under different rules from Vanilla, which confounds any architecture comparison. | Removed (the mask is applied in `forward`; use the trainer's `grad_clip`). |
| `DiagonalMatrix` (LForm) | Same pattern: a hidden clip at 0.5 plus NaN→0. A full n×n parameter for n values. `weight.data` is clamped and masked in place on every step. The gate ends up pinned at the clamp bounds (trained gates of exactly 0.000 and 1.000 seen in tutorial 03). | **Not changed** (it changes the state-dict layout). Recommended: a vector parameter `θ` with `D = sigmoid(θ)` and no hook. |
| `CustomSparseLinear` init | `init_range` is documented as a uniform range but used as a *normal* with std `a/√n`. LSpa therefore starts ≈ √(3/n) smaller than Vanilla with the same `init_range`. This confounds Vanilla vs LSpa. | Documented here; align before comparing architectures. |
| `BaseTrainer` `per_epoch` mode | Summed the losses of every batch before `backward()`, keeping the graphs of a whole epoch in memory. | Gradients now accumulated batch by batch (same maths). |
| Model selection | One validation window of 300 steps (≈ 3 Lyapunov times). At that horizon any model's error saturates, and predicting the mean scores well. Many runs selected **epoch 1**. | `train_and_validate` accepts a batch of windows (`forecast_windows`); validate at ≈ 1 Lyapunov time. |
| Ensembles | Constructors default to `init_range=(-0.75, 0.75)`, but models default to `(-0.5, 0.5)`. With the wider range most Lorenz members diverged. With the old code, seed-only members were strongly under-dispersed (spread ≪ error). After the fixes, tutorial 05 gives spread 0.11 vs error 0.15: closer, still under-dispersed. | Documented; tutorial 05 passes `init_range`. |
| Cells | Autonomous step multiplies by an identity matrix (`Ide`); `torch.as_tensor(buffer, device=...)` on every step. Pure overhead. | Low priority. |

## 3. Your list (`optim_ideas.md`), judged against HCNN specifics and measurements

Measured setup: Vanilla, `n_hid=20`, Lorenz, Adam lr 1e-2, 150 epochs, 5 seeds, 8 test windows.
Metric: **VPT**, the valid prediction time (Lyapunov times until the normalised error exceeds 0.5).
VPT spread across seeds is large (often 0.1–0.9), so only differences that hold across seeds count.

| Idea | Verdict | Why / evidence |
|---|---|---|
| **LayerNorm** (to fight vanishing/exploding gradients) | ✗ Reject for HCNN | It changes the equations. It also breaks the HCNN structure: the observables *are* state coordinates (`y = C s`), so normalising the state rescales the predictions and makes `r_obs = y` false. It discards the state's absolute scale, which carries the attractor geometry. The exploding gradients came from bug #1, not from a missing normaliser. With the correct equation, ‖∂L/∂A‖ stays bounded as the window grows. The HCNN-native answer to long-range gradient flow is LForm's gated residual. |
| Gradient clipping | ≈ No gain | `clip=1.0`: median VPT 0.20 vs 0.33 for BASE. Keep it as a safety net (`grad_clip`), not as a fix. |
| Orthogonal / Xavier init | ✗ Hurt, but scaling matters | Spectral-radius-1 init: VPT 0.10 vs 0.33. Chaotic dynamics need an expansive map, and the ±0.5 default (radius ≈ 1.4 at n = 23) works better. What matters is **keeping the radius when `n` changes**: with 50 hidden units, the fixed ±0.5 init gives 0.14, and a radius-preserving ±0.5·√(23/n) gives 1.25 (round 4). |
| Leaky integration | Already exists | `s_{t+1} = (1−α) r_t + α A tanh(r_t)` is LForm with a scalar `D`. It changes the vanilla equation, so it belongs in LForm. |
| Adam / AdamW | Keep Adam | `adamw` + `weight_decay` added. There's no evidence of overfitting (noise-free data, about 550 parameters), so decay is not expected to help. |
| LR schedules (cosine) | ✓ Helps once the model can learn | `dt = 0.01`: 0.34 vs 0.33 (no gain). `dt = 0.05`: **2.15 vs 1.52**, 4 of 5 seeds better. Added as `lr_schedule="cosine"`. |
| Momentum / Nesterov SGD | ✗ Skip | Adam already beats it on small RNNs; no reason to expect otherwise here. |
| Recurrent / variational dropout | ✗ Not on the state | Dropping state units corrupts the dynamical state that *is* the model. The HCNN-native form is PTF (dropout on the *correction*), which this repo has, now fixed (bug #2). |
| L1 / L2 | Low priority | The model underfits, not overfits. L1 on `A` is interesting as *learned* sparsity for LSpa (a research topic). |
| Early stopping | ✓ Cheap, saves time | `patience=` added. With cosine annealing, the best epochs were 132–150 of 150, so use generous patience. |
| SWA | Low priority | It averages along a trajectory in weight space. For a chaotic emulator that can average two different, individually good dynamics into a bad one. Untested; EMA of weights is the cheaper experiment. |

## 4. What actually moved the needle

Median VPT in Lyapunov times (5 seeds; values in brackets are the individual seeds).

| Change (one at a time) | dt = 0.01 | Verdict |
|---|---|---|
| Old code | 0.21 | |
| Fixed equations + 6-window validation (BASE) | 0.33 | ✓ Stable training (no loss spikes); modest skill gain |
| + shuffled windows | 0.25 (mean 0.45) | ✓ Standard, small gain in RMSE@50 (0.121 vs 0.158) |
| + validation horizon ≈ 1 Lyapunov time | 0.41 | ✓ Selection no longer picks epoch 1 |
| + washout of 10 steps | 0.12 | ✗ Lower train loss (1e-5), worse forecasts: the one-step objective ≠ forecast skill. (Option removed again.) |
| + lr 3e-3 | 0.12 | ✗ |
| + 50 hidden units (same ±0.5 init) | 0.10 | ✗ Spectral radius grows as √n (≈ 2.1). See round 4 |
| **+ sampling dt = 0.05** | **1.52** [2.25 1.52 1.66 1.47 1.22] | ✓✓ **6×, every seed** |

Why the sampling step dominates: at `dt = 0.01` with scaling 0.02, consecutive states differ by about
1 %. The vanilla map must represent a near-identity step `s_{t+1} ≈ s_t + dt·f(s_t)` through
`A tanh(s_t)`, which is badly conditioned, and its one-step error (RMSE ≈ 0.017) was *worse than
persistence* (≈ 0.01). A coarser step gives every transition real dynamics to learn.

At `dt = 0.05` (5 seeds):

| Configuration | Median VPT | Seeds | Notes |
|---|---|---|---|
| **Old code**, 1 validation window | 0.39 | 0.38 0.59 0.36 0.39 0.49 | training loss spikes to 5 and 20 |
| **Fixed equations**, same setup | **1.27** | 1.11 1.52 1.27 1.47 1.22 | every seed beats every old-code seed (3.3×) |
| + 6 validation windows | 1.52 | 2.25 1.52 1.66 1.47 1.22 | |
| + cosine LR | **2.15** | 2.46 1.74 2.15 2.25 1.68 | best overall |
| + PTF (adaptive, p → 0.3) instead of Vanilla | 1.34 | 1.80 1.91 1.31 1.22 1.34 | no gain on noise-free data |
| 50 hidden units, radius-preserving init | 1.25 | 1.39 1.46 1.17 1.25 1.08 | capacity is not the bottleneck |
| `dt = 0.1` | 1.52 | 1.52 1.66 1.01 2.23 1.47 | gain plateaus beyond 0.05 |

The matched comparison (rows 1–2) is the cleanest evidence for the equation fixes: same data, same
protocol, and the fixed model forecasts 3.3× longer with no overlap between seeds.

## 5. Recommendations, in priority order

1. **Merge the equation fixes** (bugs 1–4). Results produced with the refactored code before this
   fix used a different model from the one in the thesis.
2. **Treat the sampling step as a first-class hyper-parameter.** Report it, and sweep it
   (`dt ∈ {0.02, 0.05, 0.1}` for Lorenz) before tuning anything else. The tutorials now use
   `dt = 0.05`.
3. **Validate on several windows at a horizon of ≈ 1 Lyapunov time**, and report VPT in Lyapunov
   times alongside RMSE. RMSE at 3+ Lyapunov times rewards predicting the mean.
4. **Shuffle training windows and use the cosine schedule** (`lr_schedule="cosine"`).
5. **Scale `init_range` with the state size** to keep the spectral radius when changing `n_hid_vars`.
6. **Remove the hidden gradient hooks in LForm** and parametrise `D = sigmoid(θ)`. Align LSpa's
   initialisation with Vanilla's before comparing architectures. Since LSpa's hidden gradient clip was
   removed, LSpa is the weakest architecture in tutorial 06 (test RMSE 0.55 vs 0.09 for Vanilla,
   single seed). Its small, undocumented init is the first suspect; this needs a multi-seed check.
7. **Research directions for students** (these change the objective, not the equations):
   multi-step / free-running loss and curricula, where PTF is the HCNN-native version and is now
   correctly implemented; ensemble calibration (members are under-dispersed); learned sparsity for
   LSpa (L1 or RigL instead of a random mask).

## Reproducing

The ablation script and raw results are not part of the package. To reproduce, train
`hcnn.Vanilla_Model(3, 20)` with `HCNNTrainer(lr=1e-2)` for 150 epochs on
`LorenzSolver(0, 6000·dt, (1, 0, 1.25), dt, burn_in=20)`, split 60/20/20 with
`train_val_test_split(..., scaling_factor=0.02)`, windows of 100, batch 32, and evaluate VPT on 8
evenly spaced test windows (`forecast_windows(test, 200, 300, 8)`) with threshold 0.5 on the error
normalised by the training attractor's RMS.
