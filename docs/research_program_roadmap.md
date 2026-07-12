# HCNN × Solar-PV Research Program — Execution Roadmap

How the group actually delivers the topics in [`msc_topics_solar_pv.md`](msc_topics_solar_pv.md):
shared infrastructure, data strategy, a repeatable per-topic recipe, a 12-month MSc
timeline, governance, and risks. The guiding idea is that **students should not each
rebuild the plumbing** — we build a shared foundation once (on the clean, CI-guarded
`hcnn` package) so every thesis is a focused scientific contribution on top of it.

---

## 1. Guiding principles

1. **Simulation-first, real-data-second.** Every topic is prototyped on a
   high-fidelity physics simulation (PVLib / Simulink single-diode + thermal +
   soiling + array models) that emits **both the observables and the ground-truth
   latent states**. This is uniquely valuable for HCNN: we can *validate the
   inferred hidden state* (soiling, cell temperature, diode parameters, shading map)
   against simulation truth before touching noisy field data — and it de-risks the
   biggest threat (scarce real labels).
2. **Physics as a prior, not a postprocessor.** The single-diode equation, thermal
   energy balance, and monotone degradation enter as differentiable
   constraints/regularizers on the HCNN state, not as separate models.
3. **One shared evaluation standard.** Valid-prediction-time, CRPS/coverage, and the
   *domain engineering baselines* (P&O, INC, PSO, HSU/Kimber, Faiman) are implemented
   once, so "HCNN outperforms" is comparable and defensible across theses.
4. **Leakage-safe by construction.** Everyone uses the repo's data contract
   (split-then-fit normalization, burn-in, val ≠ test) enforced by `CONTRIBUTING.md`
   and CI.
5. **Everything through the PR + CI process.** Shared-foundation code is reviewed and
   test-gated like any contribution.

---

## 2. Phase 0 — The shared foundation (build once, reuse everywhere)

This is the critical enabler and should exist before students start topic work. It
extends `hcnn` the same way `chaotic_data/` supports the chaotic experiments.

| Module (proposed) | Purpose | Analogue in repo |
|-------------------|---------|------------------|
| `hcnn/systems/pv/single_diode.py` | Differentiable single-diode I–V, MPP solver | `chaotic_data/systems.py` |
| `hcnn/systems/pv/thermal.py` | Lumped thermal energy-balance (cell temp dynamics) | — |
| `hcnn/systems/pv/soiling.py` | Soiling accumulation / rain wash-off (hysteretic) | — |
| `hcnn/systems/pv/array.py` | Multi-module array, bypass diodes, partial-shading P–V, spatial topology | — |
| `hcnn/systems/pv/faults.py` | Fault injectors (arc, hot-spot, diode, disconnect, degradation) | — |
| `hcnn/systems/pv/generators.py` | Emit trajectories **with latent ground truth** + burn-in, leakage-safe loaders | `LorenzSolver` + `SlidingWindowDataset` |
| `hcnn/physics/priors.py` | Diode residual, thermal balance, monotonicity as loss terms | `model_utils/custom_losses` |
| `hcnn/evaluation/metrics.py` | Valid-prediction-time, CRPS, coverage, tracking efficiency | — |
| `hcnn/evaluation/baselines.py` | Reference P&O/INC/PSO/HSU/Kimber/Faiman + adapters to `hcnn.baselines` LSTM/RNN | `hcnn.baselines` |
| `benchmarks/` | Standard task suite + a small CI smoke benchmark | `tests/` |

**Ownership:** advisor + one "infrastructure lead" student build Phase 0; it is not a
thesis by itself but every thesis depends on it. Target: **~6–8 weeks**.

---

## 3. Data strategy

The **simulation source is the workhorse**; the **real source is for validation and
the novelty claim**. The colleague's African field data is the differentiator for the
efficiency topics and must be secured early.

| Topic(s) | Simulation (primary) | Real validation | What to obtain from colleague |
|----------|----------------------|-----------------|-------------------------------|
| 1.1, 1.2 MPPT | PVLib / Simulink single-diode under measured G/T | NREL PVDAQ, Sandia I–V | high-rate inverter I/V logs |
| 2.1, 2.3 Faults | Simulink fault injection | **GPVS-Faults** (labeled), **DKASC** healthy | any logged fault events |
| 2.2 Degradation/RUL | synthetic multi-year degradation | NREL / DKASC multi-year | multi-year performance history |
| 3.1 Soiling | `soiling.py` accumulation model | **DKASC Alice Springs** (arid analogue), NREL soiling | cleaning logs + African power/weather |
| 3.2 Climate transfer | multi-site simulated micro-climates | multi-site African data | **≥2–3 sites, different zones** |
| 3.3 Thermal | `thermal.py` energy balance | back-of-module T datasets | back-of-module temperature |
| 4.1–4.3 Shading | `array.py` moving-shadow scenarios | module-level array datasets (scarce → sim-led) | module/string-level telemetry |
| A, B Microgrid | aggregated simulated fleet | **Open Power System Data**, NREL fleet + local demand | fleet + demand co-located data |

**Weather everywhere:** NASA POWER and the Global Solar Atlas provide irradiance /
temperature / humidity for any African coordinate — the exogenous driver for all
topics.

**Data governance:** agree a data-sharing arrangement with the colleague in month 0
(licensing, anonymization, storage); standardize every source into the repo's
`SlidingWindowDataset` contract; keep raw data out of git (already gitignored),
version only the loaders and a small sample.

---

## 4. Per-topic execution recipe (every student follows this)

A repeatable 8-step pipeline on the shared foundation. Each step maps to a concrete
interface we already have.

1. **Name the hidden-state prior** — the unobserved dynamical variable (soiling,
   diode params, cell temp, degradation, shading regime). *This is the thesis.*
2. **Simulate** with `hcnn.systems.pv` → observables **+ latent ground truth**.
3. **Reproduce baselines** — engineering (`hcnn.evaluation.baselines`) + ML
   (`hcnn.baselines` LSTM/RNN). Establishes the bar.
4. **Build the HCNN variant** — subclass `BaseHCNNCell` if the recurrence changes;
   pick Vanilla/PTF/LForm/LSpa; add the physics prior from `hcnn.physics`.
5. **Train leakage-safe** — `HCNNTrainer`/`BaseTrainer`, split-then-fit, burn-in.
6. **Validate hidden-state recovery** — compare the inferred latent state to the
   simulation ground truth (the step black-box ML *cannot* do).
7. **Transfer to real data** — fine-tune / adapt; report the sim-to-real gap.
8. **Ablate & compare** — vs baselines with the shared metrics; write.

---

## 5. Master timeline (per MSc, ~12 months)

| Months | Phase | Student deliverable | Gate |
|--------|-------|---------------------|------|
| 0–1 | Onboarding | Run `hcnn`, reproduce a Lorenz forecast, HCNN + domain lit review | Can run + explain the observer view |
| 1–3 | Foundation + data | Contribute to / consume Phase-0 PV modules; simulate topic data | Simulated dataset with latent truth |
| 2–4 | Baselines | Engineering + ML baselines reproduced on the task | Baseline numbers locked |
| 4–8 | HCNN method | Hidden-state prior + physics constraint + variant; hidden-state recovery on sim | HCNN beats ML baseline on sim |
| 6–9 | Sim→real | Transfer to real data; quantify gap | Result holds on real data |
| 9–11 | Experiments | Ablations, comparison vs engineering baseline, UQ | HCNN ≥ engineering baseline |
| 11–12 | Writing | Thesis + conference/journal draft | Submitted |

Phase 0 runs in parallel with the first cohort's onboarding; later cohorts inherit it.

---

## 6. Team, governance, cadence

- **Advisor (you):** methodology, the observer/Kalman-gain framing, cross-topic
  consistency, review PRs.
- **PV colleague:** domain correctness (physics, baselines), field data, validation.
- **Infrastructure-lead student:** owns Phase 0; second-authors on others' methods.
- **Topic students:** one topic each, following the §4 recipe.
- **Cadence:** weekly student 1:1s, **monthly cohort review** (everyone presents
  against the shared metrics), shared code via PR + green CI only.

---

## 7. Infrastructure & tooling

- **Repo:** `hcnn` (single source of truth), `hcnn.baselines` for comparison,
  new `hcnn.systems.pv` / `hcnn.physics` / `hcnn.evaluation` for Phase 0.
- **CI:** extend the existing GitHub Actions to also run the `benchmarks/` smoke
  suite so a regression in a shared PV model is caught immediately.
- **Compute:** these models are small — the demo trains on CPU in ~1 min; MPS/CUDA is
  a nice-to-have, not a blocker. A shared results/leaderboard table (CSV in repo) per
  benchmark keeps comparisons honest.
- **Reproducibility:** seeds fixed (`torch.manual_seed`), env pinned via
  `requirements-dev.txt`, data loaders versioned, raw data out of git.

---

## 8. Risk register

| # | Risk | Likelihood | Mitigation |
|---|------|-----------|------------|
| R1 | African field data scarce / low quality | **High** | Simulation-first; DKASC (arid analogue) + NASA POWER; secure colleague data in month 0 |
| R2 | Sim-to-real gap | Medium | Validate on real early (step 7); Topic 3.2 (climate adaptation) doubles as the mitigation study |
| R3 | MSc scope creep | Medium | The §4 recipe + simulation-first bound the work; one hidden-state prior per thesis |
| R4 | Reproducibility / leakage | Low (now) | Enforced by `CONTRIBUTING.md` invariants + CI |
| R5 | Shared-foundation delays block students | Medium | Advisor + lead own Phase 0; ship `single_diode.py` + metrics harness first (unblocks 1.x, 4.x) |
| R6 | Weak baseline → indefensible claim | Medium | `hcnn.evaluation.baselines` mandates the domain baseline, not just LSTM |

---

## 9. Definition of done (per phase)

- **Phase 0:** PV generators emit latent truth; physics priors + metrics + baselines
  importable; smoke benchmark green in CI.
- **Method:** HCNN recovers the latent state on simulation (quantified) and beats the
  ML baseline.
- **Thesis:** HCNN ≥ the *engineering* baseline on real data, with calibrated
  uncertainty, on the shared metrics; reproducible from the repo.

---

## 10. Immediate next actions (first 2–4 weeks)

1. **Confirm the cohort:** which of the 12 topics, which students, who is the
   infrastructure lead.
2. **Secure data:** data-sharing agreement + sample from the colleague; download
   DKASC, GPVS-Faults, NASA POWER for the target sites.
3. **Ship the first Phase-0 slice:** `hcnn.systems.pv.single_diode` (+ MPP solver) and
   `hcnn.evaluation.metrics` — this alone unblocks Topics 1.1, 1.2, 4.1.
4. **Stand up the benchmark:** one task (e.g. MPP tracking on simulated cloudy days)
   with P&O + LSTM baselines, as the template all topics copy.
5. **Pick the two lead theses** (2.1 residual faults; 3.1 latent soiling) to start —
   highest distinctiveness, clearest hidden-state story.
