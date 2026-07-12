# MSc Research Topics — HCNN × Solar Photovoltaics

Candidate Master's thesis topics combining the **Historically Consistent Neural
Network (HCNN)** framework with Solar PV engineering. Each topic is deliberately
framed so the PV system is treated as a **partially-observed nonlinear dynamical
system**, which is what makes HCNN *required* rather than optional.

## The two levers that make these HCNN theses (not "ML on solar")

1. **Unmeasured drivers become HCNN hidden states.** Soiling load, cell
   temperature, single-diode parameters, degradation state, and shading
   configuration are physically real but rarely instrumented. HCNN infers them as
   latent state from the observable I/V/P/weather stream.
2. **The teacher-forcing innovation is a physical residual.** The correction
   `δ_t = measured − expected` is a Kalman/observer innovation. Because the HCNN
   explains weather-driven variation in its hidden states, `δ` isolates *genuine*
   deviations (faults, drift), and the **autonomous rollout** is a
   trajectory-preserving forecast rather than a smooth extrapolation.

Variant guide: **Vanilla** (baseline), **PTF** (noisy field telemetry), **LForm**
(slow / long-memory dynamics such as degradation and thermal inertia), **LSpa**
(high-dimensional arrays / fleets); **Ensembles** for calibrated uncertainty.

---

## Suggestion 1 — ML-Based MPPT for PV Systems

### Topic 1.1
🔬 **Formal Thesis Title:** *A Historically Consistent Neural Observer for Predictive Maximum Power Point Tracking under Stochastically Fluctuating Irradiance*

❓ **Core Scientific Problem:** The MPP is a moving target on the nonlinear P–V manifold, driven by turbulent (effectively chaotic) cloud dynamics. Perturb-and-Observe / Incremental-Conductance oscillate and lose energy on fast ramps; LSTM/ANN predictors are accurate one step ahead but drift and violate I–V physics over the multi-step control horizon.

📐 **How HCNN Philosophy Fits:** The historically-consistent variable is the **(V_mp, I_mp) locus trajectory**; latent irradiance/thermal dynamics are hidden states. The innovation feedback corrects unmodeled cloud transients; the rollout forecasts the MPP set-point ahead of time → predictive, not reactive, MPPT.

🛠️ **Methodology Summary:** Observed state = {V, I, P, module temp, POA irradiance if available}; hidden vars for latent cloud/thermal dynamics. Roll out a short-horizon MPP forecast as the converter reference; single-diode I–V relation as a soft physics prior.

📊 **Key Evaluation Metrics:** Tracking efficiency (%), harvested energy vs P&O/INC/PSO, transient recovery time, steady-state oscillation amplitude, valid-prediction-time under ramps.

### Topic 1.2
🔬 **Formal Thesis Title:** *Single-Diode-Constrained Historically Consistent Networks for Model-Based MPPT with Online Parameter Estimation*

❓ **Core Scientific Problem:** Purely data-driven MPPT discards the known I–V physics and extrapolates poorly to unseen irradiance/temperature; the five single-diode parameters drift slowly with temperature and aging.

📐 **How HCNN Philosophy Fits:** The dynamical prior is the **single-diode equation**; the slowly-drifting diode parameters are the hidden states (a deep Extended-Kalman-style parameter observer). **LForm** suits the slow drift.

🛠️ **Methodology Summary:** Hidden state encodes diode parameters; observation predicts current from voltage via the diode equation; teacher-force on measured current; locate the MPP analytically from the estimated curve.

📊 **Key Evaluation Metrics:** Parameter-ID accuracy vs offline curve-fitting, MPP tracking efficiency, noise robustness (PTF vs Vanilla), generalization to unseen G/T.

---

## Suggestion 2 — Fault Detection & Diagnosis

### Topic 2.1
🔬 **Formal Thesis Title:** *Innovation-Residual Fault Detection in PV Systems via Historically Consistent Neural Observers*

❓ **Core Scientific Problem:** Faults are deviations from normal dynamics, but supervised classifiers need scarce labels and raise false alarms because they cannot separate weather-driven from fault-driven change.

📐 **How HCNN Philosophy Fits:** The strongest fit in the set — the observer **innovation `δ_t` is the physically-grounded residual**. Weather variation is explained by the hidden states, so a residual shift isolates a genuine fault. Trains only on healthy data (unsupervised / open-set).

🛠️ **Methodology Summary:** Train on healthy-plant series; monitor the innovation online with CUSUM/statistical change-detection; classify fault type by residual signature and (arrays) spatial pattern via LSpa.

📊 **Key Evaluation Metrics:** Detection rate vs false-alarm rate under variable weather, time-to-detect incipient faults, AUC vs supervised CNN/SVM, unseen-fault-type detection (open-set).

### Topic 2.2
🔬 **Formal Thesis Title:** *Historically Consistent Prognostics of PV Degradation Trajectories for Remaining-Useful-Life Estimation*

❓ **Core Scientific Problem:** Degradation (PID, LID, corrosion) is slow, nonlinear and path-dependent; regression gives points, not physically-plausible monotone trajectories, and multi-year extrapolation is unstable.

📐 **How HCNN Philosophy Fits:** A latent **degradation state** evolving monotonically; the long-horizon rollout preserves the trajectory shape. **LForm** for long memory; ensembles for RUL intervals.

🛠️ **Methodology Summary:** Hidden degradation states on multi-year performance + weather; monotonicity constraint; roll out years ahead; `predict_with_uncertainty` + conformal calibration for RUL bounds.

📊 **Key Evaluation Metrics:** RUL error, long-horizon efficiency forecast error, trajectory plausibility, RUL-interval coverage/calibration, vs Kalman/particle-filter prognostics.

### Topic 2.3
🔬 **Formal Thesis Title:** *Large-Sparse HCNN for Spatially-Resolved Fault Localization in PV Arrays*

❓ **Core Scientific Problem:** Localizing *which* module/string is faulty is a high-dimensional, spatially-coupled problem; thermography/image methods need dedicated hardware and labels.

📐 **How HCNN Philosophy Fits:** **LSpa**'s sparsity mask encodes the physical electrical topology; localized residual spikes point to the faulty node; historical consistency separates spatially-correlated weather from local faults.

🛠️ **Methodology Summary:** LSpa mask from the wiring graph; string-level measurements; map spatial innovation pattern to module/string localization.

📊 **Key Evaluation Metrics:** Localization accuracy, sensor economy, vs thermographic-CNN pipelines under variable irradiance.

---

## Suggestion 3 — PV Efficiency Estimation under African Climate

### Topic 3.1
🔬 **Formal Thesis Title:** *Latent Soiling and Thermal Dynamics in Historically Consistent PV Efficiency Forecasting under Arid African Climates*

❓ **Core Scientific Problem:** In arid/Sahelian conditions, **dust soiling** is the dominant *unmeasured* loss — accumulating nonlinearly and resetting hysteretically with rain/cleaning. Efficiency regressors have no state for it, so performance ratio drifts and cleaning is scheduled blindly.

📐 **How HCNN Philosophy Fits:** Hidden state = soiling load (+ thermal dynamics); historically-consistent variable = the soiling accumulation/wash-off trajectory. The observer infers soiling from the power residual with no soiling sensor. **PTF** for noisy telemetry.

🛠️ **Methodology Summary:** Inputs = POA irradiance, temperature, humidity, rainfall, AC power; hidden soiling+thermal states; teacher-force on power; validate soiling against known cleaning/rain events.

📊 **Key Evaluation Metrics:** Efficiency/PR forecast error, inferred vs measured soiling ratio, economic value of an HCNN-optimized cleaning schedule, vs HSU/Kimber soiling models.

### Topic 3.2
🔬 **Formal Thesis Title:** *Climate-Adaptive Historically Consistent Networks for PV Efficiency Estimation across African Micro-Climates*

❓ **Core Scientific Problem:** Models fit on one zone (coastal humid vs highland vs Sahel) fail elsewhere due to distribution shift, though device physics is shared; per-site ANNs retrain wastefully.

📐 **How HCNN Philosophy Fits:** The shared dynamical prior transfers; climate-specific behavior is absorbed into hidden-state initialization and the observer gain, adapted per site with little data.

🛠️ **Methodology Summary:** Pretrain a multi-site HCNN; few-shot adapt the hidden-state/gain to a new site; ensemble across sites for regional UQ.

📊 **Key Evaluation Metrics:** Few-shot cross-climate forecast error, data-efficiency curves, vs site-specific ANN retraining, cross-zone uncertainty calibration.

### Topic 3.3
🔬 **Formal Thesis Title:** *Coupled Thermal–Electrical Historically Consistent Modeling of PV Efficiency under High-Temperature African Conditions*

❓ **Core Scientific Problem:** Cell temperature governs efficiency and has real dynamics (thermal mass, wind, mounting) under strong diurnal heating; steady-state NOCT/Faiman models miss transients and cell temperature is partially observed.

📐 **How HCNN Philosophy Fits:** A hidden **thermal state** coupled to electrical output; the energy-balance equation is the prior; the observer estimates cell temperature from the electrical residual.

🛠️ **Methodology Summary:** Lumped thermal energy-balance prior; jointly forecast module temperature and efficiency; LForm (thermal inertia) vs Vanilla.

📊 **Key Evaluation Metrics:** Cell-temperature RMSE vs back-of-module sensors, efficiency forecast error, vs Faiman/Sandia/Ross thermal models.

---

## Suggestion 4 — MPPT under Partial Shading

### Topic 4.1
🔬 **Formal Thesis Title:** *Historically Consistent Global-MPP Tracking on Multi-Peak P–V Manifolds under Dynamic Partial Shading*

❓ **Core Scientific Problem:** Partial shading makes the P–V curve multi-modal; conventional MPPT traps in local peaks, and the global MPP jumps discontinuously (bifurcation-like) as shadows move — a regime-switching behavior smooth LSTMs cannot represent.

📐 **How HCNN Philosophy Fits:** Model the manifold as a dynamical system with regime switches; hidden state encodes the bypass-diode activation configuration; the innovation detects a regime change (new peak) before the tracker stalls.

🛠️ **Methodology Summary:** State includes the discrete bypass-diode regime + latent shading dynamics; forecast GMPP location and feed the converter; train/validate on moving-shadow scenarios. A selective/input-dependent transition (Mamba-style) is a natural upgrade.

📊 **Key Evaluation Metrics:** GMPP tracking-success rate, local-MPP trapping rate, energy vs PSO/GWO global search, convergence speed under moving shadows.

### Topic 4.2
🔬 **Formal Thesis Title:** *Large-Sparse HCNN for Spatiotemporal Forecasting of Partial-Shading Propagation across PV Arrays*

❓ **Core Scientific Problem:** Shadows propagate spatially across an array; predicting which modules shade next enables proactive reconfiguration — a high-dimensional coupled spatiotemporal problem point-wise MPPT ignores.

📐 **How HCNN Philosophy Fits:** **LSpa** with a spatial-adjacency mask forecasts the shading front as a historically-consistent spatial trajectory.

🛠️ **Methodology Summary:** LSpa mask = spatial neighborhood/topology; module-level irradiance/power; forecast the shading map to drive dynamic reconfiguration / distributed MPPT.

📊 **Key Evaluation Metrics:** Shading-map forecast accuracy, energy gain from proactive vs static reconfiguration, valid prediction time of the shading front.

### Topic 4.3
🔬 **Formal Thesis Title:** *Uncertainty-Aware Historically Consistent GMPP Forecasting under Intermittent Shading using HCNN Ensembles*

❓ **Core Scientific Problem:** Fast intermittent shading makes the near-future GMPP genuinely uncertain; deterministic MPPT over/under-commits and communicates no risk.

📐 **How HCNN Philosophy Fits:** The HCNN ensemble (`predict_with_uncertainty`, `model_agreement`) yields a predictive distribution over the GMPP for risk-aware set-point control.

🛠️ **Methodology Summary:** Deep-ensemble HCNN + conformal intervals; a risk-aware control rule weighting the set-point by forecast confidence.

📊 **Key Evaluation Metrics:** CRPS and interval coverage of the GMPP forecast, harvested energy vs deterministic MPPT, robustness across intermittency levels.

---

## ⭐ Bonus — Cross-Disciplinary: PV → Microgrid / Smart-Grid Scaling

### Bonus Topic A
🔬 **Formal Thesis Title:** *Hierarchical Historically Consistent Forecasting of Distributed PV Fleets for Microgrid Dispatch under Regional Climate Patterns*

❓ **Core Scientific Problem:** Microgrid dispatch needs multi-horizon net-generation forecasts aggregated over many heterogeneous distributed plants correlated by regional weather (Harmattan dust, monsoon fronts); independent per-plant models miss the spatiotemporal correlation driving system-level ramps.

📐 **How HCNN Philosophy Fits:** A hierarchical / LSpa HCNN where each plant is a sub-state coupled through a shared regional-climate hidden state; the aggregate trajectory preserves fleet-level ramp dynamics; ensemble UQ sizes reserves.

🛠️ **Methodology Summary:** Graph/hierarchical HCNN over plants with a regional climate driver; forecast aggregate and per-node generation with calibrated uncertainty for dispatch.

📊 **Key Evaluation Metrics:** Aggregate net-load forecast skill, ramp-event forecasting, reserve/dispatch cost savings, vs persistence / LSTM / graph-NN baselines.

### Bonus Topic B
🔬 **Formal Thesis Title:** *Regime-Switching Historically Consistent Co-Forecasting of PV Generation and Demand for Smart-Grid Stability under Seasonal African Climate Cycles*

❓ **Core Scientific Problem:** Grid stability requires joint PV-and-demand forecasting that survives seasonal regime shifts (dry/wet, cool/hot) and the coupling between chaotic weather and human demand — a coupled non-stationary dynamical system single-target models handle poorly.

📐 **How HCNN Philosophy Fits:** A multivariate HCNN with a regime-switching hidden variable (latent seasonal/climate index) drives a selective transition, co-evolving PV, load and weather across regimes for long-horizon stability planning.

🛠️ **Methodology Summary:** Multivariate HCNN (PV, load, weather) with a latent regime state; long-horizon rollout; evaluate against grid-stability proxies (ramp coverage, reserve adequacy).

📊 **Key Evaluation Metrics:** Joint PV–demand forecast skill, regime-transition handling, grid-stability KPIs, economic-dispatch improvement.

---

## Selection matrix (topic × variant × prior × baseline × data × metric)

Use this to help a student pick a topic that matches their interest and available data. "Hidden-state prior" is the unobserved dynamical variable the HCNN must infer — the crux of each thesis.

| ID | Short title | HCNN variant | Hidden-state prior (the crux) | Engineering baseline to beat | Representative data | Headline metric |
|----|-------------|--------------|-------------------------------|------------------------------|---------------------|-----------------|
| 1.1 | Predictive MPP observer | Vanilla / PTF | latent irradiance + thermal dynamics | P&O, INC, PSO-MPPT | Simulink PV + measured irradiance (NREL/African station) | tracking efficiency, energy vs P&O |
| 1.2 | Diode-constrained MPPT | LForm | single-diode parameters (drift) | offline curve-fit + P&O | I–V sweeps + G/T (Sandia, PVLib synthetic) | param-ID accuracy, MPP efficiency |
| 2.1 | Innovation-residual faults | Vanilla + LSpa | weather-explained normal dynamics | supervised CNN/SVM, threshold | GPVS-Faults, DKASC healthy+fault | detection vs false-alarm, latency |
| 2.2 | Degradation prognostics (RUL) | LForm + Ensemble | latent degradation state (monotone) | Kalman/particle-filter prognostics | multi-year fleet data (NREL, DKASC) | RUL error, interval coverage |
| 2.3 | Array fault localization | LSpa | electrical-topology coupling | thermography-CNN | string/module-level plant SCADA | localization accuracy, sensor economy |
| 3.1 | Latent soiling forecasting | PTF + LForm | soiling load (hysteretic) | HSU/Kimber soiling models | arid-site power + weather + rain (NASA POWER) | PR forecast, cleaning-schedule value |
| 3.2 | Climate-adaptive efficiency | Vanilla + Ensemble | site-specific latent init/gain | site-specific ANN retrain | multi-site African micro-climate data | few-shot transfer error |
| 3.3 | Thermal–electrical coupling | LForm | module thermal state | Faiman/Sandia/Ross thermal | high-T site power + back-of-module T | cell-T RMSE, efficiency error |
| 4.1 | GMPP on multi-peak manifold | Vanilla (selective A) | bypass-diode regime + shading dynamics | PSO/GWO global MPPT | Simulink multi-module moving-shadow | GMPP success, trapping rate |
| 4.2 | Shading-front spatiotemporal | LSpa | spatial-adjacency coupling | static config MPPT | module-level array irradiance/power | shading-map accuracy, energy gain |
| 4.3 | Uncertainty-aware GMPP | Ensemble | GMPP predictive distribution | deterministic MPPT | intermittent-shading field/sim data | CRPS, coverage, energy |
| A | PV-fleet microgrid forecast | LSpa / hierarchical + Ensemble | regional-climate coupling state | persistence, LSTM, graph-NN | distributed PV fleet + weather (OPSD) | aggregate skill, ramp forecast |
| B | PV–demand co-forecast | Multivariate + selective A | seasonal regime index | independent PV & load models | co-located PV + load + weather | joint skill, grid-stability KPI |

### Cross-cutting evaluation standards (adopt group-wide)
- Report multi-step forecasts in **valid-prediction-time / horizon** terms, not just aggregate MSE.
- For any predictive/UQ topic, report **calibrated uncertainty (CRPS + interval coverage)** — the ensemble UQ (`predict_with_uncertainty`, `model_agreement`) is built for this.
- Always compare against the **domain engineering baseline** in the table, not only an LSTM, so "HCNN outperforms" is defensible.
- Lead theses: **2.1** (residual = the science) and **1.2 / 3.1** (physically-real but unmeasured hidden states) are the hardest to reframe as "basic ML" and the most distinctively HCNN.
