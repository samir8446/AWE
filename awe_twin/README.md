# AWE Digital Twin — platform v0.3 (DRT, degradation analysis, anomaly detection)

Streamlit app + tested Python engine for alkaline water electrolysers: diagnostics, degradation forecasts with
held-out uncertainty, a self-updating twin, and operation/maintenance decisions.

## Run
```
pip install -r requirements.txt
streamlit run app.py
python -m pytest -q tests          # 43 tests: engine, no-leakage, DRT, degradation, anomalies, figures, headless app
python study.py --synthetic-ss --cells 3 --out results/     # batch study: all tables + 4 paper figures
python study.py --data my_folder --area 25 --jop 0.5 --T 60   # same on your files
```
Keep `app.py` and `engine.py` from the same release side by side (exact version guard: engine and modules 0.3.0).

## Files
- `engine.py` — all computation (no Streamlit): physics, data loading, windows, health indicators, degradation
  model v1, forecasting models, assessment, live twin, operations. Each equation cites its article.
- `app.py` — 9 views: Home, Diagnostics, Live twin, Models & forecasting, Operations & control, DRT & mechanisms,
  Degradation analysis, Anomalies, Study results (incl. paper figures). Data sources: Ni fleet synthetic,
  stainless-steel ageing synthetic, or your upload.
- `tests/` — `test_engine.py`, `test_app.py`, `harness/` (headless Streamlit stand-in, CSS check).
- `sample_data/` — example files showing the upload formats (synthetic values, format only).

## New in v0.3
New modules (no Streamlit inside, each tested on its own):
- `drt.py` — DRT by non-negative Tikhonov regression on a Gaussian basis (DRTtools method), lambda by mGCV / GCV /
  L-curve, bootstrap confidence bands; lin-KK Kramers–Kronig test with an adaptive validity gate; peak finding inside
  the measured tau window; peak tracking over time (persistence clustering, labels aligned across currents);
  peak-to-mechanism assignment from the current dependence of each peak (slope of log R vs log j); DRT-guided ZARC
  equivalent circuit.
- `degradation.py` — reset detection with cause (event log / temperature / electrochemical); reversible vs
  irreversible split (isotonic baseline, Theil–Sen rate, bootstrap CI); sawtooth cycle model and next-reset
  back-test; piecewise-linear segmentation into stages (activation / steady / accelerated / terminal); **AC–DC
  closure** U(j_op) = U_LSV(j_start) + ∫[R_inf + Σ R_k(j)] dj using EIS at several currents; **DRT-informed
  Kalman twin** (one state per mechanism + instrument offset + closure residual), with ablation without DRT;
  replicate statistics (mean, SD, t-based 95 % CI).
- `anomaly.py` — three layers: (1) Kalman-filter residuals with NIS, CUSUM, EWMA; (2) PCA T²/Q and isolation forest
  trained on a reference period; (3) event-aware cause classification (hardware / operating / sensor /
  electrochemical). Evaluation: detection rate and delay per anomaly type, false alarms per 100 h, lead time before
  failure, DRT precursor check.
- `synthetic_ss.py` — stainless-steel ageing scenario with **known truth**: 4 mechanisms, EIS at 5 currents,
  spalling resets, KOH refills, power-supply changes, measurement pauses, temperature excursion, sensor spikes,
  8 % drifting (KK-invalid) spectra, terminal failure. Used only to check that the methods recover what was put in.
- `figures.py` + `study.py` — 4 paper figures (300 dpi PNG + the plotted data as CSV + a README stating SYNTHETIC
  or REAL): Fig 1 DRT map + tracking + assignment, Fig 2 closure + attribution + twin, Fig 3 degradation stages and
  cycles, Fig 4 anomalies and early warning.
- `views_advanced.py` — app views "DRT & mechanisms", "Degradation analysis", "Anomalies", and the paper-figure
  download under "Study results".

## Results on synthetic data (3 replicate cells, seed 0) — method check only
| Check | Result |
|---|---|
| DRT peak resistances at 0.5 A/cm² | within a few % of truth |
| KK gate | valid spectra ≤ 0.35 %, drifting ≥ 0.7 %; catches ≥ 80 % of bad spectra, ≤ 5 % wrong rejections |
| Peak-to-mechanism assignment | correct for all 3 cells |
| AC–DC closure RMSE (DRT, multi-current) | 12.6 / 6.9 / 17.3 mV (mean 12.3 ± 5.2 mV) |
| Same, single-current j·ΔR approximation | 10.2 / 10.1 / 20.0 mV |
| Twin mechanism error / SD | 3–9 mV, SD ≈ 3 mV with DRT vs ≈ 37 mV without DRT |
| Irreversible rate | 74.9 ± 5.9 µV/h (95 % CI 60–89, n = 3) |
| Stages | all 4 found; terminal onset 852 ± 77 h |
| Anomaly detection rate | 1.0 for all types except sensor spikes in SS-C (0.6) |
| False alarms | 0.11 / 0.20 / 0.40 per 100 h |
| Early-warning lead time | 196 / 176 / 298 h before failure |

## Honest caveats (v0.3)
- **Everything above is synthetic.** It shows the code recovers known parameters; it says nothing yet about real
  cells. The synthetic truth was written by us, so agreement is partly by construction.
- The single-current approximation does as well as the multi-current closure here, because the synthetic
  degradation is mostly film-type (linear in j). Real cells may differ — that is the experiment to run.
- After the last EIS, mechanism attribution is extrapolated (shaded in the twin plot). The terminal phase in the
  synthetic runs has no EIS, so attribution there is not verified.
- SS-B's terminal stage is detected early (765 h vs true ≈ 984 h): the accelerated stage and terminal stage are hard
  to separate with voltage alone.
- The early-warning alarm starts 155–177 h before the labelled terminal onset, during the accelerating stage. It
  is a warning of acceleration, not proof that failure can be predicted ~200 h ahead on real cells.
- n = 3 replicates gives wide CIs (e.g. closure RMSE CI includes 0–25 mV).

## What is needed from experiments for a publishable study
Replicate cells (≥ 3); EIS at the operating current and at several currents, down to ≤ 0.01 Hz (0.05 Hz → 0.01 Hz
improved closure from 22 to 13 mV in one test); a reference electrode to separate anode and cathode; ICP-OES of the
electrolyte (Fe/Cr/Ni dissolution); post-mortem SEM/XPS/Raman to confirm the mechanism assigned to each DRT peak.

## Inputs the user chooses (sidebar)
1. **Data**: synthetic demo (known parameters) or upload. Data types: operating time series, polarisation (U-i)
   curves, EIS spectra (incl. headerless DRTtools 3-column format), events, DRT results. Formats: csv/txt (any
   delimiter), xlsx, parquet. Columns are auto-detected and can be re-mapped by hand.
2. **Cell / stack design**: number of cells, active area, electrode type (Ni literature kinetics / stainless steel
   fitted / custom), electrode thickness and conductivity, electrode–separator gaps, separator model (empirical /
   area resistance / structural), separator thickness, porosity, tortuosity, H2 permeability, circulation
   (separated/mixed), supersaturation factor.
3. **Operating conditions**: T, p, j, KOH wt%, electrolyte flow, anode–cathode pressure difference, H2 solubility,
   viscosity, HTO shutdown limit.
4. **Kinetics & Faraday efficiency**: literature Butler–Volmer (Ni), Tafel fitted from your U-i curves, or custom;
   alpha fit choice, j0, Ea, T_ref/P_ref (assumption, labelled), bubble effects, Faraday coefficients.
5. **Analysis**: window length, "on" threshold, stability criterion, recovery-jump threshold, end-of-life threshold.

## What is implemented
| Part | Status |
|---|---|
| Physics (Nernst, Butler–Volmer/Tafel + bubbles, ohmic incl. bubbles, separator, KOH conductivity, V_tn, Faraday, production, crossover Fick/Darcy/mixing, HTO, min load, thermal step) | done, cited |
| Literature check vs Ham25 (E_rev ≤ 2.4 mV, η_act ≤ 6.6 mV, θ ≤ 0.008) | done (unit test) |
| Operating-window table, normalised voltage, recovery flags (causal), start-up counting | done |
| Health indicators + ranking; U-i fit (Tafel + R); EIS HF intercept; quick DRT | done |
| Degradation model v1 (per-unit drift, pooled Arrhenius + start-up law, identifiability notes) | done |
| Level 1 (persistence, drift, fleet curve), Level 2 (Bayesian ridge, RF, SVM, GP), Level 3 MLP | done |
| XGBoost / LightGBM | used if installed |
| GRU, Transformer, PINN + ablation | **next steps** (need PyTorch) |
| Held-out units or held-out periods, anchored forecasts, held-out bands with coverage, 95 % CI, Wilcoxon vs best baseline, per-group tables, EOL error | done |
| Bayesian-optimisation tuning on held-out data, self-optimise switch, ensemble 1/RMSE², recommended model, influence (leave-one-unit-out), no-leakage test | done |
| Live twin: Kalman filter (ΔR, Δη) with EIS, particle filter EOL with fleet prior, update-frequency study, measurement ablation | done |
| Operations: T×p map with HTO feasibility, load-following simulation + grid optimisation, maintenance timing from bands | done |

## Verified on what
Everything above is verified on **synthetic data and on Ham25's published values only**. No real electrolyser
data have been run yet. The synthetic degradation parameters are invented test values (declared in
`SYNTH_LAW`), not literature values.

## Assumptions (shown in the app)
T_ref/P_ref = 298.15 K/1 bar for eq 5–6 and 13 (not stated in the articles; reproduces Ham25); anode alpha from Jang;
V_tn eq A.2/A.6 with T in °C (otherwise 1.54 V at 25 °C, contradicting the article's 1.48 V); H2 solubility basis
per bar; η_F = 1 unless coefficients given; Zirfon permeability constant in T; membrane resistance in Ham25's units.
