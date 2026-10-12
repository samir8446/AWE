# AWE Digital Twin — platform v0.2 (general form)

Streamlit app + tested Python engine for alkaline water electrolysers: diagnostics, degradation forecasts with
held-out uncertainty, a self-updating twin, and operation/maintenance decisions.

## Run
```
pip install -r requirements.txt
streamlit run app.py
python -m pytest -q tests          # 29 tests: engine, no-leakage, headless app (every view, buttons, 2nd render)
```
Keep `app.py` and `engine.py` from the same release side by side (exact version guard: engine 0.2.0).

## Files
- `engine.py` — all computation (no Streamlit): physics, data loading, windows, health indicators, degradation
  model v1, forecasting models, assessment, live twin, operations. Each equation cites its article.
- `app.py` — 6 views: Home, Diagnostics, Live twin, Models & forecasting, Operations & control, Study results.
- `tests/` — `test_engine.py`, `test_app.py`, `harness/` (headless Streamlit stand-in, CSS check).
- `sample_data/` — example files showing the upload formats (synthetic values, format only).

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
