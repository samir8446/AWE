# awe_synth — synthetic AWE test-bench data with known ground truth (v1.0.0)

```
python -m awe_synth.generate --seed 0 --out datasets          # 18 cells x 3000 h, ~25 s
python -m awe_synth.report datasets/awe_synth_v1.0.0_seed0   # plausibility checks + figures (report/)
python -m pytest -q tests/test_synth.py                       # 9 tests
```

## Design rules
- `data/` has the same schema a real bench delivers (column names follow the platform's loader aliases, so real files
  can replace it with no code change). `truth/` is read only by validation code (`report.py`, tests).
- Every parameter lives in `config.py` with unit and source; `ASSUMPTION` marks values no source in hand provides.
- Fixed seeds per unit (`SeedSequence`); `manifest.json` stores the version, config hash and SHA-256 of every file.
  Same seed, same files (tested).
- Physics written independently of the platform engine (same papers): they agree within 4.8 mV, so the platform's
  physics is close to correctly specified here, which is OPTIMISTIC compared with real cells.

## Modules
| file | role |
|---|---|
| config.py | sources, parameter registry, condition groups, injected faults |
| physics.py | BoL physics: Nernst, Butler-Volmer + bubbles, KOH conductivity, separator, Faraday efficiency, crossover, HTO, V_tn |
| simulate.py | 1-min simulation per unit: schedule (constant / renewable with start-stops, diagnostics, trips, electrolyte changes), thermal + pressure + electrolyte loop (water consumption, make-up, carbonation), hidden degradation state, HTO interlock, mechanism decomposition |
| measure.py | hourly log with noise, drift, outliers, dropouts, logger outages; LSV, EIS (3 currents), CV (double layer + Ni redox/OER), titration; truth tables |
| generate.py | writes data/, truth/, README, data dictionary, manifest |
| report.py | plausibility checks with literature criteria + 7 figures + self-contained HTML |
