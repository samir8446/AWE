"""Generate a full synthetic AWE dataset.

    python -m awe_synth.generate --seed 0 --out datasets

writes datasets/awe_synth_v<version>_seed<seed>/
    data/   what the platform may read (same schema as a real bench)
    truth/  hidden ground truth (validation code only)
    manifest.json (version, seed, config hash, SHA-256 of every file)
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from . import GENERATOR_VERSION, config
from .config import BENCH, DESIGN, FAULTS, GROUPS
from .measure import diagnostics, hourly_log, minute_sample
from .simulate import UnitSpec, simulate_unit

DICTIONARY = [
    # file, column, unit, meaning, sampling
    ("units.csv", "unit", "-", "Unit identifier (group-index)", "one row per unit"),
    ("units.csv", "group", "-", "Condition group", ""),
    ("units.csv", "operating_mode", "-", "constant | dynamic (renewable-like, with start/stops)", ""),
    ("units.csv", "T_set_C", "C", "Operating temperature set-point", ""),
    ("units.csv", "P_set_bar", "bar(a)", "System pressure set-point (both sides)", ""),
    ("units.csv", "koh_nominal_wt", "wt%", "KOH concentration of fresh electrolyte", ""),
    ("units.csv", "j_nominal_A_cm2", "A/cm2", "Constant load, or rated load for dynamic units", ""),
    ("units.csv", "area_cm2", "cm2", "Active electrode area", ""),
    ("units.csv", "electrodes / separator / gap_mm", "-", "Cell build", ""),
    ("units.csv", "diag_T_C / diag_P_bar / diag_every_h", "C / bar / h", "Diagnostic reference conditions and period", ""),
    ("timeseries_1h.csv", "unit", "-", "Unit identifier", "1 h"),
    ("timeseries_1h.csv", "time_h", "h", "Start of the hourly bin; values are means over [t, t+1 h) of 1-min data", "1 h"),
    ("timeseries_1h.csv", "state", "-", "run | off | mixed (start/stop inside the hour) | diagnostic", "1 h"),
    ("timeseries_1h.csv", "run_fraction", "-", "Fraction of minutes with current on", "1 h"),
    ("timeseries_1h.csv", "current_A", "A", "Cell current (mean incl. zero minutes)", "1 h"),
    ("timeseries_1h.csv", "j_A_cm2", "A/cm2", "current_A / area", "1 h"),
    ("timeseries_1h.csv", "voltage_V", "V", "Cell voltage (mean of all minutes; open-circuit decay when off)", "1 h"),
    ("timeseries_1h.csv", "T_in_C", "C", "Lye inlet temperature", "1 h"),
    ("timeseries_1h.csv", "T_out_C", "C", "Lye outlet temperature (= cell temperature, well mixed)", "1 h"),
    ("timeseries_1h.csv", "T_C", "C", "Copy of T_out_C (platform's standard name)", "1 h"),
    ("timeseries_1h.csv", "P_bar", "bar(a)", "System pressure", "1 h"),
    ("timeseries_1h.csv", "flow_L_min", "L/min", "Electrolyte flow through the cell", "1 h"),
    ("timeseries_1h.csv", "conductivity_S_cm", "S/cm", "Online electrolyte conductivity at loop temperature", "1 h"),
    ("timeseries_1h.csv", "h2_flow_Nm3_h", "Nm3/h", "H2 outlet flow (dry, normal conditions)", "1 h"),
    ("timeseries_1h.csv", "o2_flow_Nm3_h", "Nm3/h", "O2 outlet flow (dry)", "1 h"),
    ("timeseries_1h.csv", "HTO_pct", "% (mol)", "H2 in O2 at anode outlet, mean over minutes with current; bench "
     "interlock trips at 2 %", "1 h"),
    ("timeseries_1min_sample.csv", "(as above)", "", "Raw 1-min record for two units, one week (1000-1168 h in a 3000 h run)", "1 min"),
    ("events.csv", "unit / time_h / event / note", "- / h / - / -", "startup, shutdown, trip, measurement "
     "(diagnostic session), electrolyte_change (drain + fresh KOH), water_topup", "event"),
    ("lsv.csv", "unit, time_h, session", "-", "Diagnostic session", "every 250 h (first at 24 h)"),
    ("lsv.csv", "j_A_cm2 / current_A / voltage_V", "A/cm2 / A / V", "Galvanostatic steps (quasi-steady), voltage NOT "
     "iR-corrected", "18 points"),
    ("lsv.csv", "T_C / P_bar", "C / bar", "Diagnostic conditions (70 C, 1 bar)", ""),
    ("eis.csv", "unit, time_h, session, j_A_cm2", "-", "Galvanostatic EIS at 0.05, 0.2, 0.4 A/cm2", "per session"),
    ("eis.csv", "freq_Hz / z_re / z_im_neg", "Hz / ohm / ohm", "Cell impedance (not area-normalised), 100 kHz-0.1 Hz, "
     "10 points/decade", ""),
    ("cv.csv", "electrode", "-", "cathode | anode (three-electrode measurement with a reference electrode)", ""),
    ("cv.csv", "window", "-", "double_layer (non-faradaic, 4 scan rates) | Ni_redox_OER (anode, 10 mV/s)", ""),
    ("cv.csv", "scan_rate_V_s / cycle_point / E_V_vs_RHE / current_A", "V/s / - / V / A", "One cycle per scan rate", ""),
    ("titration.csv", "koh_free_wt / k2co3_wt", "wt%", "Electrolyte sample titration (free KOH, carbonate as K2CO3)",
     "per session"),
]


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def config_hash() -> str:
    return hashlib.sha256(inspect.getsource(config).encode()).hexdigest()[:16]


def unit_specs(seed: int, units_per_group: int = 3) -> List[UnitSpec]:
    out = []
    for gi, g in enumerate(GROUPS):
        for i in range(1, units_per_group + 1):
            out.append(UnitSpec(f"{g.name}-{i}", g, i, seed * 100000 + gi * 100 + i))
    return out


def generate(seed: int = 0, out: str = "datasets", hours: int = None, units_per_group: int = 3,
             progress=print) -> str:
    hours = int(hours or BENCH["hours"].value)
    root = os.path.join(out, f"awe_synth_v{GENERATOR_VERSION}_seed{seed}")
    dd, td = os.path.join(root, "data"), os.path.join(root, "truth")
    os.makedirs(dd, exist_ok=True)
    os.makedirs(td, exist_ok=True)
    logs, truths, mechs, events, labels, lsv, eis, cv, titr, eis_truth, units, uparams, samples = ([] for _ in range(13))
    for spec in unit_specs(seed, units_per_group):
        progress(f"simulating {spec.unit}")
        r = simulate_unit(spec, hours)
        lg, th, md, lb = hourly_log(r, spec.seed)
        dg, lb2, et = diagnostics(r, spec.seed)
        logs.append(lg)
        truths.append(th)
        mechs.append(md)
        eis_truth.append(et)
        labels += lb + lb2
        for k, v in (("lsv", lsv), ("eis", eis), ("cv", cv), ("titration", titr)):
            if len(dg[k]):
                v.append(dg[k])
        events += r["events"]
        g = spec.group
        units.append(dict(unit=spec.unit, group=g.name, operating_mode=g.mode, T_set_C=g.T_C, P_set_bar=g.P_bar,
                          koh_nominal_wt=g.koh_wt, j_nominal_A_cm2=g.j_A_cm2, area_cm2=DESIGN["area_cm2"].value,
                          electrodes=DESIGN["electrodes"].value, separator=DESIGN["separator"].value,
                          gap_mm=2 * 10 * DESIGN["gap_each_side_cm"].value, test_duration_h=hours,
                          diag_T_C=BENCH["diag_T_C"].value, diag_P_bar=BENCH["diag_P_bar"].value,
                          diag_every_h=BENCH["diag_every_h"].value))
        ex = r["extra"]
        uparams.append(dict(unit=spec.unit, seed=spec.seed, **{f"draw_{k}": v for k, v in r["draws"].items()},
                            D_star_h=ex["Dstar"], accel_onset_h=(ex["k_on"] / 60.0 if ex["k_on"] is not None else np.nan),
                            trip_h=(r["trip_min"] / 60.0 if r["trip_min"] is not None else np.nan),
                            n_stops=int(((r["mode"][1:] == 0) & (r["mode"][:-1] > 0)).sum()),
                            final_t_eq_h=float(ex["t_eq"][-1]), final_stop_dose=float(ex["S"][-1])))
        if spec.unit in (f"{GROUPS[2].name}-1", f"{GROUPS[5].name}-1"):
            h0 = min(1000, hours // 3)
            samples.append(minute_sample(r, spec.seed, h0, min(h0 + 168, hours)))
    cat = lambda L: pd.concat(L, ignore_index=True)
    w = lambda df, d, name: df.to_csv(os.path.join(d, name), index=False, float_format="%.6g")
    w(pd.DataFrame(units), dd, "units.csv")
    w(cat(logs), dd, "timeseries_1h.csv")
    w(cat(samples), dd, "timeseries_1min_sample.csv")
    ev = pd.DataFrame(events, columns=["unit", "time_h", "event", "note"]).sort_values(["unit", "time_h"], kind="stable")
    w(ev, dd, "events.csv")
    w(cat(lsv), dd, "lsv.csv")
    w(cat(eis), dd, "eis.csv")
    w(cat(cv), dd, "cv.csv")
    w(cat(titr), dd, "titration.csv")
    w(pd.DataFrame(DICTIONARY, columns=["file", "column", "unit", "meaning", "sampling"]), dd, "DATA_DICTIONARY.csv")
    # truth
    w(cat(truths), td, "hourly_state.csv")
    w(cat(mechs), td, "mechanisms_at_reference.csv")
    w(cat(eis_truth), td, "diagnostics_truth.csv")
    w(pd.DataFrame(uparams), td, "unit_parameters.csv")
    fl = pd.DataFrame(FAULTS)
    w(fl, td, "faults.csv")
    w(pd.DataFrame(labels, columns=["unit", "channel", "kind", "start_h", "end_h", "detail"]), td, "measurement_labels.csv")
    reg = [dict(block=b, name=k, value=json.dumps(p.value) if not isinstance(p.value, str) else p.value, unit=p.unit,
                source=p.source, note=p.note)
           for b, D in (("design", config.DESIGN), ("physics", config.PHYS), ("bench", config.BENCH),
                        ("degradation_truth", config.DEG), ("measurement", config.MEAS)) for k, p in D.items()]
    w(pd.DataFrame(reg), td, "parameter_registry.csv")
    write_readmes(root, hours, units_per_group)
    man = dict(generator="awe_synth", version=GENERATOR_VERSION, seed=seed, hours=hours,
               units_per_group=units_per_group, config_hash=config_hash(),
               files={os.path.relpath(os.path.join(d, f), root): _sha(os.path.join(d, f))
                      for d in (dd, td) for f in sorted(os.listdir(d))})
    with open(os.path.join(root, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    progress(f"written {root}")
    return root


def write_readmes(root, hours, upg):
    from .config import SOURCES
    groups = "\n".join(f"| {g.name} | {g.mode} | {g.T_C:g} | {g.P_bar:g} | {g.koh_wt:g} | {g.j_A_cm2:g} | {upg} | "
                       f"{', '.join(f'{x:g}' for x in g.refresh_h)} | {g.why} |" for g in GROUPS)

    def reg(D):
        return "\n".join(f"| {k} | {p.value} | {p.unit} | {p.source} | {p.note} |" for k, p in D.items())

    src = "\n".join(f"- **[{k}]** {v}" for k, v in SOURCES.items())
    txt = f"""# Synthetic AWE bench dataset (awe_synth v{GENERATOR_VERSION})

**SYNTHETIC DATA.** Generated from documented physics with a hidden, known degradation truth. It is for checking
methods, not for drawing conclusions about real electrolysers.

- `data/` = what a real test bench would deliver. The platform reads only this folder.
- `truth/` = hidden ground truth. Only validation code may read it.
- `manifest.json` = generator version, seed, config hash and SHA-256 of every file (same seed, same files).

## Units and conditions ({len(GROUPS) * upg} single cells, {hours} h each)
| group | mode | T (C) | P (bar a) | KOH (wt%) | j (A/cm2) | units | electrolyte change (h) | why |
|---|---|---|---|---|---|---|---|---|
{groups}

Diagnostics every {BENCH['diag_every_h'].value:g} h (first at {BENCH['diag_first_h'].value:g} h) at
{BENCH['diag_T_C'].value:g} C and {BENCH['diag_P_bar'].value:g} bar: LSV (18 steps to 1 A/cm2), EIS at 0.05/0.2/0.4
A/cm2, CV (double layer both electrodes, Ni redox + OER on the anode), electrolyte titration.

## Model (cell voltage)
U = E_rev(T, P, KOH) + eta_act,c + eta_act,a + j (R_gap + R_sep + R_contact) + eta_conc

| term | equation | source |
|---|---|---|
| E_rev | E0(T) + RT/2F ln[(P - p_w)^1.5 / a_w] | Hu22 eq 11, Ham25 eq 2-3, Hu22 eq B.2/B.4 |
| KOH molarity | from wt% and T | Ham25 eq 12 |
| eta_act | RT/(n alpha F) asinh(j / 2 j0 (1 - theta)) | Hu22 eq 13/18, Ham25 eq 14 |
| j0(T, P), Ni | j0,ref (P/Pref)^0.1 exp[-Ea/RT (1 - T/Tref)] | Ham25 eq 5-6 |
| alpha_c, alpha_a | 0.1175 + 0.00095 T ; 0.07835 + 0.001 T | Hu22 Table B3 |
| theta (bubbles) | 0.023 j^0.3 (T/Tref Pref/P)^(2/3) | Hu22 eq 17, Ham25 eq 13 |
| sigma_KOH(m, T) | polynomial, 0-12 M, 273-373 K | Hu22 Table 4, Ham25 eq 11 |
| R_gap | 2 d / sigma / (1 - 2 theta/3)^1.5 | Ham25 eq 15 |
| R_sep | delta tau / (eps sigma), tau = (2 - eps)^2/eps | Hu22 eq 24-25 (gives 0.25-0.32 ohm cm2; Zirfon 0.27 in Qu26) |
| eta_conc | RT/2F ln(1/(1 - j/j_lim)) | ASSUMPTION (Hu22 3.2.5: negligible) |
| Faraday efficiency | j^2/(f1(T) + j^2) | Hu22 eq 7, Ulleberg03 |
| H2/O2 production | eta_F I / 2F, eta_F I / 4F | Hu22 eq 5, 9 |
| crossover | perm p_gas / delta x supersaturation x (1 - theta) | Hu22 eq 40-41, Ham25 eq 21-24, Qu26 Table 2, Barros23 |
| HTO | N_H2,cross / (N_H2,cross + N_O2,out) | Hu22 eq 46-47; shutdown 2 % (Hu22 3.3) |
| V_tn, heat | V_tn(T, P) ; Q = U I - eta_F I V_tn | Hu22 App. A, eq 53-56 |

## Degradation truth (hidden)
Seven mechanisms, each a time-varying parameter (see `truth/parameter_registry.csv`):
cathode intrinsic activity (j0 down, Tafel slope up), reversible cathode hydride loss (recovers at long stops and
electrolyte change), anode intrinsic activity that **accelerates** after a dose threshold (onset per unit in
`truth/unit_parameters.csv: accel_onset_h`), ECSA loss (Cdl down), separator fouling, contact degradation,
electrolyte carbonation and concentration drift, plus reverse-current damage at each stop (intermittent operation).
Rates are **accelerated about 10x** relative to industrial AWE so that 3000 h shows clear ageing.

## Injected faults
Listed in `truth/faults.csv` only. Units without an entry there are fault-free (false-alarm testing).

## Realism
Hourly means of 1-min data; measurement noise; sensor drift (T, conductivity probe, HTO analyser); outliers in
T_in, flow and HTO; single NaN dropouts; logger outages (rows absent); 5 % missing diagnostic sessions;
5 % non-stationary (KK-invalid) EIS spectra; start-up transients (thermal warm-up); water make-up and electrolyte
changes that partly RECOVER performance. All labelled in `truth/measurement_labels.csv`.

## Parameters
### Design
| name | value | unit | source | note |
|---|---|---|---|---|
{reg(config.DESIGN)}

### Physics
| name | value | unit | source | note |
|---|---|---|---|---|
{reg(config.PHYS)}

### Bench
| name | value | unit | source | note |
|---|---|---|---|---|
{reg(config.BENCH)}

### Degradation truth
| name | value | unit | source | note |
|---|---|---|---|---|
{reg(config.DEG)}

### Measurement
| name | value | unit | source | note |
|---|---|---|---|---|
{reg(config.MEAS)}

## Sources
{src}
"""
    with open(os.path.join(root, "README.md"), "w") as fh:
        fh.write(txt)
    with open(os.path.join(root, "truth", "README_TRUTH.md"), "w") as fh:
        fh.write("""# Ground truth - DO NOT read from the platform

| file | content |
|---|---|
| hourly_state.csv | true signals and every hidden parameter per hour (j0 factors, Tafel slopes, roughness, Cdl, resistances, carbonate, crossover factor, doses, onset flag, heat and mole flows) |
| mechanisms_at_reference.csv | voltage change at 80 C, 1 bar, 0.4 A/cm2 caused by each mechanism alone (mV), total and interaction residual |
| diagnostics_truth.csv | true EIS elements per session and current (R_ohm, R_ct cathode/anode, R_gas, Cdl, Tafel, j0, KK validity) |
| unit_parameters.csv | per-unit random draws, acceleration threshold and onset time, trip time, stop count |
| faults.csv | injected faults: unit, type, onset, magnitude |
| measurement_labels.csv | sensor drifts, outliers, logger outages, missing sessions, non-stationary spectra |
| parameter_registry.csv | every parameter with unit and source |
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="datasets")
    ap.add_argument("--hours", type=int, default=None)
    ap.add_argument("--units-per-group", type=int, default=3)
    a = ap.parse_args()
    generate(a.seed, a.out, a.hours, a.units_per_group)


if __name__ == "__main__":
    main()
