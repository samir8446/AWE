"""Batch study (no app): run DRT, degradation analysis, closure, DRT-informed twin and anomaly detection on a dataset
and write every table and figure to an output folder.

  python study.py --synthetic-ss --cells 3 --out results/            # synthetic stainless-steel scenario
  python study.py --data my_folder/ --area 25 --jop 0.5 --out results/

my_folder/ may contain timeseries.csv, eis.csv, polcurve.csv, events.csv in the standard column names
(see README: unit, time_h, j_A_cm2, voltage_V, T_C, ...; EIS: unit, time_h, j_A_cm2, freq_Hz, z_re, z_im_neg).
"""
from __future__ import annotations

import argparse
import json
import os
import re

import pandas as pd

import anomaly as AN
import degradation as G
import drt as D
import engine as E
import figures as FG
import synthetic_ss as SS


KEYWORDS = {"timeseries": ("timeseries", "time_series", "operation", "cp_"), "eis": ("eis", "impedance"),
            "polcurve": ("polcurve", "polarisation", "polarization", "lsv", "ui_curve"), "events": ("event",)}


def load_folder(path: str, area: float, default_unit: str = "cell-1"):
    """Read every csv/txt/xlsx/parquet in `path`; the data type is taken from the file name (e.g. *timeseries*,
    *eis*, *polcurve*/*lsv*, *events*). Several files of one type are stacked (file name = unit if no unit column)."""
    design = E.CellDesign(area_cm2=area, electrode="stainless")
    parts: dict = {}
    for fn in sorted(os.listdir(path)):
        low = fn.lower()
        if not low.endswith((".csv", ".txt", ".xlsx", ".parquet")):
            continue
        kind = next((k for k, keys in KEYWORDS.items() if any(x in low for x in keys)), None)
        if kind is None:
            print(f"skipped {fn}: name does not say which data type it is")
            continue
        with open(os.path.join(path, fn), "rb") as fh:
            raw = E.read_table(fh, fn)
        m = re.search(r"[_-]t?(\d+(?:\.\d+)?)h\b", os.path.splitext(low)[0] + " ")
        t_file = float(m.group(1)) if m else 0.0          # time of an EIS / U-i file from its name, e.g. eis_t250h
        df, notes = E.apply_mapping(raw, E.guess_mapping(list(raw.columns), kind), kind, design, default_unit,
                                    t_file if kind in ("eis", "polcurve") else None)
        for n in notes:
            print(f"{fn}: {n}")
        parts.setdefault(kind, []).append(df)
    data = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    if "timeseries" not in data:
        raise SystemExit("No operating time series found: name the file e.g. 'cell1_timeseries.csv'.")
    return data, design


def run(data, design, ref, out, labels=None, truth=None):
    os.makedirs(out, exist_ok=True)
    kin = E.kinetics_from_fit(E.fit_polcurve(data["polcurve"], design, E.Kinetics())) if "polcurve" in data else E.Kinetics()
    W = E.build_windows(data, design, kin, ref)
    W.to_csv(os.path.join(out, "windows.csv"), index=False)
    summary = {"units": sorted(W["unit"].unique())}
    A = D.analyse_all(data["eis"], design.area_cm2) if "eis" in data else None
    tr = D.track_peaks(A["peaks"], spectra=A["spectra"]) if A is not None else None
    cl = D.classify_by_current(tr, A["spectra"]) if A is not None else None
    if A is not None:
        A["spectra"].to_csv(os.path.join(out, "drt_spectra.csv"), index=False)
        tr.to_csv(os.path.join(out, "drt_tracked_peaks.csv"), index=False)
        cl.to_csv(os.path.join(out, "drt_peak_assignment.csv"), index=False)
    rep, leads, ev_rows = [], [], []
    fig2 = {}
    for u in summary["units"]:
        ser = G.unit_series(W, u, 2.0, data.get("events"))
        rs = G.detect_resets(ser["t_h"].to_numpy(), ser["dV_mV"].to_numpy(), events=data.get("events"), unit=u,
                             T=ser["T_C"].to_numpy())
        ri = G.reversible_irreversible(ser["t_h"].to_numpy(), ser["dV_mV"].to_numpy(), rs)
        seg = G.label_stages(G.segment_piecewise_linear(ser["t_h"].to_numpy(), ser["dV_mV"].to_numpy()), resets=rs)
        rs.to_csv(os.path.join(out, f"{u}_resets.csv"), index=False)
        seg.to_csv(os.path.join(out, f"{u}_stages.csv"), index=False)
        ec = rs[rs["cause"] == "electrochemical reset"] if len(rs) else rs
        rep.append(dict(unit=u, irreversible_uV_h=1000 * ri.get("irreversible_rate_mV_h", float("nan")),
                        n_resets=len(ec), terminal_onset_h=seg.loc[seg["stage"] == "4 terminal", "start_h"].min()))
        if A is not None and tr is not None and len(cl):
            pm = D.auto_peak_map(cl, u)
            c = G.closure_test(tr, A["spectra"], data.get("polcurve"), ser, u, ref.j_A_cm2, pm)
            if c.get("ok"):
                c["table"].to_csv(os.path.join(out, f"{u}_closure.csv"), index=False)
                tw = G.drt_informed_twin(ser[ser["t_h"] >= c["table"]["time_h"].iloc[0]], c["table"], data.get("events"), u)
                tw.to_csv(os.path.join(out, f"{u}_drt_twin.csv"), index=False)
                rep[-1].update(closure_rmse_mV=c["rmse_mV"], closure_single_rmse_mV=c["rmse_single_mV"])
                if u == summary["units"][0]:
                    tw0 = G.drt_informed_twin(ser[ser["t_h"] >= c["table"]["time_h"].iloc[0]], c["table"],
                                              data.get("events"), u, use_drt=False)
                    fig2 = FG.fig_closure(c, tw, tw0)
        r = AN.run_unit(data["timeseries"], W, data.get("events"), u)
        r["episodes"].to_csv(os.path.join(out, f"{u}_anomaly_episodes.csv"), index=False)
        tend = float(data["timeseries"].loc[data["timeseries"]["unit"] == u, "time_h"].max())
        onset = truth.loc[(truth["unit"] == u) & (truth["phase"] == "4 terminal"), "time_h"].min() if truth is not None else None
        leads.append(dict(unit=u, **AN.lead_time(r["episodes"], tend, onset)))
        if labels is not None:
            ev = AN.evaluate(r["episodes"], labels, u, tend)
            ev_rows.append(ev["summary"].assign(unit=u, false_per_100h=ev["false_per_100h"]))
        if u == summary["units"][0]:
            P = dict(series=ser, resets=rs, ri=ri, seg=seg, saw=G.sawtooth_model(ser["t_h"].to_numpy(), ser["dV_mV"].to_numpy(), ec))
            files = FG.fig_degradation(P, None)
            if A is not None and len(cl):
                pm = D.auto_peak_map(cl, u)
                files.update(FG.fig_drt(A, tr, cl, pm, u, ref.j_A_cm2))
            files.update(fig2)
            first_files = files
            first_r = r
    lt = pd.DataFrame(leads)
    files = dict(first_files)
    files.update(FG.fig_anomalies({summary["units"][0]: first_r}, summary["units"][0], lt))
    files["README.txt"] = (f"Data: {'SYNTHETIC (method check)' if truth is not None else 'REAL'}\n"
                           "Engine/modules 0.3.0. Each PNG has CSV files with the plotted data.\n").encode()
    for n, b in files.items():
        if not b:
            continue
        with open(os.path.join(out, n), "wb") as fh:
            fh.write(b)
    rep = pd.DataFrame(rep)
    rep.to_csv(os.path.join(out, "replicates.csv"), index=False)
    G.replicate_summary(rep, [c for c in rep.columns if c != "unit"]).to_csv(os.path.join(out, "replicate_summary.csv"), index=False)
    pd.DataFrame(leads).to_csv(os.path.join(out, "lead_times.csv"), index=False)
    if ev_rows:
        pd.concat(ev_rows).to_csv(os.path.join(out, "anomaly_evaluation.csv"), index=False)
    summary["replicates"] = rep.to_dict(orient="records")
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic-ss", action="store_true")
    ap.add_argument("--cells", type=int, default=3)
    ap.add_argument("--data")
    ap.add_argument("--area", type=float, default=25.0)
    ap.add_argument("--jop", type=float, default=0.5)
    ap.add_argument("--T", type=float, default=30.0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if a.synthetic_ss:
        s = SS.synthetic_ss_dataset(n_cells=a.cells)
        res = run(s["data"], s["design"], s["ref"], a.out, s["anomalies"], s["truth"])
    else:
        data, design = load_folder(a.data, a.area)
        res = run(data, design, E.OperatingConditions(T_C=a.T, j_A_cm2=a.jop, P_bar=1.0), a.out)
    print(json.dumps(res["replicates"], indent=1, default=str))
