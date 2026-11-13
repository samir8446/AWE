"""Tests for DRT, degradation analysis, anomaly detection and figures on the synthetic stainless-steel scenario
(known truth). Run: python -m pytest -q tests/test_advanced.py"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import anomaly as AN  # noqa: E402
import degradation as G  # noqa: E402
import drt as D  # noqa: E402
import engine as E  # noqa: E402
import synthetic_ss as SS  # noqa: E402


@pytest.fixture(scope="module")
def scen():
    s = SS.synthetic_ss_dataset(n_cells=1)
    data, d = s["data"], s["design"]
    kin = E.kinetics_from_fit(E.fit_polcurve(data["polcurve"], d, E.Kinetics()))
    W = E.build_windows(data, d, kin, s["ref"])
    A = D.analyse_all(data["eis"], d.area_cm2)
    tr = D.track_peaks(A["peaks"], spectra=A["spectra"])
    return dict(s=s, W=W, A=A, tr=tr, ser=G.unit_series(W, "SS-A", 2.0, data["events"]))


def test_versions():
    assert D.__version__ == G.__version__ == AN.__version__ == SS.__version__ == "0.3.0" == E.ENGINE_VERSION


def test_drt_recovers_known_peaks():
    f = np.logspace(4, -2, 61)
    z = 0.2 + 0.1 / (1 + (1j * 2 * np.pi * f * 1e-3) ** 0.95) + 0.3 / (1 + (1j * 2 * np.pi * f * 0.1) ** 0.9)
    z = z * (1 + 0.003 * np.random.default_rng(0).standard_normal(len(f)))
    r = D.fit_drt(f, z.real, -z.imag)
    p = D.find_drt_peaks(r, 0.02, min_share=0.05)
    assert abs(r["R_inf"] - 0.2) < 0.01
    assert len(p) == 2
    assert abs(p["R"].iloc[0] - 0.1) < 0.015 and abs(p["R"].iloc[1] - 0.3) < 0.03
    assert (p["R_lo"] <= p["R"] + 1e-9).all() and (p["R_hi"] >= p["R"] - 1e-9).all()


def test_kk_gate_catches_drift(scen):
    m = scen["A"]["spectra"].merge(scen["s"]["eis_truth"][["unit", "time_h", "j_A_cm2", "valid"]],
                                   on=["unit", "time_h", "j_A_cm2"])
    caught = (~m["kk_valid"] & ~m["valid"]).sum() / max((~m["valid"]).sum(), 1)
    wrong = (~m["kk_valid"] & m["valid"]).sum() / max(m["valid"].sum(), 1)
    assert caught >= 0.8 and wrong <= 0.05


def test_peak_assignment_by_current(scen):
    cl = D.classify_by_current(scen["tr"], scen["A"]["spectra"])
    pm = D.auto_peak_map(cl)
    assert set(pm.values()) >= set(G.MECHANISMS)          # all four mechanisms found by behaviour
    p2 = cl[cl["peak"].map(pm) == "HER (cathode)"]["slope_logR_logj"].iloc[0]
    assert -1.2 < p2 < -0.8                                  # Tafel: R ~ 1/j


def test_closure_and_attribution(scen):
    cl = D.classify_by_current(scen["tr"], scen["A"]["spectra"])
    c = G.closure_test(scen["tr"], scen["A"]["spectra"], scen["s"]["data"]["polcurve"], scen["ser"], "SS-A", 0.5,
                       D.auto_peak_map(cl))
    assert c["ok"] and c["rmse_mV"] < 20                    # includes a deliberate +12 mV instrument offset
    tw = G.drt_informed_twin(scen["ser"][scen["ser"]["t_h"] >= c["table"]["time_h"].iloc[0]], c["table"],
                             scen["s"]["data"]["events"], "SS-A")
    tw0 = G.drt_informed_twin(scen["ser"][scen["ser"]["t_h"] >= c["table"]["time_h"].iloc[0]], c["table"],
                              scen["s"]["data"]["events"], "SS-A", use_drt=False)
    m = G.MECHANISMS[2]
    assert tw[f"{m}_sd"].median() * 5 < tw0[f"{m}_sd"].median()   # DRT makes the mechanisms identifiable


def test_resets_and_stages(scen):
    ser = scen["ser"]
    rs = G.detect_resets(ser["t_h"].to_numpy(), ser["dV_mV"].to_numpy(), events=scen["s"]["data"]["events"],
                         unit="SS-A", T=ser["T_C"].to_numpy())
    ec = rs[rs["cause"] == "electrochemical reset"]["time_h"].to_numpy()
    true = scen["s"]["resets"].query("unit == 'SS-A'")["time_h"].to_numpy()
    hits = sum(np.any(np.abs(ec - t) < 10) for t in true)
    assert hits >= len(true) - 2
    assert (rs["cause"] == "electrolyte or service recovery").sum() >= 1     # KOH refills recognised
    ri = G.reversible_irreversible(ser["t_h"].to_numpy(), ser["dV_mV"].to_numpy(), rs)
    assert ri["ok"] and ri["irreversible_rate_mV_h"] > 0
    seg = G.label_stages(G.segment_piecewise_linear(ser["t_h"].to_numpy(), ser["dV_mV"].to_numpy()), resets=rs)
    assert {"1 activation", "2 passivation", "3 cyclic spalling", "4 terminal"} <= set(seg["stage"])


def test_anomaly_detection_quality(scen):
    s = scen["s"]
    r = AN.run_unit(s["data"]["timeseries"], scen["W"], s["data"]["events"], "SS-A")
    tend = float(s["data"]["timeseries"]["time_h"].max())
    ev = AN.evaluate(r["episodes"], s["anomalies"], "SS-A", tend)
    summ = ev["summary"].set_index("kind")
    assert summ.loc["spalling reset", "rate"] >= 0.8
    assert summ.loc["sensor spike", "rate"] >= 0.6
    assert ev["false_per_100h"] < 2.0
    onset = s["truth"].query("unit == 'SS-A' and phase == '4 terminal'")["time_h"].min()
    lt = AN.lead_time(r["episodes"], tend, onset)
    assert np.isfinite(lt["lead_h"]) and lt["lead_h"] > 10


def test_anomaly_detection_is_causal(scen):
    """Alarms before time t must not change when the data after t are altered."""
    s, W = scen["s"], scen["W"]
    cut = 500.0
    W2 = W.copy()
    m = (W2["unit"] == "SS-A") & (W2["t_end_h"] > cut)
    W2.loc[m, "U_cell_V"] = W2.loc[m, "U_cell_V"] + 0.2
    a = AN.residual_layer(AN.window_features(s["data"]["timeseries"], W, "SS-A"))
    b = AN.residual_layer(AN.window_features(s["data"]["timeseries"], W2, "SS-A"))
    k = a["t_h"] <= cut
    assert np.allclose(a.loc[k, "z"], b.loc[k, "z"])


def test_figures(scen, tmp_path):
    import figures as FG
    cl = D.classify_by_current(scen["tr"], scen["A"]["spectra"])
    pm = D.auto_peak_map(cl)
    out = FG.fig_drt(scen["A"], scen["tr"], cl, pm, "SS-A", 0.5)
    assert out["fig1_drt_map_tracking.png"][:4] == b"\x89PNG"


def test_study_cli_on_upload_folder(tmp_path):
    """Batch study runs on the sample upload files (format check only; values are not meaningful)."""
    import study
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")
    data, design = study.load_folder(here, 25.0)
    assert {"timeseries", "eis", "events"} <= set(data)
    res = study.run(data, design, E.OperatingConditions(T_C=30, j_A_cm2=0.5, P_bar=1.0), str(tmp_path))
    assert res["units"] == ["cell-1"]
    assert (tmp_path / "summary.json").exists() and (tmp_path / "README.txt").read_text().startswith("Data: REAL")
