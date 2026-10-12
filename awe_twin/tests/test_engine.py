"""Engine unit tests (no Streamlit). Run: python -m pytest -q tests"""
import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import engine as E  # noqa: E402


@pytest.fixture(scope="module")
def synth():
    s = E.synthetic_dataset(hours=1000)
    W = E.build_windows(s["data"], s["design"], s["kin"], s["ref"])
    return s, E.attach_indicators(W, s["data"], s["design"], s["kin"])


def test_version():
    assert E.ENGINE_VERSION == "0.2.0"


def test_reproduces_ham25():
    lc = E.literature_check()
    assert (lc.loc[lc["quantity"] == "E_rev", "diff"].abs() < 0.003).all()
    assert (lc.loc[lc["quantity"] == "eta_act", "diff"].abs() < 0.008).all()
    assert (lc.loc[lc["quantity"] == "theta", "diff"].abs() < 0.01).all()


def test_thermodynamics():
    assert abs(E.molarity_from_wt(30, 343.15) - 6.76) < 0.02           # Ham25 eq 12
    assert abs(E.p_sat_water_bar(303.15) - 0.0424) < 0.002               # steam tables
    assert abs(E.V_thermoneutral(298.15, 1.0, 1e-3) - 1.48) < 0.02       # Hu22: 1.48 V at 25 C, 1 bar
    assert 0.5 < E.sigma_koh(7.0, 303.15) < 0.8                          # Gilliam-type values (S/cm)


def test_voltage_trends():
    d, k = E.CellDesign(), E.Kinetics()
    U = lambda T, P: E.cell_voltage(0.4, E.OperatingConditions(T_C=T, P_bar=P), d, k)["U"][0]
    assert U(80, 3) < U(60, 3)                                          # temperature lowers U [Ham25]
    assert E.cell_voltage([0.1, 0.4], E.OperatingConditions(), d, k)["U"][1] > E.cell_voltage([0.1], E.OperatingConditions(), d, k)["U"][0]
    deg = E.DegradationState(dR_ohm_cm2=0.1)
    assert abs(E.cell_voltage(0.4, E.OperatingConditions(), d, k, deg)["U"][0] - E.cell_voltage(0.4, E.OperatingConditions(), d, k)["U"][0] - 0.04) < 1e-9


def test_hto_monotone_and_min_load():
    op, d, k = E.OperatingConditions(P_bar=5), E.CellDesign(), E.Kinetics()
    h = E.hto_percent([0.02, 0.1, 0.4], op, d, k)["HTO_pct"]
    assert h[0] > h[1] > h[2]                                            # HTO rises at low load [Hu22]
    assert E.hto_percent(0.1, E.OperatingConditions(P_bar=10), d, k)["HTO_pct"][0] > E.hto_percent(0.1, E.OperatingConditions(P_bar=1), d, k)["HTO_pct"][0]
    jm = E.min_load_for_hto(op, d, k, 2.0)
    assert abs(E.hto_percent(jm, op, d, k)["HTO_pct"][0] - 2.0) < 1e-3


def test_windows_and_recovery(synth):
    s, W = synth
    assert set(W["unit"]) == {"S1", "S2", "S3", "S4", "S5", "S6"}
    assert W.groupby("unit")["op_hours"].apply(lambda x: x.is_monotonic_increasing).all()
    assert W.loc[W["unit"] == "S5", "starts"].max() > 5
    assert "R_EIS_ohm_cm2" in W and W["R_EIS_ohm_cm2"].notna().any()


def test_degradation_v1_recovers_order_of_truth(synth):
    s, W = synth
    r = E.degradation_v1(W)
    hot = r["per_unit"][r["per_unit"]["T_C"] > 75]["drift_uV_h"]
    cold = r["per_unit"][r["per_unit"]["T_C"] < 65]["drift_uV_h"]
    assert hot.mean() > cold.mean()                                      # Arrhenius in the synthetic truth
    assert 0 < r["law"]["Ea_J_mol"] < 60000


def test_no_leakage(synth):
    s, W = synth
    for m in ["Linear drift", "Fleet curve", "Bayesian ridge", "Random forest"]:
        assert E.no_leakage_check(W, "S3", 0.5, m), m


def test_forecast_anchored(synth):
    s, W = synth
    u = E.split_units(W)["S1"]
    k = E.origin_index(u, 0.5)
    past = u.iloc[:k + 1]
    fut = u.iloc[k + 1:k + 2].drop(columns=["dV_mV"]).copy()
    fut["op_hours"] = past["op_hours"].iloc[-1]
    fut["starts"] = past["starts"].iloc[-1]
    fut[["j_A_cm2", "T_C", "P_bar"]] = past[["j_A_cm2", "T_C", "P_bar"]].tail(24).median().to_numpy()
    now = E.state_at(past)[0]
    for m in ["Persistence", "Linear drift", "Bayesian ridge"]:
        assert abs(E.forecast(m, {x: v for x, v in E.split_units(W).items() if x != "S1"}, past, fut)[0] - now) < 1e-6


def test_future_answer_rejected(synth):
    s, W = synth
    u = E.split_units(W)["S1"]
    with pytest.raises(AssertionError):
        E.forecast("Persistence", {}, u.iloc[:100], u.iloc[100:])


def test_periods_mode_single_unit(synth):
    s, W = synth
    one = W[W["unit"] == "S3"]
    r = E.run_task(one, "S3", 0.5, "Random forest", mode="periods")
    assert len(r) and r["pred"].notna().all()


def test_scoreboard_small_sample_wording(synth):
    s, W = synth
    res = pd.concat([E.run_task(W, u, 0.5, m) for u in ["S1", "S3"] for m in ["Persistence", "Linear drift"]])
    sb, note = E.scoreboard(E.score_tasks(res, 10))
    assert "too few" in " ".join(sb["vs best baseline"]) or "fewer than 6" in note


def test_kalman_tracks_truth(synth):
    s, W = synth
    d = W[W["unit"] == "S5"]
    kf = E.kalman_replay(d, s["design"], s["kin"], s["data"]["eis"], ref=s["ref"])
    tr = s["truth"][s["truth"]["unit"] == "S5"].iloc[-1]
    assert abs(kf["d_eta_V"].iloc[-1] - tr["d_eta_V"]) < 0.004


def test_upload_roundtrip():
    csv = "Time (h),Ewe/V,I (A),Temperature\n0,2.17,12.5,30\n0.25,2.171,12.5,30\n0.5,2.172,12.5,30.1\n"
    raw = E.read_table(io.BytesIO(csv.encode()), "x.csv")
    m = E.guess_mapping(list(raw.columns), "timeseries")
    assert m["time_h"] == "Time (h)" and m["voltage_V"] == "Ewe/V" and m["current_A"] == "I (A)"
    df, notes = E.apply_mapping(raw, m, "timeseries", E.CellDesign(area_cm2=25.0))
    assert abs(df["j_A_cm2"].iloc[0] - 0.5) < 1e-12
    eis = E.read_table(io.BytesIO(b"1000,0.1,0.01\n100,0.12,0.03\n10,0.2,0.02\n"), "e.csv")
    assert list(eis.columns) == ["freq_Hz", "z_re", "z_im_neg"]


def test_drt_recovers_resistances():
    f = np.logspace(4, -1, 50)
    Z = 0.02 + 0.1 / (1 + 1j * 2 * np.pi * f * 0.01)
    r = E.drt_tikhonov(f, Z.real, -Z.imag, 1e-4)
    assert abs(r["R_inf"] - 0.02) < 0.005 and abs(r["R_pol"] - 0.1) < 0.02


def test_operations_feasible(synth):
    s, W = synth
    p = E.renewable_profile(24 * 7)
    r = E.optimise_operation(p, np.full(len(p), 50.0), s["design"], s["kin"], E.OperatingConditions(), 0.5,
                             dict(a=0.008, Ea=24000, b=0.25), [60, 80], [0.1, 0.3])
    assert len(r) == 4 and r["H2_kg"].gt(0).all()
