"""Tests for the synthetic AWE data generator (awe_synth). A short dataset (1 unit per group, 600 h) keeps them fast;
the full 18-unit / 3000 h dataset is checked by `python -m awe_synth.report`."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import engine as E  # noqa: E402
from awe_synth import GENERATOR_VERSION, physics as ph  # noqa: E402
from awe_synth.config import DEG, DESIGN, FAULTS, GROUPS, PHYS  # noqa: E402
from awe_synth.generate import generate  # noqa: E402
from awe_synth.simulate import UnitSpec, simulate_unit  # noqa: E402

BOL = dict(j0c_f=1.0, j0a_f=1.0, ac_f=1.0, aa_f=1.0, sep_f=1.0, R_c=0.03)


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    root = generate(seed=7, out=str(out), hours=600, units_per_group=1, progress=lambda *_: None)
    return root


def test_bol_physics_literature_shape():
    T, m = 353.15, ph.molarity_from_wt(30, 353.15)
    U = ph.cell(np.array([0.1, 0.4, 0.8]), T, 1.0, m, BOL)["U"]
    assert np.all(np.diff(U) > 0) and 1.6 <= U[1] <= 2.0          # Qu26 industrial window
    assert abs(float(ph.r_separator(ph.sigma_koh(m, T))) - 0.27) < 0.03     # Zirfon, Qu26 Table 2
    assert abs(float(ph.V_tn(298.15, 1.0, 1e-6)) - 1.48) < 0.015            # Hu22: 1.48 V at 25 C (+ vapour term)
    hto1 = ph.gases(np.array([0.4]), T, 1.0, m)["HTO_pct"][0]
    hto16 = ph.gases(np.array([0.4]), T, 16.0, m)["HTO_pct"][0]
    assert hto1 < 0.2 < hto16 < 2.0                                       # crossover grows with pressure


def test_generator_matches_independent_engine():
    dz = E.CellDesign(area_cm2=100, gap_anode_mm=0.5, gap_cathode_mm=0.5, membrane_model="structural",
                      sep_porosity=DESIGN["sep_porosity"].value, sep_thickness_mm=0.5, electrode_sigma_S_m=1e12)
    for Tc in (60, 80):
        ue = E.cell_voltage(0.4, E.OperatingConditions(T_C=Tc, P_bar=1, koh_wt=30), dz, E.Kinetics(),
                            E.DegradationState(dR_ohm_cm2=0.03))["U"][0]
        ug = ph.cell(np.array([0.4]), Tc + 273.15, 1.0, ph.molarity_from_wt(30, Tc + 273.15), BOL)["U"][0]
        assert abs(ue - ug) < 0.01


def test_files_schema_and_truth_separated(ds):
    data, truth = os.path.join(ds, "data"), os.path.join(ds, "truth")
    assert set(os.listdir(data)) >= {"units.csv", "timeseries_1h.csv", "events.csv", "lsv.csv", "eis.csv", "cv.csv",
                                     "titration.csv", "DATA_DICTIONARY.csv"}
    for f in os.listdir(data):       # nothing in data/ reveals the hidden truth
        if f.endswith(".csv") and f != "DATA_DICTIONARY.csv":
            cols = set(pd.read_csv(os.path.join(data, f), nrows=5).columns)
            assert not cols & {"j0c_factor", "fault", "accel_onset_h", "dV_total_mV", "kk_valid", "carbonate_fraction"}
    ev = pd.read_csv(os.path.join(data, "events.csv"))
    assert not ev["note"].fillna("").str.contains("fault|resistance_step|activity", case=False).any()
    units = pd.read_csv(os.path.join(data, "units.csv"))
    assert "fault" not in " ".join(units.columns) and len(units) == len(GROUPS)
    assert {"faults.csv", "hourly_state.csv", "mechanisms_at_reference.csv", "unit_parameters.csv"} <= set(os.listdir(truth))


def test_data_dictionary_covers_columns(ds):
    dd = pd.read_csv(os.path.join(ds, "data", "DATA_DICTIONARY.csv"))
    text = " ".join(dd["column"].astype(str))
    ts = pd.read_csv(os.path.join(ds, "data", "timeseries_1h.csv"), nrows=3)
    for c in ts.columns:
        assert c in text, c


def test_reproducible_same_seed(tmp_path):
    a = generate(seed=3, out=str(tmp_path / "a"), hours=300, units_per_group=1, progress=lambda *_: None)
    b = generate(seed=3, out=str(tmp_path / "b"), hours=300, units_per_group=1, progress=lambda *_: None)
    ma, mb = (json.load(open(os.path.join(x, "manifest.json"))) for x in (a, b))
    assert ma["files"] == mb["files"] and ma["version"] == GENERATOR_VERSION
    c = generate(seed=4, out=str(tmp_path / "c"), hours=300, units_per_group=1, progress=lambda *_: None)
    mc = json.load(open(os.path.join(c, "manifest.json")))
    assert mc["files"]["data/timeseries_1h.csv"] != ma["files"]["data/timeseries_1h.csv"]


def test_mass_and_energy_balance_minute_level():
    g = next(x for x in GROUPS if x.mode == "dynamic")
    r = simulate_unit(UnitSpec("G6_dynamic-1", g, 1, 11), 400)
    gas = r["gas"]
    assert np.allclose(gas["H2_out"] + gas["nH2_cross"], gas["nH2_prod"], rtol=0, atol=1e-15)
    assert np.allclose(gas["nH2_prod"], gas["eta_F"] * gas["I_A"] / (2 * ph.F))
    on = r["j"] > 0
    lhs = r["U"][on] * gas["I_A"][on]
    rhs = gas["eta_F"][on] * gas["I_A"][on] * r["Vtn"][on] + r["Q_gen"][on]
    assert np.allclose(lhs, rhs)
    assert np.allclose(r["Q_gen"], r["Q_loss"] + r["Q_store"] + r["Q_lye"])


def test_acceleration_and_recovery_known_answers():
    g = next(x for x in GROUPS if x.name == "G3_T80_ref")
    r = simulate_unit(UnitSpec("G3_T80_ref-1", g, 1, 21), 3000)
    k_on = r["extra"]["k_on"]
    assert k_on is not None
    aa = r["parts"]["a_intrinsic"]
    h = np.arange(len(aa)) / 60
    pre = np.polyfit(h[k_on - 400 * 60:k_on], np.log(aa[k_on - 400 * 60:k_on]), 1)[0]
    post = np.polyfit(h[k_on + 60:k_on + 400 * 60], np.log(aa[k_on + 60:k_on + 400 * 60]), 1)[0]
    assert post / pre > 0.8 * DEG["accel_factor"].value * 0.9
    # electrolyte change resets carbonation and the reversible hydride loss -> voltage recovers
    k = r["refresh_end"][0]
    assert r["xco3"][k + 5] < 1e-3 < r["xco3"][k - 5]
    assert r["parts"]["c_hydride"][k + 30] > r["parts"]["c_hydride"][k - 30]


def test_faults_only_on_listed_units_and_hto_trip(ds):
    st = pd.read_csv(os.path.join(ds, "truth", "hourly_state.csv"))
    faulty = {f["unit"] for f in FAULTS}
    ok = st[~st["unit"].isin(faulty)]
    assert (ok["crossover_factor"] == 1).all()
    # full-length HTO fault unit trips at 2 %
    g = next(x for x in GROUPS if x.name == "G4_P16")
    r = simulate_unit(UnitSpec("G4_P16-1", g, 1, 31), 3000)
    f = next(x for x in FAULTS if x["kind"] == "HTO_rise")
    assert r["trip_min"] is not None and r["trip_min"] / 60 > f["t_h"]
    assert any(e[2] == "trip" and "HTO" in e[3] for e in r["events"])
    assert np.nanmax(r["gas"]["HTO_pct"][: r["trip_min"] - 5]) <= PHYS["HTO_shutdown_pct"].value + 0.2


def test_platform_reads_generated_files_without_code_change(ds):
    data = os.path.join(ds, "data")
    design = E.CellDesign(area_cm2=DESIGN["area_cm2"].value)
    for f, kind in (("timeseries_1h.csv", "timeseries"), ("lsv.csv", "polcurve"), ("eis.csv", "eis"), ("events.csv", "events")):
        raw = pd.read_csv(os.path.join(data, f))
        mp = E.guess_mapping(list(raw.columns), kind)
        for req in E.DATA_TYPES[kind]["required"]:
            assert mp[req] is not None, (f, req)
        df, _ = E.apply_mapping(raw, mp, kind, design)
        assert len(df) and df["unit"].nunique() == len(GROUPS)
