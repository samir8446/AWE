"""What a test bench would record: 1-min signals aggregated to 1 h with noise, drift, outliers, gaps; periodic
diagnostics (LSV, EIS at 3 current densities, CV, titration). Also builds the hidden truth tables."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from . import physics as ph
from .config import BENCH, DESIGN, FAULTS, MEAS
from .simulate import MIN_PER_H, MECHANISMS, _rng, compose, mechanism_contributions

REF = dict(T_C=80.0, P_bar=1.0, j=0.4)       # reference condition for the truth decomposition
EIS_J = (0.05, 0.2, 0.4)
LSV_J = np.array([0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
EIS_F = np.logspace(5, -1, 61)


def _hourly(x: np.ndarray, how="mean") -> np.ndarray:
    n = len(x) // MIN_PER_H
    a = x[: n * MIN_PER_H].reshape(n, MIN_PER_H)
    if how == "mean":
        return a.mean(1)
    if how == "first":
        return a[:, 0]
    raise ValueError(how)


def _hourly_masked(x, mask):
    n = len(x) // MIN_PER_H
    a = np.where(mask, x, np.nan)[: n * MIN_PER_H].reshape(n, MIN_PER_H)
    with np.errstate(all="ignore"):
        cnt = np.isfinite(a).sum(1)
        return np.where(cnt > 0, np.nansum(a, 1) / np.maximum(cnt, 1), np.nan)


# ------------------------------------------------------------------------------------------------ continuous log
def hourly_log(r: Dict, seed: int):
    """Returns (log dataframe, truth-hourly dataframe, measurement-label rows)."""
    spec = r["spec"]
    u = spec.unit
    rng = _rng(seed, 3)
    area = DESIGN["area_cm2"].value
    mode = r["mode"]
    on = r["j"] > 0
    H = len(mode) // MIN_PER_H
    t_h = np.arange(H, dtype=float)
    run_frac = _hourly(on.astype(float))
    diag_frac = _hourly((mode == 2).astype(float))
    state = np.where(diag_frac > 0.5, "diagnostic", np.where(run_frac > 0.99, "run",
                                                                np.where(run_frac < 0.01, "off", "mixed")))
    g = r["gas"]
    true = dict(
        current_A=_hourly(g["I_A"]), voltage_V=_hourly(r["U"]), T_out_C=_hourly(r["T_C"]), T_in_C=_hourly(r["T_in_C"]),
        P_bar=_hourly(r["P"]), h2_flow_Nm3_h=_hourly(ph.molar_to_Nm3_h(g["H2_out"])),
        o2_flow_Nm3_h=_hourly(ph.molar_to_Nm3_h(g["O2_out"])), HTO_pct=_hourly_masked(g["HTO_pct"], on),
        conductivity_S_cm=_hourly(ph.sigma_koh(r["m_free"], r["T_C"] + 273.15)))
    labels: List[dict] = []
    meas = {}
    meas["current_A"] = true["current_A"] * (1 + MEAS["sd_I_rel"].value * rng.standard_normal(H))
    meas["voltage_V"] = true["voltage_V"] + MEAS["sd_U_V"].value * rng.standard_normal(H)
    # sensor drifts (labelled in truth)
    drift_T = 0.8 * t_h / 3000.0 if spec.index == 2 else np.zeros(H)
    if drift_T.any():
        labels.append(dict(unit=u, channel="T_out_C", kind="sensor_drift", start_h=0.0, end_h=float(H),
                           detail="+0.8 C linear over 3000 h"))
    meas["T_out_C"] = true["T_out_C"] + drift_T + MEAS["sd_T_C"].value * rng.standard_normal(H)
    meas["T_in_C"] = true["T_in_C"] + MEAS["sd_T_C"].value * rng.standard_normal(H)
    meas["P_bar"] = true["P_bar"] * (1 + MEAS["sd_P_rel"].value * rng.standard_normal(H))
    meas["flow_L_min"] = BENCH["flow_L_min"].value * (1 + MEAS["sd_flow_rel"].value * rng.standard_normal(H))
    meas["h2_flow_Nm3_h"] = true["h2_flow_Nm3_h"] * (1 + MEAS["sd_gas_rel"].value * rng.standard_normal(H))
    meas["o2_flow_Nm3_h"] = true["o2_flow_Nm3_h"] * (1 + MEAS["sd_gas_rel"].value * rng.standard_normal(H))
    hto_zero = (0.05 * t_h / 1000.0) if spec.group.P_bar > 1 else np.zeros(H)
    if hto_zero.any():
        labels.append(dict(unit=u, channel="HTO_pct", kind="sensor_drift", start_h=0.0, end_h=float(H),
                           detail="analyser zero drift +0.05 %/1000 h"))
    meas["HTO_pct"] = (true["HTO_pct"] * (1 + MEAS["sd_HTO_rel"].value * rng.standard_normal(H)) + hto_zero
                       + MEAS["sd_HTO_abs"].value * rng.standard_normal(H))
    # conductivity probe fouls (-2 %/1000 h) and is cleaned at each electrolyte change
    last_clean = np.zeros(H)
    for k in r["refresh_end"]:
        last_clean[int(k // MIN_PER_H):] = k / MIN_PER_H
    foul = 1 - 0.02 * (t_h - last_clean) / 1000.0
    labels.append(dict(unit=u, channel="conductivity_S_cm", kind="sensor_drift", start_h=0.0, end_h=float(H),
                       detail="probe fouling -2 %/1000 h, reset at electrolyte change"))
    meas["conductivity_S_cm"] = true["conductivity_S_cm"] * foul * (1 + MEAS["sd_cond_rel"].value * rng.standard_normal(H))
    # outliers in T_in, flow, HTO
    for ch, mag in (("T_in_C", 15.0), ("flow_L_min", 0.15), ("HTO_pct", 1.5)):
        idx = np.nonzero(rng.random(H) < MEAS["outlier_rate"].value)[0]
        for k in idx:
            meas[ch][k] += mag * rng.choice([-1, 1]) if ch != "HTO_pct" else mag
            labels.append(dict(unit=u, channel=ch, kind="outlier", start_h=float(k), end_h=float(k + 1), detail=f"{mag:+}"))
    # injected sensor fault (single bad voltage reading)
    for f in FAULTS:
        if f["unit"] == u and f["kind"] == "sensor_bad_reading":
            k = int(f["t_h"])
            if k < H:
                meas["voltage_V"][k] += f["magnitude"]
    df = pd.DataFrame(dict(unit=u, time_h=t_h, state=state, run_fraction=np.round(run_frac, 3), **meas))
    df["j_A_cm2"] = df["current_A"] / area
    df["T_C"] = df["T_out_C"]
    # single dropouts
    num = ["current_A", "voltage_V", "T_in_C", "T_out_C", "P_bar", "flow_L_min", "h2_flow_Nm3_h", "o2_flow_Nm3_h",
           "HTO_pct", "conductivity_S_cm"]
    for ch in num:
        k = rng.random(H) < MEAS["dropout_rate"].value
        df.loc[k, ch] = np.nan
    # logger outages: rows missing
    keep = np.ones(H, bool)
    for _ in range(int(rng.integers(1, 4))):
        a = int(rng.uniform(100, H - 100))
        d = int(rng.uniform(2, 40))
        keep[a:a + d] = False
        labels.append(dict(unit=u, channel="all", kind="logger_outage", start_h=float(a), end_h=float(a + d), detail="rows absent"))
    # nothing after the end of test (trip)
    if r["trip_min"] is not None:
        keep[int(np.ceil(r["trip_min"] / MIN_PER_H)) + 1:] = False
    df = df[keep].reset_index(drop=True)
    df["T_C"] = df["T_out_C"]
    cols = ["unit", "time_h", "state", "run_fraction", "current_A", "j_A_cm2", "voltage_V", "T_in_C", "T_out_C", "T_C",
            "P_bar", "flow_L_min", "conductivity_S_cm", "h2_flow_Nm3_h", "o2_flow_Nm3_h", "HTO_pct"]
    df = df[cols]
    # hourly truth
    st = r["state"]
    ex = r["extra"]
    TK = r["T_C"] + 273.15
    c = r["cell"]
    th = pd.DataFrame(dict(
        unit=u, time_h=t_h, state=state, U_true_V=true["voltage_V"], j_true_A_cm2=_hourly(r["j"]), T_true_C=true["T_out_C"],
        T_in_true_C=true["T_in_C"], P_true_bar=true["P_bar"], HTO_true_pct=true["HTO_pct"],
        OTH_true_pct=_hourly_masked(g["OTH_pct"], on), eta_F=_hourly_masked(g["eta_F"], on),
        koh_wt_true=_hourly(r["wt"]), carbonate_fraction=_hourly(r["xco3"]), m_free_mol_L=_hourly(r["m_free"]),
        sigma_true_S_cm=true["conductivity_S_cm"],
        j0c_factor=_hourly(st["j0c_f"]), j0a_factor=_hourly(st["j0a_f"]), alpha_c_factor=_hourly(st["ac_f"]),
        alpha_a_factor=_hourly(st["aa_f"]),
        tafel_c_mV_dec_at_T=_hourly(np.broadcast_to(c["b_c"], TK.shape) * 1000),
        tafel_a_mV_dec_at_T=_hourly(np.broadcast_to(c["b_a"], TK.shape) * 1000),
        RF_cathode=_hourly(ex["RFc"]), RF_anode=_hourly(ex["RFa"]), Cdl_cathode_mF_cm2=_hourly(ex["Cdl_c"]) * 1e3,
        Cdl_anode_mF_cm2=_hourly(ex["Cdl_a"]) * 1e3, sep_fouling_factor=_hourly(st["sep_f"]),
        R_contact_ohm_cm2=_hourly(st["R_c"]), R_sep_ohm_cm2=_hourly(c["R_sep"]), R_gap_ohm_cm2=_hourly(c["R_gap"]),
        crossover_factor=_hourly(st["x_f"]), t_eq_h=_hourly(ex["t_eq"]), op_hours=_hourly(ex["t_run"]),
        stop_dose=_hourly(ex["S"]), anode_dose_h=_hourly(ex["D"]),
        anode_accelerated=(_hourly(ex["D"]) >= ex["Dstar"]).astype(int),
        Q_gen_W=_hourly(r["Q_gen"]), Q_loss_W=_hourly(r["Q_loss"]), Q_store_W=_hourly(r["Q_store"]),
        Q_lye_W=_hourly(r["Q_lye"]), V_tn_V=_hourly(r["Vtn"]),
        nH2_prod_mol_s=_hourly(g["nH2_prod"]), nH2_cross_mol_s=_hourly(g["nH2_cross"]),
        nH2_out_mol_s=_hourly(g["H2_out"]), nO2_prod_mol_s=_hourly(g["nO2_prod"]), nO2_out_mol_s=_hourly(g["O2_out"]),
        nO2_cross_mol_s=_hourly(g["nO2_cross"])))
    # mechanism contributions at the reference condition (hourly states)
    parts_h = {k: (_hourly(np.broadcast_to(v, r["j"].shape).astype(float)) if np.ndim(v) else np.full(H, float(v)))
               for k, v in r["parts"].items()}
    m_nom = float(ph.molarity_from_wt(spec.group.koh_wt, REF["T_C"] + 273.15))
    m_ref = ph.molarity_from_wt(_hourly(r["wt"]), REF["T_C"] + 273.15) * (1 - _hourly(r["xco3"]))
    mech = mechanism_contributions(parts_h, m_ref, REF, m_nom)
    mdf = pd.DataFrame({"unit": u, "time_h": t_h, **{f"dV_{k}_mV": 1000 * v for k, v in mech.items()
                                                    if k != "U_ref_BoL"}})
    mdf["U_ref_BoL_V"] = mech["U_ref_BoL"]
    return df, th, mdf, labels


def minute_sample(r: Dict, seed: int, h0: int, h1: int) -> pd.DataFrame:
    """Raw 1-min record for a short window (shows what the hourly aggregation hides)."""
    rng = _rng(seed, 9)
    k = slice(h0 * MIN_PER_H, h1 * MIN_PER_H)
    n = h1 * MIN_PER_H - h0 * MIN_PER_H
    g = r["gas"]
    return pd.DataFrame(dict(
        unit=r["spec"].unit, time_h=np.arange(h0 * MIN_PER_H, h1 * MIN_PER_H) / MIN_PER_H,
        current_A=g["I_A"][k] * (1 + 0.002 * rng.standard_normal(n)),
        voltage_V=r["U"][k] + 0.004 * rng.standard_normal(n), T_out_C=r["T_C"][k] + 0.3 * rng.standard_normal(n),
        T_in_C=r["T_in_C"][k] + 0.3 * rng.standard_normal(n), P_bar=r["P"][k] * (1 + 0.01 * rng.standard_normal(n)),
        h2_flow_Nm3_h=ph.molar_to_Nm3_h(g["H2_out"][k]) * (1 + 0.03 * rng.standard_normal(n)),
        HTO_pct=g["HTO_pct"][k] * (1 + 0.06 * rng.standard_normal(n))))


# ------------------------------------------------------------------------------------------------ diagnostics
def _state_at(r, k):
    return {kk: (v[k] if np.ndim(v) else v) for kk, v in r["state"].items()}


def diagnostics(r: Dict, seed: int):
    """LSV, EIS (3 current densities), CV (double layer, both electrodes; Ni redox, anode), titration."""
    spec = r["spec"]
    u = spec.unit
    rng = _rng(seed, 4)
    area = DESIGN["area_cm2"].value
    Td = BENCH["diag_T_C"].value + 273.15
    Pd = BENCH["diag_P_bar"].value
    lsv, eis, cv, titr, labels, truth_rows = [], [], [], [], [], []
    for n_s, k in enumerate(r["diag_start"]):
        t = k / MIN_PER_H
        if n_s > 0 and rng.random() < 0.05:
            labels.append(dict(unit=u, channel="diagnostics", kind="missing_session", start_h=t, end_h=t + 3,
                               detail="session not recorded"))
            continue
        s = _state_at(r, k)
        m = float(ph.molarity_from_wt(r["wt"][k], Td) * (1 - r["xco3"][k]))
        ses = f"{u}-D{n_s:02d}"
        # LSV (galvanostatic steps, quasi-steady; voltage includes iR)
        c = ph.cell(LSV_J, Td, Pd, m, s)
        lsv.append(pd.DataFrame(dict(unit=u, time_h=t, session=ses, j_A_cm2=LSV_J, current_A=LSV_J * area,
                                     voltage_V=c["U"] + MEAS["sd_lsv_V"].value * rng.standard_normal(len(LSV_J)),
                                     T_C=Td - 273.15, P_bar=Pd)))
        # EIS: R_ohm + ZARC(cathode) + ZARC(anode) + ZARC(gas/bubbles), low-f limit = dU/dj of the DC model
        RFc, RFa = r["extra"]["RFc"][k], r["extra"]["RFa"][k]
        Cc = DESIGN["C_specific_uF_cm2"].value * 1e-6 * RFc
        Ca = DESIGN["C_specific_uF_cm2"].value * 1e-6 * RFa
        for jj in EIS_J:
            cj = ph.cell(np.array([jj]), Td, Pd, m, s)
            dj = 1e-4
            dUdj = float((ph.cell(np.array([jj + dj]), Td, Pd, m, s)["U"] - ph.cell(np.array([jj - dj]), Td, Pd, m, s)["U"])[0] / (2 * dj))
            Rs, Rc_, Ra_ = float(cj["R_ohm"][0]), float(cj["Rct_c"][0]), float(cj["Rct_a"][0])
            Rg = max(dUdj - Rs - Rc_ - Ra_, 1e-4)
            w = 2 * np.pi * EIS_F
            zarc = lambda R_, Q, n: R_ / (1 + R_ * Q * (1j * w) ** n)
            Z = Rs + zarc(Rc_, Cc, 0.92) + zarc(Ra_, Ca, 0.88) + zarc(Rg, 0.05 / Rg, 0.95)
            Z = Z / area + 1j * w * DESIGN["L_cell_H"].value
            drift = rng.random() < MEAS["eis_drift_rate"].value
            if drift:      # slow drift during the sweep (high -> low f): not Kramers-Kronig consistent
                prog = np.linspace(0, 1, len(EIS_F)) ** 2
                Z = Z + 0.08 * (Rc_ + Ra_) / area * prog
                labels.append(dict(unit=u, channel="eis", kind="nonstationary_spectrum", start_h=t, end_h=t,
                                   detail=f"j={jj} A/cm2; drift 8 % of R_pol"))
            e = MEAS["sd_eis_rel"].value * np.abs(Z)
            Z = Z + e * rng.standard_normal(len(Z)) + 1j * e * rng.standard_normal(len(Z))
            eis.append(pd.DataFrame(dict(unit=u, time_h=t, session=ses, j_A_cm2=jj, T_C=Td - 273.15, freq_Hz=EIS_F,
                                         z_re=Z.real, z_im_neg=-Z.imag)))
            truth_rows.append(dict(unit=u, time_h=t, session=ses, j_A_cm2=jj, R_ohm_ohm_cm2=Rs, R_ct_cathode_ohm_cm2=Rc_,
                                   R_ct_anode_ohm_cm2=Ra_, R_gas_ohm_cm2=Rg, dU_dj_ohm_cm2=dUdj,
                                   Cdl_cathode_mF_cm2=Cc * 1e3, Cdl_anode_mF_cm2=Ca * 1e3,
                                   tafel_c_mV_dec=float(cj["b_c"]) * 1000, tafel_a_mV_dec=float(cj["b_a"]) * 1000,
                                   j0c_A_cm2=float(cj["j0c"][0]), j0a_A_cm2=float(cj["j0a"][0]), U_V=float(cj["U"][0]),
                                   kk_valid=not drift))
        # CV double layer (both electrodes): box current +- Cdl A nu, small leak slope
        for el, Cdl, (e0, e1) in (("cathode", Cc, (0.10, 0.20)), ("anode", Ca, (0.90, 1.00))):
            for nu in (0.01, 0.02, 0.04, 0.08):
                E = np.r_[np.linspace(e0, e1, 50), np.linspace(e1, e0, 50)]
                sgn = np.r_[np.ones(50), -np.ones(50)]
                i_dl = sgn * Cdl * area * nu * (1 - np.exp(-np.r_[np.arange(50), np.arange(50)] / 3.0)) + (E - (e0 + e1) / 2) / 200.0
                i_dl = i_dl * (1 + MEAS["sd_cv_rel"].value * rng.standard_normal(100)) + 2e-5 * rng.standard_normal(100)
                cv.append(pd.DataFrame(dict(unit=u, time_h=t, session=ses, electrode=el, window="double_layer",
                                            scan_rate_V_s=nu, cycle_point=np.arange(100), E_V_vs_RHE=E, current_A=i_dl)))
        # CV Ni(OH)2/NiOOH redox + OER onset, anode, 10 mV/s
        nu = 0.01
        E = np.r_[np.linspace(1.00, 1.55, 110), np.linspace(1.55, 1.00, 110)]
        fwd = np.r_[np.ones(110, bool), np.zeros(110, bool)]
        Q = DESIGN["q_redox_mC_cm2"].value * 1e-3 * RFa                 # C/cm2 geometric
        sig = 0.02
        Ep_a = 1.37 + 0.002 * t / 1000.0
        gauss = lambda x, mu: np.exp(-(x - mu) ** 2 / (2 * sig ** 2)) / (sig * np.sqrt(2 * np.pi))
        j_red = np.where(fwd, Q * nu * gauss(E, Ep_a), -0.9 * Q * nu * gauss(E, Ep_a - 0.07))
        E_eq = float(ph.E_rev(Td, 1.0, m))
        b = float(ph.R * Td / (2 * ph.alpha_a(Td) * s["aa_f"] * ph.F))
        j_oer = float(ph.j0_bol(Td, 1.0)[1]) * s["j0a_f"] * np.exp(np.clip((E - E_eq) / b, -50, 50))
        j_tot = (j_red + np.minimum(j_oer, 0.5) + np.where(fwd, 1, -1) * Ca * nu) * area
        j_tot = j_tot * (1 + MEAS["sd_cv_rel"].value * rng.standard_normal(220)) + 5e-5 * rng.standard_normal(220)
        cv.append(pd.DataFrame(dict(unit=u, time_h=t, session=ses, electrode="anode", window="Ni_redox_OER",
                                    scan_rate_V_s=nu, cycle_point=np.arange(220), E_V_vs_RHE=E, current_A=j_tot)))
        # titration of the electrolyte sample
        w = r["wt"][k]
        x = r["xco3"][k]
        titr.append(dict(unit=u, time_h=t, session=ses, koh_free_wt=w * (1 - x) + MEAS["sd_titr_wt"].value * rng.standard_normal(),
                         k2co3_wt=w * x * (138.2 / 2) / 56.1 + 0.3 * MEAS["sd_titr_wt"].value * abs(rng.standard_normal())))
    cat = lambda L: pd.concat(L, ignore_index=True) if L else pd.DataFrame()
    return dict(lsv=cat(lsv), eis=cat(eis), cv=cat(cv), titration=pd.DataFrame(titr)), labels, pd.DataFrame(truth_rows)
