"""Synthetic stainless-steel accelerated-ageing scenario with KNOWN truth (for testing the DRT, degradation, anomaly
and closure methods). It mimics the qualitative behaviour reported in 'Cell and results' (activation, passivation,
cyclic spalling resets, terminal gas suffocation, hardware events), but every number here is an invented test value,
NOT a measurement or a literature value.

Process model (area-specific, ohm cm2; each overpotential has an exact differential resistance R_k = d eta_k / dj,
which is what an EIS measured at current j sees):
  U(j) = E_rev + j (R_inf + R_1) + b_c ln(1 + j/j0c) + b_a ln(1 + j/j0a) + j R_film + b_mt ln(1/(1 - j/j_lim))
  EIS(j): R_inf + R_1 (ion transport, ~kHz) + b_c/(j0c+j) (HER, P2) + [b_a/(j0a+j) + R_film] (OER + passive film, P3)
          + b_mt/(j_lim - j) (mass transport / trapped gas, P4)
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

import engine as E

__version__ = "0.3.0"

SS_TRUTH_PARAMS = dict(
    area_cm2=25.0, T_C=30.0, P_bar=1.0, koh_M=7.0, j_op=0.5, hours=1000.0, dt_h=0.25,
    R_inf0=0.22, R_inf_drift_per_h=2.0e-4, R1=0.06, R1_damage=0.5,
    b_c=0.045, j0c=2e-3, C2=0.01, a2=0.92,
    b_a=0.050, j0a=5e-5, C3=0.05, a3=0.90,
    R_film_native=0.50, tau_activation_h=15.0, R_film_act=0.02,
    k_pass=2.0e-4, spall_thr_mean=0.12, spall_thr_sd=0.012, spall_keep=0.3, pit_per_spall=0.010,
    regrowth_accel=1.6, g1=8.0, g2=15.0, b_mt=0.04, j_lim0=1.5, tau4=1.5, a4=0.85,
    tau1=5e-5, a1=0.95, noise_V=0.002, eis_noise=0.004,
    refill_h=(300.0, 650.0), power_change=((400.0, 0.012), (800.0, -0.008)),
    eis_every_h=50.0, eis_currents=(0.05, 0.1, 0.2, 0.35, 0.5), bad_spectrum_share=0.08,
    terminal_damage=0.05, f_min_Hz=0.05,
)


def _state_series(rng, P, n):
    """Hidden degradation states on the time grid: R_film, pit damage, gas trapping, phase labels, resets."""
    t = np.arange(n) * P["dt_h"]
    film = np.empty(n)
    dmg = np.zeros(n)
    phase = np.empty(n, dtype=object)
    resets = []
    f, d, k = P["R_film_native"], 0.0, P["k_pass"]
    thr = rng.normal(P["spall_thr_mean"], P["spall_thr_sd"])
    for i, ti in enumerate(t):
        if ti < 50:
            f = P["R_film_act"] + (P["R_film_native"] - P["R_film_act"]) * math.exp(-ti / P["tau_activation_h"])
            ph = "1 activation"
        else:
            f += k * P["dt_h"]
            ph = "2 passivation" if not resets else "3 cyclic spalling"
            if f > thr + d and d < P["terminal_damage"]:   # a deeply scarred surface no longer sheds
                drop_to = d + P["R_film_act"] + P["spall_keep"] * (f - d - P["R_film_act"])
                resets.append(dict(time_h=float(ti), drop_ohm_cm2=float(f - drop_to), film_before=float(f)))
                f = drop_to
                d += P["pit_per_spall"] * (1 + 0.3 * rng.standard_normal())
                k *= P["regrowth_accel"]
                thr = rng.normal(P["spall_thr_mean"], P["spall_thr_sd"]) * max(0.88 ** len(resets), 0.6)
            if d >= P["terminal_damage"]:
                ph = "4 terminal"
        film[i], dmg[i], phase[i] = f, d, ph
    excess = np.maximum(film - P["R_film_act"] - dmg, 0) * np.where(t < 50, 0.1, 1.0)   # native oxide traps little gas
    G = P["g1"] * excess + P["g2"] * dmg
    # terminal: gas suffocation accelerates once the damage threshold is passed
    term = np.nonzero(dmg >= P["terminal_damage"])[0]
    t_term = float(t[term[0]]) if len(term) else float("nan")
    if len(term):
        G[term[0]:] += 0.01 * (t[term[0]:] - t[term[0]]) ** 1.5
    return t, film, dmg, G, phase, resets, t_term


def _overpotentials(j, P, R_inf, R1, film, G, T_C):
    j_lim = P["j_lim0"] / (1 + G)
    scale = (T_C + 273.15) / (P["T_C"] + 273.15)
    eta = dict(ohmic=j * R_inf, ion=j * R1,
               her=P["b_c"] * scale * np.log1p(j / P["j0c"]),
               oer=P["b_a"] * scale * np.log1p(j / P["j0a"]) + j * film,
               mt=P["b_mt"] * scale * -np.log(np.clip(1 - j / j_lim, 1e-4, None)))
    return eta


def eis_spectrum(j, P, R_inf, R1, film, G, freq):
    """Area-specific impedance (ohm cm2) at current j from the process model (ZARC elements)."""
    w = 2 * np.pi * freq
    j_lim = P["j_lim0"] / (1 + G)
    R2 = P["b_c"] / (P["j0c"] + j)
    R3 = P["b_a"] / (P["j0a"] + j) + film
    R4 = P["b_mt"] / max(j_lim - j, 0.02)
    z = R_inf + R1 / (1 + (1j * w * P["tau1"]) ** P["a1"])
    z = z + R2 / (1 + (1j * w * R2 * P["C2"]) ** P["a2"]) + R3 / (1 + (1j * w * R3 * P["C3"]) ** P["a3"])
    z = z + R4 / (1 + (1j * w * P["tau4"]) ** P["a4"])
    return z, dict(R_inf=R_inf, R_P1=R1, R_P2=R2, R_P3=R3, R_P4=R4,
                   tau_P1=P["tau1"], tau_P2=R2 * P["C2"], tau_P3=R3 * P["C3"], tau_P4=P["tau4"])


def synthetic_ss_dataset(seed: int = 0, n_cells: int = 3, params: Optional[dict] = None) -> Dict[str, object]:
    """Replicate stainless-steel cells at 0.5 A/cm2, 30 C, 7 M KOH with known DRT-resolved truth, resets, hardware
    events, measurement pauses with restart transients, injected sensor/temperature anomalies, and a terminal failure."""
    P = dict(SS_TRUTH_PARAMS, **(params or {}))
    koh_wt = E.wt_from_molarity(P["koh_M"], P["T_C"] + 273.15)
    ts, ev, pcs, eis, truth, eis_truth, anomalies, reset_rows = [], [], [], [], [], [], [], []
    freq = np.logspace(4, np.log10(P["f_min_Hz"]), int(round((4 - np.log10(P["f_min_Hz"])) * 10)) + 1)
    for c in range(n_cells):
        rng = np.random.default_rng(seed * 100 + c)
        Pc = dict(P)
        for key in ("k_pass", "spall_thr_mean", "pit_per_spall", "b_a", "R_inf0"):
            Pc[key] = P[key] * (1 + 0.08 * rng.standard_normal())
        u = f"SS-{chr(65 + c)}"
        n = int(P["hours"] / P["dt_h"]) + 1
        t, film, dmg, G, phase, resets, t_term = _state_series(rng, Pc, n)
        # failure: voltage spike then gasket failure (end of test)
        Erev = float(E.E_rev(P["T_C"] + 273.15, P["P_bar"], P["koh_M"]))
        # electrolyte drift between refills (evaporation), reset at each refill
        last_refill = np.zeros(n)
        for rf in P["refill_h"]:
            last_refill = np.where(t >= rf, rf, last_refill)
        R_inf = Pc["R_inf0"] + P["R_inf_drift_per_h"] * (t - last_refill)
        R1 = P["R1"] + P["R1_damage"] * dmg
        T = np.full(n, P["T_C"]) + 0.15 * rng.standard_normal(n)
        # injected temperature excursion (operating anomaly)
        t_ex = float(rng.uniform(150, 850))
        ex = (t >= t_ex) & (t < t_ex + 8)
        T[ex] += 6.0
        anomalies.append(dict(unit=u, start_h=t_ex, end_h=t_ex + 8, kind="temperature excursion", cls="operating"))
        on = np.ones(n, bool)
        pause = np.zeros(n, bool)
        for tm in np.arange(P["eis_every_h"], P["hours"], P["eis_every_h"]):
            pause |= (t >= tm) & (t < tm + 1.0)
            ev.append((u, float(tm), "measurement"))
            anomalies.append(dict(unit=u, start_h=float(tm + 1.0), end_h=float(tm + 4.0), kind="restart transient", cls="known"))
        on &= ~pause
        jj = np.where(on, P["j_op"], 0.0)
        U = np.full(n, np.nan)
        comp = {k: np.zeros(n) for k in ("ohmic", "ion", "her", "oer", "mt")}
        for i in np.nonzero(on)[0]:
            eta = _overpotentials(P["j_op"], Pc, R_inf[i], R1[i], film[i], G[i], T[i])
            for k in comp:
                comp[k][i] = eta[k]
            U[i] = Erev - 0.0009 * (T[i] - P["T_C"]) + sum(eta.values())
        # restart transient: lower voltage that relaxes over ~2 h after each pause (reversible, benign)
        since = np.full(n, np.inf)
        for tm in np.arange(P["eis_every_h"], P["hours"], P["eis_every_h"]):
            m = t >= tm + 1.0
            since[m] = np.minimum(since[m], t[m] - (tm + 1.0))
        U = U - 0.012 * np.exp(-since / 2.0)
        # hardware: power-supply changes add an instrument offset to the logged voltage
        offset = np.zeros(n)
        for tp, dv in P["power_change"]:
            offset[t >= tp] += dv
            ev.append((u, float(tp), "power_supply_change"))
            anomalies.append(dict(unit=u, start_h=float(tp), end_h=float(tp) + 1, kind="power-supply change", cls="hardware"))
        for rf in P["refill_h"]:
            ev.append((u, float(rf), "koh_refill"))
            anomalies.append(dict(unit=u, start_h=float(rf), end_h=float(rf) + 1, kind="KOH refill", cls="hardware"))
        Um = U + offset + P["noise_V"] * rng.standard_normal(n)
        # sensor spikes
        for _ in range(5):
            i = int(rng.integers(40, n - 40))
            if on[i]:
                Um[i] += rng.choice([-1, 1]) * rng.uniform(0.04, 0.08)
                anomalies.append(dict(unit=u, start_h=float(t[i]), end_h=float(t[i] + P["dt_h"]), kind="sensor spike", cls="sensor"))
        for r in resets:
            anomalies.append(dict(unit=u, start_h=r["time_h"], end_h=r["time_h"] + 1, kind="spalling reset", cls="electrochemical"))
            reset_rows.append(dict(unit=u, **r))
        if np.isfinite(t_term):
            anomalies.append(dict(unit=u, start_h=t_term, end_h=float(t[-1]), kind="terminal degradation", cls="electrochemical"))
        # end of test: voltage above 2.6 V (or the planned duration) -> gasket failure logged
        over = np.nonzero(Um > 2.6)[0]
        end = int(over[0]) + 4 if len(over) else n
        end = min(end, n)
        ev.append((u, float(t[end - 1]), "gasket_failure"))
        ts.append(pd.DataFrame(dict(unit=u, time_h=t[:end], j_A_cm2=jj[:end], current_A=jj[:end] * P["area_cm2"],
                                    voltage_V=Um[:end], U_cell_V=Um[:end], T_C=T[:end], P_bar=P["P_bar"], koh_wt=koh_wt)))
        truth.append(pd.DataFrame(dict(unit=u, time_h=t[:end], phase=phase[:end], R_film=film[:end], damage=dmg[:end],
                                       G=G[:end], R_inf=R_inf[:end], R1=R1[:end],
                                       **{f"eta_{k}": comp[k][:end] for k in comp},
                                       U_true=U[:end] + 0.012 * np.exp(-since[:end] / 2.0))))
        # EIS at several currents and LSV, at t = 0 and every eis_every_h (during the pauses)
        for tm in np.r_[0.0, np.arange(P["eis_every_h"], t[end - 1], P["eis_every_h"])]:
            i = int(round(tm / P["dt_h"]))
            for jc in P["eis_currents"]:
                z, tr = eis_spectrum(jc, Pc, R_inf[i], R1[i], film[i], G[i], freq)
                z = z * (1 + P["eis_noise"] * (rng.standard_normal(len(z)) + 1j * rng.standard_normal(len(z))))
                bad = rng.random() < P["bad_spectrum_share"]
                if bad:   # non-stationary measurement: drift during the sweep (violates Kramers-Kronig)
                    z = z + np.linspace(0, 0.08 * abs(z[-1]), len(z))
                zo = z / P["area_cm2"]
                eis.append(pd.DataFrame(dict(unit=u, time_h=float(tm), j_A_cm2=jc, freq_Hz=freq, z_re=zo.real,
                                             z_im_neg=-zo.imag)))
                eis_truth.append(dict(unit=u, time_h=float(tm), j_A_cm2=jc, valid=not bad, **tr))
            jg = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            Ug = np.array([Erev + sum(_overpotentials(x, Pc, R_inf[i], R1[i], film[i], G[i], P["T_C"]).values()) for x in jg])
            pcs.append(pd.DataFrame(dict(unit=u, time_h=float(tm), j_A_cm2=jg, voltage_V=Ug + 0.001 * rng.standard_normal(len(jg)),
                                         T_C=P["T_C"], P_bar=P["P_bar"])))
    data = dict(timeseries=pd.concat(ts, ignore_index=True), events=pd.DataFrame(ev, columns=["unit", "time_h", "event"])
                .sort_values(["unit", "time_h"]).reset_index(drop=True),
                polcurve=pd.concat(pcs, ignore_index=True), eis=pd.concat(eis, ignore_index=True))
    design = E.CellDesign(name="synthetic SS cell", n_cells=1, area_cm2=P["area_cm2"], electrode="stainless")
    ref = E.OperatingConditions(T_C=P["T_C"], P_bar=P["P_bar"], j_A_cm2=P["j_op"], koh_wt=koh_wt)
    return dict(data=data, truth=pd.concat(truth, ignore_index=True), eis_truth=pd.DataFrame(eis_truth),
                anomalies=pd.DataFrame(anomalies), resets=pd.DataFrame(reset_rows), design=design, ref=ref,
                params=P, freq=freq)
