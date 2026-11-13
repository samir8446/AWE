"""Minute-level simulation of one unit: schedule -> thermal/pressure/electrolyte loop -> hidden degradation state ->
noise-free signals. Measurement effects are added in measure.py; nothing here is seen by the platform."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from . import physics as ph
from .config import BENCH, DEG, DESIGN, FAULTS, PHYS, Group

MIN_PER_H = 60
M_W = 18.015e-3   # kg/mol water


@dataclass
class UnitSpec:
    unit: str
    group: Group
    index: int          # 1..n within group
    seed: int


def _rng(seed: int, *salt: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, *salt]))


# ------------------------------------------------------------------------------------------------ schedule
def renewable_minutes(hours: int, rng) -> np.ndarray:
    """Normalised power 0..1: daily solar + slow wind + minute fluctuations (synthetic, not a measured site)."""
    h = np.arange(hours + 1)
    day = rng.uniform(0.5, 1.0, hours // 24 + 2)[h // 24]
    solar = np.clip(np.sin((h % 24 - 6) / 12 * np.pi), 0, None) * day
    ou = np.zeros(hours + 1)
    z = rng.standard_normal(hours + 1)
    for k in range(1, hours + 1):         # Ornstein-Uhlenbeck, ~1 day memory
        ou[k] = ou[k - 1] * np.exp(-1 / 24) + 0.2 * z[k]
    wind = np.clip(0.3 + 0.15 * np.sin(h / 41.0 + rng.uniform(0, 6)) + 0.35 * ou, 0, 1)
    p_h = np.clip(0.65 * solar + 0.45 * wind, 0, 1)
    t = np.arange(hours * MIN_PER_H) / MIN_PER_H
    p = np.interp(t, h, p_h)
    e = np.zeros_like(t)
    eps = rng.standard_normal(len(t)) * 0.03
    for k in range(1, len(t)):            # AR(1) minute fluctuation
        e[k] = 0.97 * e[k - 1] + eps[k]
    return np.clip(p + e * 0.25, 0, 1)


def schedule(spec: UnitSpec, hours: int) -> Dict[str, object]:
    """Mode per minute (0 off, 1 run, 2 diagnostic), set-points and bench events."""
    g = spec.group
    rng = _rng(spec.seed, 1)
    n = hours * MIN_PER_H
    mode = np.ones(n, np.int8)
    j = np.full(n, g.j_A_cm2)
    events: List[tuple] = []
    planned_off: List[tuple] = []        # (start_min, end_min, reason)
    # electrolyte refresh (drain/refill, 12 h off), staggered per unit
    for t in g.refresh_h:
        t0 = int((t + rng.uniform(-48, 48)) * MIN_PER_H)
        planned_off.append((t0, t0 + 12 * MIN_PER_H, "electrolyte_change"))
    # unplanned trips (power supply / utilities), 1-2 per unit
    for _ in range(int(rng.integers(1, 3))):
        t0 = int(rng.uniform(150, hours - 150) * MIN_PER_H)
        planned_off.append((t0, t0 + int(rng.uniform(2, 10) * MIN_PER_H), "trip"))
    if g.mode == "dynamic":
        p = renewable_minutes(hours, rng)
        req = g.j_A_cm2 * p
        on = np.zeros(n, bool)
        state, last = False, -10 ** 9
        for k in range(n):               # hysteresis + 30 min minimum on/off time
            if state and req[k] < 0.10 and k - last >= 30:
                state, last = False, k
            elif not state and req[k] > 0.15 and k - last >= 30:
                state, last = True, k
            on[k] = state
        mode = np.where(on, 1, 0).astype(np.int8)
        j = np.where(on, np.clip(req, 0.10, g.j_A_cm2), 0.0)
    for t0, t1, why in planned_off:
        mode[t0:t1] = 0
        j[t0:t1] = 0.0
    # diagnostics every 250 h from 24 h; forced (bench grid power), 3 h at reference conditions
    diag_start = []
    dur = BENCH["diag_duration_h"].value
    times = [BENCH["diag_first_h"].value] + list(np.arange(BENCH["diag_every_h"].value, hours - dur + 1e-9,
                                                           BENCH["diag_every_h"].value))
    for t in times:
        k0 = int(t * MIN_PER_H)
        k1 = k0 + int(dur * MIN_PER_H)
        if not any(a <= k0 < b for a, b, _ in planned_off):
            mode[k0:k1] = 2
            j[k0:k1] = 0.2
            if g.mode == "constant":       # depressurise first, then lower the current (avoids an HTO excursion)
                j[k0:k0 + 30] = g.j_A_cm2
            diag_start.append(k0)
    return dict(mode=mode, j=j, planned_off=planned_off, diag_start=diag_start)


def transition_events(unit: str, mode: np.ndarray, planned_off, diag_start) -> List[tuple]:
    ev = []
    run = mode > 0
    starts = np.nonzero(run[1:] & ~run[:-1])[0] + 1
    stops = np.nonzero(~run[1:] & run[:-1])[0] + 1
    reason = {}
    for a, b, why in planned_off:
        reason[a] = why
    for k in stops:
        why = reason.get(int(k))
        if why == "trip":
            ev.append((unit, k / MIN_PER_H, "trip", "unplanned shutdown (power supply / utilities)"))
        elif why == "electrolyte_change":
            ev.append((unit, k / MIN_PER_H, "shutdown", "planned: drain electrolyte"))
        else:
            ev.append((unit, k / MIN_PER_H, "shutdown", ""))
    for k in starts:
        ev.append((unit, k / MIN_PER_H, "startup", ""))
    for a, b, why in planned_off:
        if why == "electrolyte_change":
            ev.append((unit, (b - 1) / MIN_PER_H, "electrolyte_change", "fresh KOH at nominal concentration"))
    for k in diag_start:
        ev.append((unit, k / MIN_PER_H, "measurement", "diagnostics at 70 C, 1 bar: LSV, EIS, CV, titration"))
    return ev


# ------------------------------------------------------------------------------------------------ loop
def thermal_electrolyte_loop(spec: UnitSpec, mode, j, refresh_end_min):
    """Cell temperature, pressure, KOH wt% (water consumption + make-up) and carbonation, minute by minute."""
    g = spec.group
    n = len(mode)
    dt = 1.0 / MIN_PER_H
    a_h, a_c = 1 - np.exp(-dt / BENCH["tau_heat_h"].value), 1 - np.exp(-dt / BENCH["tau_cool_h"].value)
    a_p = 1 - np.exp(-dt / (10 / 60))
    Tamb = BENCH["T_amb_C"].value
    T = np.empty(n)
    P = np.empty(n)
    wt = np.empty(n)
    xco3 = np.empty(n)
    Tc, Pc = g.T_C, g.P_bar           # test starts warm and pressurised after conditioning
    M_koh = BENCH["lye_mass_kg"].value * g.koh_wt / 100
    M_water = BENCH["lye_mass_kg"].value - M_koh
    t_since_refresh = 0.0
    topups = []
    k_co3 = DEG["k_co3"].value / 1000.0
    refresh = set(refresh_end_min)
    area = DESIGN["area_cm2"].value
    band = BENCH["topup_band_wt"].value
    for k in range(n):
        m = mode[k]
        if k in refresh:
            M_water = BENCH["lye_mass_kg"].value - M_koh
            t_since_refresh = 0.0
        if m == 1:
            Tc += (g.T_C - Tc) * a_h
            Pc += (g.P_bar - Pc) * a_p
        elif m == 2:
            Tc += (BENCH["diag_T_C"].value - Tc) * a_h
            Pc += (BENCH["diag_P_bar"].value - Pc) * a_p
        else:
            Tc += (Tamb - Tc) * a_c
            Pc += (1.0 - Pc) * a_p
        T[k], P[k] = Tc, Pc
        if j[k] > 0:
            eF = float(ph.faraday_eff(j[k], Tc + 273.15))
            M_water -= eF * j[k] * area / (2 * ph.F) * M_W * 60.0      # water electrolysed in one minute
        w = 100 * M_koh / (M_koh + M_water)
        if w >= g.koh_wt + band:
            add = (M_koh * 100 / g.koh_wt - M_koh) - M_water
            M_water += add
            topups.append((k, add))
            w = 100 * M_koh / (M_koh + M_water)
        wt[k] = w
        t_since_refresh += dt
        xco3[k] = min(k_co3 * t_since_refresh, 0.5)
    return T, P, wt, xco3, topups


# ------------------------------------------------------------------------------------------------ degradation
def unit_draws(spec: UnitSpec) -> Dict[str, float]:
    rng = _rng(spec.seed, 2)
    cv = DEG["unit_cv"].value
    names = ["j0c_bol", "j0a_bol", "RFc_bol", "RFa_bol", "Rc_bol", "u_j0c", "u_bc", "u_hyd", "u_j0a", "u_ba", "u_RF",
             "u_sep", "u_contact", "u_D", "u_stop"]
    d = {k: float(np.exp(rng.normal(0, cv))) for k in names}
    for k in ("RFc_bol", "RFa_bol", "Rc_bol", "j0c_bol", "j0a_bol"):
        d[k] = float(np.exp(rng.normal(0, cv * 0.6)))      # BoL spread smaller than rate spread
    return d


def stop_doses(mode: np.ndarray):
    """Dose of every off period that follows operation: 1 - exp(-duration / tau)  [Uchino18, Kim22]."""
    run = mode > 0
    n = len(mode)
    dose = np.zeros(n)
    long_stop_end = []
    stops = np.nonzero(~run[1:] & run[:-1])[0] + 1
    starts = np.nonzero(run[1:] & ~run[:-1])[0] + 1
    for s in stops:
        nxt = starts[starts > s]
        e = int(nxt[0]) if len(nxt) else n
        dur_h = (e - s) / MIN_PER_H
        if e < n:
            dose[e] = 1 - np.exp(-dur_h / DEG["stop_tau_h"].value)
            if dur_h > 2.0:
                long_stop_end.append(e)
    return np.cumsum(dose), long_stop_end


def degradation_state(spec: UnitSpec, mode, j, T_C, refresh_end_min, draws, t_trip_min=None):
    """Hidden truth: all state multipliers per minute + mechanism bookkeeping."""
    n = len(mode)
    T_K = T_C + 273.15
    running = (mode > 0) & (j > 0)
    g = np.exp(-DEG["Ea_deg"].value / ph.R * (1 / T_K - 1 / DEG["T_ref_deg_K"].value)) * \
        np.power(np.maximum(j, 0) / DEG["j_ref_deg"].value, DEG["n_j"].value) * running
    t_eq = np.cumsum(g) / MIN_PER_H
    t_run = np.cumsum(running) / MIN_PER_H
    S, long_stop_end = stop_doses(mode)
    S_unit = S * draws["u_stop"]
    # reversible hydride: run hours since the last reset (long stop or refresh)
    resets = np.zeros(n, bool)
    resets[list(long_stop_end)] = True
    resets[[k for k in refresh_end_min if k < n]] = True
    seg = np.cumsum(resets)
    run_cum = np.cumsum(running) / MIN_PER_H
    base = np.zeros(seg.max() + 1)
    first_idx = np.r_[0, np.nonzero(resets)[0]]
    base[np.arange(len(first_idx))] = run_cum[first_idx] - running[first_idx] / MIN_PER_H
    t_since = run_cum - base[seg]
    f_hyd = 1 - DEG["a_hyd"].value * draws["u_hyd"] * (1 - np.exp(-t_since / DEG["tau_hyd_h"].value))
    f_act = 1 - 0.4 * np.exp(-t_run / DEG["activation_h"].value)
    # ECSA
    loss = DEG["dRF_max"].value * draws["u_RF"] * (1 - np.exp(-t_eq / DEG["tau_RF_h"].value))
    RFc_ecsa = 1 - loss
    RFc_stop = np.maximum(1 - DEG["stop_RF"].value * S_unit, 0.2)
    RFa_ecsa = 1 - loss
    RFc = DESIGN["RF0_cathode"].value * draws["RFc_bol"] * RFc_ecsa * RFc_stop
    RFa = DESIGN["RF0_anode"].value * draws["RFa_bol"] * RFa_ecsa
    # cathode intrinsic + stop damage
    c_intr = np.exp(-DEG["k_j0c"].value * draws["u_j0c"] * t_eq)
    c_stop = (1 - DEG["stop_j0c"].value) ** S_unit
    ac_f = 1 / (1 + DEG["k_bc"].value * draws["u_bc"] * t_eq)
    # anode with acceleration after the dose crosses D*
    D = t_eq + DEG["w_stop_h"].value * S
    Dstar = DEG["D_star_h"].value * draws["u_D"]
    on_idx = np.nonzero(D >= Dstar)[0]
    k_on = int(on_idx[0]) if len(on_idx) else None
    A_eq = t_eq.copy()
    if k_on is not None:
        A_eq = t_eq + (DEG["accel_factor"].value - 1) * np.maximum(t_eq - t_eq[k_on], 0)
    a_intr = np.exp(-DEG["k_j0a"].value * draws["u_j0a"] * A_eq)
    aa_f = 1 / (1 + DEG["k_ba"].value * draws["u_ba"] * A_eq)
    sep_f = 1 + DEG["k_sep"].value * draws["u_sep"] * t_eq
    Rc0 = DESIGN["R_contact0_ohm_cm2"].value * draws["Rc_bol"]
    Rc_growth = DEG["k_contact"].value * draws["u_contact"] * t_eq
    Rc_stop = DEG["stop_contact"].value * S_unit
    # faults
    t_h = np.arange(n) / MIN_PER_H
    f_c_fault = np.ones(n)
    ac_fault = np.ones(n)
    R_fault = np.zeros(n)
    x_f = np.ones(n)
    for f in FAULTS:
        if f["unit"] != spec.unit:
            continue
        ramp = np.clip((t_h - f["t_h"]) / (200.0 if f["kind"] == "fast_activity_loss" else 300.0), 0, 1)
        if f["kind"] == "resistance_step":
            R_fault = np.where(t_h >= f["t_h"], f["magnitude"], 0.0)
        elif f["kind"] == "fast_activity_loss":
            f_c_fault = f["magnitude"] ** ramp
            ac_fault = 1 / (1 + 0.15 * ramp)
        elif f["kind"] == "HTO_rise":
            x_f = 1 + (f["magnitude"] - 1) * ramp
    RFc_nom, RFa_nom = DESIGN["RF0_cathode"].value, DESIGN["RF0_anode"].value
    parts = dict(  # every factor separately, so mechanism contributions can be computed exactly
        j0c_bol=draws["j0c_bol"] * draws["RFc_bol"], j0a_bol=draws["j0a_bol"] * draws["RFa_bol"],
        c_activation=f_act, c_hydride=f_hyd, c_intrinsic=c_intr, c_ecsa=RFc_ecsa, c_stop=c_stop * RFc_stop,
        c_fault=f_c_fault, ac_intrinsic=ac_f, ac_fault=ac_fault, a_intrinsic=a_intr, a_ecsa=RFa_ecsa,
        aa_intrinsic=aa_f, sep_f=sep_f, Rc0=Rc0, Rc_growth=Rc_growth, Rc_stop=Rc_stop, Rc_fault=R_fault, x_f=x_f)
    state = compose(parts)
    extra = dict(t_eq=t_eq, t_run=t_run, S=S, D=D, Dstar=Dstar, k_on=k_on, RFc=RFc, RFa=RFa,
                 Cdl_c=DESIGN["C_specific_uF_cm2"].value * 1e-6 * RFc, Cdl_a=DESIGN["C_specific_uF_cm2"].value * 1e-6 * RFa)
    return state, parts, extra


def compose(p: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return dict(
        j0c_f=p["j0c_bol"] * p["c_activation"] * p["c_hydride"] * p["c_intrinsic"] * p["c_ecsa"] * p["c_stop"] * p["c_fault"],
        j0a_f=p["j0a_bol"] * p["a_intrinsic"] * p["a_ecsa"],
        ac_f=p["ac_intrinsic"] * p["ac_fault"], aa_f=p["aa_intrinsic"], sep_f=p["sep_f"],
        R_c=p["Rc0"] + p["Rc_growth"] + p["Rc_stop"] + p["Rc_fault"], x_f=p["x_f"])


# mechanism -> which factors it owns (others are set to their BoL value when isolating it)
MECHANISMS = {
    "cathode_activity_intrinsic": ["c_intrinsic", "ac_intrinsic"],
    "cathode_hydride_reversible": ["c_hydride"],
    "anode_activity_incl_acceleration": ["a_intrinsic", "aa_intrinsic"],
    "ECSA_loss": ["c_ecsa", "a_ecsa"],
    "intermittent_reverse_current": ["c_stop", "Rc_stop"],
    "separator_fouling": ["sep_f"],
    "contact_degradation": ["Rc_growth"],
    "electrolyte_carbonation_concentration": ["_electrolyte"],
    "conditioning_BoL": ["c_activation"],
    "injected_fault": ["c_fault", "ac_fault", "Rc_fault", "x_f"],
}
BOL_VALUE = {"c_activation": 1.0, "c_hydride": 1.0, "c_intrinsic": 1.0, "c_ecsa": 1.0, "c_stop": 1.0, "c_fault": 1.0,
             "ac_intrinsic": 1.0, "ac_fault": 1.0, "a_intrinsic": 1.0, "a_ecsa": 1.0, "aa_intrinsic": 1.0, "sep_f": 1.0,
             "Rc_growth": 0.0, "Rc_stop": 0.0, "Rc_fault": 0.0, "x_f": 1.0}


def mechanism_contributions(parts_h: Dict[str, np.ndarray], m_free_h, ref: Dict[str, float], m_nominal: float):
    """Voltage change at the reference condition caused by each mechanism alone (others at BoL), V.
    Also returns the total and the interaction residual (total - sum)."""
    T_K, P, jr = ref["T_C"] + 273.15, ref["P_bar"], ref["j"]
    n = len(m_free_h)

    def U(p, m):
        return ph.cell(jr, T_K, P, m, compose(p))["U"]

    bol = {k: (np.full(n, v) if k in BOL_VALUE else parts_h[k]) for k, v in
           list(BOL_VALUE.items()) + [(k, None) for k in ("j0c_bol", "j0a_bol", "Rc0")]}
    U0 = U(bol, m_nominal)
    out = {}
    for name, keys in MECHANISMS.items():
        p = dict(bol)
        m = m_nominal
        for key in keys:
            if key == "_electrolyte":
                m = m_free_h
            else:
                p[key] = parts_h[key]
        out[name] = U(p, m) - U0
    total = U(parts_h, m_free_h) - U0
    out["total"] = total
    out["interaction_residual"] = total - sum(out[k] for k in MECHANISMS)
    out["U_ref_BoL"] = U0
    return out


# ------------------------------------------------------------------------------------------------ one unit
def simulate_unit(spec: UnitSpec, hours: int) -> Dict[str, object]:
    sch = schedule(spec, hours)
    mode, j = sch["mode"].copy(), sch["j"].copy()
    refresh_end = [b for a, b, why in sch["planned_off"] if why == "electrolyte_change"]
    draws = unit_draws(spec)
    trip_min = None
    for _ in range(2):                                   # second pass only if the HTO interlock trips
        T_C, P, wt, xco3, topups = thermal_electrolyte_loop(spec, mode, j, refresh_end)
        T_K = T_C + 273.15
        m_K = ph.molarity_from_wt(wt, T_K)
        m_free = m_K * (1 - xco3)
        state, parts, extra = degradation_state(spec, mode, j, T_C, refresh_end, draws)
        gas = ph.gases(j, T_K, P, m_free, state["x_f"])
        if trip_min is not None:
            break
        lim = PHYS["HTO_shutdown_pct"].value
        over = np.nan_to_num(gas["HTO_pct"]) > lim
        run5 = np.convolve(over.astype(int), np.ones(5, int), "full")[:len(over)] >= 5
        if run5.any():
            trip_min = int(np.argmax(run5))
            mode[trip_min:] = 0
            j[trip_min:] = 0.0
            sch["diag_start"] = [k for k in sch["diag_start"] if k < trip_min]
        else:
            break
    c = ph.cell(j, T_K, P, m_free, state)
    on = j > 0
    # open-circuit voltage after a stop decays as the electrode charge is discharged (reverse current) [Uchino18]
    U = c["U"].copy()
    off = ~on
    if off.any():
        k_off_start = np.maximum.accumulate(np.where(off & np.r_[True, ~off[:-1]], np.arange(len(off)), 0))
        t_off = (np.arange(len(off)) - k_off_start) / MIN_PER_H
        U = np.where(off, c["E_rev"] * np.exp(-t_off / 1.5), U)
    Vtn = ph.V_tn(T_K, P, m_free)
    I = gas["I_A"]
    Q_gen = np.where(on, U * I - gas["eta_F"] * I * Vtn, 0.0)
    dTdt = np.gradient(T_C) * MIN_PER_H / 3600.0               # K/s
    Q_loss = BENCH["UA_W_K"].value * (T_C - BENCH["T_amb_C"].value)
    Q_store = BENCH["C_th_J_K"].value * dTdt
    mcp = BENCH["flow_L_min"].value / 60 * PHYS["rho_lye"].value * PHYS["cp_lye"].value     # W/K
    Q_lye = Q_gen - Q_loss - Q_store                            # heat leaving with the lye (negative = heater)
    T_in = T_C - Q_lye / mcp
    events = transition_events(spec.unit, mode, sch["planned_off"], sch["diag_start"])
    for k, add in topups:
        events.append((spec.unit, k / MIN_PER_H, "water_topup", f"{add * 1000:.0f} g deionised water"))
    if trip_min is not None:
        events = [e for e in events if not (e[2] in ("shutdown", "trip") and abs(e[1] - trip_min / MIN_PER_H) < 1e-9)]
        events.append((spec.unit, trip_min / MIN_PER_H, "trip", "HTO interlock: HTO > 2 % (shutdown limit); test ended"))
    return dict(spec=spec, mode=mode, j=j, T_C=T_C, T_in_C=T_in, P=P, wt=wt, xco3=xco3, m_free=m_free, U=U,
                cell=c, gas=gas, state=state, parts=parts, extra=extra, draws=draws, Q_gen=Q_gen, Q_loss=Q_loss,
                Q_store=Q_store, Q_lye=Q_lye, Vtn=Vtn, topups=topups, events=events, diag_start=sch["diag_start"],
                refresh_end=refresh_end, trip_min=trip_min, planned_off=sch["planned_off"])
