"""Degradation analysis for the AWE twin (no Streamlit): reversible / irreversible split, sawtooth (reset) model,
stage detection by piecewise-linear segmentation, AC-DC closure test (DRT -> voltage) and a DRT-informed Kalman twin
with mechanism attribution. All functions are descriptive (offline) unless stated causal; nothing here is used to
build forecasting features, so the no-look-ahead rule of the forecasting module is not affected.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import theilslopes
from sklearn.isotonic import IsotonicRegression

__version__ = "0.3.0"

MECHANISMS = ["ohmic + ion transport", "HER (cathode)", "OER + passive film (anode)", "mass transport / gas"]


# ----------------------------------------------------------------------------------------------- series helpers
def unit_series(W: pd.DataFrame, unit: str, bin_h: float = 1.0, events: Optional[pd.DataFrame] = None,
                exclude_after_measurement_h: float = 4.0) -> pd.DataFrame:
    """Stable windows of one unit, binned to `bin_h` hours (median). Windows within `exclude_after_measurement_h`
    after a logged 'measurement' pause are dropped (restart relaxation is a known, benign transient)."""
    d = W[(W["unit"] == unit) & W["stable"]].copy()
    if events is not None and len(events) and exclude_after_measurement_h:
        mt = events.loc[(events["unit"] == unit) & (events["event"] == "measurement"), "time_h"].to_numpy()
        if len(mt):
            dt = d["t_end_h"].to_numpy()[:, None] - (mt[None, :] + 1.0)
            d = d[~((dt >= 0) & (dt < exclude_after_measurement_h)).any(axis=1)]
    if not len(d):
        return pd.DataFrame()
    d["b"] = np.floor(d["t_end_h"] / bin_h)
    g = d.groupby("b").agg(t_h=("t_end_h", "median"), op_hours=("op_hours", "max"), U=("U_cell_V", "median"),
                           dV_mV=("dV_mV", "median"), T_C=("T_C", "median")).reset_index(drop=True)
    return g


def detect_resets(t: np.ndarray, y: np.ndarray, min_drop_mV: float = 8.0, before: int = 6, after: int = 3,
                  min_gap_h: float = 15.0, t_start: float = 50.0, events: Optional[pd.DataFrame] = None,
                  unit: Optional[str] = None, T: Optional[np.ndarray] = None, event_tol_h: float = 8.0) -> pd.DataFrame:
    """Sudden sustained drops (offline): median of the next `after` points below the median of the previous `before`
    points by more than min_drop_mV, after `t_start` (activation excluded). Each drop gets a cause: a logged
    KOH refill / electrolyte change / service ('electrolyte or service recovery'), a logged power-supply change
    ('instrument offset'), a temperature excursion ('operating condition'), otherwise 'electrochemical reset'."""
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    out, last = [], -np.inf
    ev = events[events["unit"] == unit] if events is not None and len(events) and unit is not None else None
    for i in range(before, len(y) - after):
        if t[i] < t_start:
            continue
        pre = np.median(y[i - before:i])
        post = np.median(y[i:i + after])
        if pre - post > min_drop_mV and t[i] - last > min_gap_h:
            # refine: largest single-step drop in the neighbourhood
            k = i - 2 + int(np.argmin(np.diff(y[max(i - 3, 0):i + 3]))) if i > 3 else i
            th = float(t[k + 1] if k + 1 < len(t) else t[i])
            cause = "electrochemical reset"
            if ev is not None:
                near = ev[(ev["time_h"] - th).abs() <= event_tol_h]["event"].tolist()
                if any(e in ("koh_refill", "electrolyte_change", "stack_service") for e in near):
                    cause = "electrolyte or service recovery"
                elif "power_supply_change" in near:
                    cause = "instrument offset"
            if cause == "electrochemical reset" and T is not None:
                Tm = np.median(T)
                if np.max(np.abs(np.asarray(T)[max(i - 2, 0):i + after] - Tm)) > 2.0:
                    cause = "operating condition (temperature)"
            out.append(dict(time_h=th, drop_mV=float(pre - post), level_before=float(pre), level_after=float(post),
                            cause=cause))
            last = t[i]
    return pd.DataFrame(out, columns=["time_h", "drop_mV", "level_before", "level_after", "cause"])


def reversible_irreversible(t: np.ndarray, y: np.ndarray, resets: pd.DataFrame, t_start: float = 50.0) -> Dict[str, object]:
    """Irreversible baseline = non-decreasing (isotonic) line through the post-reset levels (and the post-activation
    minimum); reversible = signal minus baseline. Rates by Theil-Sen with bootstrap CI over anchor points."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = t >= t_start
    if m.sum() < 10:
        return dict(ok=False)
    anchors_t = [float(t[m][int(np.argmin(y[m][:max(int(m.sum() * 0.1), 5)]))])]
    anchors_y = [float(np.min(y[m][:max(int(m.sum() * 0.1), 5)]))]
    for r in resets.itertuples():
        if r.time_h > t_start:
            anchors_t.append(float(r.time_h))
            anchors_y.append(float(r.level_after))
    at, ay = np.array(anchors_t), np.array(anchors_y)
    if len(at) >= 2:
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(at, ay)
        base = np.interp(t, at, iso.predict(at))
        base[t > at[-1]] = iso.predict(at)[-1]
        slope = theilslopes(ay, at)[0]
        rng = np.random.default_rng(0)
        bs = [theilslopes(ay[i], at[i])[0] for i in (rng.choice(len(at), len(at)) for _ in range(300))
              if len(set(at[i])) > 1]
        ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if bs else (np.nan, np.nan)
    else:
        base = np.full_like(y, ay[0])
        slope, ci = np.nan, (np.nan, np.nan)
    rev = y - base
    return dict(ok=True, t=t, baseline=base, reversible=rev, anchors=pd.DataFrame(dict(time_h=at, level=ay)),
                irreversible_rate_mV_h=float(slope), irreversible_ci=ci,
                reversible_share_end=float(rev[-1] / max(y[-1] - y[m][0], 1e-9)) if len(y) else np.nan)


def sawtooth_model(t, y, resets: pd.DataFrame, t_start: float = 50.0) -> Dict[str, object]:
    """Cycle table between resets: growth rate (Theil-Sen, mV/h), duration, drop, reversible level at the reset.
    Trends over cycles (does regrowth accelerate? do intervals shrink?) and a next-reset predictor:
    t_next = t_now + (mean reset level - current reversible level) / current growth rate."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    edges = [t_start] + [float(x) for x in resets["time_h"] if x > t_start] + [float(t[-1])]
    rows = []
    for k in range(len(edges) - 1):
        m = (t >= edges[k] + 1) & (t < edges[k + 1] - 0.5)
        if m.sum() < 5:
            continue
        s = theilslopes(y[m], t[m])[0]
        rows.append(dict(cycle=k + 1, start_h=edges[k], end_h=edges[k + 1], duration_h=edges[k + 1] - edges[k],
                         growth_mV_h=float(s), ended_by_reset=k < len(edges) - 2,
                         drop_mV=float(resets["drop_mV"].iloc[k]) if k < len(resets) else np.nan))
    cyc = pd.DataFrame(rows)
    trend = {}
    if len(cyc) >= 3:
        trend["growth_trend_per_cycle"] = float(np.polyfit(cyc["cycle"], np.log(np.maximum(cyc["growth_mV_h"], 1e-6)), 1)[0])
        c2 = cyc[cyc["ended_by_reset"]]
        if len(c2) >= 3:
            trend["interval_trend_per_cycle"] = float(np.polyfit(c2["cycle"], np.log(c2["duration_h"]), 1)[0])
    return dict(cycles=cyc, trend=trend)


def predict_next_reset(cycles: pd.DataFrame, rev_level_now: float, t_now: float, growth_now: float,
                       reset_levels: Sequence[float]) -> float:
    if growth_now <= 0 or not len(reset_levels):
        return float("nan")
    return t_now + max(float(np.mean(reset_levels)) - rev_level_now, 0) / growth_now


# ----------------------------------------------------------------------------------------------- stages
def segment_piecewise_linear(t: np.ndarray, y: np.ndarray, penalty: Optional[float] = None, min_size: int = 4,
                             max_points: int = 600, penalty_factor: float = 10.0) -> pd.DataFrame:
    """Optimal segmentation into linear pieces with free jumps (dynamic programming, cost = SSE, BIC-type penalty
    penalty_factor * sigma^2 * ln n per segment). Offline stage detection."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    if len(t) > max_points:
        idx = np.linspace(0, len(t) - 1, max_points).astype(int)
        t, y = t[idx], y[idx]
    n = len(t)
    sig2 = float(np.median(np.diff(y) ** 2) / 2) if n > 2 else 1.0
    pen = penalty if penalty is not None else penalty_factor * max(sig2, 1e-9) * math.log(n)
    S1, St, Stt, Sy, Sty, Syy = (np.r_[0, np.cumsum(a)] for a in (np.ones(n), t, t * t, y, t * y, y * y))

    def cost(i, j):          # SSE of the line fit on [i, j)
        m = S1[j] - S1[i]
        st, stt, sy, sty, syy = St[j] - St[i], Stt[j] - Stt[i], Sy[j] - Sy[i], Sty[j] - Sty[i], Syy[j] - Syy[i]
        den = m * stt - st * st
        if den <= 1e-12:
            return syy - sy * sy / max(m, 1)
        b = (m * sty - st * sy) / den
        a = (sy - b * st) / m
        return max(syy - 2 * a * sy - 2 * b * sty + m * a * a + 2 * a * b * st + b * b * stt, 0.0)

    F = np.full(n + 1, np.inf)
    F[0] = -pen
    last = np.zeros(n + 1, int)
    for j in range(min_size, n + 1):
        best, arg = np.inf, 0
        for i in range(0, j - min_size + 1):
            if not np.isfinite(F[i]):
                continue
            c = F[i] + cost(i, j) + pen
            if c < best:
                best, arg = c, i
        F[j], last[j] = best, arg
    cps, j = [], n
    while j > 0:
        cps.append((last[j], j))
        j = last[j]
    rows = []
    for i, j in reversed(cps):
        b = np.polyfit(t[i:j], y[i:j], 1) if j - i >= 2 else (0.0, y[i])
        rows.append(dict(start_h=float(t[i]), end_h=float(t[j - 1]), slope_mV_h=float(b[0]),
                         level_start=float(np.polyval(b, t[i])), level_end=float(np.polyval(b, t[j - 1])), n=j - i))
    return pd.DataFrame(rows)


def label_stages(seg: pd.DataFrame, jump_mV: float = 8.0, terminal_factor: float = 3.0,
                 resets: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Name the segments: activation (falling at the start), passivation (rising, before the first reset),
    cyclic spalling (after a reset), terminal (slope > terminal_factor x the median rising slope, last third)."""
    if seg is None or not len(seg):
        return seg
    seg = seg.copy()
    pos = seg.loc[seg["slope_mV_h"] > 0, "slope_mV_h"]
    med = float(pos.median()) if len(pos) else 0.0
    T = float(seg["end_h"].max())
    labels, seen_reset = [], False
    first_reset = np.inf
    if resets is not None and len(resets):
        ec = resets[resets.get("cause", pd.Series("electrochemical reset", index=resets.index)) == "electrochemical reset"]
        first_reset = float(ec["time_h"].min()) if len(ec) else np.inf
    for k, r in seg.iterrows():
        jump = (r["level_start"] - seg.loc[k - 1, "level_end"]) if k > 0 else 0.0
        if resets is None and jump < -jump_mV and r["start_h"] > 0.12 * T:
            seen_reset = True
        if r["start_h"] >= first_reset - 2:
            seen_reset = True
        if r["start_h"] < 0.12 * T and r["slope_mV_h"] < 0:
            labels.append("1 activation")
        elif med > 0 and r["slope_mV_h"] > terminal_factor * med and r["start_h"] > 0.66 * T:
            labels.append("4 terminal")
        elif seen_reset:
            labels.append("3 cyclic spalling")
        else:
            labels.append("2 passivation")
        seg.loc[k, "jump_mV"] = jump
    seg["stage"] = labels
    return seg


# ----------------------------------------------------------------------------------------------- AC-DC closure
def _integrate_powerlaw(j: np.ndarray, R: np.ndarray, j_from: float, j_to: float) -> float:
    """Integral of R(j) dj with R interpolated as a power law between measured currents (exact for R ~ a j^b)."""
    o = np.argsort(j)
    j, R = np.asarray(j, float)[o], np.maximum(np.asarray(R, float)[o], 1e-12)
    total = 0.0
    for k in range(len(j) - 1):
        a, b = max(j[k], j_from), min(j[k + 1], j_to)
        if b <= a:
            continue
        p = math.log(R[k + 1] / R[k]) / math.log(j[k + 1] / j[k])
        c = R[k] / j[k] ** p
        total += c * (math.log(b / a) if abs(p + 1) < 1e-9 else (b ** (p + 1) - a ** (p + 1)) / (p + 1))
    return total


def closure_test(tracked: pd.DataFrame, spectra: pd.DataFrame, polcurve: Optional[pd.DataFrame], series: pd.DataFrame,
                 unit: str, j_op: float, peak_map: Optional[Dict[str, str]] = None, pre_h: float = 3.0,
                 t_ref: float = 50.0, cp_window: str = "post", j_start: Optional[float] = None,
                 require_all_peaks: bool = True, runaway_mV: float = 150.0) -> Dict[str, object]:
    """AC-DC closure: the voltage at the operating current predicted from impedance alone,
        U(j_op, t) = U_LSV(j_min, t) + int_{j_min}^{j_op} [R_inf + sum_k R_k(j)] dj,
    with R_k(j) from the DRT at several EIS currents (power-law interpolation), compared with the measured
    chronopotentiometry voltage right after the EIS pause (cp_window='post', restart transient excluded) or just before
    it ('pre'). A residual that persists after a logged power-supply change reveals an instrument offset. Also per-mechanism attribution of the change since the first
    EIS time, and the single-current approximation dU ~ j_op * dR_tot(j_op) for comparison.
    peak_map: tracked peak label -> mechanism name (default P1..P4 in MECHANISMS order).
    j_start: lower integration limit (default: lowest EIS current). Starting at a higher current avoids the lowest-current
    spectra where kinetic peaks overlap most; the measured LSV voltage at j_start carries the rest, and its change is
    split between mechanisms in proportion to j_start * dR_k(j_start) (exact for film-type, current-independent
    resistance changes; approximate otherwise)."""
    tr = tracked[tracked["unit"] == unit]
    sp = spectra[(spectra["unit"] == unit) & spectra["kk_valid"]]
    if not len(tr) or tr["j_A_cm2"].nunique() < 2:
        return dict(ok=False, reason="needs DRT at >= 2 EIS currents")
    peak_map = peak_map or {f"P{k + 1}": m for k, m in enumerate(MECHANISMS)}
    expected_all = [pk for pk in peak_map if (tr["peak"] == pk).any()]
    ntimes = max(tr["time_h"].nunique(), 1)
    # currents at which every mechanism peak is resolved in most spectra (merged peaks at low current are excluded)
    full = sorted(j for j in tr["j_A_cm2"].dropna().unique()
                  if all(tr[(tr["j_A_cm2"] == j) & (tr["peak"] == pk)]["time_h"].nunique() >= 0.5 * ntimes
                         for pk in expected_all))
    if len(full) < 2:
        return dict(ok=False, reason="fewer than 2 EIS currents where all mechanism peaks are resolved")
    rows = []
    for t, d in tr.groupby("time_h"):
        d = d[d["j_A_cm2"].isin(full)]
        js = sorted(d["j_A_cm2"].unique())
        s = sp[(sp["time_h"] == t) & sp["j_A_cm2"].isin(full)]
        if js != full or sorted(s["j_A_cm2"].unique()) != full:     # only complete, KK-valid current sets
            continue
        jmin = min(js) if j_start is None else max(min(js), min(j_start, max(js)))
        contrib = {m: 0.0 for m in MECHANISMS}
        r_at = {m: 0.0 for m in MECHANISMS}
        rinf = s.groupby("j_A_cm2")["R_inf"].median()
        contrib[MECHANISMS[0]] += _integrate_powerlaw(rinf.index.to_numpy(), rinf.to_numpy(), jmin, j_op)
        r_at[MECHANISMS[0]] += float(np.exp(np.interp(np.log(jmin), np.log(rinf.index), np.log(np.maximum(rinf, 1e-12)))))
        complete = True
        expected = [pk for pk in peak_map if (tr["peak"] == pk).any()]
        present = all(set(d.loc[d["peak"] == pk, "j_A_cm2"]) >= set(full) for pk in expected)
        if require_all_peaks and not present:      # a missing peak means it merged with a neighbour: skip this time
            continue
        for pk, mech in peak_map.items():
            q = d[d["peak"] == pk].groupby("j_A_cm2")["R"].sum().reindex(full).fillna(0.0)
            if (q > 0).sum() < 2:
                continue
            q = q[q > 0]
            contrib[mech] += _integrate_powerlaw(q.index.to_numpy(), q.to_numpy(), jmin, j_op)
            r_at[mech] += float(np.exp(np.interp(np.log(jmin), np.log(q.index), np.log(q.to_numpy()))))
        U_low = np.nan
        if polcurve is not None and len(polcurve):
            pc = polcurve[(polcurve["unit"] == unit) & (polcurve["time_h"] == t)].sort_values("j_A_cm2")
            if len(pc) >= 2:
                U_low = float(np.interp(jmin, pc["j_A_cm2"], pc["voltage_V"]))
        Rtot_op = float(rinf.reindex([j_op]).fillna(rinf.iloc[-1]).iloc[0]) + float(
            d[d["j_A_cm2"] == max(js)]["R"].sum())
        if cp_window == "post":     # after the pause (the EIS state, incl. a refill done during the pause)
            meas = series[(series["t_h"] > t) & (series["t_h"] <= t + 5 + pre_h)]["U"].median() if len(series) else np.nan
        else:
            meas = series[(series["t_h"] < t) & (series["t_h"] >= t - pre_h)]["U"].median() if len(series) else np.nan
        rows.append(dict(time_h=t, j_min=jmin, U_low=U_low, complete=complete, R_tot_op=Rtot_op, U_meas=float(meas),
                         **{f"int_{m}": v for m, v in contrib.items()}, **{f"Rj_{m}": v for m, v in r_at.items()}))
    df = pd.DataFrame(rows)
    if not len(df):
        return dict(ok=False, reason="no EIS time with >= 2 valid currents")
    ints = [f"int_{m}" for m in MECHANISMS]
    df["U_pred"] = df["U_low"] + df[ints].sum(axis=1)
    df = df[(df["time_h"] >= t_ref) & df["U_meas"].notna()].reset_index(drop=True)
    if not len(df):
        return dict(ok=False, reason="no complete EIS time after the reference time")
    b = df.iloc[0]
    df["dU_meas_mV"] = 1000 * (df["U_meas"] - b["U_meas"])
    df["dU_pred_mV"] = 1000 * (df["U_pred"] - b["U_pred"])
    df["dU_single_mV"] = 1000 * j_op * (df["R_tot_op"] - b["R_tot_op"])
    low = 1000 * (df["U_low"] - b["U_low"])
    w = pd.DataFrame({m: df["j_min"] * (df[f"Rj_{m}"] - b[f"Rj_{m}"]) for m in MECHANISMS})
    wsum = w.sum(axis=1)
    for m in MECHANISMS:
        share = (w[m] / wsum).where(wsum.abs() > 1e-6, 0.25).clip(-1, 2)
        df[f"d_{m}_mV"] = 1000 * (df[f"int_{m}"] - b[f"int_{m}"]) + share * low
    df["d_low-current part_mV"] = low
    # near the limiting current (terminal gas suffocation) the small-signal impedance no longer integrates to the
    # DC voltage: such rows are flagged and scored separately (the breakdown itself marks the terminal regime)
    df["nonlinear_regime"] = df["dU_meas_mV"].abs() > runaway_mV
    ok = df["dU_meas_mV"].notna() & df["dU_pred_mV"].notna() & ~df["nonlinear_regime"]
    err = (df.loc[ok, "dU_pred_mV"] - df.loc[ok, "dU_meas_mV"])
    err1 = (df.loc[ok, "dU_single_mV"] - df.loc[ok, "dU_meas_mV"])
    return dict(ok=True, table=df, rmse_mV=float(np.sqrt(np.mean(err ** 2))) if len(err) else np.nan,
                n_nonlinear=int(df["nonlinear_regime"].sum()),
                rmse_single_mV=float(np.sqrt(np.mean(err1 ** 2))) if len(err1) else np.nan,
                abs_pred_vs_meas_mV=float(1000 * np.nanmedian(df["U_pred"] - df["U_meas"])))


# ----------------------------------------------------------------------------------------------- DRT-informed twin
def drt_informed_twin(series: pd.DataFrame, closure: pd.DataFrame, events: Optional[pd.DataFrame] = None,
                      unit: Optional[str] = None, use_drt: bool = True, q_level: float = 0.002, q_rate: float = 2e-5,
                      q_resid: float = 0.02, r_cp_mV: float = 2.0, r_drt_mV: float = 3.0,
                      hardware_events=("power_supply_change",), q_offset_mV: float = 30.0,
                      adaptive: bool = True) -> pd.DataFrame:
    """Kalman filter with, per mechanism m, a level x_m (mV, change since the reference EIS) and a rate r_m (mV/h),
    an instrument-offset state that may jump only at logged hardware events, and a slowly varying closure residual
    (voltage change NOT explained by the impedance: integration/model error, unlogged offsets).
    Observations: every CP window dU = sum_m x_m + offset + residual; at each EIS time the closure gives each x_m.
    Between EIS times the mechanisms move with their learned rates; with adaptive=True the process noise of each
    mechanism is scaled by its recent DRT activity (squared change between successive EIS), so voltage changes seen
    between EIS are attributed to the mechanisms that have been changing, not split evenly. Without DRT (use_drt=False) only the sum is
    observed, so the attribution is not identifiable (the posterior spread shows it). Causal (filter only)."""
    nm = len(MECHANISMS)
    nx = 2 * nm + 2                       # levels, rates, offset, residual
    x = np.zeros(nx)
    P = np.diag([25.0] * nm + [1e-2] * nm + [1e-6, 1.0])
    ev_t = np.array([]) if events is None or not len(events) else events.loc[
        (events["unit"] == unit) & events["event"].isin(hardware_events), "time_h"].to_numpy()
    U0 = float(closure["U_meas"].iloc[0]) if closure is not None and len(closure) else float(series["U"].iloc[0])
    ctimes = closure["time_h"].to_numpy() if closure is not None and len(closure) else np.array([])
    used = set()
    rows = []
    act = np.ones(nm)                     # adaptive activity weights from successive DRT observations
    last_obs = {}
    tprev = float(series["t_h"].iloc[0])
    for r in series.itertuples(index=False):
        dt = max(r.t_h - tprev, 0.0)
        tprev = r.t_h
        Fm = np.eye(nx)
        for m in range(nm):
            Fm[m, nm + m] = dt
        wq = act / act.mean() if adaptive else np.ones(nm)
        Q = np.diag(list(q_level * dt * wq) + list(q_rate * dt * wq) + [0.0, q_resid * dt])
        if len(ev_t) and np.any((ev_t <= r.t_h) & (ev_t > r.t_h - max(dt, 1.0))):
            Q[2 * nm, 2 * nm] = q_offset_mV ** 2
        x = Fm @ x
        P = Fm @ P @ Fm.T + Q
        H = [np.r_[np.ones(nm), np.zeros(nm), 1.0, 1.0]]
        z = [1000 * (r.U - U0)]
        Rr = [r_cp_mV ** 2]
        if use_drt:
            for k, tc in enumerate(ctimes):
                if k not in used and tc <= r.t_h:
                    used.add(k)
                    row = closure.iloc[k]
                    for m, mech in enumerate(MECHANISMS):
                        v = row.get(f"d_{mech}_mV", np.nan)
                        if np.isfinite(v) and mech in last_obs:
                            act[m] = 0.5 * act[m] + 0.5 * (abs(v - last_obs[mech]) + 0.5) ** 2
                        if np.isfinite(v):
                            last_obs[mech] = v
                            h = np.zeros(nx)
                            h[m] = 1.0
                            H.append(h)
                            z.append(v)
                            Rr.append(r_drt_mV ** 2)
        H, z, Rr = np.array(H), np.array(z), np.diag(Rr)
        S = H @ P @ H.T + Rr
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (z - H @ x)
        P = (np.eye(nx) - K @ H) @ P
        out = dict(t_h=r.t_h, dU_meas_mV=z[0], offset_mV=x[2 * nm], residual_mV=x[2 * nm + 1])
        for m, mech in enumerate(MECHANISMS):
            out[mech] = x[m]
            out[f"{mech}_sd"] = math.sqrt(max(P[m, m], 0))
            out[f"{mech}_rate"] = x[nm + m]
        rows.append(out)
    return pd.DataFrame(rows)


def mechanism_forecast(twin: pd.DataFrame, horizon_h: float, steps: int = 20) -> pd.DataFrame:
    """Linear extrapolation of each mechanism from its filtered level and rate (with level uncertainty)."""
    last = twin.iloc[-1]
    h = np.linspace(0, horizon_h, steps)
    rows = []
    for mech in MECHANISMS:
        for hh in h:
            rows.append(dict(mechanism=mech, t_h=last["t_h"] + hh, level=last[mech] + last[f"{mech}_rate"] * hh,
                             sd=last[f"{mech}_sd"]))
    return pd.DataFrame(rows)


def next_reset_backtest(series_by_unit: Dict[str, pd.DataFrame], resets_by_unit: Dict[str, pd.DataFrame],
                        t_start: float = 50.0, at_share: float = 0.5) -> pd.DataFrame:
    """Causal test of the sawtooth predictor: for every cycle that ended in an electrochemical reset, stand at
    `at_share` of the cycle, estimate the current growth rate from that cycle's data so far, take the mean reset level
    (reversible part at reset) from the OTHER units and this unit's earlier cycles, and predict the reset time."""
    rows = []
    levels = {}
    for u, ser in series_by_unit.items():
        rs = resets_by_unit.get(u, pd.DataFrame())
        rs = rs[rs["cause"] == "electrochemical reset"] if len(rs) and "cause" in rs else rs
        ri = reversible_irreversible(ser["t_h"].to_numpy(), ser["dV_mV"].to_numpy(), resets_by_unit.get(u, pd.DataFrame(
            columns=["time_h", "level_after"])), t_start)
        if not ri.get("ok"):
            continue
        lv = []
        for r in rs.itertuples():
            k = int(np.argmin(np.abs(ri["t"] - r.time_h)))
            lv.append(dict(time_h=r.time_h, rev_before=float(ri["reversible"][max(k - 2, 0)])))
        levels[u] = (ri, pd.DataFrame(lv), rs)
    for u, (ri, lv, rs) in levels.items():
        edges = [t_start] + list(rs["time_h"])
        for k in range(1, len(edges)):
            a, b = edges[k - 1], edges[k]
            tnow = a + at_share * (b - a)
            m = (ri["t"] > a + 1) & (ri["t"] <= tnow)
            if m.sum() < 4:
                continue
            g = theilslopes(ri["reversible"][m], ri["t"][m])[0]
            pool = [x for v, (_, l2, _) in levels.items() if v != u for x in l2["rev_before"]] + \
                   [x for t_, x in zip(lv["time_h"], lv["rev_before"]) if t_ < a]
            if not pool:
                continue
            rev_now = float(np.median(ri["reversible"][m][-3:]))
            pred = predict_next_reset(None, rev_now, tnow, g, pool)
            rows.append(dict(unit=u, cycle=k, standing_at_h=tnow, actual_reset_h=b, predicted_reset_h=pred,
                             error_h=pred - b if np.isfinite(pred) else np.nan, growth_mV_h=g,
                             mean_reset_level_mV=float(np.mean(pool))))
    return pd.DataFrame(rows)


def replicate_summary(per_unit: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Mean, SD and 95 % CI (Student t) across replicate cells, with n."""
    from scipy.stats import t as tdist
    rows = []
    for c in cols:
        x = per_unit[c].dropna().to_numpy(float)
        n = len(x)
        mu = float(np.mean(x)) if n else np.nan
        sd = float(np.std(x, ddof=1)) if n > 1 else np.nan
        h = float(tdist.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else np.nan
        rows.append(dict(quantity=c, n=n, mean=mu, sd=sd, ci95_low=mu - h if n > 1 else np.nan,
                         ci95_high=mu + h if n > 1 else np.nan))
    return pd.DataFrame(rows)
