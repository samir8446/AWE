"""Anomaly detection for the AWE twin (no Streamlit). Three layers, all CAUSAL (an alarm at time t uses data up to t):
 1. twin residuals: local-trend Kalman filter on the window voltage; normalised innovation squared (NIS) test,
    two-sided CUSUM and EWMA on the standardised innovations;
 2. multivariate: PCA T^2 / Q (SPE) and an isolation forest on per-window features, trained on an initial
    reference period only;
 3. event-aware classification: alarms coinciding with logged hardware events, temperature excursions, isolated
    sensor glitches, downward resets or sustained upward drift.
Evaluation against labelled anomalies: detection rate per kind, detection delay, false alarms per 100 h and the
lead time before a terminal failure.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

__version__ = "0.3.0"

HARDWARE_EVENTS = ("power_supply_change", "koh_refill", "electrolyte_change", "stack_service", "trip",
                   "gasket_failure", "measurement", "startup", "shutdown")


def window_features(ts: pd.DataFrame, W: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Per stable window: voltage, within-window voltage spread (sensor glitches), temperature and current deviation
    from their running medians, time since the last logged event."""
    d = W[(W["unit"] == unit) & W["stable"]].copy().reset_index(drop=True)
    raw = ts[ts["unit"] == unit]
    if len(raw) and len(d):
        t = raw["time_h"].to_numpy()
        u = raw["voltage_V"].to_numpy()
        spread = []
        for a, b in zip(d["t_start_h"], d["t_end_h"]):
            m = (t >= a) & (t <= b)
            x = u[m]
            x = x[np.isfinite(x)]
            spread.append(float(np.max(np.abs(x - np.median(x)))) if len(x) else 0.0)
        d["U_spread_mV"] = 1000 * np.array(spread)
    else:
        d["U_spread_mV"] = 0.0
    d["T_dev"] = d["T_C"] - d["T_C"].rolling(48, min_periods=1).median().shift(1).fillna(d["T_C"].iloc[0] if len(d) else 0)
    d["j_dev"] = d["j_A_cm2"] - d["j_A_cm2"].rolling(48, min_periods=1).median().shift(1).fillna(
        d["j_A_cm2"].iloc[0] if len(d) else 0)
    return d


def residual_layer(d: pd.DataFrame, q_level: float = 0.01, q_slope: float = 1e-5, r_mV: float = 2.0,
                   nis_p: float = 0.999, cusum_k: float = 0.5, cusum_h: float = 8.0, ewma_l: float = 0.2,
                   ewma_L: float = 4.0, reinit_after: int = 3, ignore_before_h: float = 50.0) -> pd.DataFrame:
    """Local linear trend Kalman filter on U (mV): x = [level, slope]. Innovation z = (y - y_pred)/sqrt(S).
    Alarms: NIS > chi2_1(nis_p); CUSUM+/- > cusum_h; |EWMA| > ewma_L * sigma_EWMA. Outlier-robust: a window with
    NIS above the threshold is not used to update the state (so one glitch does not drag the twin); after
    `reinit_after` consecutive rejections the shift is accepted as a new level (step change)."""
    y = 1000 * d["U_cell_V"].to_numpy()
    th = d["t_end_h"].to_numpy()
    x = np.array([y[0], 0.0])
    P = np.diag([25.0, 1e-2])
    nis_thr = chi2.ppf(nis_p, 1)
    gp = gn = ew = 0.0
    n_rej = 0
    started = ignore_before_h <= th[0]
    sig_e = math.sqrt(ewma_l / (2 - ewma_l))
    rows = []
    tp = th[0]
    for k in range(len(y)):
        dt = max(th[k] - tp, 1e-3)
        tp = th[k]
        Fm = np.array([[1.0, dt], [0.0, 1.0]])
        Q = np.diag([q_level * dt, q_slope * dt])
        x = Fm @ x
        P = Fm @ P @ Fm.T + Q
        S = P[0, 0] + r_mV ** 2
        nu = y[k] - x[0]
        z = nu / math.sqrt(S)
        nis = z * z
        if nis <= nis_thr or k < 5:
            K = P[:, 0] / S
            x = x + K * nu
            P = P - np.outer(K, P[0, :])
            n_rej = 0
        else:
            n_rej += 1
            if n_rej >= reinit_after:        # persistent level shift: accept it and restart the filter there
                x = np.array([y[k], x[1]])
                P = np.diag([r_mV ** 2, P[1, 1] + 1e-3])
                n_rej = 0
        gp = max(0.0, gp + z - cusum_k)
        gn = max(0.0, gn - z - cusum_k)
        ew = ewma_l * z + (1 - ewma_l) * ew
        live = th[k] >= ignore_before_h                  # e.g. skip the activation stage
        if live and not started:                         # restart the detector after the ignored period
            started = True
            x = np.array([y[k], 0.0])
            P = np.diag([r_mV ** 2, 1e-2])
            gp = gn = ew = 0.0
            nu, z, nis = 0.0, 0.0, 0.0
        a_cus = live and (gp > cusum_h or gn > cusum_h)
        rows.append(dict(t_h=th[k], U_mV=y[k], pred_mV=y[k] - nu, z=z, nis=nis,
                         cusum_pos=gp, cusum_neg=gn, ewma=ew, slope_mV_h=x[1],
                         alarm_nis=live and nis > nis_thr, alarm_cusum=a_cus,
                         alarm_ewma=live and abs(ew) > ewma_L * sig_e))
        if a_cus:
            gp = gn = 0.0          # restart the CUSUM after an alarm
    return pd.DataFrame(rows)


def multivariate_layer(d: pd.DataFrame, res: pd.DataFrame, train_h: float = 200.0, train_from_h: float = 60.0, quantile: float = 0.995,
                       seed: int = 0) -> pd.DataFrame:
    """PCA T^2 / Q and isolation forest on [innovation z, slope, voltage spread, T deviation, j deviation], trained on
    windows before `train_h` (reference period, assumed healthy); thresholds = training quantiles."""
    X = pd.DataFrame(dict(z=res["z"].to_numpy(), slope=res["slope_mV_h"].to_numpy(), spread=d["U_spread_mV"].to_numpy(),
                          T_dev=d["T_dev"].to_numpy(), j_dev=d["j_dev"].to_numpy())).fillna(0.0)
    tr = ((res["t_h"] < train_h) & (res["t_h"] >= train_from_h)).to_numpy()
    if tr.sum() < 20:
        tr = np.arange(len(X)) < max(len(X) // 4, 10)
    sc = StandardScaler().fit(X[tr])
    Z = sc.transform(X)
    Z = np.nan_to_num(Z)
    k = min(3, Z.shape[1])
    pca = PCA(n_components=k).fit(Z[tr])
    T = pca.transform(Z)
    t2 = np.sum(T ** 2 / np.maximum(pca.explained_variance_, 1e-9), axis=1)
    q = np.sum((Z - pca.inverse_transform(T)) ** 2, axis=1)
    iso = IsolationForest(n_estimators=200, random_state=seed).fit(Z[tr])
    s = -iso.score_samples(Z)
    out = pd.DataFrame(dict(t_h=res["t_h"].to_numpy(), T2=t2, Q=q, iforest=s))
    live = (res["t_h"] >= res["t_h"].iloc[0] + 0).to_numpy() & (res["t_h"] >= train_from_h).to_numpy()
    for c in ("T2", "Q", "iforest"):
        thr = float(np.quantile(out.loc[tr, c], quantile)) * 1.2
        out[f"alarm_{c}"] = (out[c] > thr) & live
        out[f"thr_{c}"] = thr
    return out


MV_DETECTORS = {"T2", "Q", "iforest"}


def combine_and_classify(d: pd.DataFrame, res: pd.DataFrame, mv: pd.DataFrame, events: Optional[pd.DataFrame],
                         unit: str, merge_h: float = 3.0, event_tol_h: float = 3.0, use: Sequence[str] = (
                             "nis", "cusum", "ewma", "T2", "Q", "iforest"), min_mv_windows: int = 2) -> pd.DataFrame:
    """Merge alarms of the chosen detectors into episodes and give each a probable cause."""
    a = np.zeros(len(res), bool)
    srcs = []
    for k in range(len(res)):
        s = [n for n in use if (f"alarm_{n}" in res and res[f"alarm_{n}"].iloc[k]) or
             (f"alarm_{n}" in mv and mv[f"alarm_{n}"].iloc[k])]
        srcs.append(s)
        a[k] = bool(s)
    th = res["t_h"].to_numpy()
    ev = events[events["unit"] == unit] if events is not None and len(events) else pd.DataFrame(columns=["time_h", "event"])
    eps, cur = [], None
    for k in np.nonzero(a)[0]:
        if cur and th[k] - cur["end_h"] <= merge_h:
            cur["end_h"] = th[k]
            cur["idx"].append(k)
        else:
            if cur:
                eps.append(cur)
            cur = dict(start_h=th[k], end_h=th[k], idx=[k])
    if cur:
        eps.append(cur)
    rows = []
    for e in eps:
        idx = e["idx"]
        zmean = float(res["z"].iloc[idx].mean())
        short = len(idx) <= 4
        near_all = ev[(ev["time_h"] >= e["start_h"] - event_tol_h) & (ev["time_h"] <= e["start_h"] + 1)]
        near = []
        for evn, evt in zip(near_all["event"], near_all["time_h"]):
            if evn == "measurement":       # restart after a measurement pause: short dip within ~5 h
                if short and zmean < 0 and 0 <= e["start_h"] - evt <= 6:
                    near.append(evn)
            elif evn in ("gasket_failure", "trip"):   # logged at the end: only the episode that starts with it
                if e["start_h"] >= evt - 1:
                    near.append(evn)
            elif short or evn in ("power_supply_change",):
                near.append(evn)
        Tdev = float(np.max(np.abs(d["T_dev"].iloc[idx])))
        spread = float(np.max(d["U_spread_mV"].iloc[idx]))
        dets = {s for i in idx for s in srcs[i]}
        # persistence rule: a single-window alarm raised ONLY by the multivariate layer (T2/Q/iforest), with no
        # sensor-like spread, no temperature excursion and no logged event, is treated as noise and dropped
        if (len(idx) < min_mv_windows and dets <= MV_DETECTORS and spread <= 15 and Tdev <= 2.0 and not near):
            continue
        if any(x in HARDWARE_EVENTS for x in near):
            cause = "hardware / logged event: " + ", ".join(sorted(set(near)))
        elif Tdev > 2.0 and float(np.mean(np.abs(d["T_dev"].iloc[idx]) > 2.0)) >= 0.5:
            cause = "operating condition (temperature)"
        elif len(idx) <= 2 and spread > 15:
            cause = "sensor glitch"
        elif zmean < -2:
            cause = "electrochemical: voltage drop (recovery / reset)"
        elif zmean > 0:
            cause = "electrochemical: accelerated degradation"
        else:
            cause = "unclassified"
        rows.append(dict(unit=unit, start_h=e["start_h"], end_h=e["end_h"], n_windows=len(idx), mean_z=zmean,
                         max_T_dev=Tdev, max_spread_mV=spread,
                         detectors=",".join(sorted({s for i in idx for s in srcs[i]})), cause=cause))
    return pd.DataFrame(rows, columns=["unit", "start_h", "end_h", "n_windows", "mean_z", "max_T_dev", "max_spread_mV",
                                       "detectors", "cause"])


def run_unit(ts: pd.DataFrame, W: pd.DataFrame, events: Optional[pd.DataFrame], unit: str, train_h: float = 200.0,
             use: Sequence[str] = ("nis", "cusum", "ewma", "T2", "Q", "iforest"), **kw) -> Dict[str, pd.DataFrame]:
    d = window_features(ts, W, unit)
    if len(d) < 30:
        return dict(features=d, residuals=pd.DataFrame(), multivariate=pd.DataFrame(), episodes=pd.DataFrame())
    res = residual_layer(d, **kw)
    mv = multivariate_layer(d, res, train_h)
    ep = combine_and_classify(d, res, mv, events, unit, use=use)
    return dict(features=d, residuals=res, multivariate=mv, episodes=ep)


def evaluate(episodes: pd.DataFrame, labels: pd.DataFrame, unit: str, t_end: float, pre_h: float = 1.0,
             post_h: float = 6.0, max_extra_h: float = 12.0) -> Dict[str, object]:
    """Detection rate and delay per anomaly kind, false alarms per 100 h (episodes overlapping no labelled interval),
    and cause-classification agreement (episode cause vs label class). An episode much longer than the event it
    overlaps (> event duration + max_extra_h) is NOT counted as explained (strict: long spurious episodes that happen
    to touch a short labelled event remain false alarms)."""
    lab = labels[labels["unit"] == unit]
    ep = episodes if episodes is not None else pd.DataFrame()
    rows = []
    hit_ep = np.zeros(len(ep), bool)
    dur_ep = (ep["end_h"] - ep["start_h"]).to_numpy() if len(ep) else np.array([])
    for r in lab.itertuples():
        m = (ep["end_h"] >= r.start_h - pre_h) & (ep["start_h"] <= r.end_h + post_h) if len(ep) else pd.Series(dtype=bool)
        det = bool(m.any()) if len(ep) else False
        if det:
            # an episode is 'explained' by this label only if it is not much longer than the labelled event
            hit_ep |= m.to_numpy() & (dur_ep <= (r.end_h - r.start_h) + max_extra_h)
            first = ep[m].iloc[0]
            delay = max(float(first["start_h"]) - r.start_h, 0.0)
            cause = first["cause"]
        else:
            delay, cause = np.nan, ""
        rows.append(dict(kind=r.kind, cls=r.cls, start_h=r.start_h, detected=det, delay_h=delay, cause=cause))
    per = pd.DataFrame(rows)
    false = int((~hit_ep).sum()) if len(ep) else 0
    dur = max(t_end, 1.0)
    summary = per.groupby(["cls", "kind"]).agg(n=("detected", "size"), detected=("detected", "sum"),
                                               median_delay_h=("delay_h", "median")).reset_index() if len(per) else per
    if len(summary):
        summary["rate"] = summary["detected"] / summary["n"]
    return dict(per_label=per, summary=summary, false_alarms=false, false_per_100h=100 * false / dur,
                false_episodes=ep[~hit_ep] if len(ep) else ep)


def lead_time(episodes: pd.DataFrame, t_failure: float, t_onset: Optional[float] = None,
              persistent_windows: int = 3) -> Dict[str, float]:
    """Earliest persistent 'accelerated degradation' alarm before the failure: lead time = t_failure - t_alarm
    (also relative to the true onset when known, e.g. synthetic data)."""
    if episodes is None or not len(episodes):
        return dict(t_alarm=np.nan, lead_h=np.nan, after_onset_h=np.nan)
    e = episodes[(episodes["cause"].str.contains("accelerated")) & (episodes["n_windows"] >= persistent_windows)
                 & (episodes["start_h"] < t_failure)]
    if t_onset is not None and np.isfinite(t_onset):
        e = e[e["start_h"] >= t_onset - 200]     # ignore much earlier episodes for this metric
    if not len(e):
        return dict(t_alarm=np.nan, lead_h=np.nan, after_onset_h=np.nan)
    ta = float(e["start_h"].iloc[0])
    return dict(t_alarm=ta, lead_h=t_failure - ta, after_onset_h=(ta - t_onset) if t_onset is not None else np.nan)


def drt_precursor(tracked: pd.DataFrame, unit: str, peak: str, j: float, k: float = 1.5,
                  baseline_times: int = 3) -> Dict[str, object]:
    """DRT early-warning indicator: the chosen peak's resistance at current j relative to its median over the first
    `baseline_times` post-activation spectra; first time it exceeds k x baseline."""
    d = tracked[(tracked["unit"] == unit) & (tracked["peak"] == peak) & (np.isclose(tracked["j_A_cm2"], j))]
    d = d.groupby("time_h", as_index=False)["R"].sum().sort_values("time_h")
    d = d[d["time_h"] >= 50]
    if len(d) < baseline_times + 1:
        return dict(series=d, t_warn=np.nan)
    base = float(d["R"].head(baseline_times).median())
    d = d.assign(ratio=d["R"] / base)
    w = d[d["ratio"] > k]
    return dict(series=d, t_warn=float(w["time_h"].iloc[0]) if len(w) else np.nan, baseline=base)
