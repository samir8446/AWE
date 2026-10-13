"""Publication figures (matplotlib, 300 dpi) with the data behind each figure as CSV. Used by the app (Study results)
and by study.py. Every caption states whether the data are synthetic."""
from __future__ import annotations

import io
from typing import Dict

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MECH_COL = {"ohmic + ion transport": "#2a78d6", "HER (cathode)": "#1baf7a", "OER + passive film (anode)": "#eb6834",
            "mass transport / gas": "#6250d6"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
                     "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False})


def _png(fig) -> bytes:
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return b.getvalue()


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def _jmask(s, j):
    return s.isna() if not np.isfinite(j) else np.isclose(s.astype(float), j)


def fig_drt(A, tr, cl, pmap, unit, j) -> Dict[str, bytes]:
    g = A["gamma"]
    v = A["spectra"][(A["spectra"]["unit"] == unit) & A["spectra"]["kk_valid"]][["time_h"]]
    g = g[(g["unit"] == unit) & _jmask(g["j_A_cm2"], j)].merge(v.drop_duplicates(), on="time_h")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
    if len(g):
        pv = g.pivot_table(index="time_h", columns="ln_tau", values="gamma")
        f = np.log10(1 / (2 * np.pi * np.exp(pv.columns.to_numpy())))
        vmax = float(np.nanpercentile(pv.values[1:] if len(pv) > 2 else pv.values, 99.5))   # t=0 (native oxide) excluded
        ax[0].pcolormesh(f, pv.index, pv.values, shading="auto", cmap="Blues", vmin=0, vmax=vmax)
        ax[0].invert_xaxis()
        ax[0].set(xlabel="log$_{10}$ f (Hz)", ylabel="time (h)", title=f"(a) DRT map, {j:g} A cm$^{{-2}}$")
    t = tr[(tr["unit"] == unit) & _jmask(tr["j_A_cm2"], j) & (tr["peak"] != "minor")] if len(tr) else tr
    for pk, d in t.groupby("peak"):
        mech = pmap.get(pk, pk)
        ax[1].errorbar(d["time_h"], d["R"], yerr=[(d["R"] - d["R_lo"]).clip(lower=0), (d["R_hi"] - d["R"]).clip(lower=0)],
                       fmt="o-", ms=2.5, lw=1, capsize=1.5, color=MECH_COL.get(mech, "k"), label=f"{pk}: {mech}")
    ax[1].set(xlabel="time (h)", ylabel="R ($\\Omega$ cm$^2$)", title="(b) tracked peaks")
    ax[1].legend(fontsize=6)
    t0 = tr[(tr["unit"] == unit) & (tr["peak"] != "minor")] if len(tr) else tr
    if len(t0) and t0["j_A_cm2"].nunique() >= 2:
        tt = sorted(t0["time_h"].unique())[len(sorted(t0["time_h"].unique())) // 2]
        for pk, d in t0[t0["time_h"] == tt].groupby("peak"):
            ax[2].loglog(d["j_A_cm2"], d["R"], "o-", ms=3, color=MECH_COL.get(pmap.get(pk, ""), "k"), label=pk)
        ax[2].set(xlabel="j (A cm$^{-2}$)", ylabel="R ($\\Omega$ cm$^2$)", title=f"(c) current dependence, {tt:g} h")
        ax[2].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax[2].set_xticks([0.05, 0.1, 0.2, 0.5])
        ax[2].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    fig.tight_layout()
    data = t.copy()
    return {"fig1_drt_map_tracking.png": _png(fig), "fig1_tracked_peaks.csv": _csv(data),
            "fig1_peak_assignment.csv": _csv(cl) if cl is not None and len(cl) else b""}


def fig_closure(clo, tw, tw0) -> Dict[str, bytes]:
    tb = clo["table"]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
    ax[0].plot(tb["time_h"], tb["dU_meas_mV"], "o-", color="0.5", ms=3, label="measured (CP)")
    ax[0].plot(tb["time_h"], tb["dU_pred_mV"], "s-", color="#2a78d6", ms=3, label="from impedance (DRT)")
    ax[0].set(xlabel="time (h)", ylabel="$\\Delta U$ (mV)", title=f"(a) AC-DC closure, RMSE {clo['rmse_mV']:.1f} mV")
    ax[0].plot(tb["time_h"], tb["dU_single_mV"], "--", color="#eb6834", lw=0.8, label="single-current j$\\Delta$R")
    ax[0].legend(fontsize=6)
    bottom_p = np.zeros(len(tb))
    bottom_n = np.zeros(len(tb))
    for mech, col in MECH_COL.items():
        y = tb[f"d_{mech}_mV"].to_numpy()
        b = np.where(y >= 0, bottom_p, bottom_n)
        ax[1].bar(tb["time_h"], y, width=30, bottom=b, color=col, label=mech)
        bottom_p += np.where(y >= 0, y, 0)
        bottom_n += np.where(y < 0, y, 0)
    ax[1].set(xlabel="time (h)", ylabel="mV", title="(b) attribution")
    ax[1].legend(fontsize=5)
    for mech, col in MECH_COL.items():
        ax[2].plot(tw["t_h"], tw[mech], color=col, lw=1.2)
        ax[2].fill_between(tw["t_h"], tw[mech] - 2 * tw[f"{mech}_sd"], tw[mech] + 2 * tw[f"{mech}_sd"], color=col, alpha=0.15, lw=0)
    if tw0 is not None:
        ax[2].plot(tw0["t_h"], 2 * tw0[f"{list(MECH_COL)[0]}_sd"], color="0.4", lw=0.8, ls=":",
                   label="2$\\sigma$ without DRT (any mechanism)")
        ax[2].legend(fontsize=5, loc="upper left")
    le = float(tb["time_h"].max()) + 50
    if tw["t_h"].max() > le:
        ax[2].axvspan(le, float(tw["t_h"].max()), color="#e34948", alpha=0.08, lw=0)
        ax[2].text(le, ax[2].get_ylim()[1] * 0.9, " no EIS", fontsize=6, color="#e34948")
    ax[2].set(xlabel="time (h)", ylabel="mV", title="(c) DRT-informed twin ($\\pm2\\sigma$)")
    fig.tight_layout()
    return {"fig2_closure_attribution_twin.png": _png(fig), "fig2_closure.csv": _csv(tb), "fig2_twin.csv": _csv(tw),
            "fig2_twin_without_drt.csv": _csv(tw0) if tw0 is not None else b""}


def fig_degradation(P, rep) -> Dict[str, bytes]:
    ser, ri, seg, rs = P["series"], P["ri"], P["seg"], P["resets"]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
    ax[0].plot(ser["t_h"], ser["dV_mV"], color="0.6", lw=0.8)
    if ri.get("ok"):
        ax[0].plot(ri["t"], ri["baseline"], color="#e34948", lw=1.5, label="irreversible")
    if len(rs):
        e = rs[rs["cause"] == "electrochemical reset"]
        o = rs[rs["cause"] != "electrochemical reset"]
        ax[0].plot(e["time_h"], e["level_after"], "v", color="#1baf7a", ms=4, label="electrochemical reset")
        ax[0].plot(o["time_h"], o["level_after"], "v", color="0.4", ms=4, label="refill / instrument")
    ax[0].set(xlabel="time (h)", ylabel="$\\Delta V$ (mV)", title="(a) resets and baseline")
    ax[0].legend(fontsize=5)
    if ri.get("ok"):
        ax[1].stackplot(ri["t"], ri["baseline"] - ri["baseline"][0], np.maximum(ri["reversible"], 0),
                        colors=["#e34948", "#eda100"], labels=["irreversible", "reversible"])
        ax[1].legend(fontsize=6, loc="upper left")
    ax[1].set(xlabel="time (h)", ylabel="mV", title="(b) reversible vs irreversible")
    sc = {"1 activation": "#2a78d6", "2 passivation": "#eda100", "3 cyclic spalling": "#eb6834", "4 terminal": "#e34948"}
    ax[2].plot(ser["t_h"], ser["dV_mV"], ".", color="0.75", ms=1.5)
    for r in seg.itertuples():
        ax[2].plot([r.start_h, r.end_h], [r.level_start, r.level_end], color=sc.get(r.stage, "k"), lw=2)
    for k, v in sc.items():
        ax[2].plot([], [], color=v, lw=2, label=k)
    ax[2].legend(fontsize=5)
    ax[2].set(xlabel="time (h)", ylabel="mV", title="(c) detected stages")
    fig.tight_layout()
    out = {"fig3_degradation.png": _png(fig), "fig3_resets.csv": _csv(rs), "fig3_stages.csv": _csv(seg),
           "fig3_cycles.csv": _csv(P["saw"]["cycles"])}
    if rep is not None and len(rep):
        out["fig3_replicates.csv"] = _csv(rep)
    return out


def fig_anomalies(res_an, unit, leads) -> Dict[str, bytes]:
    r = res_an[unit]
    rr, ep = r["residuals"], r["episodes"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw=dict(width_ratios=[2.2, 1]))
    ax[0].plot(rr["t_h"], rr["U_mV"], color="0.5", lw=0.7)
    col = lambda c: "#9aa4b2" if c.startswith("hardware") else "#eda100" if c.startswith("operating") else \
        "#e87ba4" if c.startswith("sensor") else "#1baf7a" if "drop" in c else "#e34948" if "accelerated" in c else "#6250d6"  # noqa: E731
    for e in ep.itertuples():
        ax[0].axvspan(e.start_h - 0.5, e.end_h + 0.5, color=col(e.cause), alpha=0.35, lw=0)
    ax[0].set(xlabel="time (h)", ylabel="U (mV)", title=f"(a) {unit}: alarms by cause")
    if leads is not None and len(leads):
        col_l = "lead_time_h" if "lead_time_h" in leads else "lead_h"
        ax[1].barh(leads["unit"].astype(str), leads[col_l], color="#e34948")
        ax[1].set(xlabel="lead time before failure (h)", title="(b) early warning")
    fig.tight_layout()
    return {"fig4_anomalies.png": _png(fig), "fig4_episodes.csv": _csv(ep),
            "fig4_lead_times.csv": _csv(leads) if leads is not None else b""}


def figure_set(ctx) -> Dict[str, bytes]:
    """Build all figures from the app context (computes what is missing for the first unit)."""
    import views_advanced as VA
    ss, G, D = ctx["ss"], ctx["G"], ctx["D"]
    files: Dict[str, bytes] = {}
    tag = "SYNTHETIC data (method check)" if ctx["source"] == "synthetic" else "REAL data"
    notes = [f"Data: {tag}", "Engine/modules 0.3.0"]
    W = ctx["W"]
    units = sorted(W["unit"].unique()) if len(W) else []
    A = VA.drt_results(ctx) if ctx["data"].get("eis") is not None else None
    if A is not None and len(A["peaks"]):
        tr = VA.tracked(ctx, A)
        if not len(tr):
            tr = pd.DataFrame(columns=["unit", "j_A_cm2", "time_h", "peak", "R", "R_lo", "R_hi", "tau_s", "f_Hz"])
        cl = D.classify_by_current(tr, A["spectra"])
        u = units[0] if units else sorted(A["spectra"]["unit"].unique())[0]
        pmap = D.auto_peak_map(cl, u) if len(cl) else {}
        jop = float(ctx["op"].j_A_cm2)
        jlist = sorted(tr["j_A_cm2"].dropna().unique()) if len(tr) else []
        jj = jop if jop in set(jlist) else (float(jlist[-1]) if jlist else float("nan"))
        files.update(fig_drt(A, tr, cl, pmap, u, jj))
        ser = VA.unit_series(ctx, u)
        clo = G.closure_test(tr, A["spectra"], ctx["data"].get("polcurve"), ser, u, jop, pmap or None)
        if clo.get("ok"):
            s2 = ser[ser["t_h"] >= clo["table"]["time_h"].iloc[0]]
            tw = G.drt_informed_twin(s2, clo["table"], ctx["data"].get("events"), u)
            tw0 = G.drt_informed_twin(s2, clo["table"], ctx["data"].get("events"), u, use_drt=False)
            files.update(fig_closure(clo, tw, tw0))
            notes.append(f"Closure RMSE {clo['rmse_mV']:.1f} mV (single-current {clo['rmse_single_mV']:.1f} mV), unit {u}")
    if units:
        P = VA.degradation_pipeline(ctx, units[0])
        if P is not None:
            files.update(fig_degradation(P, ss.get("res_replicates")))
    if ss.get("res_an"):
        leads = (ss.get("res_an_eval") or {}).get("leads")
        files.update(fig_anomalies(ss["res_an"], sorted(ss["res_an"])[0], leads))
        if (ss.get("res_an_eval") or {}).get("summary") is not None:
            files["fig4_detection_summary.csv"] = _csv(ss["res_an_eval"]["summary"])
    files["README.txt"] = ("\n".join(notes) + "\nEach PNG has CSV files with the plotted data.\n").encode()
    return {k: v for k, v in files.items() if v}
