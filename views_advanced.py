"""Advanced views of the AWE twin app: DRT & mechanisms, degradation analysis, anomalies, paper figures.
Drawing only; all computation in drt.py, degradation.py, anomaly.py, engine.py, figures.py."""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go

MECH_COL = {"ohmic + ion transport": "#2a78d6", "HER (cathode)": "#1baf7a", "OER + passive film (anode)": "#eb6834",
            "mass transport / gas": "#6250d6"}
CAUSE_COL = {"hardware": "#9aa4b2", "operating": "#eda100", "sensor": "#e87ba4", "electrochemical: voltage drop": "#1baf7a",
             "electrochemical: accelerated": "#e34948", "unclassified": "#6250d6"}


def _cause_color(c: str) -> str:
    for k, v in CAUSE_COL.items():
        if c.startswith(k) or k in c:
            return v
    return "#6250d6"


# ============================================================================================ shared pipelines
def drt_results(ctx):
    st, ss, data = ctx["st"], ctx["ss"], ctx["data"]
    eis = data.get("eis")
    if eis is None or not len(eis):
        return None
    units = tuple(sorted(ss.get("drt_units") or sorted(eis["unit"].unique())))
    lam = None if ss.get("drt_lam_mode", "mGCV (automatic)") != "fixed" else float(ss.get("drt_lam", 1e-3))
    method = {"mGCV (automatic)": "mgcv", "GCV (automatic)": "gcv", "L-curve (automatic)": "lcurve",
              "fixed": "fixed"}[ss.get("drt_lam_mode", "mGCV (automatic)")]
    return ctx["c_drt"](eis, ctx["key_of"](eis), float(ctx["design"].area_cm2), method, lam,
                        float(ss.get("drt_prom", 0.01)), bool(ss.get("drt_ind", False)), units)


def tracked(ctx, A):
    ss, D = ctx["ss"], ctx["D"]
    method = ss.get("drt_track", "cluster")
    return D.track_peaks(A["peaks"], method=method, spectra=A["spectra"], min_share=float(ss.get("drt_minshare", 0.02)))


def unit_series(ctx, unit):
    return ctx["G"].unit_series(ctx["W"], unit, float(ctx["ss"].get("dg_bin", 2.0)), ctx["data"].get("events"))


def degradation_pipeline(ctx, unit):
    G, ss = ctx["G"], ctx["ss"]
    ser = unit_series(ctx, unit)
    if not len(ser):
        return None
    t, y = ser["t_h"].to_numpy(), ser["dV_mV"].to_numpy()
    rs = G.detect_resets(t, y, float(ss.get("dg_drop", 8.0)), t_start=float(ss.get("dg_tstart", 50.0)),
                         events=ctx["data"].get("events"), unit=unit, T=ser["T_C"].to_numpy())
    ec = rs[rs["cause"] == "electrochemical reset"] if len(rs) else rs
    ri = G.reversible_irreversible(t, y, rs, float(ss.get("dg_tstart", 50.0)))
    sw = G.sawtooth_model(t, y, ec, float(ss.get("dg_tstart", 50.0)))
    seg = G.label_stages(G.segment_piecewise_linear(t, y, penalty_factor=float(ss.get("dg_pen", 10.0))), resets=rs)
    return dict(series=ser, resets=rs, electro=ec, ri=ri, saw=sw, seg=seg)


def _need(ctx, what="W"):
    st = ctx["st"]
    if what == "W" and not len(ctx["W"]):
        st.info("Load data with an operating time series (sidebar → 1 · Data).")
        return False
    if what == "eis" and (ctx["data"].get("eis") is None or not len(ctx["data"]["eis"])):
        st.info("No EIS spectra loaded. Choose 'Synthetic: stainless-steel ageing' or upload EIS files (data type 'EIS').")
        return False
    return True


# ============================================================================================ DRT & MECHANISMS
def view_drt(ctx):
    st, ss, D, G, style, PAL, explain = ctx["st"], ctx["ss"], ctx["D"], ctx["G"], ctx["style"], ctx["PAL"], ctx["explain"]
    if not _need(ctx, "eis"):
        return
    eis = ctx["data"]["eis"]
    with st.expander("DRT settings (regularisation, peaks, tracking)"):
        c1, c2, c3, c4 = st.columns(4)
        c1.selectbox("Lambda selection", ["mGCV (automatic)", "GCV (automatic)", "L-curve (automatic)", "fixed"], key="drt_lam_mode")
        c2.select_slider("Fixed lambda", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1], value=1e-3, key="drt_lam")
        c3.number_input("Peak prominence (share of max)", 0.001, 0.5, 0.01, format="%.3f", key="drt_prom")
        c4.toggle("Fit inductance", False, key="drt_ind")
        c1.selectbox("Peak tracking", ["cluster", "track", "bands"], key="drt_track",
                     help="cluster: persistent groups in log tau (default); track: nearest to previous; bands: fixed bands")
        c2.number_input("Minimum peak share of R_pol", 0.0, 0.2, 0.02, format="%.3f", key="drt_minshare")
        c3.multiselect("Units to analyse", sorted(eis["unit"].unique()), key="drt_units",
                       help="Empty = all units. Each spectrum takes ~0.2 s the first time, then it is cached.")
        st.caption(f"Bands for the 'bands' method (from the literature mapping table in 'Cell and results'; edit in "
                   f"drt.DEFAULT_BANDS_HZ): {D.DEFAULT_BANDS_HZ}")
    ctx["eq_popover"]("DRT")
    A = drt_results(ctx)
    if A is None or not len(A["spectra"]):
        st.warning("No spectra could be analysed.")
        return
    spec, peaks = A["spectra"], A["peaks"]
    tr = tracked(ctx, A)
    units = sorted(spec["unit"].unique())
    st.markdown("#### 1 · Spectrum explorer")
    c1, c2, c3 = st.columns(3)
    u = c1.selectbox("Unit", units, key="drt_u")
    times = sorted(spec.loc[spec["unit"] == u, "time_h"].unique())
    t = c2.selectbox("Time (h)", times, key="drt_t")
    js = sorted(spec.loc[(spec["unit"] == u) & (spec["time_h"] == t), "j_A_cm2"].dropna().unique())
    j = c3.selectbox("EIS current (A/cm2)", js or [np.nan], key="drt_j")
    m = (eis["unit"] == u) & (eis["time_h"] == t) & ((eis["j_A_cm2"] == j) if "j_A_cm2" in eis and np.isfinite(j) else True)
    d = eis[m].sort_values("freq_Hz", ascending=False)
    area = ctx["design"].area_cm2
    r = D.fit_drt(d["freq_Hz"], d["z_re"] * area, d["z_im_neg"] * area,
                  method="mgcv" if ss.get("drt_lam_mode", "mGCV (automatic)").startswith("mGCV") else "gcv")
    kk = D.kk_test(d["freq_Hz"], d["z_re"], d["z_im_neg"])
    pk = D.find_drt_peaks(r, float(ss.get("drt_prom", 0.01)))
    c1, c2, c3 = st.columns(3)
    fig = go.Figure(go.Scatter(x=d["z_re"] * area, y=d["z_im_neg"] * area, mode="markers", name="measured",
                               marker=dict(size=6, color="#9aa4b2")))
    fig.add_trace(go.Scatter(x=r["z_fit"].real, y=-r["z_fit"].imag, name="DRT fit", line=dict(color=PAL[0], width=2)))
    fig.update_yaxes(scaleanchor="x")
    c1.plotly_chart(style(fig, "Nyquist (ohm cm2)", "Z'", "-Z''", 340), key="drt_nyq")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[r["freq_peak"], r["freq_peak"][::-1]], y=np.r_[r["gamma_hi"], r["gamma_lo"][::-1]],
                             fill="toself", fillcolor=PAL[0], opacity=0.18, line=dict(width=0), name="5-95 % bootstrap"))
    fig.add_trace(go.Scatter(x=r["freq_peak"], y=r["gamma"], name="gamma", line=dict(color=PAL[0], width=2)))
    for _, p in pk.iterrows():
        fig.add_vline(x=p["f_Hz"], line=dict(color="#8a93a3", dash="dot"))
    fig.update_xaxes(type="log", autorange="reversed")
    c2.plotly_chart(style(fig, f"DRT (lambda {r['lam']:.1e}, R_pol {r['R_pol']:.3f} ohm cm2)", "relaxation frequency (Hz)",
                          "gamma (ohm cm2)", 340), key="drt_gamma")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=kk["freq"], y=kk["residual_re_pct"], name="Δ re", mode="lines+markers", line=dict(color=PAL[0])))
    fig.add_trace(go.Scatter(x=kk["freq"], y=kk["residual_im_pct"], name="Δ im", mode="lines+markers", line=dict(color=PAL[1])))
    fig.update_xaxes(type="log")
    c3.plotly_chart(style(fig, f"Kramers-Kronig residuals: mean {kk['mean_abs_pct']:.2f} %", "frequency (Hz)", "%", 340),
                    key="drt_kk")
    st.dataframe(pk.round(5), hide_index=True)
    if st.button("Fit DRT-guided equivalent circuit (R_inf + one ZARC per peak)", key="drt_ecm_btn") and len(pk):
        ss["res_drt_ecm"] = D.fit_ecm_from_drt(d["freq_Hz"], d["z_re"] * area, d["z_im_neg"] * area, pk, r["R_inf"])
    if ss.get("res_drt_ecm") is not None and ss["res_drt_ecm"].get("ok"):
        e = ss["res_drt_ecm"]
        st.caption(f"ECM: R_inf = {e['R_inf']:.4f} ohm cm2, fit error {e['rel_rmse_pct']:.2f} %")
        st.dataframe(e["elements"].round(5), hide_index=True)
    explain("DRT by non-negative Tikhonov regression with Gaussian basis; lambda by modified GCV; bands by parametric "
            "bootstrap (noise only, not regularisation bias). The KK test (lin-KK) gates data quality: invalid spectra "
            "are excluded from tracking and closure.")

    st.markdown("#### 2 · Data quality (Kramers-Kronig gate)")
    q = spec.copy()
    fig = go.Figure()
    for k, (uu, dd) in enumerate(q.groupby("unit")):
        fig.add_trace(go.Scatter(x=dd["time_h"], y=dd["kk_mean_abs_pct"], mode="markers", name=uu,
                                 marker=dict(size=7, color=PAL[k % len(PAL)],
                                             symbol=np.where(dd["kk_valid"], "circle", "x"))))
    st.plotly_chart(style(fig, "KK residual per spectrum (x = rejected)", "time (h)", "mean |residual| (%)", 300), key="drt_kkall")
    st.caption(f"{int((~q['kk_valid']).sum())} of {len(q)} spectra rejected (threshold: 2 x the unit's median residual, "
               "min 0.3 %).")
    if ctx["source"] == "synthetic" and ctx["scen"] == "ss":
        et = ctx["syn"]["eis_truth"]
        mm = q.merge(et[["unit", "time_h", "j_A_cm2", "valid"]], on=["unit", "time_h", "j_A_cm2"])
        if len(mm):
            tp = int((~mm["kk_valid"] & ~mm["valid"]).sum())
            fp = int((~mm["kk_valid"] & mm["valid"]).sum())
            st.caption(f"Synthetic check: {tp}/{int((~mm['valid']).sum())} drifting spectra caught, {fp} valid spectra "
                       f"wrongly rejected.")

    st.markdown("#### 3 · DRT map over time")
    c1, c2 = st.columns(2)
    mu = c1.selectbox("Unit (map)", units, key="drt_map_u")
    jm = c2.selectbox("Current (map)", sorted(A["gamma"].loc[A["gamma"]["unit"] == mu, "j_A_cm2"].dropna().unique()) or [np.nan],
                      key="drt_map_j")
    g = A["gamma"]
    g = g[(g["unit"] == mu) & ((g["j_A_cm2"] == jm) if np.isfinite(jm) else True)]
    vs = q[(q["unit"] == mu) & q["kk_valid"]][["time_h", "j_A_cm2"]]
    g = g.merge(vs, on=["time_h", "j_A_cm2"]) if np.isfinite(jm) else g
    if len(g):
        pv = g.pivot_table(index="time_h", columns="ln_tau", values="gamma")
        fr = 1 / (2 * np.pi * np.exp(pv.columns.to_numpy()))
        zmax = float(np.nanpercentile(pv.values[1:] if len(pv) > 2 else pv.values, 99.5))
        fig = go.Figure(go.Heatmap(z=pv.values, x=np.log10(fr), y=pv.index, colorscale="Blues", zmin=0, zmax=zmax,
                                   colorbar=dict(title="gamma")))
        fig.update_xaxes(autorange="reversed")
        st.plotly_chart(style(fig, f"{mu}: DRT map at {jm:g} A/cm2 (valid spectra)", "log10 relaxation frequency (Hz)",
                              "time (h)", 420), key="drt_map")
        explain("Each row is one DRT. Peaks that grow, shift or collapse together over time show the degradation "
                "mechanism and its resets directly.")

    st.markdown("#### 4 · Tracked peaks and their physical assignment")
    cl = D.classify_by_current(tr, spec)
    pmap = D.auto_peak_map(cl, mu) if len(cl) else {}
    if len(cl):
        st.dataframe(cl.assign(mechanism=cl["peak"].map(pmap)).round(3), hide_index=True)
        explain("Assignment by behaviour, not by literature frequency tables: R ~ 1/j (slope -1) is charge transfer, "
                "current independent is ohmic/ion transport, rising with j is mass transport/gas.")
    else:
        st.info("Peak assignment needs EIS at two or more currents.")
    tu = tr[(tr["unit"] == mu) & ((tr["j_A_cm2"] == jm) if np.isfinite(jm) else True) & (tr["peak"] != "minor")]
    fig = go.Figure()
    for k, (pn, dd) in enumerate(tu.groupby("peak")):
        col = MECH_COL.get(pmap.get(pn, ""), PAL[k % len(PAL)])
        fig.add_trace(go.Scatter(x=dd["time_h"], y=dd["R"], name=f"{pn} · {pmap.get(pn, '?')}", mode="lines+markers",
                                 line=dict(color=col, width=2),
                                 error_y=dict(type="data", symmetric=False, array=(dd["R_hi"] - dd["R"]).clip(lower=0),
                                              arrayminus=(dd["R"] - dd["R_lo"]).clip(lower=0), visible=True)))
    st.plotly_chart(style(fig, f"{mu}: peak resistances at {jm:g} A/cm2", "time (h)", "R (ohm cm2)", 380), key="drt_track_plot")

    st.markdown("#### 5 · AC-DC closure: does the DRT explain the measured voltage?")
    c1, c2, c3 = st.columns(3)
    cu = c1.selectbox("Unit (closure)", units, key="drt_cl_u")
    jop = c2.number_input("Operating current density (A/cm2)", 0.01, 3.0, float(ctx["op"].j_A_cm2), key="drt_cl_jop")
    jst = c3.selectbox("Integration starts at", ["lowest EIS current"] + [f"{x:g}" for x in sorted(spec["j_A_cm2"].dropna().unique())],
                       key="drt_cl_js")
    ser = unit_series(ctx, cu)
    pm = D.auto_peak_map(cl, cu) if len(cl) else None
    clo = G.closure_test(tr, spec, ctx["data"].get("polcurve"), ser, cu, jop, pm,
                         j_start=None if jst.startswith("lowest") else float(jst))
    ctx["ss"]["res_closure"] = dict(unit=cu, result=clo)
    if not clo.get("ok"):
        st.info(f"Closure not possible: {clo.get('reason')}.")
        return
    tb = clo["table"]
    m3 = st.columns(3)
    m3[0].metric("Closure RMSE (full, multi-current)", f"{clo['rmse_mV']:.1f} mV")
    m3[1].metric("Single-current approximation j·ΔR", f"{clo['rmse_single_mV']:.1f} mV")
    m3[2].metric("Absolute U_pred - U_meas (median)", f"{clo['abs_pred_vs_meas_mV']:.1f} mV")
    if clo.get("n_nonlinear"):
        st.warning(f"{clo['n_nonlinear']} EIS time(s) with a voltage rise > 150 mV (near the limiting current) are "
                   "excluded from the RMSE: small-signal impedance no longer integrates to the DC voltage there.")
    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tb["time_h"], y=tb["dU_meas_mV"], name="measured (CP after pause)", mode="lines+markers",
                             line=dict(color="#9aa4b2", width=2)))
    fig.add_trace(go.Scatter(x=tb["time_h"], y=tb["dU_pred_mV"], name="predicted from impedance", mode="lines+markers",
                             line=dict(color=PAL[0], width=2)))
    fig.add_trace(go.Scatter(x=tb["time_h"], y=tb["dU_single_mV"], name="single-current j·ΔR", mode="lines",
                             line=dict(color=PAL[1], width=1.5, dash="dash")))
    c1.plotly_chart(style(fig, "Voltage change since the reference EIS", "time (h)", "ΔU (mV)", 380), key="drt_cl_plot")
    fig = go.Figure()
    for mech in G.MECHANISMS:
        fig.add_trace(go.Bar(x=tb["time_h"], y=tb[f"d_{mech}_mV"], name=mech, marker_color=MECH_COL[mech]))
    fig.update_layout(barmode="relative")
    c2.plotly_chart(style(fig, "Mechanism attribution (mV at the operating current)", "time (h)", "mV", 380),
                    key="drt_cl_attr")
    st.dataframe(tb.round(4), hide_index=True)
    explain("U(j_op) = U_LSV(j_start) + ∫ [R_inf + Σ R_k(j)] dj, with each DRT resistance interpolated as a power law "
            "between the EIS currents. The residual between measured and predicted is itself a result: a persistent "
            "step after a logged power-supply change is an instrument offset, not cell ageing.")

    st.markdown("#### 6 · DRT-informed twin: mechanism-resolved, self-updating")
    use = st.toggle("Use DRT observations (off = voltage only: ablation)", True, key="drt_tw_use")
    tw = G.drt_informed_twin(ser[ser["t_h"] >= tb["time_h"].iloc[0]], tb, ctx["data"].get("events"), cu, use_drt=use)
    ss["res_twin"] = dict(unit=cu, twin=tw, use=use)
    fig = go.Figure()
    for mech in G.MECHANISMS:
        col = MECH_COL[mech]
        fig.add_trace(go.Scatter(x=np.r_[tw["t_h"], tw["t_h"][::-1]],
                                 y=np.r_[tw[mech] + 2 * tw[f"{mech}_sd"], (tw[mech] - 2 * tw[f"{mech}_sd"])[::-1]],
                                 fill="toself", fillcolor=col, opacity=0.12, line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=tw["t_h"], y=tw[mech], name=mech, line=dict(color=col, width=2)))
    fig.add_trace(go.Scatter(x=tw["t_h"], y=tw["offset_mV"], name="instrument offset", line=dict(color="#9aa4b2", dash="dot")))
    fig.add_trace(go.Scatter(x=tw["t_h"], y=tw["residual_mV"], name="unexplained (closure residual)",
                             line=dict(color="#8a93a3", dash="dash")))
    last_eis = float(tb["time_h"].max())
    if tw["t_h"].max() > last_eis + 50:
        fig.add_vrect(x0=last_eis + 50, x1=float(tw["t_h"].max()), fillcolor="#e34948", opacity=0.08, line_width=0,
                      annotation_text="no EIS: attribution extrapolated", annotation_position="top left")
    st.plotly_chart(style(fig, f"{cu}: voltage change per mechanism (±2σ)", "time (h)", "mV", 440), key="drt_twin_plot")
    last = tw.iloc[-1]
    st.dataframe(pd.DataFrame([dict(mechanism=m, level_mV=last[m], sd_mV=last[f"{m}_sd"], rate_uV_h=1000 * last[f"{m}_rate"])
                               for m in G.MECHANISMS]).round(3), hide_index=True)
    if ctx["truth"] is not None and ctx["scen"] == "ss":
        tt = ctx["truth"]
        tt = tt[(tt["unit"] == cu) & (tt["eta_oer"] > 0)].set_index("time_h")
        if len(tt):
            near = lambda x: tt.iloc[int(np.abs(tt.index.to_numpy() - x).argmin())]  # noqa: E731
            b = near(float(tb["time_h"].iloc[0]) + 5)
            rows = []
            for rr in tw[tw["t_h"] <= last_eis + 50].iloc[::10].itertuples():
                x = near(rr.t_h)
                tru = {G.MECHANISMS[0]: x.eta_ohmic + x.eta_ion - b.eta_ohmic - b.eta_ion,
                       G.MECHANISMS[1]: x.eta_her - b.eta_her, G.MECHANISMS[2]: x.eta_oer - b.eta_oer,
                       G.MECHANISMS[3]: x.eta_mt - b.eta_mt}
                rows.append({m: tw.loc[rr.Index, m] - 1000 * tru[m] for m in G.MECHANISMS})
            er = pd.DataFrame(rows)
            st.caption(f"Synthetic check up to {last_eis + 50:.0f} h (DRT coverage), RMSE vs true mechanism contributions (mV): " + ", ".join(
                f"{m}: {np.sqrt(np.mean(er[m] ** 2)):.1f}" for m in G.MECHANISMS))
    explain("Kalman filter: levels and rates per mechanism, an instrument-offset state that can only jump at logged "
            "hardware events, and an unexplained residual. With DRT, each mechanism is identified (narrow bands); "
            "without DRT only the sum is observed and the bands stay wide - the ablation that shows what DRT adds.")


# ============================================================================================ DEGRADATION ANALYSIS
def view_degradation(ctx):
    st, ss, G, style, PAL, explain = ctx["st"], ctx["ss"], ctx["G"], ctx["style"], ctx["PAL"], ctx["explain"]
    if not _need(ctx):
        return
    W = ctx["W"]
    units = sorted(W["unit"].unique())
    with st.expander("Settings"):
        c1, c2, c3, c4 = st.columns(4)
        c1.number_input("Bin length (h)", 0.25, 24.0, 2.0, key="dg_bin")
        c2.number_input("Reset: minimum drop (mV)", 1.0, 200.0, 8.0, key="dg_drop")
        c3.number_input("Analysis starts at (h) (after activation)", 0.0, 1000.0, 50.0, key="dg_tstart")
        c4.number_input("Stage segmentation penalty factor", 1.0, 100.0, 10.0, key="dg_pen")
    u = st.selectbox("Unit", units, key="dg_u")
    P = degradation_pipeline(ctx, u)
    if P is None:
        st.info("No stable windows for this unit.")
        return
    ser, rs, ri, sw, seg = P["series"], P["resets"], P["ri"], P["saw"], P["seg"]
    st.markdown("#### 1 · Resets and their causes")
    fig = go.Figure(go.Scatter(x=ser["t_h"], y=ser["dV_mV"], mode="lines", name="dV (binned)", line=dict(color="#9aa4b2", width=1.5)))
    if ri.get("ok"):
        fig.add_trace(go.Scatter(x=ri["t"], y=ri["baseline"], name="irreversible baseline", line=dict(color=PAL[7], width=2.5)))
    for cause, dd in rs.groupby("cause") if len(rs) else []:
        fig.add_trace(go.Scatter(x=dd["time_h"], y=dd["level_after"], mode="markers", name=cause,
                                 marker=dict(size=11, symbol="triangle-down", color=_cause_color(
                                     "electrochemical: voltage drop" if cause.startswith("electrochem") else
                                     "hardware" if "instrument" in cause or "electrolyte" in cause else "operating"))))
    if ctx["truth"] is not None and ctx["scen"] == "ss":
        tr = ctx["syn"]["resets"]
        tr = tr[tr["unit"] == u]
        for x in tr["time_h"]:
            fig.add_vline(x=x, line=dict(color=PAL[2], dash="dot", width=1))
    st.plotly_chart(style(fig, f"{u}: voltage rise, resets and irreversible baseline" +
                          (" (dotted: true spalling times)" if ctx["scen"] == "ss" else ""), "time (h)", "dV (mV)", 420),
                    key="dg_main")
    st.dataframe(rs.round(2), hide_index=True)
    st.markdown("#### 2 · Reversible vs irreversible degradation")
    if ri.get("ok"):
        m = st.columns(3)
        lo, hi = ri["irreversible_ci"]
        m[0].metric("Irreversible rate", f"{1000 * ri['irreversible_rate_mV_h']:.1f} uV/h", help="Theil-Sen on post-reset levels")
        m[1].metric("95 % CI (bootstrap)", f"{1000 * lo:.1f} – {1000 * hi:.1f} uV/h")
        m[2].metric("Reversible share at the end", f"{100 * ri['reversible_share_end']:.0f} %")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ri["t"], y=ri["baseline"] - ri["baseline"][0], name="irreversible", stackgroup="a",
                                 line=dict(color=PAL[7])))
        fig.add_trace(go.Scatter(x=ri["t"], y=np.maximum(ri["reversible"], 0), name="reversible", stackgroup="a",
                                 line=dict(color=PAL[3])))
        st.plotly_chart(style(fig, "Split of the voltage rise", "time (h)", "mV", 340), key="dg_split")
        explain("Irreversible = non-decreasing line through the levels right after each reset (permanent damage); "
                "reversible = what a reset can recover. Recoveries caused by KOH refills or service are counted as "
                "reversible but labelled separately.")
    st.markdown("#### 3 · Sawtooth model (cyclic passivation and spalling)")
    st.dataframe(sw["cycles"].round(3), hide_index=True)
    if sw["trend"]:
        tr_ = sw["trend"]
        st.caption("Per cycle: regrowth rate ×" + f"{np.exp(tr_.get('growth_trend_per_cycle', 0)):.2f}" +
                   (f", interval ×{np.exp(tr_['interval_trend_per_cycle']):.2f}" if "interval_trend_per_cycle" in tr_ else "") +
                   " (log-linear trend over cycles). Faster regrowth and shorter intervals = accelerating cycle.")
    if st.button("Back-test next-reset prediction on all units (causal)", key="dg_bt_btn"):
        sers, rss = {}, {}
        for uu in units:
            p = degradation_pipeline(ctx, uu)
            if p:
                sers[uu], rss[uu] = p["series"], p["resets"]
        ss["res_dg_bt"] = G.next_reset_backtest(sers, rss)
    if ss.get("res_dg_bt") is not None:
        bt = ss["res_dg_bt"]
        st.dataframe(bt.round(1), hide_index=True)
        if len(bt):
            st.caption(f"Median absolute error {bt['error_h'].abs().median():.0f} h over {len(bt)} cycles "
                       f"(standing at half of each cycle).")
    st.markdown("#### 4 · Stages (automatic piecewise-linear segmentation)")
    stage_col = {"1 activation": PAL[0], "2 passivation": PAL[3], "3 cyclic spalling": PAL[1], "4 terminal": PAL[7]}
    fig = go.Figure(go.Scatter(x=ser["t_h"], y=ser["dV_mV"], mode="markers", marker=dict(size=3, color="#9aa4b2"), name="data"))
    for stg, dd in seg.groupby("stage"):
        xs, ys = [], []
        for r in dd.itertuples():
            xs += [r.start_h, r.end_h, None]
            ys += [r.level_start, r.level_end, None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=stg, line=dict(color=stage_col.get(stg, PAL[6]), width=4)))
    st.plotly_chart(style(fig, f"{u}: detected stages", "time (h)", "dV (mV)", 380), key="dg_stages")
    if ctx["truth"] is not None and ctx["scen"] == "ss":
        tt = ctx["truth"][ctx["truth"]["unit"] == u]
        lab = np.full(len(ser), "", dtype=object)
        for r in seg.itertuples():
            lab[(ser["t_h"] >= r.start_h - 1e-9) & (ser["t_h"] <= r.end_h + 1e-9)] = r.stage
        truth_lab = np.interp(ser["t_h"], tt["time_h"], np.arange(len(tt)))
        tl = tt["phase"].to_numpy()[np.clip(np.round(truth_lab).astype(int), 0, len(tt) - 1)]
        acc = float(np.mean(lab == tl))
        st.caption(f"Synthetic check: stage label agrees with the true stage on {100 * acc:.0f} % of the time.")
    st.markdown("#### 5 · Replicates")
    rows = []
    for uu in units:
        p = degradation_pipeline(ctx, uu)
        if not p or not p["ri"].get("ok"):
            continue
        ec = p["electro"]
        term = p["seg"][p["seg"]["stage"] == "4 terminal"]
        rows.append(dict(unit=uu, irreversible_uV_h=1000 * p["ri"]["irreversible_rate_mV_h"], n_resets=len(ec),
                         first_reset_h=float(ec["time_h"].min()) if len(ec) else np.nan,
                         mean_interval_h=float(np.diff(ec["time_h"]).mean()) if len(ec) > 1 else np.nan,
                         terminal_onset_h=float(term["start_h"].min()) if len(term) else np.nan,
                         end_h=float(p["series"]["t_h"].max())))
    rep = pd.DataFrame(rows)
    st.dataframe(rep.round(2), hide_index=True)
    if len(rep) >= 2:
        st.dataframe(G.replicate_summary(rep, ["irreversible_uV_h", "n_resets", "first_reset_h", "mean_interval_h",
                                               "terminal_onset_h"]).round(2), hide_index=True)
        st.caption(f"n = {len(rep)} cells: confidence intervals are wide by construction; report them, not only means.")
    ss["res_replicates"] = rep


# ============================================================================================ ANOMALIES
def view_anomalies(ctx):
    st, ss, AN, style, PAL, explain = ctx["st"], ctx["ss"], ctx["AN"], ctx["style"], ctx["PAL"], ctx["explain"]
    if not _need(ctx):
        return
    W, data = ctx["W"], ctx["data"]
    units = sorted(W["unit"].unique())
    with st.expander("Detector settings"):
        c1, c2, c3, c4 = st.columns(4)
        use = c1.multiselect("Detectors", ["nis", "cusum", "ewma", "T2", "Q", "iforest"],
                             default=["nis", "cusum", "ewma", "T2", "Q", "iforest"], key="an_use")
        c2.number_input("Reference (training) period ends at (h)", 20.0, 2000.0, 200.0, key="an_train")
        c3.number_input("Ignore alarms before (h) (activation)", 0.0, 500.0, 50.0, key="an_ignore")
        c4.number_input("CUSUM threshold h", 2.0, 30.0, 8.0, key="an_cus")
        c1.number_input("Voltage noise per window (mV)", 0.2, 20.0, 2.0, key="an_r")
        c2.number_input("EWMA limit (sigma)", 2.0, 8.0, 4.0, key="an_ewma")
    if st.button("Run anomaly detection on all units", key="an_run", type="primary"):
        out = {}
        for u in units:
            out[u] = AN.run_unit(data["timeseries"], W, data.get("events"), u, train_h=float(ss.get("an_train", 200.0)),
                                 use=tuple(use) or ("nis",), r_mV=float(ss.get("an_r", 2.0)),
                                 cusum_h=float(ss.get("an_cus", 8.0)), ewma_L=float(ss.get("an_ewma", 4.0)),
                                 ignore_before_h=float(ss.get("an_ignore", 50.0)))
        ss["res_an"] = out
    res = ss.get("res_an")
    if not res:
        st.info("Press Run to detect anomalies (causal: every alarm uses only past data).")
        return
    u = st.selectbox("Unit", [x for x in units if x in res], key="an_u")
    r = res[u]
    if not len(r["residuals"]):
        st.info("Too few windows.")
        return
    ep = r["episodes"]
    fig = go.Figure(go.Scatter(x=r["residuals"]["t_h"], y=r["residuals"]["U_mV"], mode="lines", name="U (window)",
                               line=dict(color="#9aa4b2", width=1.5)))
    for e in ep.itertuples():
        fig.add_vrect(x0=e.start_h - 0.5, x1=e.end_h + 0.5, fillcolor=_cause_color(e.cause), opacity=0.25, line_width=0)
    if labels_ok(ctx):
        lab = ctx["labels"][ctx["labels"]["unit"] == u]
        fig.add_trace(go.Scatter(x=lab["start_h"], y=np.full(len(lab), r["residuals"]["U_mV"].min()), mode="markers",
                                 name="true anomaly (synthetic)", marker=dict(symbol="line-ns-open", size=14, color=PAL[2])))
    st.plotly_chart(style(fig, f"{u}: alarms coloured by probable cause", "time (h)", "U (mV)", 420), key="an_main")
    c1, c2 = st.columns(2)
    rr = r["residuals"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rr["t_h"], y=rr["cusum_pos"], name="CUSUM +", line=dict(color=PAL[7])))
    fig.add_trace(go.Scatter(x=rr["t_h"], y=rr["cusum_neg"], name="CUSUM −", line=dict(color=PAL[2])))
    fig.add_hline(y=float(ss.get("an_cus", 8.0)), line=dict(dash="dash", color="#8a93a3"))
    c1.plotly_chart(style(fig, "Residual layer (CUSUM on twin innovations)", "time (h)", "", 300), key="an_cusum")
    mv = r["multivariate"]
    fig = go.Figure(go.Scatter(x=mv["t_h"], y=mv["Q"], name="Q (SPE)", line=dict(color=PAL[0])))
    fig.add_hline(y=float(mv["thr_Q"].iloc[0]), line=dict(dash="dash", color="#8a93a3"))
    fig.update_yaxes(type="log")
    c2.plotly_chart(style(fig, "Multivariate layer (PCA Q statistic)", "time (h)", "", 300), key="an_q")
    st.dataframe(ep.round(2), hide_index=True)
    explain("Layer 1: innovations of a local-trend Kalman filter (NIS, CUSUM, EWMA). Layer 2: PCA T²/Q and isolation "
            "forest trained on a reference period. Layer 3: cause from the event log, temperature, isolation and sign. "
            "Hardware events (power supply, refills) are separated from electrochemical changes.")
    st.markdown("#### Evaluation and early warning")
    rows, leads = [], []
    for uu, rv in res.items():
        tend = float(data["timeseries"].loc[data["timeseries"]["unit"] == uu, "time_h"].max())
        if labels_ok(ctx):
            ev = AN.evaluate(rv["episodes"], ctx["labels"], uu, tend)
            if len(ev["summary"]):
                rows.append(ev["summary"].assign(unit=uu, false_per_100h=ev["false_per_100h"]))
        onset = None
        if ctx["truth"] is not None and "phase" in ctx["truth"]:
            tt = ctx["truth"]
            onset = tt.loc[(tt["unit"] == uu) & (tt["phase"] == "4 terminal"), "time_h"].min()
        lt = AN.lead_time(rv["episodes"], tend, onset)
        leads.append(dict(unit=uu, failure_or_end_h=tend, first_persistent_alarm_h=lt["t_alarm"], lead_time_h=lt["lead_h"],
                          alarm_minus_true_onset_h=lt["after_onset_h"]))
    if rows:
        ev_all = pd.concat(rows, ignore_index=True)
        agg = ev_all.groupby(["cls", "kind"]).agg(n=("n", "sum"), detected=("detected", "sum"),
                                                  median_delay_h=("median_delay_h", "median")).reset_index()
        agg["rate"] = agg["detected"] / agg["n"]
        st.dataframe(agg.round(2), hide_index=True)
        st.caption(f"False alarms: {ev_all.groupby('unit')['false_per_100h'].first().mean():.2f} per 100 h (mean over units).")
    st.dataframe(pd.DataFrame(leads).round(1), hide_index=True)
    ss["res_an_eval"] = dict(summary=pd.concat(rows) if rows else None, leads=pd.DataFrame(leads))
    explain("Lead time = failure (or end of test) minus the first persistent 'accelerated degradation' alarm. On real "
            "data only logged events can serve as labels; injected synthetic anomalies give a controlled second test.")
    eis = data.get("eis")
    if eis is not None and len(eis):
        st.markdown("#### DRT precursor")
        A = drt_results(ctx)
        if A is not None and len(A["peaks"]):
            tr = tracked(ctx, A)
            c1, c2, c3 = st.columns(3)
            pk = c1.selectbox("Peak", sorted(tr.loc[tr["peak"] != "minor", "peak"].unique()), index=0, key="an_pk")
            jj = c2.selectbox("Current", sorted(tr["j_A_cm2"].dropna().unique()), key="an_pj")
            kf = c3.number_input("Warn when R > k × baseline, k =", 1.05, 10.0, 1.5, key="an_pkk")
            pr = ctx["AN"].drt_precursor(tr, u, pk, jj, kf)
            s_ = pr["series"]
            if len(s_):
                fig = go.Figure(go.Scatter(x=s_["time_h"], y=s_.get("ratio", s_["R"]), mode="lines+markers",
                                           line=dict(color=PAL[6], width=2), name="R / baseline"))
                fig.add_hline(y=kf, line=dict(dash="dash", color="#e34948"))
                st.plotly_chart(style(fig, f"{u}: {pk} at {jj:g} A/cm2 relative to its post-activation baseline",
                                      "time (h)", "ratio", 320), key="an_prec")
                st.caption(f"First warning: {pr['t_warn']:.0f} h" if np.isfinite(pr["t_warn"]) else "No warning.")


def labels_ok(ctx) -> bool:
    return ctx.get("labels") is not None and len(ctx["labels"]) > 0


# ============================================================================================ PAPER FIGURES
def view_paper(ctx):
    st, ss = ctx["st"], ctx["ss"]
    st.markdown("---")
    st.markdown("#### Paper figures (PNG 300 dpi + the data behind each figure)")
    st.caption("Generated from the results currently in the session (run DRT & mechanisms, Degradation analysis and "
               "Anomalies first). Every figure is saved with a CSV of its data and a caption stating whether the data "
               "are synthetic.")
    if st.button("Generate figure set", key="pf_btn"):
        import figures as FG
        try:
            files = FG.figure_set(ctx)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Figure generation failed: {exc}")
            files = {}
        if files:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for name, data in files.items():
                    z.writestr(name, data)
            ss["res_paper_zip"] = buf.getvalue()
            ss["res_paper_names"] = sorted(files)
    if ss.get("res_paper_zip"):
        st.download_button("Download figures (zip)", ss["res_paper_zip"], "awe_twin_figures.zip", key="pf_dl")
        st.caption(", ".join(ss.get("res_paper_names", [])))
