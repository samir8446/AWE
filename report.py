"""Plausibility report for a generated dataset (validation code: the only module allowed to read truth/).

    python -m awe_synth.report datasets/awe_synth_v1.0.0_seed0

writes <dataset>/report/plausibility_report.html (self-contained), checks.csv and the PNG figures.
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import physics as ph  # noqa: E402
from .config import BENCH, DESIGN, GROUPS, PHYS  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GCOL = {g.name: SLOTS[i] for i, g in enumerate(GROUPS)}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
                     "legend.frameon": False, "figure.dpi": 110})
BOL_STATE = dict(j0c_f=1.0, j0a_f=1.0, ac_f=1.0, aa_f=1.0, sep_f=1.0, R_c=DESIGN["R_contact0_ohm_cm2"].value)


def load(root):
    rd = lambda d, f: pd.read_csv(os.path.join(root, d, f))
    D = {k: rd("data", f) for k, f in [("units", "units.csv"), ("ts", "timeseries_1h.csv"), ("ev", "events.csv"),
                                       ("lsv", "lsv.csv"), ("eis", "eis.csv"), ("cv", "cv.csv"),
                                       ("titr", "titration.csv")]}
    T = {k: rd("truth", f) for k, f in [("state", "hourly_state.csv"), ("mech", "mechanisms_at_reference.csv"),
                                        ("diag", "diagnostics_truth.csv"), ("up", "unit_parameters.csv"),
                                        ("faults", "faults.csv"), ("labels", "measurement_labels.csv")]}
    with open(os.path.join(root, "manifest.json")) as fh:
        man = json.load(fh)
    return D, T, man


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, area, name, value, criterion, ok, source, note=""):
        self.rows.append(dict(area=area, check=name, value=value, criterion=criterion,
                              result="INFO" if ok is None else ("PASS" if bool(ok) else "FAIL"), source=source,
                              note=note))

    def df(self):
        return pd.DataFrame(self.rows)


def _png(fig):
    b = io.BytesIO()
    fig.savefig(b, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    return b.getvalue()


# ------------------------------------------------------------------------------------------------ checks
def hf_intercept(e: pd.DataFrame) -> float:
    """Real part where -Z'' first crosses zero coming from high frequency (inductive -> capacitive), ohm."""
    e = e.sort_values("freq_Hz", ascending=False)
    zi, zr = e["z_im_neg"].to_numpy(), e["z_re"].to_numpy()
    k = np.nonzero(zi >= 0)[0]
    if not len(k) or k[0] == 0:
        return float(zr[0])
    i = k[0]
    return float(zr[i - 1] + (zr[i] - zr[i - 1]) * (0 - zi[i - 1]) / (zi[i] - zi[i - 1]))

def bol_checks(C: Checks, D, T):
    m80 = ph.molarity_from_wt(30.0, 353.15)
    U = lambda j, Tc, P, w=30.0: float(ph.cell(np.array([j]), Tc + 273.15, P, ph.molarity_from_wt(w, Tc + 273.15),
                                               BOL_STATE)["U"][0])
    u04 = U(0.4, 80, 1)
    C.add("BoL", "U(0.4 A/cm2, 80 C, 1 bar, 30 wt%)", f"{u04:.3f} V", "1.6-2.0 V/cell (industrial window, 0.1-0.8 "
          "A/cm2)", 1.6 <= u04 <= 2.0, "Qu26 sec 1")
    u08 = U(0.8, 80, 1)
    C.add("BoL", "U(0.8 A/cm2, 80 C, 1 bar)", f"{u08:.3f} V", "<= 2.1 V (upper end of window + narrow-gap cell)",
          u08 <= 2.1, "Qu26 sec 1", "ASSUMPTION on the margin above 2.0 V")
    t = [U(0.4, x, 1) for x in (60, 70, 80)]
    C.add("BoL", "Temperature effect at 0.4 A/cm2: U(60) > U(70) > U(80)",
          " > ".join(f"{x:.3f}" for x in t), "monotonic decrease with T", t[0] > t[1] > t[2], "Hu22 sec 3.2")
    dp = 1000 * (U(0.4, 80, 16) - U(0.4, 80, 1))
    nern = 1000 * float(ph.E_rev(353.15, 16, m80) - ph.E_rev(353.15, 1, m80))
    C.add("BoL", "Pressure effect 16 vs 1 bar at 0.4 A/cm2", f"{dp:+.1f} mV (Nernst alone {nern:+.1f} mV)",
          "0 < dU < Nernst shift (bubble coverage falls with P)", 0 < dp < nern, "Hu22 eq 11, 17")
    # Tafel slopes recovered from the BoL LSV data (iR-corrected with the EIS high-frequency intercept)
    lsv, eis, diag = D["lsv"], D["eis"], T["diag"]
    rows = []
    for u, d in lsv.groupby("unit"):
        t0 = d["time_h"].min()
        L = d[d["time_h"] == t0]
        e = eis[(eis["unit"] == u) & (eis["time_h"] == t0) & (eis["j_A_cm2"] == 0.05)]
        hfr = hf_intercept(e)
        k = (L["j_A_cm2"] >= 0.01) & (L["j_A_cm2"] <= 0.1)
        Uir = L.loc[k, "voltage_V"] - L.loc[k, "current_A"] * hfr
        slope = np.polyfit(np.log10(L.loc[k, "j_A_cm2"]), Uir, 1)[0] * 1000
        tr = diag[(diag["unit"] == u) & (diag["time_h"] == t0) & (diag["j_A_cm2"] == 0.05)].iloc[0]
        rows.append(dict(unit=u, fitted=slope, truth=tr["tafel_c_mV_dec"] + tr["tafel_a_mV_dec"], hfr=hfr * DESIGN["area_cm2"].value,
                         hfr_true=tr["R_ohm_ohm_cm2"]))
    tf = pd.DataFrame(rows)
    err = (tf["fitted"] / tf["truth"] - 1).abs().max()
    C.add("BoL", "Cell Tafel slope (b_c + b_a) from BoL LSV, 10-100 mA/cm2, iR-corrected",
          f"{tf['fitted'].median():.0f} mV/dec (truth {tf['truth'].median():.0f}); max error {100 * err:.0f} %",
          "within 15 % of the truth (asinh curvature and theta(j) bias a Tafel fit)", err < 0.15, "Hu22 Table B3",
          "Truth = 2.303RT/(n alpha F) with Hu22 alphas: ~77 (HER) + ~80 (OER) mV/dec at 70 C. The OER value is on the "
          "high side for Ni (often reported lower) - flagged, not changed: it follows the cited alpha_a.")
    C.add("BoL", "High-frequency resistance at BoL (EIS, 70 C)",
          f"{tf['hfr'].median():.3f} ohm cm2 (range {tf['hfr'].min():.3f}-{tf['hfr'].max():.3f})",
          "0.25-0.60 ohm cm2: Zirfon 0.27 + gaps + contacts", tf["hfr"].between(0.25, 0.6).all(), "Qu26 Table 2")
    Rs = float(ph.r_separator(ph.sigma_koh(m80, 353.15)))
    C.add("BoL", "Separator area resistance (structural model, 80 C, 30 wt%)", f"{Rs:.3f} ohm cm2",
          "~0.27 ohm cm2 (Zirfon)", abs(Rs - 0.27) < 0.03, "Qu26 Table 2, Hu22 eq 24-25")
    try:      # independent implementation in the platform engine (same papers, separate code)
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import engine as E
        dz = E.CellDesign(area_cm2=DESIGN["area_cm2"].value, gap_anode_mm=0.5, gap_cathode_mm=0.5,
                          membrane_model="structural", sep_porosity=DESIGN["sep_porosity"].value, sep_thickness_mm=0.5,
                          electrode_sigma_S_m=1e12)
        diffs = []
        for Tc, P in ((60, 1), (80, 1), (80, 16)):
            for jv in (0.1, 0.4, 0.8):
                ue = E.cell_voltage(jv, E.OperatingConditions(T_C=Tc, P_bar=P, koh_wt=30.0), dz, E.Kinetics(),
                                    E.DegradationState(dR_ohm_cm2=DESIGN["R_contact0_ohm_cm2"].value))["U"][0]
                diffs.append(1000 * (U(jv, Tc, P) - ue))
        dmax = max(abs(x) for x in diffs)
        C.add("BoL", "Generator vs platform engine (independent code), 9 points 0.1-0.8 A/cm2",
              f"max |dU| {dmax:.1f} mV", "< 10 mV (differences: asinh vs Tafel form, eta_conc)", dmax < 10,
              "Hu22, Ham25", "Agreement means the platform's physics is nearly correctly specified on this data: "
              "an optimistic setting compared with real cells.")
    except Exception as exc:     # engine not present next to the generator
        C.add("BoL", "Generator vs platform engine", f"not run ({exc.__class__.__name__})", "INFO", None, "-")
    eF = ph.faraday_eff(np.array([0.05, 0.4, 0.8]), 353.15)
    C.add("BoL", "Faraday efficiency at 0.05 / 0.4 / 0.8 A/cm2 (80 C)", " / ".join(f"{x:.3f}" for x in eF),
          ">= 0.99 at rated load; lower at low load", eF[1] >= 0.99 and eF[0] < eF[1], "Hu22 eq 7, sec 3.2.1",
          "Ulleberg's stack f1 for a single cell probably overstates the low-load loss (no single-cell value in hand).")
    jj = np.linspace(0.02, 0.8, 400)
    for P in (1.0, 16.0):
        h = ph.gases(jj, 353.15, P, m80)["HTO_pct"]
        jmin = jj[np.argmax(h <= 2.0)] if (h <= 2.0).any() else np.nan
        C.add("Safety", f"Minimum load for HTO <= 2 % at {P:g} bar, 80 C (BoL)",
              f"{jmin:.3f} A/cm2 = {100 * jmin / 0.8:.0f} % of rated 0.8 A/cm2",
              "1 bar: below 10 % ; 16 bar: 10-40 % of rated", (jmin / 0.8 < 0.1) if P == 1 else (0.1 <= jmin / 0.8 <= 0.4),
              "Hu22 sec 3.3 (10-40 %), Zhang24 (40 % at max pressure)")
    return tf


def balance_checks(C: Checks, D, T):
    s = T["state"]
    on = s["j_true_A_cm2"] > 0
    F = ph.F
    I = s["j_true_A_cm2"] * DESIGN["area_cm2"].value
    # within each hour the mole flows are means of minute values -> compare means of means
    p = s["nH2_prod_mol_s"].where(s["nH2_prod_mol_s"] > 0)
    r1 = ((s["nH2_out_mol_s"] + s["nH2_cross_mol_s"] - s["nH2_prod_mol_s"]) / p).abs().max()
    r2 = ((s["nO2_out_mol_s"] + s["nO2_cross_mol_s"] - s["nO2_prod_mol_s"]) / (p / 2)).abs().max()
    C.add("Balance", "H2: out + crossover - produced, relative to production", f"max {r1:.1e}",
          "< 1e-4 (files store 6 significant digits)", r1 < 1e-4, "Hu22 eq 5, 9, 40")
    C.add("Balance", "O2: out + crossover - produced, relative to production", f"max {r2:.1e}", "< 1e-4", r2 < 1e-4,
          "Hu22 eq 5, 9")
    const = s["unit"].str.rsplit("-", n=1).str[0].isin([g.name for g in GROUPS if g.mode == "constant"])
    k = s["state"].eq("run") & const
    ratio = (s.loc[k, "nH2_prod_mol_s"] / (I[k] / (2 * F)) / s.loc[k, "eta_F"]).sub(1).abs().max()
    C.add("Balance", "Produced H2 = eta_F I / 2F (steady run hours)", f"max deviation {ratio:.1e}", "< 1e-4 (6-digit files)",
          ratio < 1e-4, "Hu22 eq 5")
    # measured H2 flow vs Faraday (data only)
    ts = D["ts"]
    kk = ts["state"].eq("run") & ts["h2_flow_Nm3_h"].notna() & ts["current_A"].notna()
    fe = ts.loc[kk, "h2_flow_Nm3_h"] / ph.molar_to_Nm3_h(ts.loc[kk, "current_A"] / (2 * F))
    C.add("Balance", "Apparent Faraday efficiency from MEASURED H2 flow (run hours)",
          f"median {fe.median():.4f}, 1-99 % {fe.quantile(0.01):.3f}-{fe.quantile(0.99):.3f}",
          "0.97-1.00 (eta_F x crossover loss, +-1.5 % meter noise)", 0.97 <= fe.median() <= 1.0, "Hu22 eq 5, 7")
    # energy: U I = eta_F I V_tn + Q_loss + Q_store + Q_lye (identity of the bench model, checked on hourly means)
    P_el = s["U_true_V"] * I                     # approx: mean(U) mean(I) vs mean(U I) differs in mixed hours
    kr = s["state"].eq("run") & const     # hourly means of products are exact only at steady load
    resid = (P_el - (s["eta_F"] * I * s["V_tn_V"] + s["Q_gen_W"]) * 1).loc[kr]
    res2 = (s["Q_gen_W"] - s["Q_loss_W"] - s["Q_store_W"] - s["Q_lye_W"]).abs().max()
    C.add("Balance", "Energy: U I = eta_F I V_tn + Q_gen (steady run hours)", f"max |res| {resid.abs().max():.2e} W "
          f"of {P_el[kr].mean():.0f} W", "< 0.1 % of electrical power", resid.abs().max() < 1e-3 * P_el[kr].mean(),
          "Hu22 eq 53-56, App. A")
    C.add("Balance", "Heat: Q_gen = Q_loss + Q_store + Q_lye (all hours)", f"max |res| {res2:.1e} W", "< 1e-2 W (6-digit files)",
          res2 < 1e-2, "Hu22 eq 53", "Closes by construction (bench controller); this checks the bookkeeping.")
    # measured lye temperature rise vs truth heat to lye
    mcp = BENCH["flow_L_min"].value / 60 * PHYS["rho_lye"].value * PHYS["cp_lye"].value
    j = ts.loc[kr.index.isin([]) if False else ts["state"].eq("run")].merge(
        s[["unit", "time_h", "Q_lye_W"]], on=["unit", "time_h"])
    est = mcp * (j["T_out_C"] - j["T_in_C"])
    corr = np.corrcoef(est.fillna(0), j["Q_lye_W"])[0, 1]
    C.add("Balance", "Lye heat from MEASURED T_out - T_in vs true Q_lye", f"r = {corr:.2f}; median error "
          f"{(est - j['Q_lye_W']).median():+.2f} W", "r > 0.5 (dT ~1 K vs 0.2 K sensor noise)", corr > 0.5,
          "Hu22 eq 61", "Small single cell: lye dT is ~1 K, so this balance is weak in the data by design.")
    # water: top-ups vs electrolysed water
    ev = D["ev"]
    for u in ["G3_T80_ref-1", "G6_dynamic-1"]:
        su = s[s["unit"] == u]
        used = (su["nH2_prod_mol_s"] * 3600 * 18.015e-3).sum()
        e = ev[(ev["unit"] == u) & (ev["event"] == "water_topup")]
        added = e["note"].str.extract(r"(\d+) g")[0].astype(float).sum() / 1000
        n_ref = int((ev[(ev["unit"] == u)]["event"] == "electrolyte_change").sum())
        tol = (n_ref + 1) * BENCH["lye_mass_kg"].value * (1 - 30 / 31) + 0.01
        C.add("Balance", f"Water make-up vs electrolysed water ({u})", f"electrolysed {used:.2f} kg, added {added:.2f} kg",
              f"difference <= water held in the 1 wt% band per refill cycle ({tol:.2f} kg)", abs(used - added) <= tol,
              "Faraday + mass balance")


def safety_checks(C: Checks, D, T):
    s = T["state"]
    mx = s.groupby("unit")["HTO_true_pct"].max()
    lim = PHYS["HTO_shutdown_pct"].value
    up = T["up"].set_index("unit")
    over = sorted(up.index[np.isfinite(up["trip_h"])])       # interlock: > 2 % for 5 consecutive minutes
    faulty = sorted(T["faults"].loc[T["faults"]["kind"] == "HTO_rise", "unit"])
    C.add("Safety", "Units tripped by the HTO interlock (> 2 % for 5 min)", ", ".join(over) or "none",
          f"only the HTO-fault unit(s) {faulty}", over == faulty, "Hu22 sec 3.3, Qu26 sec 3.3")
    for u in faulty:
        f = T["faults"].set_index("unit").loc[u, "t_h"]
        C.add("Safety", f"{u}: bench interlock trip", f"trip at {up.loc[u, 'trip_h']:.1f} h, fault onset {f:g} h",
              "trip after onset, test ended", up.loc[u, "trip_h"] > f, "Hu22 sec 3.3")
    normal = mx.drop(faulty)
    C.add("Safety", "Highest true HTO among fault-free units", f"{normal.max():.2f} % ({normal.idxmax()})",
          "< 2 %", normal.max() < lim, "Hu22 sec 3.3")


def degradation_checks(C: Checks, D, T):
    m = T["mech"]
    up = T["up"].set_index("unit")
    faults = T["faults"]
    rows = []
    for u, d in m.groupby("unit"):
        end = up.loc[u, "trip_h"] if np.isfinite(up.loc[u, "trip_h"]) else 2999
        d = d[d["time_h"] <= end]
        a = d[(d["time_h"] >= 100) & (d["time_h"] <= 300)]["dV_total_mV"].mean()
        b = d[d["time_h"] >= end - 200]["dV_total_mV"].mean()
        span = (end - 100) - 200
        st = T["state"][T["state"]["unit"] == u].set_index("time_h")["op_hours"]
        op_span = st.loc[end - 200:end].mean() - st.loc[100:300].mean()
        rows.append(dict(unit=u, group=u.rsplit("-", 1)[0], rate_uV_h=1000 * (b - a) / span,
                         rate_uV_per_op_h=1000 * (b - a) / op_span, faulty=u in set(faults["unit"])))
    r = pd.DataFrame(rows)
    g = r[~r["faulty"]].groupby("group")["rate_uV_h"].agg(["mean", "min", "max"])
    for grp, x in g.iterrows():
        C.add("Degradation", f"Mean ageing rate at reference (80 C, 1 bar, 0.4 A/cm2), {grp}",
              f"{x['mean']:.0f} uV/h ({x['min']:.0f}-{x['max']:.0f})", "INFO", None,
              "Buttler18, Masel, Karacan22",
              "Accelerated on purpose: lab Ni-based AWE cells report a few uV/h (Masel: 3-5 uV/h at 60 C).")
    gop = r[~r["faulty"]].groupby("group")["rate_uV_per_op_h"].mean()
    C.add("Degradation", "Ageing per OPERATING hour: dynamic vs constant reference",
          f"G6_dynamic {gop['G6_dynamic']:.0f} vs G3_T80_ref {gop['G3_T80_ref']:.0f} uV/h_op", "dynamic > constant",
          gop["G6_dynamic"] > gop["G3_T80_ref"], "Kim22, Uchino18, Hug93",
          "Per wall-clock hour the dynamic units look milder only because they run ~55 % of the time at lower mean load.")
    ref = g.loc["G3_T80_ref", "mean"]
    C.add("Degradation", "Acceleration factor vs a 3-5 uV/h lab cell (Masel)", f"{ref / 5:.0f}-{ref / 3:.0f}x", "INFO",
          None, "Masel", "STATED: rates are accelerated so 3000 h shows clear ageing.")
    C.add("Degradation", "Temperature ordering of ageing rate (G1 < G2 < G3)",
          f"{g.loc['G1_T60', 'mean']:.0f} < {g.loc['G2_T70', 'mean']:.0f} < {g.loc['G3_T80_ref', 'mean']:.0f} uV/h",
          "increasing with T (Arrhenius ASSUMPTION)", g.loc["G1_T60", "mean"] < g.loc["G2_T70", "mean"] < ref,
          "ASSUMPTION")
    # acceleration visible: slope after onset > slope before (anode mechanism at reference)
    acc = []
    for u, row in up.iterrows():
        on = row["accel_onset_h"]
        if not np.isfinite(on) or u in set(faults["unit"]):
            continue
        d = m[m["unit"] == u]
        pre = d[(d["time_h"] > on - 400) & (d["time_h"] < on)]
        post = d[(d["time_h"] > on + 50) & (d["time_h"] < min(on + 450, 2999))]
        if len(pre) > 50 and len(post) > 50:
            s1 = np.polyfit(pre["time_h"], pre["dV_anode_activity_incl_acceleration_mV"], 1)[0]
            s2 = np.polyfit(post["time_h"], post["dV_anode_activity_incl_acceleration_mV"], 1)[0]
            acc.append(s2 / s1)
    C.add("Degradation", "Anode ageing speeds up after the onset (slope ratio after/before)",
          f"median {np.median(acc):.1f}x over {len(acc)} units", "> 2x", np.median(acc) > 2, "ASSUMPTION (known answer)")
    n_on = int(np.isfinite(up["accel_onset_h"]).sum())
    C.add("Degradation", "Units with an acceleration onset inside 3000 h", f"{n_on} of {len(up)}", "INFO", None,
          "truth/unit_parameters.csv", "Units without one (mostly 60 C) are the 'no speed-up' cases.")
    # recovery at electrolyte change (measured voltage, run hours, 48 h before vs 48 h after restart)
    ts, ev = D["ts"], D["ev"]
    rec = []
    for _, e in ev[ev["event"] == "electrolyte_change"].iterrows():
        d = ts[(ts["unit"] == e["unit"]) & ts["state"].eq("run")]
        units = D["units"].set_index("unit")
        if units.loc[e["unit"], "operating_mode"] != "constant":
            continue
        b = d[(d["time_h"] < e["time_h"] - 12) & (d["time_h"] > e["time_h"] - 60)]["voltage_V"].median()
        a = d[(d["time_h"] > e["time_h"] + 6) & (d["time_h"] < e["time_h"] + 54)]["voltage_V"].median()
        rec.append(1000 * (b - a))
    rec = np.array(rec)
    C.add("Degradation", "Partial recovery after electrolyte change (measured, constant-load units)",
          f"median {np.nanmedian(rec):.1f} mV drop (range {np.nanmin(rec):.1f}-{np.nanmax(rec):.1f})",
          "> 0 (reversible part recovers, irreversible part stays)", np.nanmedian(rec) > 0, "Rommal85")
    mix = m.groupby("unit").apply(lambda d: d["dV_interaction_residual_mV"].abs().max())
    C.add("Degradation", "Mechanism decomposition: interaction residual (max over units)",
          f"{mix.max():.1f} mV ({mix.idxmax()})", "INFO", None, "-",
          "One-at-a-time contributions do not add exactly (log terms); the residual is reported, not hidden.")
    return r


def data_checks(C: Checks, D, T):
    ts = D["ts"]
    H = D["units"]["test_duration_h"].iloc[0]
    exp = len(D["units"]) * H
    C.add("Realism", "Rows present in the hourly log", f"{len(ts)} of {exp} ({100 * len(ts) / exp:.1f} %)",
          "INFO", None, "-", "Missing = logger outages + end of test after the HTO trip.")
    lab = T["labels"]
    for kind, n in lab["kind"].value_counts().items():
        C.add("Realism", f"Labelled measurement artefacts: {kind}", str(n), "INFO", None, "truth/measurement_labels.csv")
    ses = D["lsv"].groupby("unit")["time_h"].nunique()
    C.add("Realism", "Diagnostic sessions per unit", f"{ses.min()}-{ses.max()}", "INFO", None, "-")


# ------------------------------------------------------------------------------------------------ figures
def figures(D, T, tf):
    figs = {}
    # 1 BoL polarisation curves by condition (model) + LSV data at BoL
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    jj = np.linspace(0.005, 1.0, 200)
    for g in GROUPS:
        if g.mode == "dynamic":
            continue
        U = ph.cell(jj, g.T_C + 273.15, g.P_bar, ph.molarity_from_wt(g.koh_wt, g.T_C + 273.15), BOL_STATE)["U"]
        ax[0].plot(jj, U, color=GCOL[g.name], label=f"{g.name}: {g.T_C:g} C, {g.P_bar:g} bar, {g.koh_wt:g} wt%")
    ax[0].fill_between([0.1, 0.8], 1.6, 2.0, color=GRID, alpha=0.6, zorder=0, lw=0)
    ax[0].text(0.12, 1.61, "industrial window 1.6-2.0 V, 0.1-0.8 A/cm2 [Qu26]", color=MUTED, fontsize=7.5)
    ax[0].set(xlabel="j (A/cm2)", ylabel="cell voltage (V)", title="(a) BoL polarisation curves at operating conditions")
    ax[0].legend(fontsize=7, loc="lower right")
    lsv = D["lsv"]
    for i, (u, d) in enumerate(lsv.groupby("unit")):
        t0 = d["time_h"].min()
        L = d[d["time_h"] == t0]
        ax[1].plot(L["j_A_cm2"], L["voltage_V"], "o", ms=3, color=GCOL[u.rsplit("-", 1)[0]], alpha=0.6,
                   label=u.rsplit("-", 1)[0] if u.endswith("-1") else None)
    ax[1].set(xscale="log", xlabel="j (A/cm2, log)", ylabel="cell voltage (V)",
              title=f"(b) BoL LSV data, all 18 units, 70 C, 1 bar\ncell Tafel slope fitted {tf['fitted'].median():.0f} "
                    f"mV/dec (truth {tf['truth'].median():.0f})")
    ax[1].legend(fontsize=7)
    figs["fig1_bol_polarisation.png"] = _png(fig)
    # 2 voltage vs time per group (run hours)
    ts = D["ts"]
    fig, axs = plt.subplots(2, 3, figsize=(11, 5.4), sharex=True)
    ev = D["ev"]
    for ax, g in zip(axs.ravel(), GROUPS):
        for i in range(1, 4):
            u = f"{g.name}-{i}"
            d = ts[(ts["unit"] == u) & ts["state"].eq("run")]
            if g.mode == "dynamic":     # compare like with like: hours near 0.4 A/cm2
                d = d[d["j_A_cm2"].between(0.38, 0.42)]
            ax.plot(d["time_h"], d["voltage_V"], ".", ms=1.2, color=SLOTS[i - 1], label=u)
        for _, e in ev[(ev["unit"] == f"{g.name}-1") & (ev["event"] == "electrolyte_change")].iterrows():
            ax.axvline(e["time_h"], color=MUTED, lw=0.6, ls="--")
        ax.set_title(f"{g.name}" + (" (hours at 0.38-0.42 A/cm2)" if g.mode == "dynamic" else ""), fontsize=8.5)
        ax.legend(fontsize=6.5, markerscale=5, loc="upper left")
        ax.set_ylabel("U (V)")
    for ax in axs[1]:
        ax.set_xlabel("time (h)")
    fig.suptitle("Measured cell voltage (hourly, run hours); dashed = electrolyte change of unit 1", fontsize=9)
    figs["fig2_voltage_by_group.png"] = _png(fig)
    # 3 mechanism stacks at reference for 3 units
    m = T["mech"]
    groups_m = {"cathode activity (intrinsic + hydride)": ["dV_cathode_activity_intrinsic_mV", "dV_cathode_hydride_reversible_mV"],
                "anode activity (accelerates)": ["dV_anode_activity_incl_acceleration_mV"],
                "ECSA loss": ["dV_ECSA_loss_mV"], "reverse current at stops": ["dV_intermittent_reverse_current_mV"],
                "ohmic: separator + contacts": ["dV_separator_fouling_mV", "dV_contact_degradation_mV"],
                "electrolyte (carbonation, conc.)": ["dV_electrolyte_carbonation_concentration_mV"],
                "injected fault": ["dV_injected_fault_mV"]}
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    up = T["up"].set_index("unit")
    for ax, u in zip(axs, ["G3_T80_ref-1", "G6_dynamic-1", "G2_T70-3"]):
        d = m[m["unit"] == u].iloc[::6]
        ys = [d[c].sum(axis=1).clip(lower=0).to_numpy() for c in groups_m.values()]
        ax.stackplot(d["time_h"], ys, colors=SLOTS[:len(ys)], labels=list(groups_m), edgecolor="white", linewidth=0.3)
        ax.plot(d["time_h"], d["dV_total_mV"], color=INK, lw=1, label="total (incl. interaction)")
        on = up.loc[u, "accel_onset_h"]
        if np.isfinite(on):
            ax.axvline(on, color=INK, ls=":", lw=1)
            ax.text(on, ax.get_ylim()[1] * 0.02 + 2, " anode onset", fontsize=7, color=INK)
        ax.set(title=u, xlabel="time (h)")
    axs[0].set_ylabel("voltage increase at 80 C, 1 bar, 0.4 A/cm2 (mV)")
    axs[2].legend(fontsize=6.5, loc="upper left")
    fig.suptitle("Hidden truth: each mechanism alone at the reference condition (truth/mechanisms_at_reference.csv)",
                 fontsize=9)
    figs["fig3_mechanisms_truth.png"] = _png(fig)
    # 4 HTO vs j and P (BoL) + HTO data for the 16 bar group
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    jj = np.linspace(0.02, 0.8, 300)
    m80 = ph.molarity_from_wt(30, 353.15)
    for i, P in enumerate((1, 5, 16, 30)):
        ax[0].plot(jj, ph.gases(jj, 353.15, P, m80)["HTO_pct"], color=SLOTS[i], label=f"{P} bar")
    ax[0].axhline(2, color="#d03b3b", lw=1, ls="--")
    ax[0].text(0.45, 2.15, "2 % shutdown limit [Hu22]", color=INK, fontsize=7.5)
    ax[0].set(ylim=(0, 6), xlabel="j (A/cm2)", ylabel="HTO (%)", title="(a) BoL HTO vs current density, 80 C")
    ax[0].legend(fontsize=7)
    for i in range(1, 4):
        d = ts[(ts["unit"] == f"G4_P16-{i}") & ts["state"].eq("run")]
        ax[1].plot(d["time_h"], d["HTO_pct"], ".", ms=1.5, color=SLOTS[i - 1], label=f"G4_P16-{i}")
    ax[1].axhline(2, color="#d03b3b", lw=1, ls="--")
    ax[1].set(xlabel="time (h)", ylabel="measured HTO (%)", title="(b) 16 bar group: HTO-fault unit trips at 2 %")
    ax[1].legend(fontsize=7, markerscale=4)
    figs["fig4_hto.png"] = _png(fig)
    # 5 EIS BoL vs late + Cdl from CV
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    e = D["eis"]
    u = "G3_T80_ref-1"
    times = sorted(e[e["unit"] == u]["time_h"].unique())
    for i, t in enumerate([times[0], times[len(times) // 2], times[-1]]):
        d = e[(e["unit"] == u) & (e["time_h"] == t) & (e["j_A_cm2"] == 0.2) & (e["freq_Hz"] <= 1e4)]
        ax[0].plot(d["z_re"] * 100, d["z_im_neg"] * 100, "-o", ms=2, color=SLOTS[i], label=f"{t:.0f} h")
    ax[0].set(xlabel="Z' (ohm cm2)", ylabel="-Z'' (ohm cm2)", title=f"(a) EIS {u}\n0.2 A/cm2, 70 C, f <= 10 kHz")
    ax[0].set_aspect("equal", adjustable="datalim")
    ax[0].legend(fontsize=7)
    cv = D["cv"]
    d = cv[(cv["unit"] == u) & (cv["time_h"] == times[0]) & (cv["window"] == "Ni_redox_OER")]
    d2 = cv[(cv["unit"] == u) & (cv["time_h"] == times[-1]) & (cv["window"] == "Ni_redox_OER")]
    ax[1].plot(d["E_V_vs_RHE"], d["current_A"] * 1000 / 100, color=SLOTS[0], label=f"{times[0]:.0f} h")
    ax[1].plot(d2["E_V_vs_RHE"], d2["current_A"] * 1000 / 100, color=SLOTS[1], label=f"{times[-1]:.0f} h")
    ax[1].set(xlabel="E (V vs RHE)", ylabel="j (mA/cm2)", ylim=(-3, 12), title="(b) anode CV, 10 mV/s\nOER hides the Ni redox peak (flagged)")
    ax[1].legend(fontsize=7)
    # Cdl from double-layer CV (slope of half-span current vs scan rate) vs truth
    dl = cv[(cv["unit"] == u) & (cv["window"] == "double_layer")]
    est = []
    for (t, el), x in dl.groupby(["time_h", "electrode"]):
        di = x.groupby("scan_rate_V_s").apply(lambda z: (z.loc[z["cycle_point"].between(30, 45), "current_A"].mean()
                                                          - z.loc[z["cycle_point"].between(80, 95), "current_A"].mean()) / 2)
        est.append(dict(time_h=t, electrode=el, Cdl=np.polyfit(di.index, di.values, 1)[0] / 100 * 1000))
    est = pd.DataFrame(est)
    st = T["state"][T["state"]["unit"] == u]
    for i, el in enumerate(["cathode", "anode"]):
        x = est[est["electrode"] == el]
        ax[2].plot(x["time_h"], x["Cdl"], "o", ms=3.5, color=SLOTS[i], label=f"{el}: from CV data")
        ax[2].plot(st["time_h"], st[f"Cdl_{el}_mF_cm2"], color=SLOTS[i], lw=1, label=f"{el}: truth")
    ax[2].set(xlabel="time (h)", ylabel="Cdl (mF/cm2)", yscale="log", title="(c) ECSA loss\nCdl from CV vs truth")
    ax[2].legend(fontsize=6.5)
    figs["fig5_eis_cv.png"] = _png(fig)
    # 6 electrolyte
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
    u = "G5_KOH25-1"
    st = T["state"][T["state"]["unit"] == u]
    ax[0].plot(st["time_h"], st["koh_wt_true"], color=SLOTS[0], lw=0.8)
    ax[0].set(xlabel="time (h)", ylabel="KOH (wt%, total K+)", title=f"(a) {u}: water consumption + make-up")
    d = ts[(ts["unit"] == u) & ts["state"].eq("run")]
    ax[1].plot(d["time_h"], d["conductivity_S_cm"], ".", ms=1.2, color=SLOTS[0], label="measured (probe fouls)")
    ax[1].plot(st["time_h"], st["sigma_true_S_cm"], color=INK, lw=0.8, label="true")
    ax[1].set(xlabel="time (h)", ylabel="conductivity (S/cm)", title="(b) online conductivity")
    ax[1].legend(fontsize=7, markerscale=5)
    tt = D["titr"][D["titr"]["unit"] == u]
    ax[2].plot(tt["time_h"], tt["k2co3_wt"], "o", ms=3.5, color=SLOTS[1], label="titration K2CO3 (wt%)")
    ax[2].plot(st["time_h"], st["carbonate_fraction"] * st["koh_wt_true"] * 138.2 / 2 / 56.1, color=INK, lw=0.8,
               label="true")
    ax[2].set(xlabel="time (h)", ylabel="K2CO3 (wt%)", title="(c) carbonation, reset at electrolyte change")
    ax[2].legend(fontsize=7)
    figs["fig6_electrolyte.png"] = _png(fig)
    # 7 faults
    fl = T["faults"]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.0))
    for ax, (_, f) in zip(axs, fl.iterrows()):
        d = ts[(ts["unit"] == f["unit"])]
        w = d[(d["time_h"] > f["t_h"] - 400) & (d["time_h"] < f["t_h"] + 400)]
        ch = "HTO_pct" if f["kind"] == "HTO_rise" else "voltage_V"
        w = w[w["state"].eq("run")] if f["kind"] != "sensor_bad_reading" else w[w["state"].isin(["run"])]
        ax.plot(w["time_h"], w[ch], ".", ms=2, color=SLOTS[0])
        ax.axvline(f["t_h"], color="#d03b3b", lw=1, ls="--")
        ax.set(title=f"{f['unit']}\n{f['kind']}", xlabel="time (h)", ylabel=ch)
    fig.suptitle("Injected faults in the measured data (dashed = true onset; only in truth/faults.csv)", fontsize=9, y=1.1)
    figs["fig7_faults.png"] = _png(fig)
    return figs


def run(root):
    D, T, man = load(root)
    C = Checks()
    tf = bol_checks(C, D, T)
    balance_checks(C, D, T)
    safety_checks(C, D, T)
    rates = degradation_checks(C, D, T)
    data_checks(C, D, T)
    figs = figures(D, T, tf)
    out = os.path.join(root, "report")
    os.makedirs(out, exist_ok=True)
    chk = C.df()
    chk.to_csv(os.path.join(out, "checks.csv"), index=False)
    rates.to_csv(os.path.join(out, "ageing_rates_by_unit.csv"), index=False)
    for k, b in figs.items():
        with open(os.path.join(out, k), "wb") as fh:
            fh.write(b)
    html = render_html(chk, figs, man, T)
    with open(os.path.join(out, "plausibility_report.html"), "w") as fh:
        fh.write(html)
    return chk


def render_html(chk, figs, man, T):
    css = ("body{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1f1f1e;"
           "background:#fcfcfb}table{border-collapse:collapse;font-size:12px;width:100%}td,th{border-bottom:1px solid "
           "#e6e5e0;padding:4px 6px;text-align:left;vertical-align:top}.PASS{color:#006300;font-weight:600}.FAIL{color:"
           "#b42323;font-weight:600}.INFO{color:#6b6a64}img{max-width:100%;margin:8px 0 18px}h2{margin-top:28px}"
           ".banner{background:#fff4d6;border:1px solid #eda100;padding:8px 12px;border-radius:6px}")
    n_f = int((chk["result"] == "FAIL").sum())
    n_p = int((chk["result"] == "PASS").sum())
    rows = "".join(f"<tr><td>{r.area}</td><td>{r.check}</td><td>{r.value}</td><td>{r.criterion}</td>"
                   f"<td class='{r.result}'>{r.result}</td><td>{r.source}</td><td>{r.note}</td></tr>"
                   for r in chk.itertuples())
    up = T["up"][["unit", "D_star_h", "accel_onset_h", "trip_h", "n_stops"]].round(1).to_html(index=False, na_rep="-")
    fl = T["faults"].to_html(index=False)
    imgs = "".join(f"<h3>{k}</h3><img src='data:image/png;base64,{base64.b64encode(v).decode()}'/>" for k, v in figs.items())
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>AWE synthetic data - plausibility</title>
<style>{css}</style></head><body>
<h1>Synthetic AWE dataset - plausibility report</h1>
<p class='banner'><b>SYNTHETIC DATA</b> - generator awe_synth v{man['version']}, seed {man['seed']}, config
{man['config_hash']}. Degradation rates are accelerated on purpose. Checks marked INFO have no pass/fail criterion.</p>
<p><b>{n_p} PASS, {n_f} FAIL</b>, {int((chk['result'] == 'INFO').sum())} INFO.</p>
<h2>Checks</h2><table><tr><th>area</th><th>check</th><th>value</th><th>criterion</th><th>result</th><th>source</th>
<th>note</th></tr>{rows}</table>
<h2>Known answers (truth)</h2><h3>Acceleration onset and trips</h3>{up}<h3>Injected faults</h3>{fl}
<h2>Figures</h2>{imgs}
</body></html>"""


if __name__ == "__main__":
    chk = run(sys.argv[1])
    print(chk[["area", "check", "value", "result"]].to_string())
