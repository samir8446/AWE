"""AWE digital twin - Streamlit app. All computation lives in engine.py; this file only draws."""
from __future__ import annotations

import glob
import hashlib
import io
import os
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REQUIRED_ENGINE = "0.2.0"
REQUIRED_FUNCS = ["cell_voltage", "hto_percent", "build_windows", "attach_indicators", "rank_indicators",
                  "degradation_v1", "run_task", "score_tasks", "scoreboard", "kalman_replay", "particle_filter_eol",
                  "operating_map", "optimise_operation", "synthetic_dataset", "apply_mapping", "guess_mapping",
                  "literature_check", "drt_tikhonov", "tune", "influence", "ensemble", "inner_errors"]

st.set_page_config(page_title="AWE Digital Twin", page_icon=":material/water_drop:", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import engine as E
    _ok = getattr(E, "ENGINE_VERSION", None) == REQUIRED_ENGINE and all(hasattr(E, f) for f in REQUIRED_FUNCS)
except Exception as exc:  # noqa: BLE001
    E, _ok = None, False
    st.error(f"engine.py could not be imported: {exc}")
if not _ok:
    stray = [os.path.basename(p) for p in glob.glob(os.path.join(HERE, "engine*.py")) if os.path.basename(p) != "engine.py"]
    loaded = getattr(E, "__file__", "?") if E else "?"
    st.error(f"Engine version mismatch: the app needs engine {REQUIRED_ENGINE}, loaded "
             f"{getattr(E, 'ENGINE_VERSION', 'none')} from {loaded}. Stray copies: {stray or 'none'}. "
             "Put app.py and engine.py from the same release side by side and reboot.")
    st.stop()

# ----------------------------------------------------------------------------------------------- style
CSS_BASE = ("<style>[data-testid='stAppDeployButton'],[data-testid='stMainMenu'],[data-testid='stDecoration']"
            "{display:none!important} .block-container{padding-top:2.8rem;max-width:1500px} "
            ".awe-title{font-size:1.55rem;font-weight:700;letter-spacing:.02em;margin:0} "
            ".awe-sub{color:#8a93a3;font-size:.9rem;margin:0 0 .4rem 0} "
            ".awe-bar{display:flex;gap:.6rem;flex-wrap:wrap;margin:.3rem 0 .8rem 0} "
            ".awe-chip{border:1px solid rgba(128,140,160,.35);border-radius:999px;padding:.12rem .7rem;font-size:.8rem} "
            ".awe-chip b{font-weight:600} .awe-syn{border-color:#eda100} .awe-real{border-color:#1baf7a} "
            ".awe-panel{border:1px solid rgba(128,140,160,.25);border-radius:10px;padding:.8rem 1rem;margin-bottom:.6rem}</style>")
CSS_HERO = ("<style>@keyframes rise{0%{transform:translateY(0);opacity:0}15%{opacity:.9}100%{transform:translateY(-120px);"
            "opacity:0}} @keyframes pulse{0%,100%{opacity:.35}50%{opacity:1}} @keyframes flow{to{stroke-dashoffset:-40}} "
            ".b1{animation:rise 2.6s linear infinite}.b2{animation:rise 3.1s linear infinite .6s}"
            ".b3{animation:rise 2.2s linear infinite 1.1s}.b4{animation:rise 3.4s linear infinite 1.7s}"
            ".tw{animation:pulse 2.4s ease-in-out infinite}.fl{stroke-dasharray:6 6;animation:flow 1.2s linear infinite}</style>")
st.markdown(CSS_BASE, unsafe_allow_html=True)
st.markdown(CSS_HERO, unsafe_allow_html=True)

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DASH = ["solid", "dash", "dot", "dashdot"]
ss = st.session_state


def hero_svg() -> str:
    cells = "".join(f"<rect x='{40 + i * 16}' y='40' width='10' height='120' rx='2' fill='{c}' opacity='.85'/>"
                    for i, c in enumerate(["#9aa4b2", "#2a78d6"] * 5))
    twin = "".join(f"<rect x='{330 + i * 16}' y='40' width='10' height='120' rx='2' fill='none' stroke='#1baf7a' "
                   f"stroke-width='1.2' class='tw' style='animation-delay:{i * .12:.2f}s'/>" for i in range(10))
    bub = ("<circle class='b1' cx='70' cy='150' r='4' fill='#cfe3ff'/><circle class='b2' cx='118' cy='150' r='3' fill='#cfe3ff'/>"
           "<circle class='b3' cx='150' cy='150' r='5' fill='#ffd8c7'/><circle class='b4' cx='182' cy='150' r='3' fill='#ffd8c7'/>")
    return ("<svg viewBox='0 0 520 200' width='100%' height='190' xmlns='http://www.w3.org/2000/svg'>"
            "<rect x='30' y='30' width='180' height='140' rx='10' fill='none' stroke='#8a93a3'/>"
            f"{cells}{bub}<text x='120' y='192' fill='#8a93a3' font-size='12' text-anchor='middle'>electrolyser stack</text>"
            "<path class='fl' d='M215 100 C 260 60, 280 140, 320 100' stroke='#eda100' stroke-width='2' fill='none'/>"
            "<text x='267' y='80' fill='#eda100' font-size='11' text-anchor='middle'>data</text>"
            "<rect x='320' y='30' width='180' height='140' rx='10' fill='none' stroke='#1baf7a' stroke-dasharray='4 3'/>"
            f"{twin}<text x='410' y='192' fill='#1baf7a' font-size='12' text-anchor='middle'>digital twin</text></svg>")


def style(fig: go.Figure, title: str = "", x: str = "", y: str = "", h: int = 380) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=14)), height=h, margin=dict(l=50, r=20, t=45, b=45),
                      xaxis_title=x, yaxis_title=y, legend=dict(orientation="h", y=-0.22), hovermode="x unified",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor="rgba(128,140,160,.18)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,140,160,.18)", zeroline=False)
    return fig


def unit_style(W: pd.DataFrame):
    """Colour follows the condition group (fixed order), dash follows the unit within its group."""
    groups = sorted(W["group"].unique()) if W is not None and len(W) else []
    out = {}
    for gi, g in enumerate(groups):
        for k, u in enumerate(sorted(W.loc[W["group"] == g, "unit"].unique())):
            out[u] = (PAL[gi % len(PAL)], DASH[k % len(DASH)], g)
    return out


def eq_popover(name: str):
    with st.popover(f"Equations: {name}", icon=":material/functions:"):
        for text, tex in E.EQUATIONS[name]:
            st.caption(text)
            st.latex(tex)


def explain(text: str):
    if ss.get("show_explain", True):
        st.caption(text)


def key_of(*objs) -> str:
    h = hashlib.sha1()
    for o in objs:
        if isinstance(o, pd.DataFrame):
            h.update(pd.util.hash_pandas_object(o, index=False).values.tobytes())
        else:
            h.update(repr(o).encode())
    return h.hexdigest()[:16]


def none_if_zero(v):
    return None if v in (None, 0, 0.0) else float(v)


# ----------------------------------------------------------------------------------------------- caches
@st.cache_data(show_spinner="Generating synthetic demo...")
def c_synth(seed: int, hours: int, ver: str):
    return E.synthetic_dataset(seed=seed, hours=hours)


@st.cache_data(show_spinner="Building operating windows...")
def c_windows(_data, dkey: str, design: dict, kin: dict, ref: dict, wset: dict):
    d, k, r = E.CellDesign(**design), E.Kinetics(**kin), E.OperatingConditions(**ref)
    W = E.build_windows(_data, d, k, r, **wset)
    return E.attach_indicators(W, _data, d, k) if len(W) else W


@st.cache_data(show_spinner=False, max_entries=5000)
def c_band(_W, wkey: str, model: str, unit: str, share: float, mode: str, params: tuple):
    units = E.split_units(_W)
    d = units[unit]
    k = E.origin_index(d, share)
    train = {u: x for u, x in units.items() if u != unit} if mode == "units" else {}
    return E.inner_errors(model, train, d.iloc[:k + 1], dict(params), shares=(0.5,))


@st.cache_data(show_spinner=False, max_entries=5000)
def c_task(_W, wkey: str, unit: str, share: float, model: str, mode: str, params: tuple, thr: float):
    be = c_band(_W, wkey, model, unit, share, mode, params)
    return E.run_task(_W, unit, share, model, mode, dict(params), thr, band_err=be)


# ----------------------------------------------------------------------------------------------- sidebar: inputs
def sidebar_inputs():
    sb = st.sidebar
    sb.markdown("### AWE Digital Twin")
    sb.caption(f"engine {E.ENGINE_VERSION}")
    with sb.expander("1 · Data", expanded=True):
        src = st.radio("Data source", ["Synthetic demo (known parameters)", "My data (upload)"], key="src",
                       help="The synthetic demo has known degradation parameters so every method can be checked.")
        if src.startswith("Synthetic"):
            st.number_input("Demo duration (h)", 500, 6000, 2000, 500, key="syn_hours")
            st.number_input("Random seed", 0, 99, 0, key="syn_seed")
        else:
            upload_panel(st)
    with sb.expander("2 · Cell / stack design"):
        st.text_input("Name", "My cell", key="d_name")
        st.number_input("Number of cells in series", 1, 1000, 1, key="d_ncells")
        st.number_input("Active area per cell (cm2)", 0.1, 50000.0, 25.0, key="d_area")
        st.selectbox("Electrode (kinetics source)", list(E.ELECTRODE_PRESETS), key="d_electrode")
        st.number_input("Electrode thickness (mm)", 0.0, 20.0, 2.0, key="d_eth", help="Ham25 validation cell: 2.0 mm")
        st.number_input("Electrode conductivity (S/m), 0 = model/neglect", 0.0, 1e9, 0.0, key="d_esig",
                        help="0: nickel uses Hu22 eq 22; other materials are neglected (metal resistance is small).")
        c1, c2 = st.columns(2)
        c1.number_input("Gap anode-separator (mm)", 0.0, 10.0, 0.0, key="d_ga")
        c2.number_input("Gap cathode-separator (mm)", 0.0, 10.0, 0.0, key="d_gc")
        st.selectbox("Separator resistance model", list(E.MEMBRANE_MODELS), key="d_mem")
        st.number_input("Separator area resistance (ohm cm2)", 0.0, 10.0, 0.27, key="d_memR",
                        help="Zirfon 0.27 ohm cm2 [Qu26 Table 2]; used with 'Area resistance input'.")
        st.number_input("Separator thickness (mm)", 0.01, 5.0, 0.5, key="d_sept")
        c1, c2 = st.columns(2)
        c1.number_input("Porosity (0 = unknown)", 0.0, 0.99, 0.0, key="d_por",
                        help="Not given in the articles for Zirfon; needed for the structural model and Darcy.")
        c2.number_input("Tortuosity (0 = from porosity)", 0.0, 20.0, 0.0, key="d_tor")
        st.number_input("H2 permeability (1e-12 mol cm-1 s-1 bar-1)", 0.0, 1000.0, 18.5, key="d_perm",
                        help="Zirfon 1.85e-11 [Qu26 Table 2]; 0 disables diffusion crossover.")
        st.selectbox("Electrolyte circulation", ["separated", "mixed"], key="d_circ", help="Hu22 sec 3.3.1.1")
        st.number_input("Supersaturation factor (crossover)", 1.0, 20.0, 1.0, key="d_ssat",
                        help="Hu22: average concentrations underestimate crossover near the electrode.")
    with sb.expander("3 · Operating conditions"):
        st.caption("Reference conditions (voltage normalisation) and the scenario used in physics views.")
        st.number_input("Temperature (C)", 0.0, 150.0, 80.0, key="o_T")
        st.number_input("Pressure, absolute (bar)", 0.5, 100.0, 1.0, key="o_P")
        st.number_input("Current density (A/cm2)", 0.001, 3.0, 0.4, key="o_j")
        st.number_input("KOH (wt %)", 1.0, 50.0, 30.0, key="o_koh")
        st.number_input("Electrolyte flow (L/min), 0 = unknown", 0.0, 1e5, 0.0, key="o_flow")
        st.number_input("p_anode - p_cathode (bar)", -5.0, 5.0, 0.0, key="o_dP",
                        help="Negative pushes dissolved H2 to the anode by convection (Darcy, needs viscosity).")
        st.number_input("H2 solubility (mol m-3 bar-1)", 0.0, 10.0, 0.088, format="%.4f", key="o_sol",
                        help="Hu22 Table 6: 0.088 mol/m3 at 75 C, 30.2 wt% (pressure basis assumed per bar).")
        st.number_input("Electrolyte viscosity (mPa s), 0 = unknown", 0.0, 100.0, 0.0, key="o_mu")
        st.number_input("HTO shutdown limit (%)", 0.1, 4.0, 2.0, key="o_hto", help="2 % = 50 % of LEL [Hu22], [Qu26]")
    with sb.expander("4 · Kinetics and Faraday efficiency"):
        st.selectbox("Activation model", ["Literature Butler-Volmer (Ni)", "Fitted Tafel from my U-i curves",
                                          "Custom Tafel"], key="k_mode")
        st.selectbox("Anode alpha fit", ["jang", "henao"], key="k_alpha", help="Hu22 Table B3 (two fits)")
        c1, c2 = st.columns(2)
        c1.number_input("j0,c ref (A/cm2)", 1e-9, 1.0, 1.5e-4, format="%.2e", key="k_j0c")
        c2.number_input("j0,a ref (A/cm2)", 1e-9, 1.0, 0.9e-4, format="%.2e", key="k_j0a")
        c1.number_input("Ea,c (J/mol)", 0.0, 2e5, 23000.0, key="k_Eac")
        c2.number_input("Ea,a (J/mol)", 0.0, 2e5, 42000.0, key="k_Eaa")
        c1.number_input("T_ref (K) - assumption", 200.0, 500.0, 298.15, key="k_Tref")
        c2.number_input("P_ref (bar) - assumption", 0.1, 100.0, 1.0, key="k_Pref")
        st.toggle("Bubble coverage effects", True, key="k_bub")
        c1.number_input("Custom Tafel b (V)", 0.0, 1.0, 0.0, key="k_b")
        c2.number_input("Custom Tafel j0 (A/cm2)", 0.0, 1.0, 0.0, format="%.2e", key="k_tj0")
        c1.number_input("Faraday f1 ((mA/cm2)^2), 0 = unknown", 0.0, 1e6, 0.0, key="k_f1")
        c2.number_input("Faraday f2, 0 = unknown", 0.0, 1.0, 0.0, key="k_f2")
    with sb.expander("5 · Analysis settings"):
        st.number_input("Window length (h)", 0.05, 168.0, 1.0, key="a_win")
        st.number_input("Minimum current density 'on' (A/cm2)", 0.0, 1.0, 0.02, key="a_minj")
        st.number_input("Stable window: max current CV", 0.0, 1.0, 0.05, key="a_cv")
        st.number_input("Recovery jump threshold (mV)", 0.5, 100.0, 3.0, key="a_rec")
        st.number_input("End of life: voltage rise at reference (mV)", 1.0, 2000.0, 15.0, key="a_eol",
                        help="No end-of-life criterion is given in the project articles: this is your study choice.")
        st.toggle("Show explanations under charts", True, key="show_explain")


def upload_panel(c):
    dtype = c.selectbox("Data type", list(E.DATA_TYPES), format_func=lambda k: E.DATA_TYPES[k]["label"], key="up_type")
    f = c.file_uploader("File (.csv, .txt, .xlsx, .parquet)", type=["csv", "txt", "xlsx", "xls", "parquet"], key="up_file")
    c.text_input("Unit name if the file has no unit column", "cell-1", key="up_unit")
    c.number_input("Time (h) if the file has no time column (e.g. one EIS spectrum)", 0.0, 1e6, 0.0, key="up_time")
    if f is not None:
        try:
            raw = E.read_table(io.BytesIO(f.getvalue()), f.name)
        except Exception as exc:  # noqa: BLE001
            c.error(f"Could not read {f.name}: {exc}")
            return
        c.caption(f"{f.name}: {raw.shape[0]} rows x {raw.shape[1]} columns")
        guess = E.guess_mapping(list(raw.columns), dtype)
        opts = ["(none)"] + [str(x) for x in raw.columns]
        mapping = {}
        for std, col in guess.items():
            req = std in E.DATA_TYPES[dtype]["required"]
            v = c.selectbox(f"{std}{' *' if req else ''}", opts, index=opts.index(str(col)) if col is not None else 0,
                            key=f"map_{dtype}_{std}")
            mapping[std] = None if v == "(none)" else raw.columns[opts.index(v) - 1]
        if c.button("Add to dataset", key="up_add", type="primary"):
            df, notes = E.apply_mapping(raw, mapping, dtype, design_from_state(), ss.get("up_unit", "cell-1"),
                                        ss.get("up_time") if dtype in ("eis", "polcurve", "drt") else None)
            tabs = ss.setdefault("tables", {})
            tabs[dtype] = pd.concat([tabs[dtype], df], ignore_index=True) if dtype in tabs else df
            ss.setdefault("load_notes", []).extend([f"{f.name}: {n}" for n in notes])
            c.success(f"Added {len(df)} rows as '{dtype}'.")
    tabs = ss.get("tables", {})
    if tabs:
        c.caption("Loaded: " + ", ".join(f"{k} ({len(v)} rows)" for k, v in tabs.items()))
        if c.button("Clear uploaded data", key="up_clear"):
            ss["tables"] = {}
            ss["load_notes"] = []


def design_from_state() -> "E.CellDesign":
    mem = E.MEMBRANE_MODELS[ss.get("d_mem", list(E.MEMBRANE_MODELS)[0])]
    return E.CellDesign(name=ss.get("d_name", "My cell"), n_cells=int(ss.get("d_ncells", 1)),
                        area_cm2=float(ss.get("d_area", 25.0)),
                        electrode=E.ELECTRODE_PRESETS[ss.get("d_electrode", list(E.ELECTRODE_PRESETS)[0])],
                        electrode_thickness_mm=float(ss.get("d_eth", 2.0)), electrode_sigma_S_m=none_if_zero(ss.get("d_esig")),
                        gap_anode_mm=float(ss.get("d_ga", 0.0)), gap_cathode_mm=float(ss.get("d_gc", 0.0)),
                        membrane_model=mem, membrane_area_res_ohm_cm2=float(ss.get("d_memR", 0.27)),
                        sep_thickness_mm=float(ss.get("d_sept", 0.5)), sep_porosity=none_if_zero(ss.get("d_por")),
                        sep_tortuosity=none_if_zero(ss.get("d_tor")),
                        h2_permeability=float(ss.get("d_perm", 18.5)) * 1e-12 or None,
                        circulation=ss.get("d_circ", "separated"), supersaturation=float(ss.get("d_ssat", 1.0)))


def op_from_state() -> "E.OperatingConditions":
    mu = none_if_zero(ss.get("o_mu"))
    return E.OperatingConditions(T_C=float(ss.get("o_T", 80.0)), P_bar=float(ss.get("o_P", 1.0)),
                                 j_A_cm2=float(ss.get("o_j", 0.4)), koh_wt=float(ss.get("o_koh", 30.0)),
                                 lye_flow_L_min=none_if_zero(ss.get("o_flow")),
                                 dP_anode_minus_cathode_bar=float(ss.get("o_dP", 0.0)),
                                 h2_solubility_mol_m3_bar=none_if_zero(ss.get("o_sol")),
                                 lye_viscosity_Pa_s=mu / 1000 if mu else None)


def kin_from_state(fit_table=None) -> "E.Kinetics":
    mode = ss.get("k_mode", "Literature Butler-Volmer (Ni)")
    base = E.Kinetics(j0_c_ref=float(ss.get("k_j0c", 1.5e-4)), j0_a_ref=float(ss.get("k_j0a", 0.9e-4)),
                      Ea_c=float(ss.get("k_Eac", 23000.0)), Ea_a=float(ss.get("k_Eaa", 42000.0)),
                      alpha_variant=ss.get("k_alpha", "jang"), T_ref_K=float(ss.get("k_Tref", 298.15)),
                      P_ref_bar=float(ss.get("k_Pref", 1.0)), use_bubbles=bool(ss.get("k_bub", True)))
    if mode.startswith("Fitted") and fit_table is not None and len(fit_table) and "tafel_b_V" in fit_table:
        k = E.kinetics_from_fit(fit_table)
        k.use_bubbles = base.use_bubbles
        return k
    if mode.startswith("Custom") and ss.get("k_b") and ss.get("k_tj0"):
        return E.Kinetics(mode="tafel_fit", tafel_b=float(ss["k_b"]), tafel_j0=float(ss["k_tj0"]), use_bubbles=base.use_bubbles)
    return base


def faraday_from_state():
    return E.FaradayParams(f1=none_if_zero(ss.get("k_f1")), f2=none_if_zero(ss.get("k_f2")))


# ----------------------------------------------------------------------------------------------- data assembly
sidebar_inputs()
design, op, fp = design_from_state(), op_from_state(), faraday_from_state()
truth, source = None, None
if ss.get("src", "Synthetic").startswith("Synthetic"):
    syn = c_synth(int(ss.get("syn_seed", 0)), int(ss.get("syn_hours", 2000)), E.ENGINE_VERSION)
    data, truth, source = syn["data"], syn["truth"], "synthetic"
    design = syn["design"]          # the demo has its own declared cell
else:
    data = ss.get("tables", {})
    source = "real" if data else None

pc_fit = E.fit_polcurve(data["polcurve"], design, E.Kinetics()) if data.get("polcurve") is not None and len(data["polcurve"]) else None
kin = kin_from_state(pc_fit)
if source == "synthetic" and ss.get("k_mode", "").startswith("Literature"):
    kin = E.Kinetics(**{**asdict(kin), **{k: v for k, v in asdict(syn["kin"]).items() if k in ("j0_c_ref", "j0_a_ref")}})
ref = op
W = pd.DataFrame()
if data.get("timeseries") is not None and len(data["timeseries"]):
    dkey = key_of(*[data[k] for k in sorted(data)])
    wset = dict(window_h=float(ss.get("a_win", 1.0)), min_j=float(ss.get("a_minj", 0.02)),
                stable_cv=float(ss.get("a_cv", 0.05)), recovery_mV=float(ss.get("a_rec", 3.0)))
    W = c_windows(data, dkey, asdict(design), asdict(kin), asdict(ref), wset)
WKEY = key_of(W) if len(W) else "none"
THR = float(ss.get("a_eol", 15.0))

# ----------------------------------------------------------------------------------------------- header
st.markdown("<p class='awe-title'>AWE Digital Twin · alkaline water electrolyser</p>"
            "<p class='awe-sub'>health diagnostics · degradation forecasts with uncertainty · self-updating twin · "
            "operation & maintenance</p>", unsafe_allow_html=True)
VIEWS = ["Home", "Diagnostics", "Live twin", "Models & forecasting", "Operations & control", "Study results"]
view = st.segmented_control("View", VIEWS, default="Home", required=True, key="view", label_visibility="collapsed")
chip_src = ("<span class='awe-chip awe-syn'><b>SYNTHETIC</b> data, known parameters</span>" if source == "synthetic"
            else "<span class='awe-chip awe-real'><b>REAL</b> data</span>" if source == "real"
            else "<span class='awe-chip'>no data loaded</span>")
n_units = W["unit"].nunique() if len(W) else 0
st.markdown(f"<div class='awe-bar'>{chip_src}<span class='awe-chip'>units: <b>{n_units}</b></span>"
            f"<span class='awe-chip'>cell: <b>{design.n_cells} x {design.area_cm2:g} cm2</b></span>"
            f"<span class='awe-chip'>reference: <b>{op.T_C:g} C · {op.P_bar:g} bar · {op.j_A_cm2:g} A/cm2 · "
            f"{op.koh_wt:g} wt%</b></span><span class='awe-chip'>kinetics: <b>{kin.mode}</b></span>"
            f"<span class='awe-chip'>EOL: <b>+{THR:g} mV</b></span><span class='awe-chip'>engine "
            f"<b>{E.ENGINE_VERSION}</b></span></div>", unsafe_allow_html=True)
for n in ss.get("load_notes", [])[-6:]:
    st.caption("ℹ " + n)


# ============================================================================================ HOME
def view_home():
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown(hero_svg(), unsafe_allow_html=True)
        st.markdown("**What this platform does.** Turns electrolyser operating data into health indicators (voltage at "
                    "reference conditions, ohmic and activation parts, crossover), forecasts their evolution with "
                    "honest held-out uncertainty, replays the data as a self-updating twin, and evaluates operating "
                    "strategies against hydrogen output, energy cost, degradation and the HTO safety limit.")
    with c2:
        if source is None:
            st.markdown("#### Start")
            st.info("Load the synthetic demo or upload your data (sidebar → 1 · Data). The physics below already "
                    "works with the design and operating inputs.")
            st.file_uploader("Quick upload: operating time series (.csv/.xlsx/.parquet)",
                             type=["csv", "txt", "xlsx", "parquet"], key="up_main")
            if ss.get("up_main") is not None:
                st.caption("Choose 'My data (upload)' in the sidebar to map the columns and add this file.")
        else:
            inv = E.signal_inventory(data)
            st.markdown("#### Dataset at a glance")
            st.dataframe(inv, hide_index=True, height=330)
    st.markdown("---")
    st.markdown("#### Physics at the selected design and conditions")
    eq_popover("Cell voltage")
    jg = np.linspace(0.005, max(1.0, 1.5 * op.j_A_cm2), 120)
    r = E.cell_voltage(jg, op, design, kin)
    c1, c2 = st.columns(2)
    fig = go.Figure()
    base = r["E_rev"]
    for name, y, col in [("E_rev", base, "#9aa4b2"), ("+ activation", base + r["eta_act"], PAL[0]),
                         ("+ ohmic = U", r["U"], PAL[1])]:
        fig.add_trace(go.Scatter(x=jg, y=y, name=name, line=dict(color=col, width=2)))
    c1.plotly_chart(style(fig, "Polarisation curve and its parts", "j (A/cm2)", "U (V)"), key="home_pol")
    h = E.hto_percent(jg, op, design, kin)
    fig = go.Figure(go.Scatter(x=jg, y=h["HTO_pct"], name="HTO", line=dict(color=PAL[2], width=2)))
    lim = float(ss.get("o_hto", 2.0))
    fig.add_hline(y=lim, line=dict(color="#e34948", dash="dash"), annotation_text=f"shutdown {lim:g} %")
    fig.update_yaxes(type="log", range=[np.log10(max(float(np.nanmin(h["HTO_pct"])), 1e-4)) - 0.2, np.log10(max(lim, float(np.nanmax(h["HTO_pct"])))) + 0.3])
    c2.plotly_chart(style(fig, "H2 in O2 vs load (steady state)", "j (A/cm2)", "HTO (%)"), key="home_hto")
    pt = E.cell_voltage(op.j_A_cm2, op, design, kin)
    eff = E.efficiencies(pt["U"][0], op, E.faraday_efficiency(op.j_A_cm2, fp))
    pr = E.production(op.j_A_cm2, design, fp)
    m = st.columns(6)
    m[0].metric("Cell voltage", f"{pt['U'][0]:.3f} V")
    m[1].metric("E_rev", f"{pt['E_rev'][0]:.3f} V")
    m[2].metric("Specific energy", f"{float(eff['kWh_per_kg']):.1f} kWh/kg")
    m[3].metric("Thermal eff. V_tn/U", f"{float(eff['thermal_eff']) * 100:.1f} %")
    m[4].metric("H2 (stack)", f"{float(pr['kg_h']) * 1000:.2f} g/h")
    m[5].metric("Min load (HTO)", f"{E.min_load_for_hto(op, design, kin, lim):.3f} A/cm2")
    explain("Equations and parameter values come from Hu22, Ham25 and Daoudi24 (see equations button). Faraday "
            "efficiency is 1 unless you enter f1, f2. Kinetics for nickel do not apply to stainless steel: choose "
            "'Fitted Tafel from my U-i curves' once U-i data are loaded.")
    with st.expander("Literature check: engine vs values reported in Ham25 (0.4 A/cm2, 30 wt% KOH)"):
        st.dataframe(E.literature_check().round(4), hide_index=True)
        st.caption("Differences of 1-7 mV (E_rev, activation) and up to 0.008 in bubble coverage. Ham25's ohmic term "
                   "depends on a cell geometry it does not report, so the total voltage is not compared.")
    with st.expander("Assumptions in force (values the articles do not give)"):
        st.dataframe(pd.DataFrame(E.ASSUMPTIONS, columns=["assumption", "why"]), hide_index=True)


# ============================================================================================ DIAGNOSTICS
def need_windows() -> bool:
    if not len(W):
        st.info("Load the synthetic demo or upload an operating time series to see this view.")
        return False
    return True


def view_diagnostics():
    if not need_windows():
        return
    sty = unit_style(W)
    st.markdown("#### 1 · Dataset understanding")
    summ = W.groupby("unit").agg(group=("group", "first"), op_hours=("op_hours", "max"), j_min=("j_A_cm2", "min"),
                                 j_max=("j_A_cm2", "max"), T_C=("T_C", "mean"), P_bar=("P_bar", "mean"),
                                 starts=("starts", "max"), stable_share=("stable", "mean"),
                                 recoveries=("flag_recovery", "sum")).round(3).reset_index()
    st.dataframe(summ, hide_index=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(E.signal_inventory(data), hide_index=True, height=300)
    with c2:
        ev = data.get("events")
        if ev is not None and len(ev):
            fig = go.Figure()
            for k, (e, d) in enumerate(ev.groupby("event")):
                fig.add_trace(go.Scatter(x=d["time_h"], y=d["unit"], mode="markers", name=e,
                                         marker=dict(size=8, color=PAL[k % len(PAL)], symbol="line-ns-open")))
            st.plotly_chart(style(fig, "Events", "time (h)", ""), key="diag_events")
        else:
            st.info("No events table: start-ups are derived from off→on transitions of the current.")
    explain("Units are grouped by operating condition; every later result is reported per group, never only pooled.")

    st.markdown("#### 2 · Health indicators over operating hours")
    his = [k for k in E.HEALTH_INDICATORS if k in W and W[k].notna().any()]
    hi = st.selectbox("Indicator", his, format_func=lambda k: E.HEALTH_INDICATORS[k], key="diag_hi")
    show_flags = st.toggle("Mark recovery jumps", True, key="diag_flags")
    fig = go.Figure()
    for u, d in W.groupby("unit"):
        col, dash, g = sty[u]
        dd = d[d["stable"]]
        fig.add_trace(go.Scatter(x=dd["op_hours"], y=dd[hi], name=f"{u} · {g}", mode="lines",
                                 line=dict(color=col, dash=dash, width=1.6)))
        if show_flags and dd["flag_recovery"].any():
            f = dd[dd["flag_recovery"]]
            fig.add_trace(go.Scatter(x=f["op_hours"], y=f[hi], mode="markers", name=f"{u} recovery", showlegend=False,
                                     marker=dict(size=9, color=col, symbol="x")))
    if hi == "dV_mV":
        fig.add_hline(y=THR, line=dict(color="#e34948", dash="dash"), annotation_text=f"EOL +{THR:g} mV")
    st.plotly_chart(style(fig, E.HEALTH_INDICATORS[hi], "operating hours", hi, 420), key="diag_hi_plot")
    eq_popover("Degradation law")
    explain("Voltage is normalised to the reference conditions with the polarisation model (sidebar 3 and 4). Recovery "
            "jumps (after KOH refill / service, or sudden drops) are flagged: they are removed from training labels "
            "but kept in test data.")
    st.markdown("#### 3 · Which indicators represent health best?")
    st.dataframe(E.rank_indicators(W).round(3), hide_index=True)
    explain("trend = median |Spearman| with operating hours; monotonicity = share of same-sign smoothed changes; "
            "sparse indicators (EIS, U-i every few hundred hours) are scored on their own measurement points.")

    st.markdown("#### 4 · Degradation model v1")
    dv1 = E.degradation_v1(W, T_ref_C=op.T_C)
    ss["dv1"] = dv1
    st.dataframe(dv1["per_unit"].round(3), hide_index=True)
    law = dv1["law"]
    m = st.columns(4)
    m[0].metric(f"a (uV/h at {op.T_C:g} C)", f"{1000 * law['a_mV_h']:.2f}" if np.isfinite(law["a_mV_h"]) else "n/a")
    m[1].metric("Ea of ageing (kJ/mol)", f"{law['Ea_J_mol'] / 1000:.1f}" if np.isfinite(law["Ea_J_mol"]) else "not identifiable")
    m[2].metric("b (mV per start-up)", f"{law['b_mV_start']:.3f}" if np.isfinite(law["b_mV_start"]) else "not identifiable")
    m[3].metric("fit RMSE (mV)", f"{law.get('rmse_mV', np.nan):.2f}")
    for n in dv1["notes"]:
        st.warning(n)
    if source == "synthetic":
        L = E.SYNTH_LAW
        st.caption(f"Synthetic truth: kR={L['kR']} ohm cm2/h, k_act={L['k_act'] * 1e6:.1f} uV/h at 80 C, Ea_R={L['EaR'] / 1000:g} "
                   f"/ Ea_act={L['Ea_act'] / 1000:g} kJ/mol, {L['deta_start'] * 1000:.2f} mV per start-up, recovery "
                   f"{L['refill_recovery']:.0%} of activation loss at each KOH refill. At 0.4 A/cm2 this is "
                   f"{(0.4 * L['kR'] + L['k_act']) * 1e6:.1f} uV/h at 80 C before recoveries.")

    st.markdown("#### 5 · Characterisation: U-i curves, EIS, DRT")
    c1, c2 = st.columns(2)
    with c1:
        if pc_fit is not None and len(pc_fit):
            pcu = st.selectbox("Unit (U-i)", sorted(data["polcurve"]["unit"].unique()), key="diag_pcu")
            pc = data["polcurve"][data["polcurve"]["unit"] == pcu]
            fig = go.Figure()
            times = sorted(pc["time_h"].unique())
            for k, t in enumerate(times):
                d = pc[pc["time_h"] == t]
                fig.add_trace(go.Scatter(x=d["j_A_cm2"], y=d["voltage_V"], name=f"{t:g} h", mode="lines+markers",
                                         line=dict(color=PAL[0], width=1.5), opacity=0.35 + 0.65 * k / max(len(times) - 1, 1)))
            st.plotly_chart(style(fig, "Polarisation curves over time", "j (A/cm2)", "U (V)"), key="diag_pc")
            st.dataframe(pc_fit[pc_fit["unit"] == pcu].round(4), hide_index=True)
        else:
            st.info("No U-i curves loaded.")
    with c2:
        eis = data.get("eis")
        if eis is not None and len(eis):
            eu = st.selectbox("Unit (EIS)", sorted(eis["unit"].unique()), key="diag_eu")
            de = eis[eis["unit"] == eu]
            fig = go.Figure()
            times = sorted(de["time_h"].unique())
            for k, t in enumerate(times):
                d = de[de["time_h"] == t]
                fig.add_trace(go.Scatter(x=d["z_re"], y=d["z_im_neg"], name=f"{t:g} h", mode="lines",
                                         line=dict(color=PAL[1], width=1.5), opacity=0.35 + 0.65 * k / max(len(times) - 1, 1)))
            fig.update_yaxes(scaleanchor="x")
            st.plotly_chart(style(fig, "Nyquist", "Z' (ohm)", "-Z'' (ohm)"), key="diag_eis")
            tsel = st.selectbox("Spectrum for DRT", times, key="diag_drt_t")
            lam = st.select_slider("DRT regularisation lambda", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1], value=1e-3, key="diag_lam")
            d = de[de["time_h"] == tsel]
            r = E.drt_tikhonov(d["freq_Hz"], d["z_re"], d["z_im_neg"], lam)
            fig = go.Figure(go.Scatter(x=r["tau"], y=r["gamma"], line=dict(color=PAL[2], width=2), name="gamma"))
            fig.update_xaxes(type="log")
            st.plotly_chart(style(fig, f"DRT (R_inf {r['R_inf']:.4f} ohm, R_pol {r['R_pol']:.4f} ohm)", "tau (s)",
                                  "gamma (ohm)", 300), key="diag_drt")
            eq_popover("DRT")
            explain("Quick Tikhonov DRT for inspection; for publication use DRTtools (RBF basis, Bayesian credible "
                    "bands, Hilbert-transform quality scores) [DRT25].")
        else:
            st.info("No EIS loaded.")


# ============================================================================================ MODELS & FORECASTING
def param_editor(model: str) -> dict:
    spec = E.MODEL_SPECS[model]
    pend = ss.pop(f"tuned_pending_{model}", None)
    if pend:
        for k, v in pend.items():
            ss[f"p_{model}_{k}"] = v
    remembered = ss.get("tuned_params", {}).get(model, {}) if ss.get("self_opt") else {}
    params = {}
    for k, v in spec[1].items():
        key = f"p_{model}_{k}"
        if remembered and k in remembered and key not in ss:
            ss[key] = remembered[k]
        if isinstance(v, int) and not isinstance(v, bool):
            params[k] = int(st.number_input(f"{model}: {k}", value=int(ss.get(key, v)), key=key))
        else:
            params[k] = float(st.number_input(f"{model}: {k}", value=float(ss.get(key, v)), key=key, format="%.4g"))
    return params


def view_models():
    if not need_windows():
        return
    units = sorted(W["unit"].unique())
    st.markdown("#### 1 · Set-up")
    auto_mode = "units" if len(units) >= 2 else "periods"
    c1, c2, c3 = st.columns([1, 1, 1.4])
    mode = c1.selectbox("Evaluation", ["units", "periods"], index=0 if auto_mode == "units" else 1, key="m_mode",
                        format_func=lambda m: {"units": "each unit held out in turn",
                                               "periods": "held-out later period of each unit"}[m])
    if mode == "units" and len(units) < 2:
        st.warning("Only one unit: switch to held-out periods.")
        return
    shares = c2.multiselect("Forecast origins (share of test duration)", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                            default=[0.3, 0.5], key="m_shares")
    models = c3.multiselect("Models", list(E.MODEL_SPECS), default=["Persistence", "Linear drift", "Fleet curve",
                                                                     "Bayesian ridge", "Random forest"], key="m_models",
                            format_func=lambda m: f"{m} ({E.MODEL_SPECS[m][0]})")
    test_units = st.multiselect("Test units", units, default=units, key="m_units")
    st.caption("Planned (next steps): " + "; ".join(f"{k} - {v}" for k, v in E.PLANNED_MODELS.items()) +
               ("" if E.HAS_XGB and E.HAS_LGBM else " · XGBoost/LightGBM not installed (optional)."))
    with st.expander("2 · Model parameters, automatic tuning, self-optimise"):
        st.toggle("Self-optimise: reuse the last tuned settings automatically", False, key="self_opt")
        params = {}
        cols = st.columns(3)
        for k, m in enumerate(models):
            with cols[k % 3]:
                params[m] = param_editor(m)
        tm = st.selectbox("Tune (Bayesian optimisation on held-out training units only)",
                          [m for m in models if E.MODEL_SPECS[m][2]] or ["-"], key="m_tune_model")
        if st.button("Tune selected model", key="m_tune") and tm != "-":
            with st.spinner("Bayesian optimisation (10 evaluations)..."):
                r = E.tune(tm, W, exclude_unit=test_units[0] if test_units and mode == "units" else None, mode=mode)
            ss[f"tuned_pending_{tm}"] = r["best"]
            ss.setdefault("tuned_params", {})[tm] = r["best"]
            ss["tune_hist"] = r["history"]
            st.rerun()
        if ss.get("tune_hist") is not None and len(ss["tune_hist"]):
            st.dataframe(ss["tune_hist"].round(4), hide_index=True)
            st.caption("Objective = RMSE of held-out forecasts on training units; the test unit is excluded.")
    eq_popover("Models")
    tasks = [(u, s, m) for u in test_units for s in shares for m in models]
    st.markdown("#### 3 · Run the accuracy assessment")
    est = sum(0.2 if E.MODEL_SPECS[m][0].startswith("L1") else 6.0 for _, _, m in tasks)
    if st.button(f"Run assessment ({len(tasks)} forecasts, about {max(est / 60, 0.1):.1f} min)", key="m_run", type="primary"):
        prog, live, parts = st.progress(0.0), st.empty(), []
        for i, (u, s, m) in enumerate(tasks):
            prog.progress(i / max(len(tasks), 1), text=f"{m} · {u} · origin {s:.0%}")
            r = c_task(W, WKEY, u, s, m, mode, tuple(sorted(params.get(m, {}).items())), THR)
            if len(r):
                parts.append(r)
            if parts and (i % 3 == 2 or i == len(tasks) - 1):
                res = pd.concat(parts, ignore_index=True)
                ss["fc_res"], ss["fc_meta"] = res, dict(mode=mode, wkey=WKEY, thr=THR, source=source)
                live.dataframe(E.scoreboard(E.score_tasks(res, THR))[0].round(3), hide_index=True)
        prog.progress(1.0, text="done")
    res = ss.get("fc_res")
    if res is None or not len(res):
        st.info("Press Run assessment to see forecasts and scores.")
        return
    if ss.get("fc_meta", {}).get("wkey") != WKEY:
        st.warning("Data or settings changed since this assessment: results below are from the previous run.")
    # ensemble + recommended model from held-out inner errors
    inner = {}
    for m in res["model"].unique():
        errs = [c_band(W, WKEY, m, u, s, ss["fc_meta"]["mode"], tuple(sorted(params.get(m, {}).items())))
                for u in res["unit"].unique() for s in res["share"].unique()]
        e = pd.concat([x for x in errs if len(x)]) if any(len(x) for x in errs) else pd.DataFrame(columns=["err"])
        inner[m] = float(np.sqrt(np.mean(e["err"] ** 2))) if len(e) else np.nan
    ens = E.ensemble(res, inner)
    allres = pd.concat([res, ens], ignore_index=True) if len(ens) else res
    scores = E.score_tasks(allres, THR)
    ss["fc_scores"] = scores
    rec = min((m for m in inner if np.isfinite(inner[m])), key=lambda m: inner[m], default=None)
    ss["recommended"] = rec
    t1, t2, t3, t4 = st.tabs(["Forecasts", "Scores", "Per condition group", "Checks: influence & no-leakage"])
    with t1:
        c1, c2 = st.columns(2)
        u = c1.selectbox("Unit", sorted(allres["unit"].unique()), key="m_plot_unit")
        s = c2.selectbox("Origin", sorted(allres["share"].unique()), key="m_plot_share", format_func=lambda x: f"{x:.0%}")
        show = st.multiselect("Show", sorted(allres["model"].unique()), default=[x for x in [rec, "Linear drift", "Ensemble"]
                                                                                 if x in set(allres["model"])], key="m_show")
        d = W[W["unit"] == u]
        fig = go.Figure(go.Scatter(x=d.loc[d["stable"], "op_hours"], y=d.loc[d["stable"], "dV_mV"], mode="markers",
                                   name="measured", marker=dict(size=4, color="#9aa4b2")))
        for k, m in enumerate(show):
            r = allres[(allres["model"] == m) & (allres["unit"] == u) & (allres["share"] == s)]
            if not len(r):
                continue
            col = PAL[k % len(PAL)]
            if r["lo"].notna().any():
                fig.add_trace(go.Scatter(x=np.r_[r["op_hours"], r["op_hours"][::-1]], y=np.r_[r["hi"], r["lo"][::-1]],
                                         fill="toself", fillcolor=col, opacity=0.15, line=dict(width=0), showlegend=False,
                                         hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=r["op_hours"], y=r["pred"], name=m, line=dict(color=col, width=2)))
        r0 = allres[(allres["unit"] == u) & (allres["share"] == s)]
        if len(r0):
            fig.add_vline(x=float(r0["origin_h"].iloc[0]), line=dict(color="#8a93a3", dash="dot"), annotation_text="origin")
        fig.add_hline(y=THR, line=dict(color="#e34948", dash="dash"), annotation_text="EOL")
        st.plotly_chart(style(fig, f"{u}: voltage rise forecast from {s:.0%}", "operating hours", "dV at reference (mV)", 440),
                        key="m_fc_plot")
        explain("Every forecast starts at today's measured value (causal median of the last windows). Bands (90 %) "
                "come from held-out errors on the training units, never from in-sample residuals. A straight forecast "
                "means the model only sees a constant drift: with constant conditions there is nothing else to learn.")
    with t2:
        sb, note = E.scoreboard(scores)
        st.dataframe(sb.round(3), hide_index=True)
        st.caption(note)
        st.markdown(f"**Recommended model** (lowest held-out inner RMSE): **{rec}** · ensemble weights ∝ 1/RMSE².")
        st.dataframe(pd.DataFrame({"model": list(inner), "held-out inner RMSE (mV)": list(inner.values())}).round(3),
                     hide_index=True)
        eolt = scores.groupby("model")[["eol_err_h"]].agg(lambda x: np.nanmean(np.abs(x)) if np.isfinite(x).any() else np.nan)
        st.markdown("End-of-life timing error (h), where the threshold is reached in the data:")
        st.dataframe(eolt.round(1))
        st.download_button("Download scores (CSV)", scores.to_csv(index=False), "scores.csv", key="m_dl")
    with t3:
        st.dataframe(E.group_table(scores), hide_index=True if False else False)
        explain("Mean RMSE (mV) per condition group and model. With 1-2 units per group these are descriptions, not "
                "statistically established differences.")
    with t4:
        c1, c2, c3 = st.columns(3)
        iu = c1.selectbox("Held-out unit", units, key="m_inf_u")
        im = c2.selectbox("Model", models, key="m_inf_m")
        ish = c3.selectbox("Origin", shares or [0.5], key="m_inf_s")
        if st.button("Which training units perturb the model? (leave-one-out)", key="m_inf") and mode == "units":
            with st.spinner("Re-training without each unit..."):
                ss["inf_res"] = E.influence(W, iu, ish, im, params.get(im))
        if ss.get("inf_res") is not None:
            st.dataframe(ss["inf_res"].round(3), hide_index=True)
            st.caption("delta > 0: removing that unit made the forecast worse (it helped). Data are kept as they are; "
                       "recovery counts name known data issues.")
        if st.button("Run no-leakage test", key="m_leak"):
            ok = E.no_leakage_check(W, iu, ish, im)
            (st.success if ok else st.error)(f"No-leakage test {'passed' if ok else 'FAILED'}: predictions are "
                                              f"{'identical' if ok else 'different'} when the held-out future is replaced.")


# ============================================================================================ LIVE TWIN
def view_live():
    if not need_windows():
        return
    units = sorted(W["unit"].unique())
    c1, c2, c3, c4 = st.columns(4)
    u = c1.selectbox("Unit", units, key="lt_unit")
    every = c2.select_slider("Update every (windows)", [1, 6, 24, 168], value=1, key="lt_every")
    rV = c3.number_input("Voltage noise (mV)", 0.1, 50.0, 3.0, key="lt_rV")
    use_prior = c4.toggle("Fleet prior (Bayesian updating)", True, key="lt_prior")
    eq_popover("Live twin")
    d = W[W["unit"] == u]
    if st.button("Replay the data through the twin", key="lt_run", type="primary"):
        with st.spinner("Kalman filter and particle filter..."):
            kf = E.kalman_replay(d, design, kin, data.get("eis"), r_V=rV / 1000, update_every=every, ref=op)
            prior = E.fleet_prior(W, u) if use_prior else None
            pf = E.particle_filter_eol(d, THR, update_every=every, prior=prior)
        ss["lt"] = dict(unit=u, kf=kf, pf=pf, wkey=WKEY)
    lt = ss.get("lt")
    if lt is None or lt["unit"] != u:
        st.info("Press Replay to run the self-updating twin on this unit.")
        return
    kf, pf = lt["kf"], lt["pf"]
    c1, c2 = st.columns(2)
    fig = go.Figure(go.Scatter(x=d.loc[d["stable"], "op_hours"], y=d.loc[d["stable"], "dV_mV"], mode="markers",
                               marker=dict(size=4, color="#9aa4b2"), name="measured dV"))
    fig.add_trace(go.Scatter(x=kf["op_hours"], y=kf["dV_ref_mV"], name="twin (KF) at reference j", line=dict(color=PAL[0], width=2)))
    if truth is not None:
        tu = truth[truth["unit"] == u]
        fig.add_trace(go.Scatter(x=tu["op_hours"], y=tu["dV_ref_mV"] - tu["dV_ref_mV"].iloc[0], name="synthetic truth",
                                 line=dict(color=PAL[2], width=1.5, dash="dash")))
    c1.plotly_chart(style(fig, "Voltage rise: data vs twin", "operating hours", "mV"), key="lt_dv")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=kf["op_hours"], y=1000 * op.j_A_cm2 * kf["dR_ohm_cm2"], name="ohmic part (j·ΔR)",
                             line=dict(color=PAL[1], width=2)))
    fig.add_trace(go.Scatter(x=kf["op_hours"], y=1000 * kf["d_eta_V"], name="activation part (Δη)", line=dict(color=PAL[0], width=2)))
    c2.plotly_chart(style(fig, "Dominant contribution (at reference j)", "operating hours", "mV"), key="lt_split")
    if not kf["observable"].iloc[-1]:
        st.warning("Constant current and no EIS: the ohmic and activation parts are not separately observable. "
                   "Only their sum is meaningful. Add EIS (HF intercept) or periods with varying load.")
    if len(pf):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.r_[pf["op_hours"], pf["op_hours"][::-1]],
                                 y=np.r_[pf["eol_p95"].clip(upper=pf["op_hours"].max() * 5), pf["eol_p05"][::-1]],
                                 fill="toself", fillcolor=PAL[0], opacity=0.15, line=dict(width=0), name="5-95 %"))
        fig.add_trace(go.Scatter(x=pf["op_hours"], y=pf["eol_med"], name="predicted EOL (median)", line=dict(color=PAL[0], width=2)))
        te = E.eol_time(d["op_hours"].to_numpy(), d["dV_mV"].to_numpy(), THR, 3)
        if np.isfinite(te):
            fig.add_hline(y=te, line=dict(color=PAL[2], dash="dash"), annotation_text=f"observed EOL {te:.0f} h")
        fig.update_yaxes(range=[0, max(pf["op_hours"].max() * 3, (te if np.isfinite(te) else 0) * 1.5)])
        st.plotly_chart(style(fig, f"Predicted end of life over time (threshold +{THR:g} mV)", "operating hours (now)",
                              "EOL (operating hours)", 400), key="lt_eol")
        explain("Particle filter on dV = c + a·h + b·(starts-1), initialised from the other units (fleet prior) when "
                "selected. A good twin converges to the observed EOL long before it happens.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Update-frequency study", key="lt_freq"):
            with st.spinner("Running the particle filter at several update intervals..."):
                ss["res_lt_freq"] = E.update_frequency_study(d, THR, prior=E.fleet_prior(W, u) if use_prior else None)
        if ss.get("res_lt_freq") is not None:
            st.dataframe(ss["res_lt_freq"].round(1), hide_index=True)
            st.caption("mae_eol_h needs the threshold to be reached in the data; otherwise only band widths compare.")
    with c2:
        if st.button("Measurement ablation (which sensors matter?)", key="lt_abl"):
            with st.spinner("Rebuilding indicators without each sensor..."):
                ss["res_lt_abl"] = E.measurement_ablation(data, design, kin, op, u, truth)
        if ss.get("res_lt_abl") is not None:
            st.dataframe(ss["res_lt_abl"].round(3), hide_index=True)


# ============================================================================================ OPERATIONS
def view_ops():
    lim = float(ss.get("o_hto", 2.0))
    st.markdown("#### 1 · Operating map (temperature × pressure at the selected current density)")
    eq_popover("Crossover and safety")
    c1, c2 = st.columns(2)
    Tg = c1.multiselect("Temperatures (C)", [30, 40, 50, 60, 70, 80, 90], default=[40, 60, 80], key="op_T")
    Pg = c2.multiselect("Pressures (bar)", [1, 3, 5, 10, 20, 30], default=[1, 5, 10, 30], key="op_P")
    if Tg and Pg:
        mp = E.operating_map(op, design, kin, sorted(Tg), sorted(Pg), hto_limit=lim)
        c1, c2 = st.columns(2)
        for col, z, t, k in [(c1, "kWh_per_kg", "Specific energy (kWh/kg)", "op_map_e"), (c2, "HTO_pct", "HTO (%)", "op_map_h")]:
            pv = mp.pivot(index="T_C", columns="P_bar", values=z)
            fig = go.Figure(go.Heatmap(z=pv.values, x=[str(x) for x in pv.columns], y=[str(y) for y in pv.index],
                                       colorscale="Blues", text=np.round(pv.values, 3), texttemplate="%{text}",
                                       textfont=dict(size=12)))
            fig = style(fig, t, "pressure (bar)", "temperature (C)", 330)
            fig.update_xaxes(type="category")
            fig.update_yaxes(type="category")
            col.plotly_chart(fig, key=k)
        st.dataframe(mp.round(4), hide_index=True)
        explain("Higher pressure raises E_rev and crossover (HTO) and lowers bubble coverage; higher temperature lowers "
                "the voltage. min_load_j is the lowest current density keeping steady HTO under the limit.")
    st.markdown("#### 2 · Load-following case study and optimisation")
    c1, c2, c3 = st.columns(3)
    prof_kind = c1.selectbox("Renewable profile", ["solar+wind", "solar", "wind", "upload CSV (power_frac[, price])"], key="op_prof")
    days = c2.number_input("Days", 1, 365, 30, key="op_days")
    jr = c3.number_input("Rated current density (A/cm2)", 0.05, 3.0, float(op.j_A_cm2), key="op_jr")
    price = None
    if prof_kind.startswith("upload"):
        f = st.file_uploader("Profile CSV with column power_frac (0-1) and optional price (per MWh)", type=["csv"], key="op_up")
        if f is None:
            st.info("Upload a profile, or choose a synthetic one.")
            return
        pdf = pd.read_csv(f)
        pf_ = pdf["power_frac"].clip(0, 1).to_numpy()
        price = pdf["price"].to_numpy() if "price" in pdf else None
    else:
        pf_ = E.renewable_profile(int(days) * 24, 0, prof_kind)
        st.caption("Synthetic profile for demonstration - upload your site's profile and prices for a real case.")
    c1, c2, c3 = st.columns(3)
    pconst = c1.number_input("Electricity price if no price column (per MWh)", 0.0, 1e4, 0.0, key="op_price",
                             help="No price is assumed: enter yours (0 = cost not evaluated).")
    eolmv = c2.number_input("EOL voltage rise for replacement cost (mV, 0 = ignore)", 0.0, 2000.0, 0.0, key="op_eol")
    scost = c3.number_input("Stack replacement cost (same currency, 0 = ignore)", 0.0, 1e9, 0.0, key="op_scost")
    if price is None:
        price = np.full(len(pf_), pconst)
    dv1 = ss.get("dv1") or (E.degradation_v1(W, op.T_C) if len(W) else None)
    lw = dv1["law"] if dv1 else {}
    c1, c2, c3 = st.columns(3)
    a = c1.number_input("Ageing a (mV/h at reference T)", 0.0, 1.0, float(np.nan_to_num(lw.get("a_mV_h", 0.0))), format="%.5f", key="op_a")
    Ea = c2.number_input("Ea of ageing (J/mol)", 0.0, 2e5, float(np.nan_to_num(lw.get("Ea_J_mol", 0.0))), key="op_Ea")
    b = c3.number_input("b (mV per start-up)", 0.0, 50.0, float(np.nan_to_num(lw.get("b_mV_start", 0.0))), key="op_b")
    st.caption("Degradation law defaults come from degradation model v1 (Diagnostics) " +
               ("on SYNTHETIC data." if source == "synthetic" else "on your data." if source == "real" else "- no data, zero."))
    c1, c2 = st.columns(2)
    Topt = c1.multiselect("Temperature setpoints (C)", [40, 50, 60, 70, 80, 90], default=[60, 70, 80], key="op_Topt")
    mlopt = c2.multiselect("Minimum load (share of rated)", [0.05, 0.1, 0.15, 0.2, 0.3, 0.4], default=[0.1, 0.2, 0.3], key="op_ml")
    if st.button("Optimise operation", key="op_run", type="primary") and Topt and mlopt:
        with st.spinner("Simulating each policy hour by hour..."):
            ss["op_res"] = E.optimise_operation(pf_, price, design, kin, op, jr, dict(a=a, Ea=Ea, b=b),
                                                sorted(Topt), sorted(mlopt), lim, eolmv or None, scost or None)
            best = ss["op_res"].iloc[0]
            ss["op_best"] = E.simulate_policy(pf_, price, design, kin, E.OperatingConditions(**{**asdict(op), "T_C": float(best["T_C"])}),
                                              jr, float(best["min_load"]), dict(a=a, Ea=Ea, b=b), lim)
    if ss.get("op_res") is not None:
        r = ss["op_res"]
        st.dataframe(r.round(4), hide_index=True)
        fig = go.Figure()
        for k, (T, d) in enumerate(r.groupby("T_C")):
            fig.add_trace(go.Scatter(x=d["H2_kg"], y=d["dV_mV"], mode="markers+text", text=[f"{x:.0%}" for x in d["min_load"]],
                                     textposition="top center", name=f"{T:g} C", marker=dict(size=10, color=PAL[k % len(PAL)])))
        st.plotly_chart(style(fig, "Trade-off: hydrogen vs voltage rise (labels: minimum load)", "H2 (kg)", "dV (mV)"), key="op_pareto")
        hb = ss["op_best"]["hourly"]
        fig = go.Figure(go.Scatter(x=hb["hour"], y=hb["j"], name="j (A/cm2)", line=dict(color=PAL[0], width=1.5)))
        st.plotly_chart(style(fig, "Best feasible policy: current density", "hour", "A/cm2", 300), key="op_hourly")
        explain("Ranking: feasible (no HTO violation) first, then cost per kg (energy, plus stack replacement spread over "
                "the predicted life when EOL and stack cost are given). Without prices the ranking is by kWh/kg only.")
    st.markdown("#### 3 · Maintenance timing")
    res = ss.get("fc_res")
    if res is None or not len(res):
        st.info("Run the assessment in Models & forecasting to get a forecast with a band for the maintenance decision.")
        return
    c1, c2, c3 = st.columns(3)
    mu = c1.selectbox("Unit", sorted(res["unit"].unique()), key="mt_u")
    mm = c2.selectbox("Model", sorted(res["model"].unique()), index=0, key="mt_m")
    ms = c3.selectbox("Origin", sorted(res["share"].unique()), key="mt_s")
    r = res[(res["unit"] == mu) & (res["model"] == mm) & (res["share"] == ms)]
    if len(r):
        mt = E.maintenance_time(r, THR)
        m = st.columns(3)
        m[0].metric("Service by (upper band crosses EOL)", f"{mt['service_h']:.0f} h" if np.isfinite(mt["service_h"]) else "beyond data")
        m[1].metric("Median EOL", f"{mt['median_eol_h']:.0f} h" if np.isfinite(mt["median_eol_h"]) else "beyond data")
        m[2].metric("Optimistic EOL", f"{mt['optimistic_eol_h']:.0f} h" if np.isfinite(mt["optimistic_eol_h"]) else "beyond data")


# ============================================================================================ STUDY RESULTS
def view_results():
    tag = {"synthetic": "SYNTHETIC data (method check only)", "real": "REAL data", None: "no data"}[source]
    st.markdown(f"#### Mission answers so far · verified on: **{tag}**")
    dv1 = ss.get("dv1")
    sc = ss.get("fc_scores")
    rows = [("M1 · Health indicators", "See Diagnostics §3 ranking" if len(W) else "needs data"),
            ("M1 · Degradation model v1",
             (f"a = {1000 * dv1['law']['a_mV_h']:.2f} uV/h, Ea = {dv1['law']['Ea_J_mol'] / 1000:.1f} kJ/mol, "
              f"b = {dv1['law']['b_mV_start']:.3f} mV/start") if dv1 else "open Diagnostics"),
            ("M2 · Prediction accuracy", (E.scoreboard(sc)[0].iloc[0]["model"] + " lowest mean RMSE") if sc is not None and len(sc) else "run the assessment"),
            ("M2 · Recommended model", ss.get("recommended") or "run the assessment"),
            ("M2 · Update frequency", "see Live twin study" if ss.get("res_lt_freq") is not None else "run the study"),
            ("M3 · Operation", "see Operations optimisation" if ss.get("op_res") is not None else "run the optimisation"),
            ("Not yet implemented", "GRU/Transformer (L3), PINN with ablation (L4): next steps")]
    st.dataframe(pd.DataFrame(rows, columns=["question", "status"]), hide_index=True)
    if sc is not None and len(sc):
        st.markdown("**Per condition group (RMSE, mV)**")
        st.dataframe(E.group_table(sc))
        st.download_button("Download per-unit scores", sc.to_csv(index=False), "scores_per_unit.csv", key="r_dl1")
    if len(W):
        st.download_button("Download operating-window table", W.to_csv(index=False), "windows.csv", key="r_dl2")
    st.caption("Honesty rules applied: no look-ahead (tested), forecasts anchored at today's value, bands from held-out "
               "errors with coverage reported, per-unit averaging with 95 % CI and small-sample wording, results per "
               "condition group, recoveries flagged and kept in test data.")


{"Home": view_home, "Diagnostics": view_diagnostics, "Live twin": view_live, "Models & forecasting": view_models,
 "Operations & control": view_ops, "Study results": view_results}[view]()
