"""AWE digital twin - engine (no Streamlit import; everything here is testable).

Physics follows the project articles; every function names its source:
  [Hu22]     Hu et al., Applied Energy 327 (2022) 120099  - review of AWE mathematical modelling
  [Ham25]    Ham et al., Int J Hydrogen Energy 187 (2025) 152069 - 10 kW AWE system model
  [Daoudi24] Daoudi & Bounahmidi, Int J Hydrogen Energy 49 (2024) 646 - review of AWE modelling
  [Qu26]     Qu et al., Chin Sci Bull 71 (2026) 4267 - digital/intelligent AWE review
  [DRT25]    Wang et al., ACS Electrochem 1 (2025) 2680 - DRTtools
Values the articles do not give are NOT invented: they are inputs (None by default) or explicitly labelled
assumptions that the app shows next to the result.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import spearmanr, t as t_dist, theilslopes, wilcoxon

ENGINE_VERSION = "0.2.0"

F = 96485.0            # C/mol  [Hu22 nomenclature]
R_GAS = 8.314          # J/mol/K [Hu22]
M_H2 = 2.01588e-3      # kg/mol
KWH_PER_KG_PER_VOLT = 2 * F / M_H2 / 3.6e6   # specific energy per volt of cell voltage at eta_F = 1

# --------------------------------------------------------------------------------------------- inputs
ELECTRODE_PRESETS = {
    "Nickel (literature kinetics, Jang/Ham)": "nickel",
    "Stainless steel (fit kinetics from data)": "stainless",
    "Custom (enter kinetics)": "custom",
}
MEMBRANE_MODELS = {
    "Empirical 0.5 mm polysulfone/Zirfon-type in 30 wt% KOH [Ham25 eq 9]": "jang",
    "Area resistance input (ohm cm2)": "area",
    "Structural: thickness, porosity, tortuosity [Hu22 eq 24-25]": "structural",
}


@dataclass
class CellDesign:
    name: str = "Cell"
    n_cells: int = 1
    area_cm2: float = 25.0
    electrode: str = "nickel"              # nickel | stainless | custom
    electrode_thickness_mm: float = 2.0    # Ham25 Table 6 validation cell: 2.0 mm
    electrode_sigma_S_m: Optional[float] = None   # None + nickel -> Hu22 eq 22; None + other -> neglected
    gap_anode_mm: float = 0.0              # electrode-separator distance (0 = zero gap)
    gap_cathode_mm: float = 0.0
    membrane_model: str = "jang"
    membrane_area_res_ohm_cm2: Optional[float] = None   # Zirfon 0.27 ohm cm2 [Qu26 Table 2]
    sep_thickness_mm: float = 0.5
    sep_porosity: Optional[float] = None
    sep_tortuosity: Optional[float] = None    # None -> (2-eps)^2/eps [Hu22 eq 25]
    h2_permeability: Optional[float] = 1.85e-11   # mol cm-1 s-1 bar-1, Zirfon [Qu26 Table 2]
    circulation: str = "separated"          # separated | mixed  [Hu22 sec 3.3.1.1]
    supersaturation: float = 1.0            # multiplier on dissolved H2 driving force [Hu22 sec 3.3.1.2]


@dataclass
class OperatingConditions:
    T_C: float = 70.0
    P_bar: float = 1.0                      # absolute
    j_A_cm2: float = 0.4
    koh_wt: float = 30.0                     # wt %
    lye_flow_L_min: Optional[float] = None  # total electrolyte flow (needed for mixed circulation)
    dP_anode_minus_cathode_bar: float = 0.0
    h2_solubility_mol_m3_bar: Optional[float] = 0.088   # Hu22 Table 6 (75 C, 30.2 wt%); pressure basis assumed per bar
    lye_viscosity_Pa_s: Optional[float] = None          # needed only for Darcy convection; not given in the articles

    @property
    def T_K(self) -> float:
        return self.T_C + 273.15


@dataclass
class Kinetics:
    """Butler-Volmer/Tafel parameters. Defaults: nickel electrodes [Ham25 eq 5-6], [Hu22 Table B3].
    T_ref/P_ref are NOT stated in the articles; 298.15 K / 1 bar is an assumption (the app labels it) that reproduces
    Ham25's reported activation overvoltage (see tests/test_engine.py::test_reproduces_ham25)."""
    mode: str = "butler_volmer"     # butler_volmer | tafel_fit
    j0_c_ref: float = 1.5e-4        # A/cm2
    j0_a_ref: float = 0.9e-4
    Ea_c: float = 23000.0           # J/mol
    Ea_a: float = 42000.0
    alpha_variant: str = "jang"     # henao: aa = 0.0675+0.00095T ; jang: aa = 0.07835+0.001T  [Hu22 Table B3]
    n_e: int = 2                    # electrons in RT/(n alpha F) [Hu22 eq 18], [Ham25 eq 14]
    T_ref_K: float = 298.15         # ASSUMPTION
    P_ref_bar: float = 1.0          # ASSUMPTION
    # tafel_fit mode (e.g. stainless steel fitted from LSV): eta_act = b * ln(j / j0)
    tafel_b: Optional[float] = None
    tafel_j0: Optional[float] = None
    use_bubbles: bool = True


@dataclass
class FaradayParams:
    """[Hu22 eq 7]: eta_F = (j^2/(f1 + j^2)) * f2 (j in mA/cm2 here). Coefficients are cell specific and NOT given in
    the articles -> None means eta_F = 1 and the app says so."""
    f1: Optional[float] = None
    f2: Optional[float] = None


@dataclass
class DegradationState:
    """Shift of the health parameters relative to the fresh cell (used by the live twin and degradation laws)."""
    dR_ohm_cm2: float = 0.0       # added ohmic area resistance
    d_eta_act_V: float = 0.0      # added activation overvoltage at the operating point
    j0_factor: float = 1.0        # multiplies both exchange current densities
    crossover_factor: float = 1.0


ASSUMPTIONS = [
    ("Kinetics T_ref / P_ref = 298.15 K / 1 bar", "Not stated in Hu22/Ham25 for eq 5-6 and eq 13; chosen because it "
     "reproduces Ham25's activation overvoltage within ~5 mV (unit test)."),
    ("alpha_a from Jang (0.07835 + 0.001 T)", "Hu22 Table B3 lists two fits; the Jang one reproduces Ham25 (which cites Jang)."),
    ("V_tn: T in deg C in Hu22 eq A.2/A.6", "Otherwise V_tn(25 C) = 1.54 V, contradicting 1.48 V in the same article."),
    ("H2 solubility 0.088 mol m-3 per bar", "Hu22 Table 6 lists 0.088 mol/m3 at 75 C, 30.2 wt% KOH without stating the "
     "pressure basis; assumed per bar of H2. Used only for mixed-circulation crossover."),
    ("Faraday efficiency = 1 unless f1, f2 given", "Hu22 eq 7-8 coefficients are cell specific and not given."),
    ("Zirfon H2 permeability 1.85e-11 mol cm-1 s-1 bar-1", "Qu26 Table 2; temperature dependence not given, so constant."),
    ("Membrane resistance (Ham25 eq 9 with S in m2)", "Hu22 eq 28 prints 1000*A, Ham25 10000*S: units ambiguous; Ham25 "
     "form gives 0.14-0.2 ohm cm2, consistent with Zirfon 0.27 ohm cm2 (Qu26)."),
]

# --------------------------------------------------------------------------------------------- thermodynamics
def molarity_from_wt(w_pct: float, T_K: float) -> float:
    """KOH molarity (mol/L) from wt % and temperature [Ham25 eq 12], [Hu22 eq B.6]."""
    return w_pct * (183.1221 - 0.56845 * T_K + 984.5679 * math.exp(w_pct / 115.96277)) / 5610.5


def wt_from_molarity(m: float, T_K: float) -> float:
    return float(optimize.brentq(lambda w: molarity_from_wt(w, T_K) - m, 0.01, 60.0))


def E0_rev(T_K):
    """Standard reversible voltage [Ham25 eq 3], [Hu22 Table 3, ref 20]."""
    return 1.50342 - 9.956e-4 * T_K + 2.5e-7 * T_K ** 2


def p_sat_water_bar(T_K):
    """Vapour pressure of pure water (bar) [Hu22 eq B.4]."""
    return T_K ** -3.4159 * np.exp(37.034 - 6275.7 / T_K)


def water_activity_koh(m, T_K):
    """[Hu22 eq B.2], m in mol/L."""
    return 10 ** (-0.02255 * m + 0.001434 * m ** 2 + (1.38 * m - 0.9254 * m ** 2) / T_K)


def p_water_koh_bar(m, T_K):
    return water_activity_koh(m, T_K) * p_sat_water_bar(T_K)


def E_rev(T_K, P_bar, m):
    """Reversible (Nernst) voltage [Hu22 eq 11], [Ham25 eq 2]."""
    a = water_activity_koh(m, T_K)
    pw = a * p_sat_water_bar(T_K)
    return E0_rev(T_K) + R_GAS * T_K / (2 * F) * np.log(np.maximum(P_bar - pw, 1e-6) ** 1.5 / a)


def V_thermoneutral(T_K, P_bar, m):
    """Thermoneutral voltage [Hu22 App. A, eq A.1-A.6].
    Hu22 prints eq A.2 and A.6 without a temperature unit; with T in K, V_tn(25 C, 1 bar) would be 1.54 V, contradicting
    the 1.48 V stated in the same article, so T is taken in deg C there (gives 1.48 V and dH_vap ~ 44 kJ/mol)."""
    pw = p_water_koh_bar(m, T_K)
    phi = 1.5 * pw / max(P_bar - pw, 1e-6)
    Tc = T_K - 273.15
    dH = 42960 + 40.762 * Tc - 0.06682 * Tc ** 2
    v_hhv0 = 1.4756 + 2.252e-4 * Tc + 1.52e-8 * Tc ** 2
    P = P_bar
    f_hhv = (21.661e-6 - 10.941e-3 / T_K) * P + (18.578e-6 / T_K + 0.339e-3 / T_K ** 1.5 + 7.845e-3 / T_K ** 2
                                                   - 1.659 / T_K ** 3) * P ** 2
    return v_hhv0 + f_hhv + phi / (2 * F) * dH


# --------------------------------------------------------------------------------------------- transport
def sigma_koh(m, T_K):
    """KOH conductivity (S/cm), 0-12 mol/L, 273-373 K [Hu22 Table 4, ref 37], [Ham25 eq 11]."""
    return (-2.041 * m - 2.8e-3 * m ** 2 + 5.332e-3 * m * T_K + 207.2 * m / T_K + 1.043e-3 * m ** 3
            - 3e-7 * m ** 2 * T_K ** 2)


def sigma_nickel(T_K):
    """Nickel conductivity (S/m) [Hu22 eq 22]."""
    return 6.0e7 - 279650 * T_K + 532 * T_K ** 2 - 0.38057 * T_K ** 3


def bubble_coverage(j_A_cm2, T_K, P_bar, kin: Kinetics):
    """theta = 0.023 j^0.3 (T/Tref * Pref/P)^(2/3), j in A/m2 [Hu22 eq 17], [Ham25 eq 13]."""
    if not kin.use_bubbles:
        return np.zeros_like(np.asarray(j_A_cm2, float))
    th = 0.023 * (np.maximum(np.asarray(j_A_cm2, float), 0) * 1e4) ** 0.3 * (
        T_K / kin.T_ref_K * kin.P_ref_bar / P_bar) ** (2 / 3)
    return np.clip(th, 0.0, 0.95)


def alphas(T_K, kin: Kinetics):
    """Charge-transfer coefficients [Hu22 Table B3]."""
    ac = 0.1175 + 0.00095 * T_K
    aa = 0.0675 + 0.00095 * T_K if kin.alpha_variant == "henao" else 0.07835 + 0.001 * T_K
    return ac, aa


def exchange_currents(T_K, P_bar, kin: Kinetics):
    """j0 for nickel electrodes [Ham25 eq 5-6], [Hu22 Table B3, ref 11]."""
    fac = (P_bar / kin.P_ref_bar) ** 0.1
    j0c = kin.j0_c_ref * fac * math.exp(-kin.Ea_c / (R_GAS * T_K) * (1 - T_K / kin.T_ref_K))
    j0a = kin.j0_a_ref * fac * math.exp(-kin.Ea_a / (R_GAS * T_K) * (1 - T_K / kin.T_ref_K))
    return j0c, j0a


def membrane_resistance_cm2(design: CellDesign, T_K, m):
    """Area-specific separator resistance (ohm cm2)."""
    if design.membrane_model == "area":
        return design.membrane_area_res_ohm_cm2
    if design.membrane_model == "structural":
        eps = design.sep_porosity
        if eps is None:
            return None
        tau = design.sep_tortuosity if design.sep_tortuosity else (2 - eps) ** 2 / eps      # [Hu22 eq 25]
        return (design.sep_thickness_mm / 10) * tau / (eps * sigma_koh(m, T_K))            # [Hu22 eq 24]
    # Jang empirical, Ham25 eq 9: R = (0.06 + 80 exp(-T/50)) / (10000 S), S in m2 -> area-specific (ohm cm2)
    return 0.06 + 80 * math.exp(-T_K / 50)


# --------------------------------------------------------------------------------------------- cell voltage
def cell_voltage(j_A_cm2, op: OperatingConditions, design: CellDesign, kin: Kinetics,
                 deg: Optional[DegradationState] = None) -> Dict[str, np.ndarray]:
    """U = E_rev + eta_act,c + eta_act,a + eta_ohm (+ bubble effects)  [Hu22 eq 4, 18, 19], [Ham25 eq 1, 7, 14, 15].
    Returns all components (V) and the bubble coverage. j may be an array."""
    deg = deg or DegradationState()
    j = np.maximum(np.atleast_1d(np.asarray(j_A_cm2, float)), 1e-6)
    T = op.T_K
    m = molarity_from_wt(op.koh_wt, T)
    th = bubble_coverage(j, T, op.P_bar, kin)
    erev = np.full_like(j, float(E_rev(T, op.P_bar, m)))
    if kin.mode == "tafel_fit" and kin.tafel_b is not None and kin.tafel_j0 is not None:
        eta_c = np.zeros_like(j)
        eta_a = kin.tafel_b * np.log(np.maximum(j / (kin.tafel_j0 * deg.j0_factor * (1 - th)), 1.0))
    else:
        ac, aa = alphas(T, kin)
        j0c, j0a = exchange_currents(T, op.P_bar, kin)
        j0c, j0a = j0c * deg.j0_factor, j0a * deg.j0_factor
        bc, ba = R_GAS * T / (kin.n_e * ac * F), R_GAS * T / (kin.n_e * aa * F)
        # Tafel form with bubble coverage [Hu22 eq 18], [Ham25 eq 14]; clipped at 0 below j0
        eta_c = bc * np.log(np.maximum(j / (j0c * (1 - th)), 1.0))
        eta_a = ba * np.log(np.maximum(j / (j0a * (1 - th)), 1.0))
    eta_act = eta_c + eta_a + deg.d_eta_act_V
    sig = sigma_koh(m, T)
    gap_cm = (design.gap_anode_mm + design.gap_cathode_mm) / 10
    r_ele = gap_cm / sig / (1 - 2 / 3 * th) ** 1.5                    # [Ham25 eq 15] (bubble-corrected)
    r_mem = membrane_resistance_cm2(design, T, m)
    r_mem = 0.0 if r_mem is None else r_mem
    if design.electrode_sigma_S_m is not None:
        s_el = design.electrode_sigma_S_m
    elif design.electrode == "nickel":
        s_el = sigma_nickel(T)
    else:
        s_el = None
    r_el = 0.0 if s_el is None else 2 * (design.electrode_thickness_mm * 1e-3) / s_el * 1e4   # ohm cm2 [Hu22 eq 20-21]
    r_tot = r_ele + r_mem + r_el + deg.dR_ohm_cm2
    eta_ohm = j * r_tot
    U = erev + eta_act + eta_ohm
    return dict(j=j, U=U, E_rev=erev, eta_act_c=eta_c, eta_act_a=eta_a, eta_act=eta_act, eta_ohm=eta_ohm,
                R_ohm_cm2=np.full_like(j, r_tot) if np.ndim(r_tot) == 0 else r_tot, theta=th,
                R_mem_cm2=np.full_like(j, r_mem), R_ele_cm2=np.atleast_1d(r_ele) * np.ones_like(j))


def faraday_efficiency(j_A_cm2, fp: FaradayParams):
    """[Hu22 eq 7] with j in mA/cm2; returns 1 when coefficients are missing."""
    j = np.asarray(j_A_cm2, float)
    if fp.f1 is None or fp.f2 is None:
        return np.ones_like(j)
    jm = j * 1000
    return jm ** 2 / (fp.f1 + jm ** 2) * fp.f2


def production(j_A_cm2, design: CellDesign, fp: FaradayParams):
    """H2 production [Hu22 eq 5, 9]: mol/s, kg/h, Nm3/h for the whole stack."""
    I = np.asarray(j_A_cm2, float) * design.area_cm2
    n = faraday_efficiency(j_A_cm2, fp) * design.n_cells * I / (2 * F)
    return dict(mol_s=n, kg_h=n * M_H2 * 3600, Nm3_h=n * 0.0224141 * 3600)


def efficiencies(U, op: OperatingConditions, eta_F=1.0):
    """Voltage efficiency E_rev/U and thermal (HHV) efficiency V_tn/U [Hu22 Table B1]; specific energy kWh/kg."""
    T = op.T_K
    m = molarity_from_wt(op.koh_wt, T)
    U = np.asarray(U, float)
    return dict(voltage_eff=E_rev(T, op.P_bar, m) / U, thermal_eff=V_thermoneutral(T, op.P_bar, m) / U,
                kWh_per_kg=KWH_PER_KG_PER_VOLT * U / np.maximum(eta_F, 1e-6))


# --------------------------------------------------------------------------------------------- crossover / safety
def hto_percent(j_A_cm2, op: OperatingConditions, design: CellDesign, kin: Kinetics,
                deg: Optional[DegradationState] = None) -> Dict[str, np.ndarray]:
    """Hydrogen in oxygen at the stack outlet, simplified material balance [Hu22 eq 38, 41-42, 46-47], [Ham25 eq 21-24].
    Fick: N = P_perm * p_H2,cat / delta * (1-theta) * supersaturation; Darcy only if dP<0 and viscosity given;
    mixing only for mixed circulation. HTO = N_H2 / (N_H2 + N_O2)."""
    deg = deg or DegradationState()
    j = np.maximum(np.atleast_1d(np.asarray(j_A_cm2, float)), 1e-6)
    T = op.T_K
    m = molarity_from_wt(op.koh_wt, T)
    th = bubble_coverage(j, T, op.P_bar, kin)
    p_h2 = max(op.P_bar - float(p_water_koh_bar(m, T)), 1e-6)
    perm = design.h2_permeability or 0.0
    delta_cm = design.sep_thickness_mm / 10
    n_fick = perm * p_h2 / delta_cm * (1 - th) * design.supersaturation * deg.crossover_factor     # mol/s/cm2
    n_darcy = np.zeros_like(j)
    if op.dP_anode_minus_cathode_bar < 0 and op.lye_viscosity_Pa_s and op.h2_solubility_mol_m3_bar \
            and design.sep_porosity:
        pore = 0.15e-6                                   # Zirfon mean pore size [Hu22 sec 3.3.1.2]
        eps = design.sep_porosity
        tau = design.sep_tortuosity or (2 - eps) ** 2 / eps
        K = eps * pore ** 2 / (32 * tau)                 # [Hu22 eq 43]
        n_darcy = (K / op.lye_viscosity_Pa_s * op.h2_solubility_mol_m3_bar * p_h2 * (-op.dP_anode_minus_cathode_bar * 1e5)
                   / (delta_cm / 100)) * 1e-4 * (1 - th)                          # mol/s/cm2 [Hu22 eq 42]
    n_mix = np.zeros_like(j)
    if design.circulation == "mixed" and op.lye_flow_L_min and op.h2_solubility_mol_m3_bar:
        v = op.lye_flow_L_min / 1000 / 60                  # m3/s
        n_mix = np.full_like(j, op.h2_solubility_mol_m3_bar * p_h2 * v / 4 / (design.area_cm2 * design.n_cells))  # [Hu22 eq 38]
    n_h2 = n_fick + n_darcy + n_mix
    n_o2 = j / (4 * F)
    return dict(j=j, HTO_pct=100 * n_h2 / (n_h2 + n_o2), N_fick=n_fick, N_darcy=n_darcy, N_mix=n_mix)


SAFETY = dict(HTO_shutdown_pct=2.0, HTO_LEL_pct=4.0)   # [Hu22 sec 3.3], [Qu26 sec 3.3], [Ham25 sec 3.1.2]


def min_load_for_hto(op, design, kin, limit_pct=2.0, deg=None, j_max=2.0):
    """Lowest current density whose steady HTO stays below the limit (HTO falls with j) [Hu22 sec 3.3]."""
    f = lambda j: float(hto_percent(j, replace(op, j_A_cm2=j), design, kin, deg)["HTO_pct"][0]) - limit_pct
    lo, hi = 1e-4, j_max
    if f(lo) <= 0:
        return lo
    if f(hi) > 0:
        return float("nan")
    return float(optimize.brentq(f, lo, hi))


def thermal_step(T_K, U, I, n_cells, Ct, Rt, Ta_K, dt_s, m_koh=6.9, P_bar=1.0, Tcw_in_K=None, UA=None, C_cw=None):
    """Lumped thermal balance Ct dT/dt = Qgen - Qloss - Qcool  [Hu22 eq 53, 56, 58, 61-62]. Needs Ct, Rt (not given in
    the articles; Ct can be measured from a heating curve, Hu22 eq 55)."""
    q_gen = n_cells * (U - V_thermoneutral(T_K, P_bar, m_koh)) * I
    q_loss = (T_K - Ta_K) / Rt
    q_cool = 0.0
    if Tcw_in_K is not None and UA and C_cw:
        Tcw_out = Tcw_in_K + (T_K - Tcw_in_K) * (1 - math.exp(-UA / C_cw))
        q_cool = C_cw * (Tcw_out - Tcw_in_K)
    return T_K + dt_s * (q_gen - q_loss - q_cool) / Ct


# ============================================================================================ data model and loading
DATA_TYPES = {
    "timeseries": dict(label="Operating time series (current, voltage, T, p, KOH, flow, gas purity)",
                       required=["unit", "time_h", "voltage_V"],
                       optional=["current_A", "j_A_cm2", "n_cells", "cell_id", "T_C", "T_in_C", "T_out_C", "P_bar",
                                 "P_cat_bar", "P_an_bar", "koh_wt", "flow_L_min", "HTO_pct", "OTH_pct", "h2_flow_Nm3_h",
                                 "power_W"]),
    "polcurve": dict(label="Polarisation curves (U-i), e.g. LSV or galvanostatic steps",
                     required=["unit", "time_h", "j_A_cm2", "voltage_V"], optional=["T_C", "P_bar", "koh_wt"]),
    "eis": dict(label="EIS spectra (frequency, Z', -Z'')", required=["unit", "time_h", "freq_Hz", "z_re", "z_im_neg"],
                optional=["j_A_cm2", "T_C"]),
    "events": dict(label="Events (start-up, shutdown, standby, trip, KOH refill, electrolyte change, service...)",
                   required=["unit", "time_h", "event"], optional=["note"]),
    "drt": dict(label="DRT results (tau or frequency, gamma) or peak table", required=["unit", "time_h"],
                optional=["freq_Hz", "tau_s", "gamma", "peak", "R_peak_ohm"]),
}
EVENT_KINDS = ["startup", "shutdown", "standby", "load_change", "trip", "koh_refill", "electrolyte_change",
               "stack_service", "power_supply_change", "measurement", "other"]
ALIASES = {
    "unit": ["unit", "cell", "stack", "cell_name", "unit_id", "id"],
    "time_h": ["time_h", "time", "hours", "t_h", "elapsed_h", "time (h)", "test time (h)"],
    "timestamp": ["timestamp", "datetime", "date", "date_time"],
    "voltage_V": ["voltage_v", "voltage", "u", "v", "cell_voltage", "stack_voltage", "ewe", "ewe/v", "e (v)", "u (v)"],
    "current_A": ["current_a", "current", "i", "i (a)", "i/ma", "stack_current"],
    "j_A_cm2": ["j_a_cm2", "j", "current_density", "j (a/cm2)", "current density"],
    "T_C": ["t_c", "temperature", "temp", "t (c)", "temperature_c"],
    "P_bar": ["p_bar", "pressure", "p (bar)"],
    "koh_wt": ["koh_wt", "koh", "koh (wt%)", "concentration"],
    "flow_L_min": ["flow_l_min", "flow", "lye_flow"],
    "HTO_pct": ["hto_pct", "hto", "h2_in_o2"],
    "OTH_pct": ["oth_pct", "oth", "o2_in_h2"],
    "h2_flow_Nm3_h": ["h2_flow_nm3_h", "h2_flow", "hydrogen_flow"],
    "freq_Hz": ["freq_hz", "freq", "frequency", "f", "freq/hz"],
    "z_re": ["z_re", "zre", "z'", "re(z)", "re(z)/ohm", "zreal"],
    "z_im_neg": ["z_im_neg", "-z''", "-im(z)", "-im(z)/ohm", "-zimag", "zim_neg"],
    "event": ["event", "type", "event_type", "kind"],
    "tau_s": ["tau_s", "tau"], "gamma": ["gamma", "gamma_ohm", "drt"],
}


def guess_mapping(columns: Sequence[str], dtype: str) -> Dict[str, Optional[str]]:
    """Propose a column mapping {standard name: file column} from the aliases (user can override in the app)."""
    spec = DATA_TYPES[dtype]
    low = {str(c).strip().lower(): c for c in columns}
    out = {}
    for std in spec["required"] + spec["optional"] + (["timestamp"] if dtype == "timeseries" else []):
        out[std] = None
        for a in ALIASES.get(std, [std.lower()]):
            if a in low:
                out[std] = low[a]
                break
    return out


def read_table(file_like, name: str) -> pd.DataFrame:
    """Read csv / txt (any delimiter) / xlsx / parquet. EIS files without header (3 columns: f, Z', -Z'') supported."""
    n = name.lower()
    if n.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_like)
    if n.endswith(".parquet"):
        return pd.read_parquet(file_like)
    df = pd.read_csv(file_like, sep=None, engine="python")
    if all(_is_number(c) for c in df.columns):          # headerless numeric file (e.g. DRTtools EIS format)
        if hasattr(file_like, "seek"):
            file_like.seek(0)
        df = pd.read_csv(file_like, sep=None, engine="python", header=None)
        if df.shape[1] == 3:
            df.columns = ["freq_Hz", "z_re", "z_im_neg"]
    return df


def _is_number(x) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def apply_mapping(df: pd.DataFrame, mapping: Dict[str, Optional[str]], dtype: str, design: CellDesign,
                  default_unit: str = "unit-1", default_time_h: Optional[float] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Rename to standard names, convert units, derive j or current, and report what is missing."""
    notes = []
    out = pd.DataFrame(index=df.index)
    for std, col in mapping.items():
        if col is not None and col in df.columns:
            out[std] = df[col]
    if "unit" not in out:
        out["unit"] = default_unit
        notes.append(f"No unit column: all rows assigned to '{default_unit}'.")
    out["unit"] = out["unit"].astype(str)
    if "time_h" not in out and "timestamp" in out:
        ts = pd.to_datetime(out["timestamp"], errors="coerce")
        out["time_h"] = (ts - ts.min()).dt.total_seconds() / 3600
        notes.append("time_h derived from timestamp (hours since first record).")
    if "time_h" not in out:
        if default_time_h is not None:
            out["time_h"] = float(default_time_h)
        elif dtype == "timeseries":
            out["time_h"] = np.arange(len(out), dtype=float) / 3600
            notes.append("No time column: assumed 1 s sampling.")
    for c in out.columns:
        if c not in ("unit", "event", "timestamp", "note", "peak", "cell_id"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if dtype in ("timeseries", "polcurve"):
        if "j_A_cm2" not in out and "current_A" in out:
            out["j_A_cm2"] = out["current_A"] / design.area_cm2
            notes.append(f"j = I / {design.area_cm2:g} cm2 (cell design).")
        if "current_A" not in out and "j_A_cm2" in out:
            out["current_A"] = out["j_A_cm2"] * design.area_cm2
        if dtype == "timeseries":
            n = out["n_cells"] if "n_cells" in out else design.n_cells
            out["U_cell_V"] = out["voltage_V"] / n
            if design.n_cells > 1 and "n_cells" not in out:
                notes.append(f"Cell voltage = stack voltage / {design.n_cells} cells.")
    if dtype == "events":
        out["event"] = out["event"].astype(str).str.strip().str.lower().str.replace(" ", "_")
        unknown = sorted(set(out["event"]) - set(EVENT_KINDS))
        if unknown:
            notes.append(f"Events not in the standard list (kept as 'other' for logic): {unknown}")
    missing = [c for c in DATA_TYPES[dtype]["required"] if c not in out]
    if missing:
        notes.append(f"MISSING required columns: {missing}")
    return out.sort_values(["unit", "time_h"]).reset_index(drop=True), notes


def signal_inventory(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Which expected signals are present, and what their absence rules out (dataset report)."""
    ts = data.get("timeseries")
    has = lambda c: ts is not None and c in ts and ts[c].notna().any()
    nun = lambda c: int(ts[c].round(3).nunique()) if has(c) else 0
    rows = [
        ("Current / current density", has("j_A_cm2"), "No operating point: nothing can be normalised."),
        ("Voltage", has("voltage_V"), "No health indicator at all."),
        ("Temperature", has("T_C"), "Voltage cannot be corrected to reference temperature; no Arrhenius effect."),
        ("Pressure", has("P_bar"), "Pressure assumed constant (sidebar value); no pressure effect on ageing."),
        ("KOH concentration", has("koh_wt"), "Concentration drift between refills unknown (sidebar value used)."),
        ("Electrolyte flow", has("flow_L_min"), "Mixed-circulation crossover and bubble/flow effects not checked."),
        ("H2 in O2 (HTO)", has("HTO_pct"), "Crossover/safety model cannot be validated on data (literature only)."),
        ("O2 in H2 (OTH)", has("OTH_pct"), "Product purity not tracked."),
        ("H2 flow", has("h2_flow_Nm3_h"), "Faraday efficiency cannot be measured (eta_F = 1 assumed)."),
        ("Events", data.get("events") is not None and len(data["events"]) > 0,
         "Start-stop damage, recoveries after refills/service cannot be attributed."),
        ("Polarisation curves", data.get("polcurve") is not None and len(data["polcurve"]) > 0,
         "Activation/ohmic split only from load variation in the time series (if any)."),
        ("EIS", data.get("eis") is not None and len(data["eis"]) > 0,
         "No direct ohmic resistance (HF intercept); split relies on U-i data."),
    ]
    df = pd.DataFrame(rows, columns=["signal", "present", "if missing"])
    if ts is not None:
        df.loc[len(df)] = ["Distinct current levels", nun("j_A_cm2") > 3,
                           f"{nun('j_A_cm2')} level(s): load dependence of ageing not identifiable."]
        df.loc[len(df)] = ["Distinct temperatures", nun("T_C") > 1 if has("T_C") else False,
                           "Temperature dependence of ageing not identifiable."]
    return df


# ============================================================================================ synthetic demo
SYNTH_LAW = dict(   # synthetic, NOT literature values: known parameters so the methods can be checked
    kR=1.0e-5,       # ohm cm2 per operating hour at 80 C
    EaR=20000.0,     # J/mol
    k_act=5.0e-6,    # V per operating hour at 80 C
    Ea_act=30000.0,
    dR_start=2.0e-5,     # ohm cm2 per start-up
    deta_start=3.0e-4,   # V per start-up (reverse-current damage, mechanism from Qu26 / Ham25)
    k_x=2.0e-4,          # crossover growth per hour (relative)
    refill_recovery=0.3, # share of activation loss recovered at a KOH refill (like capacity recovery)
    noise_V=0.002,
)


def degradation_truth(op_hours, starts, T_K, law=SYNTH_LAW, T_ref=353.15) -> DegradationState:
    fR = math.exp(-law["EaR"] / R_GAS * (1 / T_K - 1 / T_ref))
    fA = math.exp(-law["Ea_act"] / R_GAS * (1 / T_K - 1 / T_ref))
    return DegradationState(dR_ohm_cm2=law["kR"] * fR * op_hours + law["dR_start"] * starts,
                            d_eta_act_V=law["k_act"] * fA * op_hours + law["deta_start"] * starts,
                            crossover_factor=1 + law["k_x"] * op_hours)


def renewable_profile(hours: int, seed: int = 0, kind: str = "solar+wind") -> np.ndarray:
    """Synthetic normalised power profile 0..1 (for demos; upload a real profile for a case study)."""
    rng = np.random.default_rng(seed)
    h = np.arange(hours)
    solar = np.clip(np.sin((h % 24 - 6) / 12 * np.pi), 0, None) * (0.6 + 0.4 * rng.random(hours // 24 + 1)[h // 24])
    wind = np.clip(0.35 + 0.25 * np.sin(h / 37.0) + 0.15 * rng.standard_normal(hours).cumsum() / np.sqrt(hours), 0, 1)
    p = {"solar": solar, "wind": wind}.get(kind, 0.6 * solar + 0.4 * wind)
    return np.clip(p, 0, 1)


def synthetic_dataset(seed: int = 0, hours: int = 3000, design: Optional[CellDesign] = None,
                      kin: Optional[Kinetics] = None, law=SYNTH_LAW) -> Dict[str, object]:
    """Six synthetic cells in three condition groups with KNOWN degradation parameters:
    constant 0.4 A/cm2 at 60 C / 1 bar, constant 0.4 A/cm2 at 80 C / 1 bar, renewable-following at 80 C / 5 bar
    with daily start-stops. Includes periodic U-i curves (every 500 h), EIS (every 250 h), HTO and events."""
    rng = np.random.default_rng(seed)
    design = design or CellDesign(name="synthetic", n_cells=1, area_cm2=25.0, gap_anode_mm=0.0, gap_cathode_mm=0.0)
    kin = kin or Kinetics()
    specs = [("S1", "constant", 60, 1.0), ("S2", "constant", 60, 1.0), ("S3", "constant", 80, 1.0),
             ("S4", "constant", 80, 1.0), ("S5", "dynamic", 80, 5.0), ("S6", "dynamic", 80, 5.0)]
    ts, pcs, eis, ev, truth = [], [], [], [], []
    for k, (u, prof, Tc, P) in enumerate(specs):
        if prof == "constant":
            j = np.full(hours, 0.4)
        else:
            p = renewable_profile(hours, seed + k)
            j = np.where(p > 0.15, 0.6 * p, 0.0)
        j = j * (1 + 0.01 * rng.standard_normal(hours)) * (j > 0)
        on = j > 0.02
        starts = np.cumsum(np.r_[1, (on[1:] & ~on[:-1]).astype(int)])
        op_h = np.cumsum(on.astype(float))
        refills = set(range(700 + 50 * k, hours, 700))
        rec = 0.0
        T = Tc + 273.15 + 0.3 * rng.standard_normal(hours)
        for h in range(hours):
            if h in refills:
                d0 = degradation_truth(op_h[h], starts[h], Tc + 273.15, law)
                rec = law["refill_recovery"] * (d0.d_eta_act_V - law["deta_start"] * starts[h])
                ev.append((u, float(h), "koh_refill"))
            if prof == "dynamic" and h > 0 and on[h] and not on[h - 1]:
                ev.append((u, float(h), "startup"))
            if prof == "dynamic" and h > 0 and not on[h] and on[h - 1]:
                ev.append((u, float(h), "shutdown"))
        # recovery: activation loss partly reset at each refill, then grows again
        rec_series = np.zeros(hours)
        last = 0.0
        for h in range(hours):
            if h in refills:
                d0 = degradation_truth(op_h[h], starts[h], Tc + 273.15, law)
                last = law["refill_recovery"] * (d0.d_eta_act_V - law["deta_start"] * starts[h])
            rec_series[h] = last
        op = OperatingConditions(T_C=Tc, P_bar=P, koh_wt=30.0)
        U = np.full(hours, np.nan)
        hto = np.full(hours, np.nan)
        dR_t, de_t = np.zeros(hours), np.zeros(hours)
        for h in range(hours):
            d = degradation_truth(op_h[h], starts[h], Tc + 273.15, law)
            d = replace(d, d_eta_act_V=d.d_eta_act_V - rec_series[h])
            dR_t[h], de_t[h] = d.dR_ohm_cm2, d.d_eta_act_V
            if on[h]:
                oph = replace(op, T_C=T[h] - 273.15, j_A_cm2=j[h])
                U[h] = cell_voltage(j[h], oph, design, kin, d)["U"][0] + law["noise_V"] * rng.standard_normal()
                hto[h] = hto_percent(j[h], oph, design, kin, d)["HTO_pct"][0] * (1 + 0.03 * rng.standard_normal())
        ts.append(pd.DataFrame(dict(unit=u, time_h=np.arange(hours, dtype=float), j_A_cm2=np.where(on, j, 0.0),
                                    current_A=np.where(on, j, 0.0) * design.area_cm2, voltage_V=U, U_cell_V=U,
                                    T_C=T - 273.15, P_bar=P, koh_wt=30.0, HTO_pct=hto)))
        ref = OperatingConditions(T_C=80, P_bar=1.0, j_A_cm2=0.4, koh_wt=30)
        truth.append(pd.DataFrame(dict(unit=u, time_h=np.arange(hours, dtype=float), op_hours=op_h, starts=starts,
                                       dR_ohm_cm2=dR_t, d_eta_V=de_t, dV_ref_mV=1000 * (0.4 * dR_t + de_t))))
        jg = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8])
        for h in range(0, hours, 500):
            d = DegradationState(dR_ohm_cm2=dR_t[h], d_eta_act_V=de_t[h])
            Uc = cell_voltage(jg, op, design, kin, d)["U"] + 0.001 * rng.standard_normal(len(jg))
            pcs.append(pd.DataFrame(dict(unit=u, time_h=float(h), j_A_cm2=jg, voltage_V=Uc, T_C=Tc, P_bar=P)))
        f = np.logspace(4, -1, 40)
        for h in range(0, hours, 250):
            r = cell_voltage(0.05, op, design, kin, DegradationState(dR_ohm_cm2=dR_t[h], d_eta_act_V=de_t[h]))
            Rs = r["R_ohm_cm2"][0] / design.area_cm2
            Rct = (r["eta_act"][0] / 0.05) / design.area_cm2 * 0.5
            Z = Rs + Rct / (1 + 1j * 2 * np.pi * f * 0.05)
            eis.append(pd.DataFrame(dict(unit=u, time_h=float(h), freq_Hz=f, z_re=Z.real, z_im_neg=-Z.imag)))
    data = dict(timeseries=pd.concat(ts, ignore_index=True), polcurve=pd.concat(pcs, ignore_index=True),
                eis=pd.concat(eis, ignore_index=True),
                events=pd.DataFrame(ev, columns=["unit", "time_h", "event"]).sort_values(["unit", "time_h"]))
    return dict(data=data, truth=pd.concat(truth, ignore_index=True), design=design, kin=kin, law=dict(law),
                ref=OperatingConditions(T_C=80, P_bar=1.0, j_A_cm2=0.4, koh_wt=30.0))


# ============================================================================================ operating windows
RECOVERY_EVENTS = ("koh_refill", "electrolyte_change", "stack_service")


def build_windows(data: Dict[str, pd.DataFrame], design: CellDesign, kin: Kinetics, ref: OperatingConditions,
                  window_h: float = 1.0, min_j: float = 0.02, stable_cv: float = 0.05, recovery_mV: float = 5.0,
                  use_T: bool = True, use_P: bool = True) -> pd.DataFrame:
    """Operating-window table (the electrolyser analogue of the battery cycle table).
    One row per unit and window: mean conditions, cumulative operating hours and start-ups, stable flag, voltage
    normalised to the reference conditions with the polarisation model, recovery flags."""
    ts = data["timeseries"]
    ev = data.get("events")
    rows = []
    for u, d in ts.groupby("unit", sort=True):
        d = d.sort_values("time_h")
        dt = float(np.nanmedian(np.diff(d["time_h"]))) if len(d) > 1 else window_h
        dt = dt if np.isfinite(dt) and dt > 0 else window_h
        jcol = d["j_A_cm2"] if "j_A_cm2" in d else pd.Series(np.nan, index=d.index)
        on = (jcol >= min_j) & d["U_cell_V"].notna()
        w = np.floor((d["time_h"] - d["time_h"].iloc[0]) / window_h).astype(int)
        g = d.assign(_on=on, _w=w, _dt=dt)
        agg = dict(t_start_h=("time_h", "min"), t_end_h=("time_h", "max"), frac_on=("_on", "mean"), n=("_on", "size"))
        for c in ("j_A_cm2", "U_cell_V", "T_C", "P_bar", "koh_wt", "HTO_pct", "OTH_pct", "h2_flow_Nm3_h"):
            if c in g:
                agg[c] = (c, "mean")
        a = g[g["_on"]].groupby("_w").agg(**{k: v for k, v in agg.items() if k not in ("frac_on", "n", "t_start_h", "t_end_h")})
        b = g.groupby("_w").agg(t_start_h=("time_h", "min"), t_end_h=("time_h", "max"), frac_on=("_on", "mean"))
        jstd = g[g["_on"]].groupby("_w")["j_A_cm2"].std() if "j_A_cm2" in g else None
        W = b.join(a, how="left")
        W["j_cv"] = (jstd / W["j_A_cm2"]).reindex(W.index).fillna(0.0) if jstd is not None else 0.0
        W["op_hours"] = (W["frac_on"] * (W["t_end_h"] - W["t_start_h"] + dt)).cumsum()
        # start-ups: events if given, otherwise off->on transitions
        if ev is not None and len(ev) and (ev["unit"] == u).any() and (ev.loc[ev["unit"] == u, "event"] == "startup").any():
            st_t = ev.loc[(ev["unit"] == u) & (ev["event"] == "startup"), "time_h"].to_numpy()
            W["starts"] = [int((st_t <= te).sum()) + 1 for te in W["t_end_h"]]
        else:
            tr = (on.astype(int).diff() == 1).astype(int)
            W["starts"] = (g.assign(_tr=tr).groupby("_w")["_tr"].sum().cumsum() + 1).to_numpy()
        rec_t = np.array([]) if ev is None or not len(ev) else ev.loc[(ev["unit"] == u) & ev["event"].isin(RECOVERY_EVENTS), "time_h"].to_numpy()
        W["h_since_recovery_event"] = [float(te - rec_t[rec_t <= te].max()) if (rec_t <= te).any() else np.inf for te in W["t_end_h"]]
        W.insert(0, "unit", u)
        rows.append(W.reset_index(drop=True))
    if not rows:
        return pd.DataFrame()
    W = pd.concat(rows, ignore_index=True)
    W = W[W["U_cell_V"].notna() & (W["frac_on"] > 0)].copy()
    for c, v in (("T_C", ref.T_C), ("P_bar", ref.P_bar), ("koh_wt", ref.koh_wt)):
        if c not in W or not use_T and c == "T_C" or not use_P and c == "P_bar":
            W[c] = v
        W[c] = W[c].fillna(v)
    W["stable"] = (W["frac_on"] >= 0.999) & (W["j_cv"] <= stable_cv)
    W["V_norm"] = W["U_cell_V"] - (_model_U(W, design, kin) - _model_U_ref(ref, design, kin))
    W["group"] = condition_groups(W)
    out = []
    for u, d in W.groupby("unit", sort=True):
        d = d.copy()
        base = d.loc[d["stable"], "V_norm"].head(24)
        bol = float(base.median()) if len(base) else float(d["V_norm"].head(24).median())
        d["dV_mV"] = 1000 * (d["V_norm"] - bol)
        # causal, sustained drop: median of the last 3 windows below the median of the 24 before them
        # (after a refill/service event, or by more than 3x the threshold anywhere, e.g. spalling resets)
        d3 = d["dV_mV"].rolling(3, min_periods=3).median()
        prev = d["dV_mV"].shift(3).rolling(24, min_periods=6).median()
        drop = (prev - d3).fillna(0.0)
        d["flag_recovery"] = ((drop > recovery_mV) & (d["h_since_recovery_event"] < 48)) | (drop > 3 * recovery_mV)
        U = d["U_cell_V"].to_numpy()
        d["kWh_per_kg"] = KWH_PER_KG_PER_VOLT * U
        d["thermal_eff"] = [float(V_thermoneutral(t + 273.15, p, molarity_from_wt(k, t + 273.15))) / x
                            for t, p, k, x in zip(d["T_C"], d["P_bar"], d["koh_wt"], U)]
        d["w"] = np.arange(len(d))
        out.append(d)
    return pd.concat(out, ignore_index=True)


def _model_U(W: pd.DataFrame, design, kin) -> np.ndarray:
    out = np.empty(len(W))
    for i, (j, T, P, k) in enumerate(zip(W["j_A_cm2"], W["T_C"], W["P_bar"], W["koh_wt"])):
        out[i] = cell_voltage(j, OperatingConditions(T_C=T, P_bar=P, j_A_cm2=j, koh_wt=k), design, kin)["U"][0]
    return out


def _model_U_ref(ref: OperatingConditions, design, kin) -> float:
    return float(cell_voltage(ref.j_A_cm2, ref, design, kin)["U"][0])


def condition_groups(W: pd.DataFrame) -> pd.Series:
    """Label each unit by its operating condition: profile (constant/dynamic), mean T and p."""
    lab = {}
    for u, d in W.groupby("unit"):
        cv = float(d["j_A_cm2"].std() / max(d["j_A_cm2"].mean(), 1e-9))
        prof = "constant" if cv < 0.1 else "dynamic"
        lab[u] = f"{prof} {d['T_C'].mean():.0f}C {d['P_bar'].mean():.0f}bar"
    return W["unit"].map(lab)


# ============================================================================================ health indicators
def eis_hf_intercept(eis: pd.DataFrame, area_cm2: float) -> pd.DataFrame:
    """Ohmic resistance from each spectrum: Z' where -Z'' crosses zero at high frequency (else Z' at the highest
    frequency). Returned both in ohm and area-specific ohm cm2."""
    rows = []
    for (u, t), d in eis.groupby(["unit", "time_h"]):
        d = d.sort_values("freq_Hz", ascending=False)
        zi, zr = d["z_im_neg"].to_numpy(), d["z_re"].to_numpy()
        k = np.nonzero(np.sign(zi[:-1]) != np.sign(zi[1:]))[0]
        if len(k):
            i = k[0]
            r = zr[i] - zi[i] * (zr[i + 1] - zr[i]) / (zi[i + 1] - zi[i])
        else:
            r = zr[0]
        rp = float(zr.max() - r)
        rows.append(dict(unit=u, time_h=t, R_hf_ohm=float(r), R_ohm_cm2=float(r) * area_cm2, R_pol_ohm_cm2=rp * area_cm2))
    return pd.DataFrame(rows)


def fit_polcurve(pc: pd.DataFrame, design: CellDesign, kin: Kinetics, koh_wt: float = 30.0) -> pd.DataFrame:
    """Per curve: U = E_rev(T,p) + b ln(j/j0) + R j  (lumped Tafel + ohmic, bubbles neglected) [Hu22 eq 18-19],
    the same form as the empirical Hug model U = E_rev + A1 i + A2 log i [Daoudi24 eq 35]."""
    rows = []
    for (u, t), d in pc.groupby(["unit", "time_h"]):
        d = d[(d["j_A_cm2"] > 0) & d["voltage_V"].notna()].sort_values("j_A_cm2")
        if len(d) < 4:
            continue
        T = float(d["T_C"].mean()) + 273.15 if "T_C" in d else 343.15
        P = float(d["P_bar"].mean()) if "P_bar" in d else 1.0
        er = float(E_rev(T, P, molarity_from_wt(koh_wt, T)))
        j, U = d["j_A_cm2"].to_numpy(), d["voltage_V"].to_numpy()
        f = lambda x: er + x[0] * np.log(j / x[1]) + x[2] * j - U
        try:
            res = optimize.least_squares(f, [0.06, 1e-4, 0.2], bounds=([1e-3, 1e-12, 0], [0.5, 1.0, 10]), x_scale=[0.05, 1e-4, 0.1])
            b, j0, Rr = res.x
            rows.append(dict(unit=u, time_h=t, T_C=T - 273.15, P_bar=P, tafel_b_V=b, tafel_slope_mV_dec=b * np.log(10) * 1000,
                             j0_A_cm2=j0, R_ohm_cm2=Rr, rmse_mV=1000 * float(np.sqrt(np.mean(res.fun ** 2)))))
        except Exception:                                  # noqa: BLE001 - keep the curve as a failed row
            rows.append(dict(unit=u, time_h=t, rmse_mV=np.nan))
    return pd.DataFrame(rows)


def kinetics_from_fit(fit: pd.DataFrame) -> Kinetics:
    """Kinetics object in Tafel-fit mode from the earliest fitted curves (e.g. stainless steel)."""
    f = fit.dropna(subset=["tafel_b_V"]).sort_values("time_h")
    first = f.groupby("unit").head(1)
    return Kinetics(mode="tafel_fit", tafel_b=float(first["tafel_b_V"].median()), tafel_j0=float(first["j0_A_cm2"].median()),
                    use_bubbles=False)


def attach_indicators(W: pd.DataFrame, data: Dict[str, pd.DataFrame], design: CellDesign, kin: Kinetics) -> pd.DataFrame:
    """Add EIS ohmic resistance and U-i fitted parameters to the windows (causal: last measurement at or before)."""
    W = W.copy()
    extra = []
    if data.get("eis") is not None and len(data["eis"]):
        e = eis_hf_intercept(data["eis"], design.area_cm2)[["unit", "time_h", "R_ohm_cm2"]].rename(columns={"R_ohm_cm2": "R_EIS_ohm_cm2"})
        extra.append(e)
    if data.get("polcurve") is not None and len(data["polcurve"]):
        p = fit_polcurve(data["polcurve"], design, kin)
        if len(p) and "tafel_b_V" in p:
            extra.append(p[["unit", "time_h", "tafel_slope_mV_dec", "j0_A_cm2", "R_ohm_cm2"]].rename(columns={"R_ohm_cm2": "R_UI_ohm_cm2"}))
    for x in extra:
        out = []
        for u, d in W.groupby("unit", sort=False):
            xu = x[x["unit"] == u].drop(columns="unit").sort_values("time_h")
            if not len(xu):
                out.append(d)
                continue
            m = pd.merge_asof(d.sort_values("t_end_h"), xu.rename(columns={"time_h": "_tm"}), left_on="t_end_h",
                              right_on="_tm", direction="backward").drop(columns="_tm")
            out.append(m)
        W = pd.concat(out, ignore_index=True)
    return W


HEALTH_INDICATORS = {
    "dV_mV": "Voltage rise at reference conditions (mV)",
    "R_EIS_ohm_cm2": "Ohmic resistance, EIS high-frequency intercept (ohm cm2)",
    "R_UI_ohm_cm2": "Ohmic resistance, slope of U-i fit (ohm cm2)",
    "tafel_slope_mV_dec": "Tafel slope from U-i fit (mV/dec)",
    "j0_A_cm2": "Apparent exchange current density from U-i fit (A/cm2)",
    "kWh_per_kg": "Specific energy (kWh/kg H2, eta_F=1 unless H2 flow given)",
    "thermal_eff": "Thermal (HHV) efficiency V_tn/U",
    "HTO_pct": "H2 in O2 (%)",
}


def rank_indicators(W: pd.DataFrame) -> pd.DataFrame:
    """Rank health indicators per unit: trend strength (|Spearman| with operating hours), monotonicity
    (share of same-sign 24-window differences) and coverage; median over units."""
    rows = []
    for hi, label in HEALTH_INDICATORS.items():
        if hi not in W or W[hi].notna().sum() < 5:
            rows.append(dict(indicator=label, key=hi, available=False))
            continue
        rs, mono, cov = [], [], []
        for u, d in W.groupby("unit"):
            x = d[["op_hours", hi]].dropna().drop_duplicates(subset=[hi])
            cov.append(d[hi].notna().mean())
            if len(x) < 3:
                continue
            rs.append(abs(spearmanr(x["op_hours"], x[hi]).statistic))
            s = np.sign(np.diff(x[hi].rolling(max(len(x) // 20, 1)).mean().dropna().to_numpy()[::max(len(x) // 20, 1)]))
            mono.append(abs(np.mean(s)) if len(s) else np.nan)
        rows.append(dict(indicator=label, key=hi, available=True, trend=np.nanmedian(rs) if rs else np.nan,
                         monotonicity=np.nanmedian(mono) if mono else np.nan, coverage=float(np.mean(cov)),
                         units=len(rs)))
    df = pd.DataFrame(rows)
    if "trend" in df:
        df["score"] = df[["trend", "monotonicity"]].mean(axis=1) * df["coverage"].fillna(0).clip(upper=1) ** 0.25
        df = df.sort_values("score", ascending=False, na_position="last")
    return df.reset_index(drop=True)


# ============================================================================================ degradation model v1
def usable(W: pd.DataFrame) -> pd.DataFrame:
    """Windows used for LABELS of training units: stable, not flagged as recovery."""
    return W[W["stable"] & ~W["flag_recovery"]]


def degradation_v1(W: pd.DataFrame, T_ref_C: float = 80.0) -> Dict[str, object]:
    """(a) per-unit robust drift (Theil-Sen, uV/h) of dV vs operating hours; (b) pooled law
    dV = a * h * exp(-Ea/R (1/T - 1/Tref)) + b * (starts - 1) + c_unit, fitted when conditions vary.
    Ea and b are reported as 'not identifiable' when T or start counts do not vary between units."""
    per = []
    for u, d in usable(W).groupby("unit"):
        if len(d) < 10:
            continue
        s, i, lo, hi = theilslopes(d["dV_mV"], d["op_hours"])
        per.append(dict(unit=u, group=d["group"].iloc[0], T_C=d["T_C"].mean(), j=d["j_A_cm2"].mean(),
                        starts=int(d["starts"].max()), hours=float(d["op_hours"].max()),
                        drift_uV_h=1000 * s, ci_lo=1000 * lo, ci_hi=1000 * hi))
    per = pd.DataFrame(per)
    U = usable(W)
    notes = []
    T_var = U.groupby("unit")["T_C"].mean().round(0).nunique() > 1
    s_var = U.groupby("unit")["starts"].max().nunique() > 1
    units = sorted(U["unit"].unique())
    idx = {u: k for k, u in enumerate(units)}
    h, T, st = U["op_hours"].to_numpy(), U["T_C"].to_numpy() + 273.15, (U["starts"] - 1).to_numpy()
    y, ui = U["dV_mV"].to_numpy(), U["unit"].map(idx).to_numpy()

    def model(p):
        a, Ea, b = p[0], (p[1] if T_var else 0.0), (p[2] if s_var else 0.0)
        return a * h * np.exp(-Ea / R_GAS * (1 / T - 1 / (T_ref_C + 273.15))) + b * st + p[3:][ui]

    p0 = [0.005, 20000.0, 0.1] + [0.0] * len(units)
    law = dict(a_mV_h=np.nan, Ea_J_mol=np.nan, b_mV_start=np.nan)
    if len(U) > 20:
        res = optimize.least_squares(lambda p: model(p) - y, p0, x_scale=[0.005, 1e4, 0.1] + [1.0] * len(units),
                                     bounds=([0, 0, 0] + [-50] * len(units), [1, 2e5, 50] + [50] * len(units)))
        law = dict(a_mV_h=res.x[0], Ea_J_mol=res.x[1] if T_var else np.nan, b_mV_start=res.x[2] if s_var else np.nan,
                   rmse_mV=float(np.sqrt(np.mean(res.fun ** 2))))
    if not T_var:
        notes.append("Temperature does not vary between units: activation energy of ageing not identifiable.")
    if not s_var:
        notes.append("Start-up counts do not vary: damage per start-up not identifiable.")
    if len(units) < 3:
        notes.append(f"Only {len(units)} unit(s): the pooled law is descriptive, not a validated model.")
    return dict(per_unit=per, law=law, notes=notes, T_ref_C=T_ref_C)


# ============================================================================================ forecasting
from sklearn.base import clone                                     # noqa: E402
from sklearn.ensemble import RandomForestRegressor                 # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor      # noqa: E402
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, WhiteKernel  # noqa: E402
from sklearn.isotonic import IsotonicRegression                    # noqa: E402
from sklearn.linear_model import BayesianRidge                     # noqa: E402
from sklearn.neural_network import MLPRegressor                    # noqa: E402
from sklearn.pipeline import make_pipeline                         # noqa: E402
from sklearn.preprocessing import StandardScaler                   # noqa: E402
from sklearn.svm import SVR                                        # noqa: E402

try:
    import xgboost  # noqa: F401
    HAS_XGB = True
except Exception:   # noqa: BLE001
    HAS_XGB = False
try:
    import lightgbm  # noqa: F401
    HAS_LGBM = True
except Exception:   # noqa: BLE001
    HAS_LGBM = False

FEATURES = ["op_hours", "dh", "d_starts", "j_A_cm2", "T_C", "P_bar", "dV_now", "slope_now"]
COND_COLS = ["op_hours", "starts", "j_A_cm2", "T_C", "P_bar", "group", "stable", "t_end_h"]

MODEL_SPECS = {   # name: (level, default params, param grid for tuning {name: (kind, low, high)})
    "Persistence": ("L1 baseline", {}, {}),
    "Linear drift": ("L1 baseline", {"window_h": 300}, {"window_h": ("int", 50, 1500)}),
    "Fleet curve": ("L1 baseline", {"shrink": 0.7}, {"shrink": ("float", 0.0, 1.0)}),
    "Bayesian ridge": ("L2 ML", {}, {}),
    "Random forest": ("L2 ML", {"n_estimators": 200, "min_samples_leaf": 3},
                      {"n_estimators": ("int", 50, 500), "min_samples_leaf": ("int", 1, 20)}),
    "SVM": ("L2 ML", {"C": 10.0, "epsilon": 0.2, "gamma": 0.1},
            {"C": ("log", 0.1, 1000), "epsilon": ("log", 0.01, 2.0), "gamma": ("log", 0.001, 1.0)}),
    "Gaussian process": ("L2 ML", {"max_rows": 300}, {}),
    "MLP": ("L3 DL", {"hidden": 64, "alpha": 1e-3}, {"hidden": ("int", 8, 128), "alpha": ("log", 1e-5, 1e-1)}),
}
if HAS_XGB:
    MODEL_SPECS["XGBoost"] = ("L2 ML", {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 4},
                              {"max_depth": ("int", 2, 8), "learning_rate": ("log", 0.01, 0.3)})
if HAS_LGBM:
    MODEL_SPECS["LightGBM"] = ("L2 ML", {"n_estimators": 400, "num_leaves": 15},
                               {"num_leaves": ("int", 4, 63)})
PLANNED_MODELS = {"GRU": "L3 DL (needs PyTorch; next step)", "Transformer": "L3 DL (needs PyTorch; next step)",
                  "PINN (+ ablation)": "L4 physics-informed (next step)"}


def origin_index(d: pd.DataFrame, share: float) -> int:
    """Index (position) of the last window at or before `share` of the unit's operating hours."""
    h = d["op_hours"].to_numpy()
    return int(np.searchsorted(h, share * h.max(), side="right") - 1)


def state_at(past: pd.DataFrame, window_h: float = 300.0) -> Tuple[float, float]:
    """Causal health at the origin: median dV of the last 24 usable windows and Theil-Sen slope (mV/h) over the
    last `window_h` operating hours. Uses ONLY the past."""
    u = usable(past)
    if not len(u):
        u = past
    dv = float(u["dV_mV"].tail(24).median())
    r = u[u["op_hours"] >= u["op_hours"].max() - window_h]
    slope = float(theilslopes(r["dV_mV"], r["op_hours"])[0]) if len(r) >= 8 and r["op_hours"].nunique() > 2 else 0.0
    return dv, slope


def _rows(d: pd.DataFrame, origins: Sequence[int], cut: Optional[int] = None, per_origin: int = 40,
          window_h: float = 300.0) -> pd.DataFrame:
    """Training rows from one unit: for each origin k, targets m > k (labels only up to `cut`)."""
    out = []
    lab = usable(d) if cut is None else usable(d.iloc[:cut + 1])
    for k in origins:
        if k < 10:
            continue
        past = d.iloc[:k + 1]
        dv, sl = state_at(past, window_h)
        fut = lab[lab["op_hours"] > past["op_hours"].iloc[-1]]
        if not len(fut):
            continue
        fut = fut.iloc[np.linspace(0, len(fut) - 1, min(per_origin, len(fut))).astype(int)]
        out.append(pd.DataFrame(dict(op_hours=fut["op_hours"].to_numpy(), dh=fut["op_hours"].to_numpy() - past["op_hours"].iloc[-1],
                                     d_starts=fut["starts"].to_numpy() - past["starts"].iloc[-1],
                                     j_A_cm2=fut["j_A_cm2"].to_numpy(), T_C=fut["T_C"].to_numpy(), P_bar=fut["P_bar"].to_numpy(),
                                     dV_now=dv, slope_now=sl, y=fut["dV_mV"].to_numpy() - dv)))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=FEATURES + ["y"])


_ROWS_CACHE: Dict[tuple, pd.DataFrame] = {}   # training rows of a full training history (fingerprint key)


def training_table(train: Dict[str, pd.DataFrame], past: pd.DataFrame, window_h: float = 300.0) -> pd.DataFrame:
    """Rows from the training units' full histories plus the predicted unit's own past (labels <= origin)."""
    parts = []
    for u, d in train.items():
        key = (str(u), len(d), round(float(d["dV_mV"].sum()), 6), round(float(d["op_hours"].sum()), 3), window_h)
        if key not in _ROWS_CACHE:
            d = d.reset_index(drop=True)
            _ROWS_CACHE[key] = _rows(d, [origin_index(d, s) for s in np.arange(0.1, 0.85, 0.1)], window_h=window_h)
            if len(_ROWS_CACHE) > 500:
                _ROWS_CACHE.pop(next(iter(_ROWS_CACHE)))
        parts.append(_ROWS_CACHE[key])
    p = past.reset_index(drop=True)
    if len(p) > 40:
        parts.append(_rows(p, [origin_index(p, s) for s in (0.3, 0.45, 0.6, 0.75)], cut=len(p) - 1, window_h=window_h))
    parts = [x for x in parts if len(x)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=FEATURES + ["y"])


def _estimator(name: str, params: dict):
    p = dict(MODEL_SPECS[name][1], **(params or {}))
    if name == "Bayesian ridge":
        return make_pipeline(StandardScaler(), BayesianRidge())
    if name == "Random forest":
        return RandomForestRegressor(n_estimators=int(p["n_estimators"]), min_samples_leaf=int(p["min_samples_leaf"]),
                                     random_state=0, n_jobs=-1)
    if name == "SVM":
        return make_pipeline(StandardScaler(), SVR(C=p["C"], epsilon=p["epsilon"], gamma=p["gamma"]))
    if name == "Gaussian process":
        k = ConstantKernel(1.0) * RBF(length_scale=np.ones(len(FEATURES))) + ConstantKernel(0.1) * DotProduct() + WhiteKernel(0.1)
        return make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=k, normalize_y=True, random_state=0))
    if name == "MLP":
        h = int(p["hidden"])
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(h, h), alpha=p["alpha"], max_iter=2000,
                                                            early_stopping=True, random_state=0))
    if name == "XGBoost":
        import xgboost as xgb
        return xgb.XGBRegressor(n_estimators=int(p["n_estimators"]), learning_rate=p["learning_rate"],
                                max_depth=int(p["max_depth"]), random_state=0)
    if name == "LightGBM":
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=int(p["n_estimators"]), num_leaves=int(p["num_leaves"]), verbose=-1, random_state=0)
    raise KeyError(name)


def _fleet_curve(train: Dict[str, pd.DataFrame]) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    curves = []
    hmax = max(float(d["op_hours"].max()) for d in train.values())
    grid = np.linspace(0, hmax, 300)
    late = []
    for d in train.values():
        u = usable(d)
        if len(u) < 10:
            continue
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(u["op_hours"], u["dV_mV"])
        c = np.full(len(grid), np.nan)
        ok = grid <= u["op_hours"].max()
        c[ok] = iso.predict(grid[ok])
        curves.append(c)
        q = u[u["op_hours"] >= 0.75 * u["op_hours"].max()]
        if len(q) > 5:
            late.append(theilslopes(q["dV_mV"], q["op_hours"])[0])
    if len(curves) < 2:
        return None
    M = np.array(curves)
    M = np.where(np.isfinite(M).sum(0) >= 2, M, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(M, axis=0)
    ok = np.isfinite(med)
    return grid[ok], np.maximum.accumulate(med[ok]), float(np.median(late)) if late else 0.0


def forecast(name: str, train: Dict[str, pd.DataFrame], past: pd.DataFrame, future: pd.DataFrame,
             params: Optional[dict] = None, window_h: float = 300.0) -> np.ndarray:
    """Forecast dV (mV) for the future windows. `future` holds only planned conditions (no dV), `past` the unit up to
    the origin, `train` other units' full histories. Every forecast is anchored at today's (causal) value."""
    assert "dV_mV" not in future.columns, "future must not contain the answer"
    params = dict(MODEL_SPECS[name][1], **(params or {}))
    dv, sl = state_at(past, params.get("window_h", window_h))
    h0, s0 = float(past["op_hours"].iloc[-1]), float(past["starts"].iloc[-1])
    dh = future["op_hours"].to_numpy() - h0
    if name == "Persistence":
        return np.full(len(future), dv)
    if name == "Linear drift":
        return dv + max(sl, 0.0) * dh
    if name == "Fleet curve":
        fc = _fleet_curve(train) if train else None
        if fc is None:
            return dv + max(sl, 0.0) * dh            # falls back to the drift (needs >= 2 training units)
        g, c, late = fc
        a_star = float(np.interp(dv, c, g)) if dv <= c[-1] else g[-1]
        sf = float(np.interp(a_star, g[1:], np.diff(c) / np.diff(g))) if len(g) > 2 else late
        k = float(np.clip((max(sl, 1e-6) / max(sf, 1e-6)) ** params["shrink"], 0.4, 2.5)) if sf > 0 else 1.0
        a = a_star + k * dh

        def Fc(x):
            return np.where(x <= g[-1], np.interp(x, g, c), c[-1] + max(late, 0) * (x - g[-1]))
        return dv + Fc(a) - Fc(np.array([a_star]))
    tab = training_table(train, past, window_h)
    if len(tab) < 10:
        return dv + max(sl, 0.0) * dh
    if name == "Gaussian process" and len(tab) > params["max_rows"]:
        tab = tab.sample(int(params["max_rows"]), random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = _estimator(name, params).fit(tab[FEATURES].to_numpy(float), tab["y"].to_numpy(float))
    X = pd.DataFrame(dict(op_hours=future["op_hours"].to_numpy(), dh=dh, d_starts=future["starts"].to_numpy() - s0,
                          j_A_cm2=future["j_A_cm2"].to_numpy(), T_C=future["T_C"].to_numpy(),
                          P_bar=future["P_bar"].to_numpy(), dV_now=dv, slope_now=sl))
    X0 = X.iloc[:1].copy()
    X0[["op_hours", "dh", "d_starts"]] = [h0, 0.0, 0.0]
    X0[["j_A_cm2", "T_C", "P_bar"]] = past[["j_A_cm2", "T_C", "P_bar"]].tail(24).median().to_numpy()
    f = est.predict(X[FEATURES].to_numpy(float))
    f0 = est.predict(X0[FEATURES].to_numpy(float))[0]
    return dv + f - f0


def eol_time(h: np.ndarray, dv: np.ndarray, thr: float, consecutive: int = 3) -> float:
    """First operating hour where dV >= thr for `consecutive` windows (truth) - use consecutive=1 for forecasts."""
    above = np.asarray(dv) >= thr
    if consecutive > 1:
        run = np.convolve(above.astype(int), np.ones(consecutive, int), "valid") >= consecutive
        idx = np.nonzero(run)[0]
    else:
        idx = np.nonzero(above)[0]
    return float(h[idx[0]]) if len(idx) else float("nan")


def inner_errors(name: str, train: Dict[str, pd.DataFrame], past: pd.DataFrame, params=None,
                 shares=(0.3, 0.6)) -> pd.DataFrame:
    """Held-out errors used for bands, ensemble weights and tuning, computed WITHOUT the predicted unit's future:
    each training unit forecast from the others (2 folds), or - single unit - the unit's own past split at earlier
    origins."""
    rows = []
    units = list(train)
    if len(units) >= 2:
        folds = [units[0::2], units[1::2]]
        for test_units in folds:
            tr = {u: train[u] for u in units if u not in test_units}
            for u in test_units:
                d = train[u].reset_index(drop=True)
                for s in shares:
                    k = origin_index(d, s)
                    if k < 10:
                        continue
                    fut = d.iloc[k + 1:]
                    fut = fut[fut["stable"]]
                    if not len(fut):
                        continue
                    pr = forecast(name, tr, d.iloc[:k + 1], fut.drop(columns=["dV_mV"]), params)
                    rows.append(pd.DataFrame(dict(dh=fut["op_hours"] - d["op_hours"].iloc[k], err=fut["dV_mV"].to_numpy() - pr)))
    else:
        d = past.reset_index(drop=True)
        for s in (0.4, 0.6, 0.75):
            k = origin_index(d, s)
            if k < 10:
                continue
            fut = d.iloc[k + 1:]
            fut = fut[fut["stable"]]
            if not len(fut):
                continue
            pr = forecast(name, {}, d.iloc[:k + 1], fut.drop(columns=["dV_mV"]), params)
            rows.append(pd.DataFrame(dict(dh=fut["op_hours"] - d["op_hours"].iloc[k], err=fut["dV_mV"].to_numpy() - pr)))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["dh", "err"])


def band_from_errors(err: pd.DataFrame, dh: np.ndarray, level: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    """Lower/upper offsets per horizon from held-out error quantiles (binned by horizon, interpolated)."""
    if err is None or len(err) < 10:
        return np.full(len(dh), np.nan), np.full(len(dh), np.nan)
    q = (1 - level) / 2
    e = err.copy()
    e["bin"] = pd.qcut(e["dh"], q=min(8, max(len(e) // 20, 2)), duplicates="drop")
    g = e.groupby("bin", observed=True).agg(c=("dh", "median"), lo=("err", lambda x: x.quantile(q)),
                                            hi=("err", lambda x: x.quantile(1 - q)))
    lo = np.interp(dh, g["c"], g["lo"])
    hi = np.interp(dh, g["c"], g["hi"])
    hmax = float(g["c"].max())
    scale = np.where(dh > hmax, dh / max(hmax, 1e-9), 1.0)      # widen proportionally beyond the seen horizons
    return lo * scale, hi * scale


def split_units(W: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {u: d.reset_index(drop=True) for u, d in W.groupby("unit", sort=True)}


def run_task(W: pd.DataFrame, unit: str, share: float, name: str, mode: str = "units", params=None,
             thr_mV: float = 50.0, level: float = 0.9, band_err: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One (held-out unit, origin share, model) forecast. mode 'units': train on the other units' full histories;
    'periods': train on nothing but the unit's own past (held-out period). Returns one row per future window."""
    units = split_units(W)
    d = units[unit]
    k = origin_index(d, share)
    if k < 10 or k >= len(d) - 2:
        return pd.DataFrame()
    train = {u: x for u, x in units.items() if u != unit} if mode == "units" else {}
    past, fut = d.iloc[:k + 1], d.iloc[k + 1:]
    fut = fut[fut["stable"]]
    if not len(fut):
        return pd.DataFrame()
    pred = forecast(name, train, past, fut.drop(columns=["dV_mV"]), params)
    if band_err is None:
        band_err = inner_errors(name, train, past, params, shares=(0.5,))
    dh = fut["op_hours"].to_numpy() - past["op_hours"].iloc[-1]
    lo, hi = band_from_errors(band_err, dh, level)
    return pd.DataFrame(dict(model=name, level=MODEL_SPECS[name][0], unit=unit, group=d["group"].iloc[0], share=share,
                             origin_h=float(past["op_hours"].iloc[-1]), op_hours=fut["op_hours"].to_numpy(), dh=dh,
                             truth=fut["dV_mV"].to_numpy(), pred=pred, lo=pred + lo, hi=pred + hi))


def score_tasks(res: pd.DataFrame, thr_mV: float) -> pd.DataFrame:
    """Per (model, unit, share): RMSE, MAE, bias, band coverage, end-of-life error."""
    rows = []
    for (m, u, s), d in res.groupby(["model", "unit", "share"]):
        e = d["truth"] - d["pred"]
        cov = float(((d["truth"] >= d["lo"]) & (d["truth"] <= d["hi"])).mean()) if d["lo"].notna().any() else np.nan
        te = eol_time(d["op_hours"].to_numpy(), d["truth"].to_numpy(), thr_mV, 3)
        pe = eol_time(d["op_hours"].to_numpy(), d["pred"].to_numpy(), thr_mV, 1)
        rows.append(dict(model=m, level=d["level"].iloc[0], unit=u, group=d["group"].iloc[0], share=s,
                         rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))), bias=float(np.mean(e)),
                         coverage=cov, eol_true_h=te, eol_pred_h=pe, eol_err_h=pe - te))
    return pd.DataFrame(rows)


def scoreboard(scores: pd.DataFrame, baselines=("Persistence", "Linear drift", "Fleet curve")) -> Tuple[pd.DataFrame, str]:
    """Per-unit averaging, 95 % CI (Student t over units), paired Wilcoxon vs the best baseline; honest wording."""
    pu = scores.groupby(["model", "level", "unit"], as_index=False)[["rmse", "coverage"]].mean()
    rows = []
    for (m, lv), d in pu.groupby(["model", "level"]):
        n = len(d)
        mu, sd = d["rmse"].mean(), d["rmse"].std(ddof=1) if n > 1 else np.nan
        h = t_dist.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
        rows.append(dict(model=m, level=lv, units=n, rmse_mean=mu, ci95=h, coverage=d["coverage"].mean()))
    sb = pd.DataFrame(rows).sort_values("rmse_mean").reset_index(drop=True)
    base = sb[sb["model"].isin(baselines)]
    note = ""
    if len(base):
        best = base.iloc[0]["model"]
        bv = pu[pu["model"] == best].set_index("unit")["rmse"]
        verdict = []
        for m in sb["model"]:
            if m == best:
                verdict.append("best baseline")
                continue
            mv = pu[pu["model"] == m].set_index("unit")["rmse"].reindex(bv.index)
            ok = mv.notna() & bv.notna()
            nn = int(ok.sum())
            better = int((mv[ok] < bv[ok]).sum())
            word = "better" if better * 2 > nn else "worse" if better * 2 < nn else "tied"
            if nn >= 6:
                p = wilcoxon(mv[ok], bv[ok]).pvalue
                verdict.append(f"{word}: better on {better}/{nn} units, p={p:.3f}" + ("" if p < 0.05 else " (not significant)"))
            else:
                verdict.append(f"better on {better}/{nn} units (too few for a test)")
        sb["vs best baseline"] = verdict
        note = (f"Best baseline: {best}. " + ("With fewer than 6 units no significance test is meaningful; read the "
                                              "per-unit counts." if sb["units"].max() < 6 else ""))
    return sb, note


def group_table(scores: pd.DataFrame) -> pd.DataFrame:
    return scores.groupby(["group", "model"])["rmse"].mean().unstack("model").round(2)


def influence(W: pd.DataFrame, unit: str, share: float, name: str, params=None) -> pd.DataFrame:
    """Which training units perturb a model: RMSE on the held-out unit with each training unit removed in turn.
    Data are kept as they are; positive delta = removing that unit made the forecast worse (it helped)."""
    units = split_units(W)
    d = units[unit]
    k = origin_index(d, share)
    past, fut = d.iloc[:k + 1], d.iloc[k + 1:]
    fut = fut[fut["stable"]]
    full = {u: x for u, x in units.items() if u != unit}
    base = forecast(name, full, past, fut.drop(columns=["dV_mV"]), params)
    r0 = float(np.sqrt(np.mean((fut["dV_mV"].to_numpy() - base) ** 2)))
    rows = []
    for u in full:
        tr = {x: v for x, v in full.items() if x != u}
        p = forecast(name, tr, past, fut.drop(columns=["dV_mV"]), params)
        r = float(np.sqrt(np.mean((fut["dV_mV"].to_numpy() - p) ** 2)))
        rows.append(dict(removed_unit=u, group=full[u]["group"].iloc[0], rmse=r, delta=r - r0,
                         recoveries_flagged=int(full[u]["flag_recovery"].sum())))
    return pd.DataFrame(rows).sort_values("delta").reset_index(drop=True)


def tune(name: str, W: pd.DataFrame, exclude_unit: Optional[str] = None, n_iter: int = 10, seed: int = 0,
         mode: str = "units") -> Dict[str, object]:
    """Bayesian optimisation (GP surrogate + expected improvement) of a model's parameters on HELD-OUT data only:
    the objective is the inner held-out RMSE on the training units (the excluded/test unit never enters)."""
    grid = MODEL_SPECS[name][2]
    if not grid:
        return dict(best=dict(MODEL_SPECS[name][1]), history=pd.DataFrame(), note="no tunable parameters")
    units = split_units(W)
    if exclude_unit is not None:
        units.pop(exclude_unit, None)
    rng = np.random.default_rng(seed)
    keys = list(grid)

    def to_params(z):
        p = {}
        for k, zi in zip(keys, z):
            kind, lo, hi = grid[k]
            v = math.exp(math.log(lo) + zi * (math.log(hi) - math.log(lo))) if kind == "log" else lo + zi * (hi - lo)
            p[k] = int(round(v)) if kind == "int" else float(v)
        return p

    def objective(p):
        if mode == "units" and len(units) >= 2:
            e = inner_errors(name, units, next(iter(units.values())).iloc[:0], p, shares=(0.5,))
        else:
            d = next(iter(units.values()))
            e = inner_errors(name, {}, d, p)
        return float(np.sqrt(np.mean(e["err"] ** 2))) if len(e) else np.inf

    Z, Y, hist = [], [], []
    for it in range(n_iter):
        if it < 4:
            z = rng.random(len(keys))
        else:
            gp = GaussianProcessRegressor(ConstantKernel() * RBF(np.ones(len(keys))) + WhiteKernel(1e-3), normalize_y=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(np.array(Z), np.array(Y))
            cand = rng.random((256, len(keys)))
            mu, sd = gp.predict(cand, return_std=True)
            best = min(Y)
            imp = best - mu
            zz = imp / np.maximum(sd, 1e-9)
            from scipy.stats import norm
            ei = imp * norm.cdf(zz) + sd * norm.pdf(zz)
            z = cand[int(np.argmax(ei))]
        p = to_params(z)
        y = objective(p)
        Z.append(z)
        Y.append(y if np.isfinite(y) else 1e6)
        hist.append(dict(it=it, rmse=y, **p))
    h = pd.DataFrame(hist)
    b = h.loc[h["rmse"].idxmin()]
    return dict(best={k: (int(b[k]) if grid[k][0] == "int" else float(b[k])) for k in keys}, history=h,
                note="objective = held-out RMSE on training units (inner folds)")


def ensemble(res: pd.DataFrame, inner_rmse: Dict[str, float]) -> pd.DataFrame:
    """Self-weighted ensemble: weights proportional to 1/RMSE^2 of held-out inner errors; band by moment matching."""
    w = {m: 1 / max(r, 1e-6) ** 2 for m, r in inner_rmse.items() if np.isfinite(r) and m in set(res["model"])}
    if not w:
        return pd.DataFrame()
    s = sum(w.values())
    out = []
    for (u, sh), d in res[res["model"].isin(w)].groupby(["unit", "share"]):
        p = d.pivot_table(index="op_hours", columns="model", values="pred")
        lo = d.pivot_table(index="op_hours", columns="model", values="lo")
        hi = d.pivot_table(index="op_hours", columns="model", values="hi")
        ww = np.array([w[m] / s for m in p.columns])
        mu = p.to_numpy() @ ww
        var = ((((hi - lo) / 3.29) ** 2).to_numpy() + (p.to_numpy() - mu[:, None]) ** 2) @ ww
        first = d.drop_duplicates("op_hours").set_index("op_hours").loc[p.index]
        out.append(pd.DataFrame(dict(model="Ensemble", level="Ensemble", unit=u, group=first["group"].to_numpy(), share=sh,
                                     origin_h=first["origin_h"].to_numpy(), op_hours=p.index.to_numpy(),
                                     dh=first["dh"].to_numpy(), truth=first["truth"].to_numpy(), pred=mu,
                                     lo=mu - 1.645 * np.sqrt(var), hi=mu + 1.645 * np.sqrt(var))))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def no_leakage_check(W: pd.DataFrame, unit: str, share: float, name: str) -> bool:
    """Forecast twice: original data and with the held-out unit's future voltages replaced by garbage.
    Predictions must be identical (the future never enters training, cleaning, tuning or features)."""
    d = split_units(W)[unit]
    k = origin_index(d, share)
    W2 = W.copy()
    m = (W2["unit"] == unit) & (W2["op_hours"] > d["op_hours"].iloc[k])
    W2.loc[m, "dV_mV"] = 1e3 * np.random.default_rng(1).random(int(m.sum()))
    W2.loc[m, "flag_recovery"] = ~W2.loc[m, "flag_recovery"]
    a = run_task(W, unit, share, name, band_err=pd.DataFrame(columns=["dh", "err"]))
    b = run_task(W2, unit, share, name, band_err=pd.DataFrame(columns=["dh", "err"]))
    return bool(np.allclose(a["pred"].to_numpy(), b["pred"].to_numpy()))


# ============================================================================================ live twin
def kalman_replay(d: pd.DataFrame, design: CellDesign, kin: Kinetics, eis: Optional[pd.DataFrame] = None,
                  q=(1e-9, 1e-9), r_V: float = 0.003, r_eis: float = 0.005, update_every: int = 1,
                  use_T: bool = True, use_P: bool = True, ref: Optional[OperatingConditions] = None) -> pd.DataFrame:
    """Kalman filter on the voltage-decomposition parameters x = [dR_ohm (ohm cm2), d_eta_act (V)] (random walk).
    Measurement 1 (every window): U - U_model,fresh(j, T, p) = j * dR + d_eta + v.  Measurement 2 (when an EIS
    spectrum exists): R_hf - R_model,fresh = dR + w. The model is linear in x, so the EKF reduces to a KF.
    With constant current and no EIS the two states are NOT separately observable: the filter then reports the
    sum only (column 'observable')."""
    ref = ref or OperatingConditions()
    x = np.zeros(2)
    P = np.diag([1e-3, 1e-3])
    Q = np.diag(q)
    e_tab = eis_hf_intercept(eis[eis["unit"] == d["unit"].iloc[0]], design.area_cm2) if eis is not None and len(eis) else None
    e_times = e_tab["time_h"].to_numpy() if e_tab is not None and len(e_tab) else np.array([])
    used = set()
    out = []
    jj = []
    for i, row in enumerate(d.itertuples(index=False)):
        T = row.T_C if use_T else ref.T_C
        Pb = row.P_bar if use_P else ref.P_bar
        op = OperatingConditions(T_C=T, P_bar=Pb, j_A_cm2=row.j_A_cm2, koh_wt=row.koh_wt)
        fresh = cell_voltage(row.j_A_cm2, op, design, kin)
        x_pred, P_pred = x, P + Q
        if i % update_every == 0:
            H = np.array([[row.j_A_cm2, 1.0]])
            z = np.array([row.U_cell_V - fresh["U"][0]])
            Rm = np.array([[r_V ** 2]])
            k_eis = [k for k, t in enumerate(e_times) if t <= row.t_end_h and k not in used]
            if k_eis:
                k = k_eis[-1]
                used.update(k_eis)
                H = np.vstack([H, [1.0, 0.0]])
                z = np.r_[z, e_tab["R_ohm_cm2"].iloc[k] - fresh["R_ohm_cm2"][0]]
                Rm = np.diag([r_V ** 2, r_eis ** 2])
            S = H @ P_pred @ H.T + Rm
            K = P_pred @ H.T @ np.linalg.inv(S)
            x = x_pred + K @ (z - H @ x_pred)
            P = (np.eye(2) - K @ H) @ P_pred
        else:
            x, P = x_pred, P_pred
        jj.append(row.j_A_cm2)
        jr = np.array(jj[-48:])
        obs = bool(len(e_times) and (e_times <= row.t_end_h).any()) or (jr.std() / max(jr.mean(), 1e-9) > 0.1)
        out.append(dict(op_hours=row.op_hours, t_end_h=row.t_end_h, dR_ohm_cm2=x[0], d_eta_V=x[1],
                        sd_dR=math.sqrt(P[0, 0]), sd_eta=math.sqrt(P[1, 1]),
                        dV_ref_mV=1000 * (ref.j_A_cm2 * x[0] + x[1]),
                        ohmic_share=abs(ref.j_A_cm2 * x[0]) / max(abs(ref.j_A_cm2 * x[0]) + abs(x[1]), 1e-12),
                        observable=obs))
    return pd.DataFrame(out)


def particle_filter_eol(d: pd.DataFrame, thr_mV: float, n: int = 2000, update_every: int = 1,
                        prior: Optional[Dict[str, Tuple[float, float]]] = None, sigma_mV: float = 2.0,
                        starts_rate: Optional[float] = None, seed: int = 0, checkpoints: int = 40) -> pd.DataFrame:
    """Particle filter on the degradation law dV = c + a * h + b * (starts - 1) (a: mV/h, b: mV per start-up).
    Prior: from the fleet/literature (Bayesian updating) or broad. At each checkpoint the end of life is predicted
    by extrapolating with the unit's own start-up rate, as median and 5-95 % band."""
    rng = np.random.default_rng(seed)
    prior = prior or {}
    am, asd = prior.get("a", (0.01, 0.02))
    bm, bsd = prior.get("b", (0.2, 0.5))
    a = np.abs(rng.normal(am, asd, n))
    b = np.abs(rng.normal(bm, bsd, n))
    c = rng.normal(0, 3, n)
    w = np.full(n, 1 / n)
    u = usable(d).reset_index(drop=True)
    if len(u) < 5:
        return pd.DataFrame()
    cps = set(np.linspace(5, len(u) - 1, min(checkpoints, len(u) - 5)).astype(int))
    rows = []
    for i, r in enumerate(u.itertuples(index=False)):
        if i % update_every == 0:
            pred = c + a * r.op_hours + b * (r.starts - 1)
            w *= np.exp(-0.5 * ((r.dV_mV - pred) / sigma_mV) ** 2) + 1e-300
            w /= w.sum()
            if 1 / np.sum(w ** 2) < n / 2:                      # resample + jitter
                idx = rng.choice(n, n, p=w)
                a = np.abs(a[idx] + rng.normal(0, 1e-4, n))
                b = np.abs(b[idx] + rng.normal(0, 1e-3, n))
                c = c[idx] + rng.normal(0, 0.05, n)
                w = np.full(n, 1 / n)
        if i in cps:
            rate = starts_rate if starts_rate is not None else (r.starts - 1) / max(r.op_hours, 1)
            slope = a + b * rate
            now = c + a * r.op_hours + b * (r.starts - 1)
            eol = r.op_hours + np.where(slope > 1e-9, (thr_mV - now) / np.maximum(slope, 1e-9), np.inf)
            eol = np.where(now >= thr_mV, r.op_hours, eol)
            q = _wq(eol, w, [0.05, 0.5, 0.95])
            rows.append(dict(op_hours=r.op_hours, eol_p05=q[0], eol_med=q[1], eol_p95=q[2],
                             a_med=_wq(a, w, [0.5])[0], b_med=_wq(b, w, [0.5])[0], dV_obs=r.dV_mV))
    return pd.DataFrame(rows)


def _wq(x, w, qs):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return [float(x[o][min(np.searchsorted(cw, q), len(x) - 1)]) for q in qs]


def fleet_prior(W: pd.DataFrame, exclude_unit: str) -> Dict[str, Tuple[float, float]]:
    """Prior for the particle filter from the OTHER units (Bayesian updating from a fleet prior)."""
    o = usable(W[W["unit"] != exclude_unit])
    if o["unit"].nunique() < 2:
        return {}
    A = []
    for u, d in o.groupby("unit"):
        if len(d) > 10:
            A.append(theilslopes(d["dV_mV"], d["op_hours"])[0])
    if len(A) < 2:
        return {}
    return {"a": (float(np.median(A)), float(np.std(A) + 1e-4)), "b": (0.2, 0.5)}


def update_frequency_study(d: pd.DataFrame, thr_mV: float, everies=(1, 6, 24, 168), prior=None) -> pd.DataFrame:
    """How often must the twin update? EOL prediction error vs update interval (needs a reached threshold)."""
    te = eol_time(d["op_hours"].to_numpy(), d["dV_mV"].to_numpy(), thr_mV, 3)
    rows = []
    for e in everies:
        pf = particle_filter_eol(d, thr_mV, update_every=e, prior=prior)
        if not len(pf):
            continue
        pre = pf[pf["op_hours"] < te] if np.isfinite(te) else pf
        err = (pre["eol_med"] - te).abs() if np.isfinite(te) else pd.Series(dtype=float)
        rows.append(dict(update_every_windows=e, eol_true_h=te, mae_eol_h=float(err.mean()) if len(err) else np.nan,
                         final_eol_pred_h=float(pf["eol_med"].iloc[-1]),
                         band_width_h=float((pf["eol_p95"] - pf["eol_p05"]).replace(np.inf, np.nan).median())))
    return pd.DataFrame(rows)


def measurement_ablation(data, design, kin, ref, unit: str, truth: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Which sensors keep the twin accurate? Rebuild the health indicator without temperature / pressure correction and
    re-run the Kalman filter without EIS; score against the synthetic truth when available, else against the full twin."""
    cfgs = [("all sensors", True, True, True), ("no temperature", False, True, True),
            ("no pressure", True, False, True), ("no EIS", True, True, False)]
    rows, base = [], None
    for name, uT, uP, uE in cfgs:
        W = build_windows(data, design, kin, ref, use_T=uT, use_P=uP)
        d = W[W["unit"] == unit]
        kf = kalman_replay(d, design, kin, data.get("eis") if uE else None, use_T=uT, use_P=uP, ref=ref)
        if truth is not None:
            tu = truth[truth["unit"] == unit].set_index("time_h")
            tt = tu["dV_ref_mV"].reindex(np.floor(d["t_end_h"]).to_numpy()).to_numpy()
            tR = tu["dR_ohm_cm2"].reindex(np.floor(d["t_end_h"]).to_numpy()).to_numpy()
            rows.append(dict(config=name, rmse_dV_mV=float(np.sqrt(np.nanmean((kf["dV_ref_mV"].to_numpy() - (tt - tt[0])) ** 2))),
                             rmse_dR_mohm_cm2=1000 * float(np.sqrt(np.nanmean((kf["dR_ohm_cm2"].to_numpy() - (tR - tR[0])) ** 2))),
                             reference="synthetic truth"))
        else:
            if base is None:
                base = kf
            rows.append(dict(config=name, rmse_dV_mV=float(np.sqrt(np.mean((kf["dV_ref_mV"].to_numpy()[:len(base)] - base["dV_ref_mV"].to_numpy()[:len(kf)]) ** 2))),
                             rmse_dR_mohm_cm2=np.nan, reference="full-sensor twin (no truth on real data)"))
    return pd.DataFrame(rows)


# ============================================================================================ operations
def operating_map(op: OperatingConditions, design: CellDesign, kin: Kinetics, T_grid, P_grid,
                  deg: Optional[DegradationState] = None, hto_limit: float = 2.0) -> pd.DataFrame:
    rows = []
    for T in T_grid:
        for P in P_grid:
            o = replace(op, T_C=float(T), P_bar=float(P))
            r = cell_voltage(o.j_A_cm2, o, design, kin, deg)
            h = hto_percent(o.j_A_cm2, o, design, kin, deg)["HTO_pct"][0]
            eff = efficiencies(r["U"][0], o)
            rows.append(dict(T_C=T, P_bar=P, U=r["U"][0], kWh_per_kg=float(eff["kWh_per_kg"]),
                             thermal_eff=float(eff["thermal_eff"]), HTO_pct=h, feasible=h < hto_limit,
                             min_load_j=min_load_for_hto(o, design, kin, hto_limit, deg)))
    return pd.DataFrame(rows)


def simulate_policy(power_frac: np.ndarray, price: np.ndarray, design: CellDesign, kin: Kinetics, op: OperatingConditions,
                    j_rated: float, min_load_frac: float, law: Dict[str, float], hto_limit: float = 2.0,
                    fp: Optional[FaradayParams] = None, T_ref_C: float = 80.0) -> Dict[str, object]:
    """Hourly load following: j = j_rated * available power share; below the minimum load the cell goes to standby
    (counts a start-up when it restarts). Degradation from the law dV = a(T) h + b * starts (mV), law a/Ea/b given
    by the user or fitted (degradation model v1). Returns totals and the hourly table."""
    fp = fp or FaradayParams()
    j = np.where(power_frac >= min_load_frac, power_frac * j_rated, 0.0)
    on = j > 0
    starts = int(np.sum(on[1:] & ~on[:-1]) + (1 if on[0] else 0))
    U = np.zeros(len(j))
    hto = np.zeros(len(j))
    for k in np.nonzero(on)[0]:
        o = replace(op, j_A_cm2=float(j[k]))
        U[k] = cell_voltage(j[k], o, design, kin)["U"][0]
        hto[k] = hto_percent(j[k], o, design, kin)["HTO_pct"][0]
    I = j * design.area_cm2
    P_kW = U * I * design.n_cells / 1000
    h2 = production(j, design, fp)["kg_h"]
    fa = math.exp(-law.get("Ea", 0.0) / R_GAS * (1 / op.T_K - 1 / (T_ref_C + 273.15))) if law.get("Ea") else 1.0
    dV = law.get("a", 0.0) * fa * on.sum() + law.get("b", 0.0) * starts
    hourly = pd.DataFrame(dict(hour=np.arange(len(j)), power_frac=power_frac, j=j, U=U, P_kW=P_kW, H2_kg=h2,
                               price=price, cost=P_kW * price / 1000, HTO_pct=hto))
    return dict(H2_kg=float(h2.sum()), energy_kWh=float(P_kW.sum()), cost=float(hourly["cost"].sum()), starts=starts,
                op_hours=int(on.sum()), dV_mV=float(dV), hto_violations=int((hto > hto_limit).sum()),
                kWh_per_kg=float(P_kW.sum() / max(h2.sum(), 1e-12)), hourly=hourly)


def optimise_operation(power_frac, price, design, kin, op, j_rated, law, T_options, min_load_options,
                       hto_limit=2.0, eol_mV: Optional[float] = None, stack_cost: Optional[float] = None) -> pd.DataFrame:
    """Grid search over temperature setpoint and minimum load. Objective columns are all reported (production,
    energy cost, voltage-rise rate); if an end-of-life threshold and a stack cost are given, the degradation is
    converted to a replacement cost per kg (cost of the stack spread over the hours to EOL)."""
    rows = []
    for T in T_options:
        for ml in min_load_options:
            o = replace(op, T_C=float(T))
            r = simulate_policy(power_frac, price, design, kin, o, j_rated, ml, law, hto_limit)
            dv_h = r["dV_mV"] / max(len(power_frac), 1)
            row = dict(T_C=T, min_load=ml, H2_kg=r["H2_kg"], energy_cost=r["cost"], kWh_per_kg=r["kWh_per_kg"],
                       starts=r["starts"], dV_mV=r["dV_mV"], hto_violations=r["hto_violations"])
            if eol_mV and stack_cost and dv_h > 0:
                life_h = eol_mV / dv_h
                row["degradation_cost"] = stack_cost * len(power_frac) / life_h
                row["cost_per_kg"] = (r["cost"] + row["degradation_cost"]) / max(r["H2_kg"], 1e-12)
            else:
                row["cost_per_kg"] = r["cost"] / max(r["H2_kg"], 1e-12)
            rows.append(row)
    df = pd.DataFrame(rows)
    df["feasible"] = df["hto_violations"] == 0
    return df.sort_values(["feasible", "cost_per_kg"], ascending=[False, True]).reset_index(drop=True)


def maintenance_time(res_unit: pd.DataFrame, thr_mV: float, risk: float = 0.05) -> Dict[str, float]:
    """Recommended service time: first hour where the upper band crosses the EOL threshold (risk ~ band level),
    together with the median-forecast crossing."""
    h = res_unit["op_hours"].to_numpy()
    return dict(service_h=eol_time(h, res_unit["hi"].to_numpy(), thr_mV, 1),
                median_eol_h=eol_time(h, res_unit["pred"].to_numpy(), thr_mV, 1),
                optimistic_eol_h=eol_time(h, res_unit["lo"].to_numpy(), thr_mV, 1))


# ============================================================================================ literature check
HAM25_REPORTED = [  # Ham25 sec 4.2 (text): 3 bar unless stated, 0.4 A/cm2
    dict(T_C=70, P_bar=3, quantity="E_rev", value=1.219), dict(T_C=70, P_bar=5, quantity="E_rev", value=1.231),
    dict(T_C=70, P_bar=10, quantity="E_rev", value=1.246), dict(T_C=60, P_bar=3, quantity="E_rev", value=1.226),
    dict(T_C=80, P_bar=3, quantity="E_rev", value=1.211),
    dict(T_C=70, P_bar=3, quantity="eta_act", value=0.437), dict(T_C=70, P_bar=5, quantity="eta_act", value=0.430),
    dict(T_C=70, P_bar=10, quantity="eta_act", value=0.422), dict(T_C=60, P_bar=3, quantity="eta_act", value=0.456),
    dict(T_C=80, P_bar=3, quantity="eta_act", value=0.419),
    dict(T_C=70, P_bar=3, quantity="theta", value=0.138), dict(T_C=70, P_bar=5, quantity="theta", value=0.101),
    dict(T_C=70, P_bar=10, quantity="theta", value=0.065),
]


def literature_check(kin: Optional[Kinetics] = None) -> pd.DataFrame:
    """Engine vs the values Ham25 reports at 0.4 A/cm2, 30 wt% KOH (their ohmic term depends on a geometry they do
    not give, so the total voltage is not compared)."""
    kin = kin or Kinetics()
    rows = []
    for r in HAM25_REPORTED:
        o = OperatingConditions(T_C=r["T_C"], P_bar=r["P_bar"], j_A_cm2=0.4, koh_wt=30)
        v = cell_voltage(0.4, o, CellDesign(), kin)[r["quantity"]][0]
        rows.append(dict(r, engine=float(v), diff=float(v) - r["value"]))
    return pd.DataFrame(rows)


# ============================================================================================ equations (for the app)
EQUATIONS = {
    "Cell voltage": [
        ("Decomposition [Hu22 eq 4], [Ham25 eq 1]", r"U = E_{rev} + \eta_{act,c} + \eta_{act,a} + \eta_{ohm}"),
        ("Nernst [Hu22 eq 11], [Ham25 eq 2]", r"E_{rev} = E^0_{rev}(T) + \frac{RT}{2F}\ln\frac{(P-P_{H_2O})^{1.5}}{a_{H_2O}}"),
        ("Standard reversible voltage [Ham25 eq 3]", r"E^0_{rev} = 1.50342 - 9.956\cdot10^{-4}T + 2.5\cdot10^{-7}T^2"),
        ("Activation with bubbles [Hu22 eq 18], [Ham25 eq 14]", r"\eta_{act,k} = \frac{RT}{n\alpha_k F}\ln\frac{j}{j_{0,k}(1-\theta)}"),
        ("Exchange current, Ni [Ham25 eq 5-6]", r"j_{0,k} = j^{ref}_{0,k}\left(\frac{P}{P_{ref}}\right)^{0.1}\exp\left[-\frac{E_{a,k}}{RT}\left(1-\frac{T}{T_{ref}}\right)\right]"),
        ("Bubble coverage [Hu22 eq 17], [Ham25 eq 13]", r"\theta = 0.023\,j^{0.3}\left(\frac{T}{T_{ref}}\frac{P_{ref}}{P}\right)^{2/3}"),
        ("Ohmic [Hu22 eq 19], [Ham25 eq 15]", r"\eta_{ohm} = j\left(\frac{d_a+d_c}{\sigma_{KOH}(1-\tfrac{2}{3}\theta)^{1.5}} + R_{mem} + R_{el}\right)"),
        ("KOH conductivity [Hu22 Table 4]", r"\sigma = -2.041m - 0.0028m^2 + 0.005332mT + 207.2\frac{m}{T} + 0.001043m^3 - 3\cdot10^{-7}m^2T^2"),
    ],
    "Production and efficiency": [
        ("Faraday's law [Hu22 eq 5, 9]", r"\dot n_{H_2} = \eta_F \frac{N_c I}{2F}"),
        ("Faraday efficiency [Hu22 eq 7]", r"\eta_F = \frac{j^2}{f_1 + j^2} f_2"),
        ("Thermal efficiency [Hu22 Table B1]", r"\eta_{th} = V_{tn}/U"),
        ("Specific energy", r"e = \frac{2F\,U}{\eta_F M_{H_2}} \;[\mathrm{kWh/kg}] \approx 26.6\,U/\eta_F"),
    ],
    "Crossover and safety": [
        ("Fick diffusion [Hu22 eq 41], [Ham25 eq 21]", r"N_{diff} = \frac{\varepsilon_{H_2}^{Fick}\,p_{H_2}}{\delta}(1-\theta)"),
        ("Darcy convection [Hu22 eq 42]", r"N_{conv} = \frac{K}{\mu} S_{H_2} p_{H_2} \frac{\Delta P}{\delta},\quad K = \frac{\varepsilon d^2}{32\tau}"),
        ("Mixed circulation [Hu22 eq 38]", r"N_{mix} = S_{H_2}\, p_{H_2}\, \dot V_{lye}/4"),
        ("H2 in O2 [Hu22 eq 46]", r"HTO = \frac{N_{H_2}}{N_{H_2} + j/4F}"),
        ("Limits [Hu22 3.3], [Qu26 3.3]", r"HTO < 2\%\;(\text{shutdown}),\quad LEL = 4\%"),
    ],
    "Thermal": [
        ("Lumped balance [Hu22 eq 53]", r"C_t \frac{dT}{dt} = \dot Q_{gen} - \dot Q_{loss} - \dot Q_{cool}"),
        ("Heat generation [Hu22 eq 56]", r"\dot Q_{gen} = N_c (U - V_{tn}) I"),
        ("Losses [Hu22 eq 58]", r"\dot Q_{loss} = (T - T_a)/R_t"),
    ],
    "Degradation law": [
        ("Pooled law (degradation model v1)", r"\Delta V = a\,h\,e^{-\frac{E_a}{R}\left(\frac{1}{T}-\frac{1}{T_{ref}}\right)} + b\,(N_{start}-1) + c_{unit}"),
        ("Normalised voltage", r"V_{norm} = U_{meas} - [U_{model}(j,T,P) - U_{model}(j_{ref},T_{ref},P_{ref})]"),
        ("Parameter drift (twin state)", r"\Delta U = j\,\Delta R_{ohm} + \Delta\eta_{act}"),
    ],
    "Live twin": [
        ("Kalman filter state and measurements", r"x=[\Delta R,\Delta\eta],\; z_1 = U - U_{fresh} = [j\;\;1]x + v,\; z_2 = R_{HF} - R_{fresh} = [1\;\;0]x + w"),
        ("Update", r"K = P H^T (H P H^T + R)^{-1},\; x \leftarrow x + K(z - Hx)"),
        ("Particle filter law", r"\Delta V = c + a\,h + b\,(N_{start}-1),\quad w_i \propto \exp\left[-\frac{(\Delta V_{obs}-\Delta V_i)^2}{2\sigma^2}\right]"),
    ],
    "Models": [
        ("Persistence", r"\hat{\Delta V}(h) = \Delta V(h_0)"),
        ("Linear drift (Theil-Sen)", r"\hat{\Delta V}(h) = \Delta V(h_0) + \max(b,0)(h-h_0)"),
        ("Fleet curve", r"\hat{\Delta V}(h) = \Delta V(h_0) + F(a^* + k\,\Delta h) - F(a^*)"),
        ("Bayesian ridge", r"y = w^T x + e,\; w\sim N(0,\lambda^{-1}I)"),
        ("Random forest", r"f(x) = \frac{1}{B}\sum_b T_b(x)"),
        ("SVM", r"f(x) = \sum_i(\alpha_i-\alpha_i^*)K(x_i,x) + b"),
        ("Gaussian process", r"k = s^2\exp\left(-\tfrac12\sum_d \frac{(x_d-x'_d)^2}{l_d^2}\right) + s_{lin}^2 x\cdot x' + s_n^2"),
        ("MLP (Level 3)", r"h_{l+1} = \mathrm{ReLU}(W_l h_l + b_l)"),
        ("Anchoring", r"\hat{\Delta V}(h) = \Delta V(h_0) + f(x_h) - f(x_{h_0})"),
        ("Ensemble", r"w_i \propto 1/RMSE_{i,held\text{-}out}^2"),
    ],
    "DRT": [
        ("Impedance model [DRT25 eq 1]", r"Z(f) = R_\infty + i2\pi f L_0 + \int \frac{\gamma(\log\tau)}{1 + i2\pi f\tau}\, d\log\tau"),
        ("Tikhonov [DRT25 eq 2]", r"\gamma = \arg\min_{x\ge0} \|Ax - Z_{exp}\|^2 + \lambda\|Lx\|^2"),
    ],
}


def drt_tikhonov(freq, z_re, z_im_neg, lam: float = 1e-3, n_tau: int = 60) -> Dict[str, np.ndarray]:
    """Simple DRT by non-negative Tikhonov regression (piecewise-constant basis, 1st-order penalty) [DRT25 eq 1-2].
    For publication-grade DRT (RBF, Bayesian credible bands, Hilbert test) use DRTtools; this is for quick looks."""
    f = np.asarray(freq, float)
    Z = np.asarray(z_re, float) - 1j * np.asarray(z_im_neg, float)
    tau = np.logspace(np.log10(1 / (2 * np.pi * f.max())) - 1, np.log10(1 / (2 * np.pi * f.min())) + 1, n_tau)
    w = 2 * np.pi * f[:, None]
    dlt = np.log(tau[1] / tau[0])
    K = dlt / (1 + 1j * w * tau[None, :])
    A = np.vstack([np.c_[np.ones(len(f)), K.real], np.c_[np.zeros(len(f)), K.imag]])
    b = np.r_[Z.real, Z.imag]
    L = np.diff(np.eye(n_tau + 1)[1:], axis=0)
    L = np.c_[np.zeros(len(L)), L[:, 1:]] if L.shape[1] == n_tau + 1 else np.c_[np.zeros((len(L), 1)), L]
    A2 = np.vstack([A, math.sqrt(lam) * L * np.abs(Z).max()])
    b2 = np.r_[b, np.zeros(len(L))]
    x, _ = optimize.nnls(A2, b2, maxiter=5000)
    return dict(tau=tau, gamma=x[1:], R_inf=x[0], R_pol=float(np.sum(x[1:]) * dlt))
