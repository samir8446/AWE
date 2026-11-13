"""Beginning-of-life AWE physics, written independently of the platform engine (same papers, own code).

All functions are vectorised over numpy arrays. Units: T in K, P in bar (absolute), j in A/cm2, resistances in
ohm cm2, voltages in V.
"""
from __future__ import annotations

import numpy as np

from .config import DESIGN, PHYS

F = PHYS["F"].value
R = PHYS["R"].value
NE = PHYS["n_e"].value
TREF = PHYS["T_ref_kin_K"].value
PREF = PHYS["P_ref_bar"].value


def molarity_from_wt(w_pct, T_K):
    """KOH molarity (mol/L) from wt% [Ham25 eq 12], [Hu22 eq B.6]."""
    w = np.asarray(w_pct, float)
    return w * (183.1221 - 0.56845 * T_K + 984.5679 * np.exp(w / 115.96277)) / 5610.5


def E0_rev(T_K):
    """[Ham25 eq 3], [Hu22 Table 3]."""
    return 1.50342 - 9.956e-4 * T_K + 2.5e-7 * T_K ** 2


def p_sat_water(T_K):
    """Vapour pressure of water, bar [Hu22 eq B.4]."""
    return T_K ** -3.4159 * np.exp(37.034 - 6275.7 / T_K)


def water_activity(m, T_K):
    """[Hu22 eq B.2], m in mol/L."""
    return 10 ** (-0.02255 * m + 0.001434 * m ** 2 + (1.38 * m - 0.9254 * m ** 2) / T_K)


def p_water(m, T_K):
    return water_activity(m, T_K) * p_sat_water(T_K)


def E_rev(T_K, P_bar, m):
    """Nernst [Hu22 eq 11], [Ham25 eq 2]."""
    a = water_activity(m, T_K)
    pw = a * p_sat_water(T_K)
    return E0_rev(T_K) + R * T_K / (2 * F) * np.log(np.maximum(P_bar - pw, 1e-6) ** 1.5 / a)


def V_tn(T_K, P_bar, m):
    """Thermoneutral voltage [Hu22 App. A, eq A.1-A.6], T in deg C inside eq A.2/A.6 (gives 1.48 V at 25 C)."""
    pw = p_water(m, T_K)
    phi = 1.5 * pw / np.maximum(P_bar - pw, 1e-6)
    Tc = T_K - 273.15
    dH = 42960 + 40.762 * Tc - 0.06682 * Tc ** 2
    v0 = 1.4756 + 2.252e-4 * Tc + 1.52e-8 * Tc ** 2
    f = (21.661e-6 - 10.941e-3 / T_K) * P_bar + (18.578e-6 / T_K + 0.339e-3 / T_K ** 1.5 + 7.845e-3 / T_K ** 2
                                                 - 1.659 / T_K ** 3) * P_bar ** 2
    return v0 + f + phi / (2 * F) * dH


def sigma_koh(m, T_K):
    """KOH conductivity S/cm, 0-12 mol/L, 273-373 K [Hu22 Table 4], [Ham25 eq 11]."""
    return (-2.041 * m - 2.8e-3 * m ** 2 + 5.332e-3 * m * T_K + 207.2 * m / T_K + 1.043e-3 * m ** 3
            - 3e-7 * m ** 2 * T_K ** 2)


def theta(j, T_K, P_bar):
    """Bubble coverage [Hu22 eq 17], [Ham25 eq 13], j in A/cm2 converted to A/m2."""
    j = np.maximum(np.asarray(j, float), 0.0)
    return np.clip(0.023 * (j * 1e4) ** 0.3 * (T_K / TREF * PREF / P_bar) ** (2 / 3), 0.0, 0.9)


def alpha_c(T_K):
    return 0.1175 + 0.00095 * T_K          # [Hu22 Table B3]


def alpha_a(T_K):
    return 0.07835 + 0.001 * T_K           # [Hu22 Table B3, Jang]


def j0_bol(T_K, P_bar):
    """Ni exchange current densities per geometric area at the BoL roughness [Ham25 eq 5-6]."""
    fac = (P_bar / PREF) ** 0.1
    j0c = PHYS["j0c_ref"].value * fac * np.exp(-PHYS["Ea_c"].value / (R * T_K) * (1 - T_K / TREF))
    j0a = PHYS["j0a_ref"].value * fac * np.exp(-PHYS["Ea_a"].value / (R * T_K) * (1 - T_K / TREF))
    return j0c, j0a


def eta_bv(j, j0_eff, alpha, T_K):
    """Butler-Volmer, symmetric inversion [Hu22 eq 13]: eta = RT/(n alpha F) asinh(j / (2 j0)).
    Equals the Tafel form [Hu22 eq 18] RT/(n alpha F) ln(j/j0) for j >> j0."""
    return R * T_K / (NE * alpha * F) * np.arcsinh(np.asarray(j, float) / (2 * j0_eff))


def r_ct(j, j0_eff, alpha, T_K):
    """d eta / d j at fixed coverage (charge-transfer resistance, ohm cm2)."""
    b = R * T_K / (NE * alpha * F)
    return b / np.sqrt((2 * j0_eff) ** 2 + np.asarray(j, float) ** 2)


def eta_conc(j, T_K):
    j = np.asarray(j, float)
    return R * T_K / (2 * F) * np.log(1.0 / (1.0 - np.minimum(j / PHYS["j_lim"].value, 0.99)))


def faraday_eff(j, T_K):
    """eta_F = j^2/(f1 + j^2) with j in mA/cm2 [Hu22 eq 7], f1 interpolated in T [Ulleberg03]; f2 = 1."""
    pts = PHYS["f1_FE"].value
    Tc = np.asarray(T_K, float) - 273.15
    f1 = np.interp(Tc, sorted(pts), [pts[k] for k in sorted(pts)])
    jm = np.maximum(np.asarray(j, float), 0.0) * 1000.0
    return np.where(jm > 0, jm ** 2 / (f1 + jm ** 2), 0.0)


def r_separator(sigma, foul=1.0):
    """Area resistance of the electrolyte-filled separator [Hu22 eq 24-25] x fouling factor."""
    eps = DESIGN["sep_porosity"].value
    tau = (2 - eps) ** 2 / eps
    return DESIGN["sep_thickness_cm"].value * tau / (eps * sigma) * foul


def r_gap(sigma, th):
    """Bubble-corrected electrolyte gaps [Ham25 eq 15] (Bruggeman)."""
    return 2 * DESIGN["gap_each_side_cm"].value / sigma / (1 - 2 / 3 * th) ** 1.5


def cell(j, T_K, P_bar, m_free, s):
    """Cell voltage and all components for state s (dict of arrays/scalars):
    j0c_f, j0a_f   multipliers on BoL j0 (intrinsic activity x roughness ratio)
    ac_f, aa_f     multipliers on alpha (Tafel slope = 1/alpha)
    sep_f          separator fouling factor;  R_c contact resistance (ohm cm2);  x_f crossover factor
    U = E_rev + eta_c + eta_a + j (R_gap + R_sep + R_c) + eta_conc   [Hu22 eq 4]."""
    j = np.asarray(j, float)
    th = theta(j, T_K, P_bar)
    j0c, j0a = j0_bol(T_K, P_bar)
    j0c = j0c * s["j0c_f"] * (1 - th)
    j0a = j0a * s["j0a_f"] * (1 - th)
    ac, aa = alpha_c(T_K) * s["ac_f"], alpha_a(T_K) * s["aa_f"]
    sig = sigma_koh(m_free, T_K)
    Rg, Rs = r_gap(sig, th), r_separator(sig, s["sep_f"])
    Rohm = Rg + Rs + s["R_c"]
    out = dict(theta=th, E_rev=E_rev(T_K, P_bar, m_free), eta_c=eta_bv(j, j0c, ac, T_K),
               eta_a=eta_bv(j, j0a, aa, T_K), R_gap=Rg, R_sep=Rs, R_c=np.asarray(s["R_c"], float) + 0.0 * Rg,
               R_ohm=Rohm, eta_ohm=j * Rohm, eta_conc=eta_conc(j, T_K), sigma=sig,
               Rct_c=r_ct(j, j0c, ac, T_K), Rct_a=r_ct(j, j0a, aa, T_K),
               b_c=np.log(10) * R * T_K / (NE * ac * F), b_a=np.log(10) * R * T_K / (NE * aa * F), j0c=j0c, j0a=j0a)
    out["U"] = out["E_rev"] + out["eta_c"] + out["eta_a"] + out["eta_ohm"] + out["eta_conc"]
    return out


def crossover(j, T_K, P_bar, m_free, x_f=1.0):
    """Diffusive gas crossover through the separator, mol/(s cm2) [Hu22 eq 40-41], [Ham25 eq 21-24, x (1-theta)].
    Supersaturation multiplies the H2 driving force [Barros23]. Same pressure on both sides (no Darcy term)."""
    th = theta(j, T_K, P_bar)
    p_gas = np.maximum(P_bar - p_water(m_free, T_K), 1e-6)
    delta = DESIGN["sep_thickness_cm"].value
    nH2 = PHYS["perm_H2"].value * p_gas / delta * PHYS["supersaturation"].value * (1 - th) * x_f
    nO2 = PHYS["perm_H2"].value * PHYS["perm_O2_ratio"].value * p_gas / delta * (1 - th) * x_f
    return nH2, nO2


def gases(j, T_K, P_bar, m_free, x_f=1.0, area=None):
    """Faraday's law [Hu22 eq 5, 9] with Faraday efficiency, minus crossover; HTO/OTH in % (mol fraction, dry)."""
    area = area or DESIGN["area_cm2"].value
    j = np.asarray(j, float)
    I = j * area
    eF = faraday_eff(j, T_K)
    nH2_prod, nO2_prod = eF * I / (2 * F), eF * I / (4 * F)
    cH2, cO2 = crossover(j, T_K, P_bar, m_free, x_f)
    on = j > 0
    cH2, cO2 = np.where(on, cH2 * area, 0.0), np.where(on, cO2 * area, 0.0)
    H2_out = nH2_prod - cH2 + cO2 * 0          # O2 crossing to the cathode leaves with the H2 stream
    O2_out = nO2_prod - cO2
    hto = np.where(on, 100 * cH2 / np.maximum(cH2 + O2_out, 1e-30), np.nan)
    oth = np.where(on, 100 * cO2 / np.maximum(cO2 + H2_out, 1e-30), np.nan)
    return dict(eta_F=eF, I_A=I, nH2_prod=nH2_prod, nO2_prod=nO2_prod, nH2_cross=cH2, nO2_cross=cO2,
                H2_out=H2_out, O2_out=O2_out, HTO_pct=hto, OTH_pct=oth)


def molar_to_Nm3_h(n_mol_s):
    return np.asarray(n_mol_s) * 0.0224141 * 3600
