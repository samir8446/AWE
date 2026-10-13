"""DRT analysis for the AWE twin (no Streamlit).

Method follows DRTtools [DRT25 = Wang et al., ACS Electrochem 1 (2025) 2680]:
  Z(f) = R_inf + i 2 pi f L0 + int gamma(ln tau) / (1 + i 2 pi f tau) d ln tau                      [DRT25 eq 1]
  gamma = argmin_{x >= 0} ||W (A x - Z)||^2 + lambda ||L x||^2   (Gaussian RBF basis, 2nd-derivative L) [DRT25 eq 2]
plus: lambda by generalised cross-validation or L-curve, approximate Bayesian credible bands (Gaussian posterior around
the MAP, truncated at zero), a linear Kramers-Kronig test (Schoenleber et al. 2014 lin-KK, mu criterion) as a data
quality gate, peak detection and Gaussian peak fits, peak tracking over time, multi-current peak classification and a
DRT-guided equivalent-circuit fit (R_inf + ZARC elements).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.signal import find_peaks

__version__ = "0.3.0"

# adapted from the DRT frequency mapping in "Cell and results" (slides give overlapping P2/P3 bands; confirm)
DEFAULT_BANDS_HZ = {"P1 ion transport": (100.0, 1e6), "P2 HER (cathode)": (10.0, 100.0),
                    "P3 OER (anode)": (0.5, 10.0), "P4 mass transport / gas": (1e-4, 0.5)}


# ----------------------------------------------------------------------------------------------- basis
def _basis(freq: np.ndarray, n_per_decade: int = 8, ext_decades: float = 1.0, fwhm_coeff: float = 0.5):
    f = np.asarray(freq, float)
    tmin, tmax = 1 / (2 * np.pi * f.max()), 1 / (2 * np.pi * f.min())
    lo, hi = np.log(tmin) - ext_decades * np.log(10), np.log(tmax) + ext_decades * np.log(10)
    M = int(round((hi - lo) / np.log(10) * n_per_decade)) + 1
    xm = np.linspace(lo, hi, M)
    dx = xm[1] - xm[0]
    eps = 2 * math.sqrt(math.log(2)) / (fwhm_coeff * dx * 4)        # Gaussian shape parameter (FWHM ~ 2 dx)
    y = np.linspace(-5 / eps, 5 / eps, 241)
    dy = y[1] - y[0]
    phi = np.exp(-(eps * y) ** 2)
    w = 2 * np.pi * f[:, None, None]
    tau = np.exp(xm[None, :, None] + y[None, None, :])
    den = 1 + (w * tau) ** 2
    A_re = (phi / den).sum(-1) * dy
    A_im = -(phi * w * tau / den).sum(-1) * dy
    return xm, eps, A_re, A_im


def _gamma_on(xm, eps, coef, xg):
    return (coef[None, :] * np.exp(-(eps * (xg[:, None] - xm[None, :])) ** 2)).sum(1)


def _penalty(xm, eps, n_fine: int = 400):
    xg = np.linspace(xm[0], xm[-1], n_fine)
    d = xg[:, None] - xm[None, :]
    e = np.exp(-(eps * d) ** 2)
    d2 = e * (4 * eps ** 4 * d ** 2 - 2 * eps ** 2)                 # second derivative of each Gaussian
    return d2 * math.sqrt(xg[1] - xg[0])


def _system(freq, z, xm, eps, A_re, A_im, inductance: bool):
    w = 1 / np.maximum(np.abs(z), 1e-12)
    om = 2 * np.pi * np.asarray(freq, float)
    cols_re = [np.ones(len(om))] + ([np.zeros(len(om))] if inductance else [])
    cols_im = [np.zeros(len(om))] + ([om] if inductance else [])
    A = np.vstack([np.column_stack(cols_re + [A_re]), np.column_stack(cols_im + [A_im])])
    W = np.r_[w, w]
    b = np.r_[z.real, z.imag]
    return A * W[:, None], b * W


# ----------------------------------------------------------------------------------------------- DRT fit
def fit_drt(freq, z_re, z_im_neg, lam: Optional[float] = None, method: str = "mgcv", inductance: bool = False,
            n_per_decade: int = 8, n_samples: int = 40, seed: int = 0, rho: float = 2.0) -> Dict[str, object]:
    """Regularised non-negative DRT with automatic lambda ('mgcv' = modified GCV with factor rho [Maradesa et al.,
    J Electrochem Soc 170 (2023) 030502, cited in DRT25], 'gcv', 'lcurve' or a fixed value) and uncertainty bands
    from a parametric bootstrap (noise level estimated from the fit residuals, NNLS refit n_samples times; this keeps
    gamma >= 0, unlike a truncated Gaussian posterior). Returns gamma on a fine ln(tau) grid, R_inf, R_pol, bands."""
    f = np.asarray(freq, float)
    order = np.argsort(-f)
    f = f[order]
    z = (np.asarray(z_re, float) - 1j * np.asarray(z_im_neg, float))[order]
    xm, eps, A_re, A_im = _basis(f, n_per_decade)
    Aw, bw = _system(f, z, xm, eps, A_re, A_im, inductance)
    n_extra = 1 + int(inductance)
    Lp = _penalty(xm, eps)
    L = np.hstack([np.zeros((Lp.shape[0], n_extra)), Lp])
    scale = np.linalg.norm(Aw[:, n_extra:], 2) / max(np.linalg.norm(Lp, 2), 1e-12)
    L = L * scale
    grid = np.logspace(-6, 1, 36)
    if lam is None or method in ("gcv", "mgcv", "lcurve"):
        AtA, Atb, LtL = Aw.T @ Aw, Aw.T @ bw, L.T @ L
        gcv, rn, sn = [], [], []
        n = len(bw)
        for g in grid:
            Minv = np.linalg.pinv(AtA + g * LtL + 1e-12 * np.eye(len(AtA)))
            x = Minv @ Atb
            r = bw - Aw @ x
            trH = np.trace(Aw @ Minv @ Aw.T)
            gcv.append(n * (r @ r) / max(n - (rho if method == "mgcv" else 1.0) * trH, 1e-9) ** 2)
            rn.append(np.linalg.norm(r))
            sn.append(np.linalg.norm(L @ x))
        if method == "lcurve":
            lx, ly = np.log(np.maximum(rn, 1e-300)), np.log(np.maximum(sn, 1e-300))
            dx1, dy1 = np.gradient(lx), np.gradient(ly)
            dx2, dy2 = np.gradient(dx1), np.gradient(dy1)
            curv = (dx1 * dy2 - dy1 * dx2) / np.maximum((dx1 ** 2 + dy1 ** 2) ** 1.5, 1e-12)
            lam = float(grid[int(np.argmax(curv[2:-2])) + 2])
        else:
            lam = float(grid[int(np.argmin(gcv))])
    A2 = np.vstack([Aw, math.sqrt(lam) * L])
    b2 = np.r_[bw, np.zeros(L.shape[0])]
    x, _ = optimize.nnls(A2, b2, maxiter=20 * A2.shape[1])
    res = bw - Aw @ x
    dof = max(len(bw) - np.count_nonzero(x > 1e-12), 1)
    sigma2 = float(res @ res) / dof
    xg = np.linspace(xm[0], xm[-1], 400)
    coef = x[n_extra:]
    gamma = _gamma_on(xm, eps, coef, xg)
    # parametric bootstrap bands
    rng = np.random.default_rng(seed)
    G = []
    sig = math.sqrt(max(sigma2, 0.0))
    for _ in range(n_samples):
        bs = Aw @ x + sig * rng.standard_normal(len(bw))
        xs, _ = optimize.nnls(A2, np.r_[bs, np.zeros(L.shape[0])], maxiter=20 * A2.shape[1])
        G.append(_gamma_on(xm, eps, xs[n_extra:], xg))
    G = np.array(G) if G else None
    if G is not None:      # basic (centred) bootstrap: spread of the refits placed around the estimate
        med = np.median(G, axis=0)
        lo = np.maximum(gamma - (med - np.percentile(G, 5, axis=0)), 0)
        hi = gamma + (np.percentile(G, 95, axis=0) - med)
        G = G - med + gamma
    else:
        lo = hi = np.full_like(gamma, np.nan)
    zfit_w = Aw @ x
    nN = len(f)
    zfit = (zfit_w[:nN] + 1j * zfit_w[nN:]) / (1 / np.maximum(np.abs(z), 1e-12))
    return dict(ln_tau=xg, tau=np.exp(xg), freq_peak=1 / (2 * np.pi * np.exp(xg)), gamma=gamma, gamma_lo=lo,
                gamma_hi=hi, samples=G, R_inf=float(x[0]), L0=float(x[1]) if inductance else 0.0,
                R_pol=float(np.trapezoid(gamma, xg)), lam=lam, method=method, freq=f, z=z, z_fit=zfit,
                rel_residual=float(np.sqrt(np.mean(np.abs(z - zfit) ** 2 / np.abs(z) ** 2))))


# ----------------------------------------------------------------------------------------------- KK test
def kk_test(freq, z_re, z_im_neg, mu_crit: float = 0.85, threshold_pct: float = 1.0) -> Dict[str, object]:
    """Linear Kramers-Kronig test (lin-KK, Schoenleber et al., Electrochim Acta 131 (2014) 20): Voigt elements with
    fixed log-spaced time constants, number chosen by the mu criterion; residuals relative to |Z|. A spectrum passes
    when the mean absolute residual is below `threshold_pct` %."""
    f = np.asarray(freq, float)
    z = np.asarray(z_re, float) - 1j * np.asarray(z_im_neg, float)
    om = 2 * np.pi * f
    w = 1 / np.maximum(np.abs(z), 1e-12)
    best = None
    decades = np.log10(om.max() / om.min())
    m_min = int(math.ceil(decades)) + 1          # at least ~1 element per decade before the mu criterion applies
    for M in range(m_min, min(len(f), 60)):
        tau = np.logspace(np.log10(1 / om.max()), np.log10(1 / om.min()), M)
        K = 1 / (1 + 1j * om[:, None] * tau[None, :])
        A = np.vstack([np.c_[np.ones(len(f)), K.real], np.c_[np.zeros(len(f)), K.imag]])
        b = np.r_[z.real, z.imag]
        Wt = np.r_[w, w]
        p, *_ = np.linalg.lstsq(A * Wt[:, None], b * Wt, rcond=None)
        Rk = p[1:]
        mu = 1 - np.abs(Rk[Rk < 0]).sum() / max(np.abs(Rk[Rk >= 0]).sum(), 1e-12)
        best = (M, p, tau, mu)
        if mu <= mu_crit:
            break
    M, p, tau, mu = best
    K = 1 / (1 + 1j * om[:, None] * tau[None, :])
    zf = p[0] + K @ p[1:]
    dre = (z.real - zf.real) / np.abs(z) * 100
    dim = (z.imag - zf.imag) / np.abs(z) * 100
    score = float(np.mean(np.abs(np.r_[dre, dim])))
    return dict(M=M, mu=float(mu), residual_re_pct=dre, residual_im_pct=dim, mean_abs_pct=score,
                max_abs_pct=float(np.max(np.abs(np.r_[dre, dim]))), valid=score < threshold_pct, freq=f)


# ----------------------------------------------------------------------------------------------- peaks
def find_drt_peaks(res: Dict[str, object], prominence: float = 0.01, max_peaks: int = 6, margin_decades: float = 0.2,
                   valley_merge: float = 0.85, min_share: float = 0.0) -> pd.DataFrame:
    """Peaks of gamma(ln tau): position, height, Gaussian-fit width and area, and the resistance integrated between
    neighbouring minima for the MAP and for every posterior sample (credible interval of each peak's resistance)."""
    x, g = res["ln_tau"], res["gamma"]
    if g.max() <= 0:
        return pd.DataFrame()
    idx, prop = find_peaks(g, prominence=prominence * g.max())
    # only peaks supported by data: inside the measured time-constant window (+ margin); outside is extrapolation
    f = res["freq"]
    lo_t = math.log(1 / (2 * np.pi * f.max())) - margin_decades * math.log(10)
    hi_t = math.log(1 / (2 * np.pi * f.min())) + margin_decades * math.log(10)
    keep = (x[idx] >= lo_t) & (x[idx] <= hi_t)
    idx, prop = idx[keep], {k: v[keep] for k, v in prop.items()}
    if not len(idx):
        return pd.DataFrame()
    idx = np.sort(idx[np.argsort(-prop["prominences"])][:max_peaks])
    # merge neighbours separated by a shallow valley (one broad process split by regularisation ringing)
    merged = [idx[0]]
    for i in idx[1:]:
        a = merged[-1]
        valley = g[a:i + 1].min()
        if valley > valley_merge * min(g[a], g[i]):
            merged[-1] = a if g[a] >= g[i] else i
        else:
            merged.append(i)
    idx = np.array(merged)
    # bounds: minima between peaks
    bounds = [0]
    for a, b in zip(idx[:-1], idx[1:]):
        bounds.append(a + int(np.argmin(g[a:b + 1])))
    bounds.append(len(x) - 1)
    rows = []
    for k, i in enumerate(idx):
        lo, hi = bounds[k], bounds[k + 1]
        R = float(np.trapezoid(g[lo:hi + 1], x[lo:hi + 1]))
        if res.get("samples") is not None:
            Rs = np.trapezoid(res["samples"][:, lo:hi + 1], x[lo:hi + 1], axis=1)
            Rl, Rh = min(float(np.percentile(Rs, 5)), R), max(float(np.percentile(Rs, 95)), R)
        else:
            Rl = Rh = np.nan
        rows.append(dict(tau_s=float(np.exp(x[i])), f_Hz=float(1 / (2 * np.pi * np.exp(x[i]))), height=float(g[i]),
                         R=R, R_lo=Rl, R_hi=Rh, ln_tau_lo=float(x[lo]), ln_tau_hi=float(x[hi])))
    out = pd.DataFrame(rows).sort_values("tau_s").reset_index(drop=True)
    if min_share and res.get("R_pol", 0) > 0:          # drop ringing side-lobes (< min_share of R_pol)
        out = out[out["R"] >= min_share * res["R_pol"]].reset_index(drop=True)
    return out


# ----------------------------------------------------------------------------------------------- batch over a dataset
def analyse_all(eis: pd.DataFrame, area_cm2: float, method: str = "mgcv", lam: Optional[float] = None,
                kk_threshold_pct: Optional[float] = None, prominence: float = 0.01, inductance: bool = False,
                progress=None) -> Dict[str, pd.DataFrame]:
    """KK test, DRT and peaks for every spectrum (unit, time, current). Resistances are area-specific (ohm cm2).
    Returns 'spectra' (one row per spectrum), 'peaks' (one row per peak) and 'gamma' (long table for maps)."""
    keys = ["unit", "time_h"] + (["j_A_cm2"] if "j_A_cm2" in eis and eis["j_A_cm2"].notna().any() else [])
    groups = list(eis.groupby(keys))
    spec, peaks, gam = [], [], []
    for n, (k, d) in enumerate(groups):
        if progress:
            progress(n / max(len(groups), 1))
        u, t = k[0], float(k[1])
        jj = float(k[2]) if len(k) > 2 else np.nan
        d = d.sort_values("freq_Hz", ascending=False)
        kk = kk_test(d["freq_Hz"], d["z_re"], d["z_im_neg"], threshold_pct=kk_threshold_pct or 1e9)
        r = fit_drt(d["freq_Hz"], d["z_re"] * area_cm2, d["z_im_neg"] * area_cm2, lam=lam, method=method,
                    inductance=inductance)
        p = find_drt_peaks(r, prominence)
        spec.append(dict(unit=u, time_h=t, j_A_cm2=jj, kk_mean_abs_pct=kk["mean_abs_pct"], kk_valid=kk["valid"],
                         R_inf=r["R_inf"], R_pol=r["R_pol"], lam=r["lam"], fit_residual_pct=100 * r["rel_residual"],
                         n_peaks=len(p)))
        if len(p):
            peaks.append(p.assign(unit=u, time_h=t, j_A_cm2=jj, kk_valid=kk["valid"]))
        gam.append(pd.DataFrame(dict(unit=u, time_h=t, j_A_cm2=jj, ln_tau=r["ln_tau"], gamma=r["gamma"],
                                     gamma_lo=r["gamma_lo"], gamma_hi=r["gamma_hi"])))
    sp = pd.DataFrame(spec)
    if len(sp) and kk_threshold_pct is None:          # adaptive gate: 2 x the unit's median KK residual (min 0.3 %)
        thr = sp.groupby("unit")["kk_mean_abs_pct"].transform(lambda x: max(2 * x.median(), 0.3))
        sp["kk_valid"] = sp["kk_mean_abs_pct"] < thr
        if peaks:
            vmap = sp.set_index(["unit", "time_h", "j_A_cm2"])["kk_valid"]
            for i, p in enumerate(peaks):
                peaks[i] = p.assign(kk_valid=[bool(vmap.get((a, b, c), True)) for a, b, c in
                                               zip(p["unit"], p["time_h"], p["j_A_cm2"])])
    out = dict(spectra=sp,
               peaks=pd.concat(peaks, ignore_index=True) if peaks else pd.DataFrame(),
               gamma=pd.concat(gam, ignore_index=True) if gam else pd.DataFrame())
    return out


def track_peaks(peaks: pd.DataFrame, method: str = "cluster", bands: Optional[Dict[str, Tuple[float, float]]] = None,
                max_jump_decades: float = 1.0, valid_only: bool = True, min_share: float = 0.02,
                spectra: Optional[pd.DataFrame] = None, gap_decades: float = 0.3, persistence: float = 0.5) -> pd.DataFrame:
    """Give every peak a persistent label over time, separately for each unit and EIS current.
    'track': reference = peaks of the first valid spectrum, labelled P1, P2... from high to low frequency; each later
    spectrum is matched to the previous positions (Hungarian assignment, at most `max_jump_decades` away).
    'bands': fixed frequency bands (e.g. from the literature table); several peaks in one band are summed.
    'cluster' (default): pool all peaks of a unit at one current over time, group them in log tau (gaps larger than
    `gap_decades` separate processes), keep the groups present in >= `persistence` of the spectra (real processes),
    label them P1, P2... from high to low frequency and sum split peaks of the same group; rare groups -> 'minor'."""
    from scipy.optimize import linear_sum_assignment
    if peaks is None or not len(peaks):
        return pd.DataFrame()
    pk = peaks[peaks["kk_valid"]] if valid_only and "kk_valid" in peaks else peaks
    if min_share and spectra is not None and len(spectra):      # drop weak peaks (< min_share of R_pol): ringing
        rp = spectra.set_index(["unit", "time_h", "j_A_cm2"])["R_pol"]
        tot = np.array([rp.get((a, b, c), np.nan) for a, b, c in zip(pk["unit"], pk["time_h"], pk["j_A_cm2"])])
        pk = pk[~(pk["R"].to_numpy() < min_share * tot)]
    out = []
    for (u, jj), d in pk.groupby(["unit", "j_A_cm2"], dropna=False):
        times = sorted(d["time_h"].unique())
        if method == "cluster":
            out.extend(_cluster_one(d, u, jj, times, gap_decades, persistence))
            continue
        if method == "bands":
            bands = bands or DEFAULT_BANDS_HZ
            for t in times:
                s = d[d["time_h"] == t]
                for name, (flo, fhi) in bands.items():
                    m = s[(s["f_Hz"] >= flo) & (s["f_Hz"] < fhi)]
                    if len(m):
                        out.append(dict(unit=u, j_A_cm2=jj, time_h=t, peak=name, R=m["R"].sum(), R_lo=m["R_lo"].sum(),
                                        R_hi=m["R_hi"].sum(), tau_s=float(m.loc[m["R"].idxmax(), "tau_s"]),
                                        f_Hz=float(m.loc[m["R"].idxmax(), "f_Hz"])))
            continue
        first = d[d["time_h"] == times[0]].sort_values("tau_s")
        labels = [f"P{k + 1}" for k in range(len(first))]
        ref = dict(zip(labels, np.log10(first["tau_s"].to_numpy())))
        for t in times:
            s = d[d["time_h"] == t].sort_values("tau_s").reset_index(drop=True)
            lt = np.log10(s["tau_s"].to_numpy())
            names = list(ref)
            cost = np.abs(lt[:, None] - np.array([ref[nm] for nm in names])[None, :])
            ri, ci = linear_sum_assignment(cost)
            for a, b in zip(ri, ci):
                if cost[a, b] <= max_jump_decades:
                    nm = names[b]
                    ref[nm] = lt[a]
                    out.append(dict(unit=u, j_A_cm2=jj, time_h=t, peak=nm, R=s.loc[a, "R"], R_lo=s.loc[a, "R_lo"],
                                    R_hi=s.loc[a, "R_hi"], tau_s=s.loc[a, "tau_s"], f_Hz=s.loc[a, "f_Hz"]))
            for a in set(range(len(s))) - {a for a, b in zip(ri, ci) if cost[a, b] <= max_jump_decades}:
                nm = f"P{len(ref) + 1}"
                ref[nm] = lt[a]
                out.append(dict(unit=u, j_A_cm2=jj, time_h=t, peak=nm, R=s.loc[a, "R"], R_lo=s.loc[a, "R_lo"],
                                R_hi=s.loc[a, "R_hi"], tau_s=s.loc[a, "tau_s"], f_Hz=s.loc[a, "f_Hz"], new_peak=True))
    res = pd.DataFrame(out)
    if method == "cluster" and len(res):
        res = _align_across_currents(res)
    return res


def _cluster_one(d, u, jj, times, gap_decades, persistence):
    lt = np.log10(d["tau_s"].to_numpy())
    o = np.argsort(lt)
    grp = np.zeros(len(lt), int)
    gid = 0
    for a, b in zip(o[:-1], o[1:]):
        if lt[b] - lt[a] > gap_decades:
            gid += 1
        grp[b] = gid
    dd = d.assign(_g=grp)
    pres = dd.groupby("_g")["time_h"].nunique() / len(times)
    keep = [g_ for g_ in sorted(pres.index, key=lambda g_: dd.loc[dd["_g"] == g_, "tau_s"].median())
            if pres[g_] >= persistence]
    names = {g_: f"C{k + 1}" for k, g_ in enumerate(keep)}
    rows = []
    for (t, g_), m in dd.groupby(["time_h", "_g"]):
        big = m.loc[m["R"].idxmax()]
        rows.append(dict(unit=u, j_A_cm2=jj, time_h=t, peak=names.get(g_, "minor"), R=m["R"].sum(),
                         R_lo=m["R_lo"].sum(), R_hi=m["R_hi"].sum(), tau_s=big["tau_s"], f_Hz=big["f_Hz"]))
    return rows


def _align(ref: List[float], cur: List[float], gap: float = 0.6) -> Dict[int, int]:
    """Order-preserving alignment of two sorted lists of log-tau positions (sequence alignment, gap penalty)."""
    n, m = len(ref), len(cur)
    Dm = np.zeros((n + 1, m + 1))
    Dm[:, 0] = np.arange(n + 1) * gap
    Dm[0, :] = np.arange(m + 1) * gap
    for i in range(1, n + 1):
        for k in range(1, m + 1):
            Dm[i, k] = min(Dm[i - 1, k - 1] + abs(ref[i - 1] - cur[k - 1]), Dm[i - 1, k] + gap, Dm[i, k - 1] + gap)
    i, k, mp = n, m, {}
    while i > 0 and k > 0:
        if np.isclose(Dm[i, k], Dm[i - 1, k - 1] + abs(ref[i - 1] - cur[k - 1])):
            mp[k - 1] = i - 1
            i, k = i - 1, k - 1
        elif np.isclose(Dm[i, k], Dm[i - 1, k] + gap):
            i -= 1
        else:
            k -= 1
    return mp


def _align_across_currents(res: pd.DataFrame) -> pd.DataFrame:
    """Give clusters the same label at every EIS current of a unit: start from the best-resolved current (most
    persistent clusters; ties -> highest current) and walk to neighbouring currents, aligning cluster positions
    in order (peaks shift with current, so only neighbours are compared). Unmatched clusters -> 'minor'
    (e.g. two processes merged at a low current)."""
    out = []
    for u, d in res.groupby("unit"):
        js = sorted(d["j_A_cm2"].dropna().unique())
        if len(js) < 2:
            out.append(d.assign(peak=d["peak"].str.replace("C", "P", regex=False)))
            continue
        med = {j: d[(d["j_A_cm2"] == j) & (d["peak"] != "minor")].groupby("peak")["tau_s"].median()
               .pipe(lambda s: np.log10(s)).sort_values() for j in js}
        counts = {j: len(med[j]) for j in js}
        jref = max(js, key=lambda j: (counts[j], j))
        labels = {jref: {c: f"P{k + 1}" for k, c in enumerate(med[jref].index)}}
        order = sorted(js, key=lambda j: abs(js.index(j) - js.index(jref)))
        for j in order[1:]:
            nb = min(labels, key=lambda x: abs(js.index(x) - js.index(j)))
            refpos = [med[nb][c] for c in med[nb].index if c in labels[nb]]
            reflab = [labels[nb][c] for c in med[nb].index if c in labels[nb]]
            mp = _align(refpos, list(med[j].values))
            labels[j] = {c: reflab[mp[k]] for k, c in enumerate(med[j].index) if k in mp}
        dd = d.copy()
        dd["peak"] = [labels.get(j, {}).get(p, "minor") for j, p in zip(dd["j_A_cm2"], dd["peak"])]
        out.append(dd)
    return pd.concat(out, ignore_index=True)


def classify_by_current(tracked: pd.DataFrame, spectra: pd.DataFrame) -> pd.DataFrame:
    """Assign physical meaning by behaviour, not by literature frequency tables: slope of log R vs log j across the EIS
    currents at the same time (median over times). Charge transfer: R falls ~1/j (slope ~ -1); ohmic / ion transport:
    current independent (|slope| < 0.1); mass transport / gas: R rises with j (slope > 0.1). Also returns the
    current dependence of the peak frequency."""
    if tracked is None or not len(tracked) or tracked["j_A_cm2"].nunique() < 2:
        return pd.DataFrame()
    rows = []
    for (u, pkn), d in tracked[tracked["peak"] != "minor"].groupby(["unit", "peak"]):
        sl, fs = [], []
        for t, s in d.groupby("time_h"):
            s = s.dropna(subset=["j_A_cm2"])
            if s["j_A_cm2"].nunique() >= 2 and (s["R"] > 0).all():
                sl.append(np.polyfit(np.log(s["j_A_cm2"]), np.log(s["R"]), 1)[0])
                fs.append(np.polyfit(np.log(s["j_A_cm2"]), np.log(s["f_Hz"]), 1)[0])
        if not sl:
            continue
        m = float(np.median(sl))
        kind = ("charge transfer (R ~ 1/j)" if m < -0.5 else "current independent (ohmic / ion transport)"
                if abs(m) < 0.1 else "mass transport / gas (R rises with j)" if m > 0.1 else "mixed")
        rows.append(dict(unit=u, peak=pkn, slope_logR_logj=m, slope_logf_logj=float(np.median(fs)), times=len(sl),
                         assignment=kind, f_Hz_median=float(d["f_Hz"].median())))
    # also the high-frequency resistance
    s = spectra.dropna(subset=["j_A_cm2"]) if spectra is not None and len(spectra) else pd.DataFrame()
    if len(s) and s["j_A_cm2"].nunique() >= 2:
        for u, d in s.groupby("unit"):
            sl = [np.polyfit(np.log(x["j_A_cm2"]), np.log(np.maximum(x["R_inf"], 1e-9)), 1)[0]
                  for _, x in d.groupby("time_h") if x["j_A_cm2"].nunique() >= 2]
            rows.append(dict(unit=u, peak="R_inf", slope_logR_logj=float(np.median(sl)), slope_logf_logj=np.nan,
                             times=len(sl), assignment="ohmic (high-frequency intercept)", f_Hz_median=np.nan))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- ECM from DRT
def fit_ecm_from_drt(freq, z_re, z_im_neg, peaks: pd.DataFrame, R_inf0: float) -> Dict[str, object]:
    """Equivalent circuit R_inf + sum_k ZARC(R_k, tau_k, alpha_k) with one ZARC per DRT peak; DRT gives the number of
    elements and the starting values (DRT-guided, parsimonious ECM). Impedances in the same units as the input."""
    f = np.asarray(freq, float)
    z = np.asarray(z_re, float) - 1j * np.asarray(z_im_neg, float)
    om = 2 * np.pi * f
    n = len(peaks)
    if n == 0:
        return dict(ok=False)
    p0 = [R_inf0] + sum([[max(r, 1e-6), math.log(t), 0.9] for r, t in zip(peaks["R"], peaks["tau_s"])], [])

    def model(p):
        zz = p[0] + 0j
        for k in range(n):
            R, lt, a = p[1 + 3 * k:4 + 3 * k]
            zz = zz + R / (1 + (1j * om * math.exp(lt)) ** a)
        return zz

    def resid(p):
        r = (model(p) - z) / np.abs(z)
        return np.r_[r.real, r.imag]

    lb = [0] + [0, -20, 0.5] * n
    ub = [np.inf] + [np.inf, 10, 1.0] * n
    sol = optimize.least_squares(resid, p0, bounds=(lb, ub))
    els = [dict(R=sol.x[1 + 3 * k], tau_s=math.exp(sol.x[2 + 3 * k]), alpha=sol.x[3 + 3 * k]) for k in range(n)]
    return dict(ok=bool(sol.success), R_inf=float(sol.x[0]), elements=pd.DataFrame(els),
                rel_rmse_pct=float(100 * np.sqrt(np.mean(sol.fun ** 2))), z_fit=model(sol.x))


def auto_peak_map(classification: pd.DataFrame, unit: Optional[str] = None) -> Dict[str, str]:
    """Map tracked peak labels to mechanisms from their measured behaviour: current-independent high-frequency peak ->
    ohmic/ion transport; charge-transfer peaks -> HER (faster, higher frequency) and OER (slower); peaks whose R rises
    with j, or below 1 Hz and not charge transfer -> mass transport / gas. Mechanism names as in degradation.MECHANISMS."""
    c = classification if unit is None else classification[classification["unit"] == unit]
    c = c[c["peak"] != "R_inf"]
    out = {}
    ct = c[c["assignment"].str.startswith("charge")].sort_values("f_Hz_median", ascending=False)
    for k, r in enumerate(ct.itertuples()):
        out[r.peak] = "HER (cathode)" if k == 0 and len(ct) > 1 else "OER + passive film (anode)"
    for r in c.itertuples():
        if r.peak in out:
            continue
        if r.assignment.startswith("mass") or r.f_Hz_median < 1.0:
            out[r.peak] = "mass transport / gas"
        elif r.assignment.startswith("current independent"):
            out[r.peak] = "ohmic + ion transport"
        else:
            out[r.peak] = "OER + passive film (anode)"
    return out
