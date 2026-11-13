"""Parameter registry, sources, condition groups and injected faults.

Every number used by the generator lives here with its unit and its source. Source tags:
  - a literature tag (see SOURCES) when the value or range is taken from a paper;
  - "ASSUMPTION" when no source in hand gives it; the reason is in the note. These are the values to replace first
    when real data or better papers arrive.
"[recalled]" in a note means the value is quoted from memory of the paper and was not re-read here: verify it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

SOURCES = {
    "Hu22": "Hu K. et al., Comparative study of alkaline water electrolysis, PEM and SOEC models, Applied Energy 327 "
            "(2022) 120099 - project file 'AWE mathematical modeling.pdf'",
    "Ham25": "Ham et al., Int. J. Hydrogen Energy 187 (2025) 152069 - 10 kW AWE system model - project file "
             "'1-s2.0-S0360319925050724-main.pdf'",
    "Daoudi24": "Daoudi & Bounahmidi, Int. J. Hydrogen Energy 49 (2024) 646-667 - project file 'AwE modeling.pdf'",
    "Qu26": "Qu et al., Chin. Sci. Bull. 71 (2026) 4267 - project file 'D3300712FF6D47FBAA09F952AC500FCE.pdf'",
    "Ulleberg03": "Ulleberg O., Modeling of advanced alkaline electrolyzers: a system simulation approach, IJHE 28 "
                  "(2003) 21-33. https://scispace.com/papers/modeling-of-advanced-alkaline-electrolyzers-a-system-3oai973smj",
    "Schalenbach16": "Schalenbach M. et al., Hydrogen diffusivity and electrolyte permeability of the Zirfon PERL "
                     "separator, J. Electrochem. Soc. 163 (2016). https://scispace.com/papers/hydrogen-diffusivity-and-"
                     "electrolyte-permeability-of-the-4got3p2y4x",
    "Barros23": "Lira Garcia Barros R. et al., Impact of an electrode-diaphragm gap on diffusive hydrogen crossover in "
                "AWE, IJHE (2023): zero-gap supersaturation 8-80. https://scispace.com/papers/impact-of-an-electrode-"
                "diaphragm-gap-on-diffusive-hydrogen-1wumpj2cqf",
    "Kim22": "Kim Y. et al., Cathodic protection system against a reverse-current after shut-down in zero-gap AWE, "
             "JACS Au (2022): Ni cathode irreversibly oxidised to b-Ni(OH)2/NiO by reverse current. "
             "https://scispace.com/papers/cathodic-protection-system-against-a-reverse-current-after-2xs1ljtf",
    "Uchino18": "Uchino Y. et al., Relationship between redox reactions on a bipolar plate and reverse current after "
                "AWE, Electrocatalysis (2018). https://scispace.com/papers/relationship-between-the-redox-reactions-"
                "on-a-bipolar-plate-1f2il87wxt",
    "OlguinGodoy25": "Olguin Godoy V. et al., Physics-based modeling of degradation in AWE cells due to reverse currents "
                     "(2025). https://scispace.com/papers/physics-based-modeling-of-degradation-in-alkaline-water-"
                     "ahbz4lffl4nf",
    "Rommal85": "Rommal & Moran, Time-dependent energy efficiency losses at nickel cathodes in AWE, J. Electrochem. "
                "Soc. (1985): cathode loss from hydrogen absorption, Tafel slope rises, loss fully recoverable. "
                "https://scispace.com/papers/time-dependent-energy-efficiency-losses-at-nickel-cathodes-57r8qkyb9o",
    "Huot90": "Huot & Brossard, J. Appl. Electrochem. 20 (1990): Ni cathode deactivation in 30 wt% KOH, 70 C, by "
              "hydrogen absorption then metallic-impurity deposition, mainly via Tafel slope increase. "
              "https://scispace.com/papers/in-situ-activation-of-nickel-cathodes-by-sodium-molybdate-axhf9nafom",
    "Buttler18": "Buttler & Spliethoff, Current status of water electrolysis..., Renew. Sust. Energy Rev. 82 (2018) "
                 "2440. https://scispace.com/papers/current-status-of-water-electrolysis-for-energy-storage-grid-"
                 "1mrc48k608",
    "Masel": "Masel R. et al., A high performing zero gap alkaline electrolyzer (60 C, 2000 h, 3-5 uV/h). "
             "https://scispace.com/papers/a-high-performing-zero-gap-alkaline-electrolyzer-81n8aeen9lvw",
    "Karacan22": "Karacan C. et al., Ni-based catalyst coated diaphragms, J. Electrochem. Soc. (2022): 1000 h at 2 V. "
                 "https://scispace.com/papers/fabrication-of-high-performing-and-durable-nickel-based-7w60w3ry",
    "Zhang24": "Zhang T. et al., Oxygen purity in an industrial 50 Nm3/h AWE system (PPS diaphragm), Applied Energy "
               "(2024): HTO limits load to >=40 % at max pressure; HTO responds with ~3 min delay. "
               "https://scispace.com/papers/research-on-oxygen-purity-based-on-industrial-scale-alkaline-4qjnhz3za3",
    "Wang25": "Wang S. et al., HTO changes in an AWE system (10 kW, 1.6 MPa, 90 C), Processes (2025). "
              "https://scispace.com/papers/characterization-of-hydrogen-in-oxygen-changes-in-alkaline-wk6zvmorn0o4",
    "MartinezHincapie23": "Martinez-Hincapie R. et al., ECSA of structured Ni electrodes, Electrochim. Acta (2023): "
                          "Cdl-based ECSA for Ni and its pitfalls. https://scispace.com/papers/the-determination-of-the-"
                          "electrochemically-active-surface-3nlhky7m1t",
    "Liu81": "Liu, Faulds, Cook, Conductivity of KOH electrolyte supersaturated with zincate, J. Electrochem. Soc. "
             "(1981): additive mixture rule for KOH + K salts. https://scispace.com/papers/conductivity-of-koh-"
             "electrolyte-supersaturated-with-zincate-2zxzmoyxkp",
    "Hug93": "Hug, Buss, Brinner, Intermittent operation and operation modeling of an alkaline electrolyzer (1993): "
             "6000 h constant and intermittent operation, Juelich. https://scispace.com/papers/intermittel-t-"
             "operation-and-operation-modeling-of-an-5bx075pz5h",
    "Cell": "Samir's own slides 'Cell and results.pdf': SS single cell, 1000 h at 0.5 A/cm2, 7 M KOH, 30 C",
}


@dataclass(frozen=True)
class P:
    value: object
    unit: str
    source: str
    note: str = ""


# ------------------------------------------------------------------------------------------------ cell design
DESIGN = {
    "area_cm2": P(100.0, "cm2", "ASSUMPTION", "Lab single-cell bench size (Samir's cell is 25-64 cm2 [Cell]); "
                  "area only scales currents and gas flows."),
    "electrodes": P("Ni mesh anode / Ni mesh cathode", "-", "Hu22",
                    "Ni kinetics are the only complete set in the sources (Ham25 eq 5-6, Hu22 Table B3). Samir's "
                    "stainless-steel cell needs fitted kinetics instead."),
    "separator": P("Zirfon PERL UTP 500-type", "-", "Qu26", "Area resistance 0.27 ohm cm2 [Qu26 Table 2]."),
    "sep_thickness_cm": P(0.05, "cm", "Qu26", "Zirfon UTP 500 = 500 um (nominal product thickness)."),
    "sep_porosity": P(0.55, "-", "ASSUMPTION",
                      "Chosen so the structural model (Hu22 eq 24-25) gives 0.27 ohm cm2 at 80 C, 30 wt% KOH, "
                      "matching Qu26 Table 2."),
    "gap_each_side_cm": P(0.05, "cm", "ASSUMPTION", "Narrow-gap cell (0.5 mm per side)."),
    "R_contact0_ohm_cm2": P(0.03, "ohm cm2", "ASSUMPTION", "Electrodes + current collectors + contacts at BoL; no "
                            "value in the sources. Ni bulk resistance is negligible (Hu22 eq 20-22)."),
    "RF0_cathode": P(8.0, "-", "ASSUMPTION", "Roughness factor of a Ni mesh cathode (no value in sources)."),
    "RF0_anode": P(30.0, "-", "ASSUMPTION", "Anode roughness incl. hydroxide layer; larger than cathode so the two "
                   "EIS arcs are separable, as usually observed."),
    "C_specific_uF_cm2": P(40.0, "uF/cm2 (real area)", "MartinezHincapie23",
                           "Common specific capacitance used for Cdl-based ECSA of Ni [recalled convention]; "
                           "MartinezHincapie23 discusses its uncertainty."),
    "q_redox_mC_cm2": P(0.10, "mC/cm2 (real area)", "ASSUMPTION", "Ni(OH)2/NiOOH redox charge per real area."),
    "L_cell_H": P(1e-8, "H", "ASSUMPTION", "Cable/fixture inductance seen in EIS."),
}

# ------------------------------------------------------------------------------------------------ physics constants
PHYS = {
    "F": P(96485.33212, "C/mol", "Hu22"), "R": P(8.314462618, "J/mol/K", "Hu22"),
    "n_e": P(2, "-", "Hu22", "Electrons in RT/(n alpha F), Hu22 eq 18, Ham25 eq 14."),
    "T_ref_kin_K": P(298.15, "K", "ASSUMPTION",
                     "Reference T for j0 (Ham25 eq 5-6) and theta (Hu22 eq 17). Not stated in the papers; the platform "
                     "engine uses the same value because it reproduces Ham25."),
    "P_ref_bar": P(1.0, "bar", "ASSUMPTION", "Reference pressure for the same equations."),
    "j0c_ref": P(1.5e-4, "A/cm2", "Ham25", "Ni cathode, eq 5."), "j0a_ref": P(0.9e-4, "A/cm2", "Ham25", "Ni anode, eq 6."),
    "Ea_c": P(23000.0, "J/mol", "Ham25"), "Ea_a": P(42000.0, "J/mol", "Ham25"),
    "alpha_c": P("0.1175 + 0.00095 T", "-", "Hu22", "Table B3."),
    "alpha_a": P("0.07835 + 0.001 T", "-", "Hu22", "Table B3 (Jang). Gives OER Tafel ~75-80 mV/dec at 60-80 C: "
                 "high end for Ni OER; flagged in the report."),
    "theta": P("0.023 j^0.3 (T/Tref * Pref/P)^(2/3), j in A/m2", "-", "Hu22", "Eq 17; Ham25 eq 13."),
    "j_lim": P(3.0, "A/cm2", "ASSUMPTION", "Limiting current in eta_conc = RT/2F ln(1/(1-j/jlim)); Hu22 sec 3.2.5 says "
               "eta_conc is negligible in AWE: with 3 A/cm2 it is < 5 mV at 0.8 A/cm2."),
    "perm_H2": P(1.85e-11, "mol/(cm s bar)", "Qu26", "Zirfon H2 permeability, Table 2; T dependence not given -> "
                 "constant (ASSUMPTION)."),
    "perm_O2_ratio": P(0.33, "-", "Hu22", "D_O2/D_H2 in 30 wt% KOH, Table C4 (~0.32); equal solubility assumed."),
    "supersaturation": P(2.0, "-", "Barros23", "Narrow gap: between finite gap (~1) and zero gap (8-80)."),
    "f1_FE": P({40: 150.0, 60: 200.0, 80: 250.0}, "mA2/cm4", "Ulleberg03",
               "eta_F = j^2/(f1 + j^2) f2 (Hu22 eq 7) [recalled Ulleberg Table]. f2 set to 1 for a single cell "
               "(Ulleberg's f2 < 1 includes stack shunt currents)."),
    "rho_lye": P(1.28, "kg/L", "ASSUMPTION", "30 wt% KOH density, handbook order of magnitude."),
    "cp_lye": P(3200.0, "J/(kg K)", "ASSUMPTION", "30 wt% KOH heat capacity, order of magnitude; verify."),
    "HTO_shutdown_pct": P(2.0, "%", "Hu22", "Commercial shutdown at 2 % HTO = 50 % of LEL [Hu22 sec 3.3, Qu26 sec 3.3]."),
    "HTO_LEL_pct": P(4.0, "%", "Ham25"),
}

# ------------------------------------------------------------------------------------------------ bench / thermal
BENCH = {
    "flow_L_min": P(0.2, "L/min", "ASSUMPTION", "Lye flow per cell (both half-cells)."),
    "UA_W_K": P(0.35, "W/K", "ASSUMPTION", "Heat loss of cell + lines to ambient."),
    "C_th_J_K": P(8000.0, "J/K", "ASSUMPTION", "Thermal mass of cell + lye loop (Hu22 eq 55 shows how to measure it)."),
    "tau_heat_h": P(0.75, "h", "ASSUMPTION", "Warm-up time constant with lye heater."),
    "tau_cool_h": P(3.0, "h", "ASSUMPTION", "Cool-down time constant when off."),
    "T_amb_C": P(25.0, "C", "ASSUMPTION"),
    "lye_mass_kg": P(10.0, "kg", "ASSUMPTION", "Electrolyte inventory of the loop (sets the KOH sawtooth period)."),
    "topup_band_wt": P(1.0, "wt%", "ASSUMPTION", "Water make-up when KOH rises 1 wt% above nominal."),
    "diag_T_C": P(70.0, "C", "ASSUMPTION", "Reference temperature for diagnostics (all units)."),
    "diag_P_bar": P(1.0, "bar", "ASSUMPTION", "Reference pressure for diagnostics."),
    "diag_every_h": P(250.0, "h", "ASSUMPTION", "Within the 100-250 h requested."),
    "diag_first_h": P(24.0, "h", "ASSUMPTION", "BoL session after 24 h conditioning."),
    "diag_duration_h": P(3.0, "h", "ASSUMPTION"),
    "hours": P(3000, "h", "-", "Requested test length (wall clock)."),
}

# ------------------------------------------------------------------------------------------------ hidden degradation
# Rates are per equivalent operating hour at the reference condition (80 C, 0.4 A/cm2, constant load).
# They are ACCELERATED: the reference group ages ~10x faster than industrial AWE so that 3000 h shows clear ageing.
DEG = {
    "T_ref_deg_K": P(353.15, "K", "ASSUMPTION", "Reference condition for degradation rates."),
    "j_ref_deg": P(0.4, "A/cm2", "ASSUMPTION"),
    "Ea_deg": P(40000.0, "J/mol", "ASSUMPTION", "Arrhenius factor on all electrochemical degradation rates. No "
                "degradation activation energy in the sources (physics sheet sec 4)."),
    "n_j": P(1.0, "-", "ASSUMPTION", "Rates proportional to (j/j_ref)^n_j."),
    # M1 cathode intrinsic (HER) activity
    "k_j0c": P(8e-5, "1/h", "ASSUMPTION", "j0,c factor exp(-k t_eq). Mechanism: Rommal85, Huot90."),
    "k_bc": P(3e-5, "1/h", "ASSUMPTION", "alpha_c / (1 + k t_eq): Tafel slope rises (Rommal85, Huot90)."),
    # M1r reversible hydride loss (cathode)
    "a_hyd": P(0.35, "-", "ASSUMPTION", "Max fractional j0,c loss from H absorption, recovered at stops > 2 h and "
               "electrolyte refresh (Rommal85: loss fully recoverable)."),
    "tau_hyd_h": P(150.0, "h", "ASSUMPTION"),
    # M2 anode intrinsic (OER) activity, ACCELERATES after onset
    "k_j0a": P(6e-5, "1/h", "ASSUMPTION", "Before onset."),
    "k_ba": P(1.5e-5, "1/h", "ASSUMPTION"),
    "accel_factor": P(5.0, "-", "ASSUMPTION", "Anode rate multiplier after onset (known answer to 'when does it speed "
                      "up?'). Onset = damage dose crossing D*."),
    "D_star_h": P(1600.0, "h-eq", "ASSUMPTION", "Dose threshold (equivalent hours + stop dose) for the onset."),
    "w_stop_h": P(6.0, "h-eq per stop", "ASSUMPTION", "Each stop adds this to the onset dose."),
    # M3 ECSA loss (both electrodes)
    "dRF_max": P(0.25, "-", "ASSUMPTION", "Max fractional roughness loss by slow coarsening, tau below."),
    "tau_RF_h": P(2500.0, "h", "ASSUMPTION"),
    # M4 diaphragm fouling, M5 contact degradation
    "k_sep": P(4e-5, "1/h", "ASSUMPTION", "R_sep x (1 + k t_eq): pore blocking (Qu26 sec 1.2 lists separator ageing)."),
    "k_contact": P(1.2e-5, "ohm cm2/h", "ASSUMPTION", "Contact/current-collector degradation."),
    # M6 electrolyte carbonation
    "k_co3": P(0.10, "fraction of K+ as carbonate per 1000 h", "ASSUMPTION",
               "CO2 ingress via vents; no rate in the sources. Carbonate treated as non-conducting (conservative; "
               "Liu81 mixture rule would give it some conductivity)."),
    # M7 intermittent operation (reverse current at shutdown)
    "stop_tau_h": P(2.0, "h", "ASSUMPTION", "Stop dose = 1 - exp(-duration/tau): reverse current flows until the "
                    "electrode charge is discharged (Uchino18, Kim22)."),
    "stop_j0c": P(0.008, "-", "ASSUMPTION", "Fractional irreversible j0,c loss per full stop (Kim22: Ni cathode "
                  "oxidised to b-Ni(OH)2/NiO)."),
    "stop_RF": P(0.002, "-", "ASSUMPTION", "Fractional cathode ECSA loss per full stop."),
    "stop_contact": P(3e-4, "ohm cm2", "ASSUMPTION", "Contact resistance per stop (thermal cycling)."),
    "activation_h": P(10.0, "h", "ASSUMPTION", "BoL conditioning: cathode j0 rises from 60 % to 100 %."),
    "unit_cv": P(0.15, "-", "ASSUMPTION", "Unit-to-unit log-normal spread of rates and BoL parameters."),
}

# ------------------------------------------------------------------------------------------------ measurement
MEAS = {
    "sd_U_V": P(0.0008, "V", "ASSUMPTION", "Hourly-mean voltage noise."),
    "sd_I_rel": P(0.001, "-", "ASSUMPTION"), "sd_T_C": P(0.15, "C", "ASSUMPTION"),
    "sd_P_rel": P(0.005, "-", "ASSUMPTION"), "sd_flow_rel": P(0.02, "-", "ASSUMPTION"),
    "sd_gas_rel": P(0.015, "-", "ASSUMPTION"), "sd_HTO_abs": P(0.01, "%", "ASSUMPTION"),
    "sd_HTO_rel": P(0.04, "-", "ASSUMPTION"), "sd_cond_rel": P(0.01, "-", "ASSUMPTION"),
    "sd_lsv_V": P(0.001, "V", "ASSUMPTION"), "sd_eis_rel": P(0.003, "-", "ASSUMPTION"),
    "sd_cv_rel": P(0.02, "-", "ASSUMPTION"), "sd_titr_wt": P(0.15, "wt%", "ASSUMPTION"),
    "outlier_rate": P(0.001, "per value", "ASSUMPTION", "Spikes in T_in, flow and HTO channels (labelled in truth)."),
    "eis_drift_rate": P(0.05, "per spectrum", "ASSUMPTION", "Non-stationary (KK-invalid) spectra."),
    "dropout_rate": P(0.002, "per value", "ASSUMPTION", "Single NaN readings."),
}

# ------------------------------------------------------------------------------------------------ groups
@dataclass(frozen=True)
class Group:
    name: str
    mode: str          # constant | dynamic
    T_C: float
    P_bar: float
    koh_wt: float
    j_A_cm2: float     # nominal (dynamic: rated)
    n_units: int = 3
    refresh_h: Tuple[float, ...] = (1000.0, 2000.0)
    why: str = ""


GROUPS: List[Group] = [
    Group("G1_T60", "constant", 60.0, 1.0, 30.0, 0.4, why="Temperature level 1 (Hu22: AWE 60-90 C)."),
    Group("G2_T70", "constant", 70.0, 1.0, 30.0, 0.4, why="Temperature level 2."),
    Group("G3_T80_ref", "constant", 80.0, 1.0, 30.0, 0.4, why="Reference: 80 C, 30 wt% (Qu26 sec 1; Hu22 353-363 K)."),
    Group("G4_P16", "constant", 80.0, 16.0, 30.0, 0.4, why="Pressure: 1.6 MPa as in industrial test platforms "
          "(Wang25, Qu26 1.6-3.2 MPa)."),
    Group("G5_KOH25", "constant", 80.0, 1.0, 25.0, 0.4, why="KOH 25 vs 30 wt% (requested 25-30 wt%)."),
    Group("G6_dynamic", "dynamic", 80.0, 1.0, 30.0, 0.8, refresh_h=(1500.0,),
          why="Renewable-like load 0.1-0.8 A/cm2 with start/stops (min load 10-40 % in Hu22 sec 3.3)."),
]

# ------------------------------------------------------------------------------------------------ injected faults
FAULTS = [
    dict(unit="G3_T80_ref-2", kind="resistance_step", t_h=1700.0, magnitude=0.06, unit_of_mag="ohm cm2",
         note="Sudden contact loss (clamping/connector)."),
    dict(unit="G2_T70-3", kind="fast_activity_loss", t_h=2100.0, magnitude=0.2, unit_of_mag="j0c factor over 200 h",
         note="Impurity deposition on cathode: j0,c x0.2 and Tafel slope +15 % in 200 h (Huot90 mechanism)."),
    dict(unit="G1_T60-1", kind="sensor_bad_reading", t_h=1234.0, magnitude=0.35, unit_of_mag="V",
         note="Single bad hourly voltage reading."),
    dict(unit="G4_P16-1", kind="HTO_rise", t_h=2200.0, magnitude=4.0, unit_of_mag="crossover factor over 300 h",
         note="Diaphragm damage: crossover x4 in 300 h; bench trips at HTO > 2 % and the test ends."),
]
