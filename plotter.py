#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified PyROOT plotter for NIsoMuon dimuon-mass and object-validation distributions.

Revision 2 updates:
  - reads the data-driven DY transfer-factor uncertainty from TFDown/TFUp
    templates in RunSyst/NIsoMuon_DYJets_est.root
  - checks every required systematic variation separately for each applicable
    process and selected era; --strict aborts when any required template is missing
  - propagates Run-2 correlations explicitly: experimental and data-driven
    template sources are independent between eras, affected processes within one
    era move coherently, BTV correlated terms are shared within Run2/Run3,
    and the luminosity grouping matches the final counting model
  - follows the final counting-model theory treatment for tt/ST only:
    symmetric-Hessian PDF, shared alphaS, and separate muF/muR nuisances
  - treats data-driven QCD shape as a symmetric absolute-yield uncertainty
    and data-driven DY as constant NF + NFStat + LightJetStat
  - keeps event-yield and differential-cross-section normalisations only
  - retains the manual cumulative-TH1 stack used to avoid ROOT THStack painting
    crashes in some CMSSW/PyROOT releases

Run with no arguments to print the detailed instructions:

    python3 plotter.py

Typical examples:

    python3 plotter.py --era Run2
    python3 plotter.py --era Run3 --variable all
    python3 plotter.py --era Run2 --blind
    python3 plotter.py --era 2018 --qcd-method mc --no-qcd-normalise
    python3 plotter.py --era 2018 --uncertainty syst+stat --strict
    python3 plotter.py --era Run2 --signal-scale 1000

Dimuon Mass

    for era in 2016preVFP 2016postVFP 2017 2018 2022 2022EE 2023 2023BPix Run2 Run3 Run2+3; do python3 plotter.py --era ${era} --blind --variable dimuon_mass --uncertainty syst+stat --draw-signal --xmax 80 --signal-scale 0.01 --ymin 0.01; done

Validation plots
    for era in 2016preVFP 2016postVFP 2017 2018 2022 2022EE 2023 2023BPix Run2 Run3 Run2+3; do python3 plotter.py --qcd-method mc --dy-method mc --era ${era} --blind --variable all --ymin 0.5; done
    for era in 2016preVFP 2016postVFP 2017 2018 2022 2022EE 2023 2023BPix Run2 Run3 Run2+3; do python3 plotter.py --qcd-method mc --dy-method mc --era ${era} --unblind --variable all --ymin 0.5 --dimuon-sign ss; done
    for era in 2016preVFP 2016postVFP 2017 2018 2022 2022EE 2023 2023BPix Run2 Run3 Run2+3; do python3 plotter.py --qcd-method mc --dy-method mc --era ${era} --unblind --variable all --ymin 0.5 --jet-flavour light-jet; done
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys
from array import array
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


RUN2_ERAS: Tuple[str, ...] = ("2016preVFP", "2016postVFP", "2017", "2018")
RUN3_ERAS: Tuple[str, ...] = ("2022", "2022EE", "2023", "2023BPix")
RUN2_LUMI_LABEL_FB = 138
RUN3_LUMI_LABEL_FB = 62
YEARS: Tuple[str, ...] = RUN2_ERAS + RUN3_ERAS
ERA_GROUPS: Dict[str, Tuple[str, ...]] = {
    **{era: (era,) for era in YEARS},
    "Run2": RUN2_ERAS,
    "Run3": RUN3_ERAS,
    "full": YEARS,
    "Run2+3": YEARS,
}
BKG_PROCESSES: Tuple[str, ...] = ("QCD", "tt", "ST", "DY", "Others")


@dataclass(frozen=True)
class VariableSpec:
    key: str
    hist_name: str
    x_title: str
    xmin: float
    xmax: float
    bin_width: float
    divide_by_bin_width: bool


VARIABLE_SPECS: Dict[str, VariableSpec] = {
    "dimuon_mass": VariableSpec("dimuon_mass", "Dilepton_Mass", "m_{#mu^{+}#mu^{-}} [GeV]", 5.0, 120.0, 1.0, True),
    "jet0_pt": VariableSpec("jet0_pt", "Jet_0_Pt", "p_{T}(j_{#mu#mu}) [GeV]", 30.0, 500.0, 10.0, True),
    "jet0_eta": VariableSpec("jet0_eta", "Jet_0_Eta", "#eta(j_{#mu#mu})", -2.4, 2.4, 0.1, False),
    "jet0_phi": VariableSpec("jet0_phi", "Jet_0_Phi", "#phi(j_{#mu#mu})", -3.0, 3.0, 0.2, False),
    "jet1_pt": VariableSpec("jet1_pt", "Jet_1_Pt", "p_{T}(j_{b-tag}) [GeV]", 30.0, 500.0, 10.0, True),
    "jet1_eta": VariableSpec("jet1_eta", "Jet_1_Eta", "#eta(j_{b-tag})", -2.4, 2.4, 0.1, False),
    "jet1_phi": VariableSpec("jet1_phi", "Jet_1_Phi", "#phi(j_{b-tag})", -3.0, 3.0, 0.2, False),
    "mu_lead_pt": VariableSpec("mu_lead_pt", "Lepton_0_Pt", "p_{T}(#mu_{lead}) [GeV]", 50.0, 500.0, 10.0, True),
    "mu_lead_eta": VariableSpec("mu_lead_eta", "Lepton_0_Eta", "#eta(#mu_{lead})", -2.4, 2.4, 0.1, False),
    "mu_lead_phi": VariableSpec("mu_lead_phi", "Lepton_0_Phi", "#phi(#mu_{lead})", -3.2, 3.2, 0.2, False),
    "mu_sub_pt": VariableSpec("mu_sub_pt", "Lepton_1_Pt", "p_{T}(#mu_{sub}) [GeV]", 10.0, 200.0, 10.0, True),
    "mu_sub_eta": VariableSpec("mu_sub_eta", "Lepton_1_Eta", "#eta(#mu_{sub})", -2.4, 2.4, 0.1, False),
    "mu_sub_phi": VariableSpec("mu_sub_phi", "Lepton_1_Phi", "#phi(#mu_{sub})", -3.2, 3.2, 0.2, False),
}

VARIABLE_ALIASES: Dict[str, str] = {
    "mass": "dimuon_mass", "mumu_mass": "dimuon_mass",
    "lepton0_pt": "mu_lead_pt", "lepton0_eta": "mu_lead_eta", "lepton0_phi": "mu_lead_phi",
    "lepton1_pt": "mu_sub_pt", "lepton1_eta": "mu_sub_eta", "lepton1_phi": "mu_sub_phi",
}

# Stack draw order is bottom -> top in the manual stack.
B_TAG_STACK_DRAW_ORDER: Tuple[str, ...] = ("Others", "ST", "tt", "QCD", "DY")
LIGHT_JET_STACK_DRAW_ORDER: Tuple[str, ...] = ("Others", "ST", "tt", "QCD", "DY")

B_TAG_LEGEND_BKG_ORDER: Tuple[str, ...] = ("DY", "QCD", "tt", "ST", "Others")
LIGHT_JET_LEGEND_BKG_ORDER: Tuple[str, ...] = ("DY", "QCD", "tt", "ST", "Others")

EXP_SYST: Dict[str, Tuple[str, str]] = {
    "jer": ("JetResDown", "JetResUp"),
    "jes": ("JetEnDown", "JetEnUp"),
    "pu": ("PUDown", "PUUp"),
    "mu_trig_sf": ("MuonTriggerSFDown", "MuonTriggerSFUp"),
    "mu_id_sf": ("MuonIDSFDown", "MuonIDSFUp"),
    "mu_scale": ("MuonEnDown", "MuonEnUp"),
}

BTAG_CORR_SYST: Dict[str, Tuple[str, str]] = {
    "btag_hf_corr": ("BTagHFCorrDown", "BTagHFCorrUp"),
    "btag_lf_corr": ("BTagLFCorrDown", "BTagLFCorrUp"),
}

BTAG_UNCORR_SYST: Dict[str, Tuple[str, str]] = {
    "btag_hf_uncorr": ("BTagHFUncorrDown", "BTagHFUncorrUp"),
    "btag_lf_uncorr": ("BTagLFUncorrDown", "BTagLFUncorrUp"),
}

L1_PREFIRE_SYST: Tuple[str, str] = ("L1PrefireDown", "L1PrefireUp")

QCD_SYST: Dict[str, Tuple[str, str]] = {
    "QCD_norm": ("NormDown", "NormUp"),
    "QCD_shape": ("ShapeDown", "ShapeUp"),
}

# In the final constant-NF DY model the ROOT producer retains the historical
# TFDown/TFUp directory names, but those templates represent the NF
# numerator-statistics nuisance, not an alternative TF central model.
DY_SYST: Dict[str, Tuple[str, str]] = {
    "DY_NFStat": ("TFDown", "TFUp"),
    "DY_LightJetStat": ("LightJetStatDown", "LightJetStatUp"),
}

# Final generator-theory model used by limit_workflow.py.
THEORY_PROCESSES: Tuple[str, ...] = ("tt", "ST")
PDF_ERROR_PREFIX = "PDFError"
PDF_ERROR_COUNT = 100
ALPHAS_PREFIX = "PDFAlphaS"
ALPHAS_PAIR: Tuple[int, int] = (0, 1)

# SKFlat/SKNano PDFScale convention:
#   0=(1,1), 1=(1,2), 2=(1,0.5), 3=(2,1), 4=(2,2),
#   5=(2,0.5), 6=(0.5,1), 7=(0.5,2), 8=(0.5,0.5).
# The final model keeps muF and muR as separate process-specific nuisances.
SCALE_PREFIX = "PDFScale"
SCALE_COUNT = 9
SCALE_PAIRS: Dict[str, Tuple[int, int]] = {
    "muF": (2, 1),
    "muR": (6, 3),
}

TT_MASS_FACTORS: Dict[str, Tuple[float, float]] = {
    "Run2": (0.973018, 1.027821),
    "Run3": (0.973365, 1.027501),
}


@dataclass
class Config:
    base_dir: str = "/data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13/NIsoMuon"
    era: str = "Run2"
    trigger: str = ""

    variable: str = "dimuon_mass"

    # Region definition.  Defaults intentionally use b-jet mode.
    muon_id: str = "POGMedium"
    jet_id: str = "tight"
    jet_mode: str = "bjet"  # bjet or lightjet
    dilepton_sign: str = "OS"  # OS or SS

    # Data display mode.
    blind: bool = False
    blind_point_mode: str = "data"  # data, asimov, toy
    toy_seed: int = 37829
    blind_low: float = 10.4
    blind_high: float = 80.0
    blind_visible_data_max: float = 9.0

    draw_signal: bool = False
    signal_masses: List[float] = field(default_factory=lambda: [20.0, 50.0])
    signal_scale: float = 1.0
    signal_reference_xsec_pb: float = 1.0
    signal_xsec_pb: float = -1.0

    normalisations: List[str] = field(default_factory=lambda: ["events"])

    xmin: Optional[float] = None
    xmax: Optional[float] = None
    bin_width: Optional[float] = None
    rebin_factor: int = 1
    bin_edges: List[float] = field(default_factory=list)

    qcd_method: str = "data-driven"  # data-driven or mc
    qcd_normalise: bool = True         # single (Data - non-QCD MC) / QCD MC factor
    dy_method: str = "data-driven"   # mc or data-driven
    uncertainty: str = "stat-only"   # stat-only or syst+stat
    draw_systematics: bool = False

    qcd_data_driven_file: str = "NIsoMuon_SS_fit.root"
    qcd_mc_file: str = "NIsoMuon_QCD_Inclusive.root"
    dy_data_driven_file: str = "NIsoMuon_DYJets_est.root"
    dy_mc_file: str = "NIsoMuon_DYJets_Inclusive.root"
    tt_file: str = "NIsoMuon_tt.root"
    st_file: str = "NIsoMuon_ST.root"
    others_file: str = "NIsoMuon_Others.root"

    divide_by_bin_width: bool = True

    output_dir: str = "plots"
    extensions: List[str] = field(default_factory=lambda: ["pdf", "png"])
    save_root: bool = False

    logy: bool = True
    draw_ratio: bool = True
    ratio_min: float = 0.5
    ratio_max: float = 1.5
    ymin: Optional[float] = 10.0
    ymax: Optional[float] = None
    cms_label: str = "Preliminary"

    left_margin: float = 0.120
    right_margin: float = 0.050
    main_top_margin: float = 0.100
    main_bottom_margin: float = 0.030
    no_ratio_bottom_margin: float = 0.130
    ratio_top_margin: float = 0.040
    ratio_bottom_margin: float = 0.350
    y_axis_title_ndc_x: float = 0.04
    main_y_axis_title_size: float = 0.045
    ratio_y_axis_title_size: float = 0.11

    strict: bool = False
    verbose_warnings: bool = True
    verbose_systematics: bool = True


@dataclass
class Uncertainty:
    low: List[float] = field(default_factory=list)
    high: List[float] = field(default_factory=list)


@dataclass
class PlotInputs:
    years: List[str]
    lumi_pb: float
    bkg: Dict[str, object] = field(default_factory=dict)
    bkg_by_year: Dict[str, Dict[str, object]] = field(default_factory=dict)
    bkg_total: Optional[object] = None
    data: Optional[object] = None
    qcd_normalisation_factor: float = 1.0
    signals: List[Tuple[float, object]] = field(default_factory=list)
    stat: Uncertainty = field(default_factory=Uncertainty)
    syst: Uncertainty = field(default_factory=Uncertainty)
    total: Uncertainty = field(default_factory=Uncertainty)
    warnings: List[str] = field(default_factory=list)


class NameFactory:
    def __init__(self) -> None:
        self.counter = 0

    def unique(self, prefix: str) -> str:
        self.counter += 1
        clean = str(prefix).replace("/", "_").replace(" ", "_").replace(".", "p")
        clean = re.sub(r"[^A-Za-z0-9_]+", "_", clean)
        return f"{clean}_{self.counter}"


_NAMES = NameFactory()


def import_root():
    try:
        import ROOT  # type: ignore
    except Exception as exc:  # pragma: no cover - ROOT is normally unavailable outside CMSSW/ROOT env.
        raise RuntimeError(
            "Could not import PyROOT. Run inside a ROOT/CMSSW environment, e.g. cmsenv, "
            f"or install ROOT with Python bindings. Original error: {exc}"
        )
    ROOT.gROOT.SetBatch(True)
    try:
        ROOT.TH1.AddDirectory(False)
    except Exception:
        pass
    return ROOT


def normalise_key(value: str) -> str:
    return value.lower().replace("_", "").replace("-", "").replace("+", "").replace(" ", "")


def canonical_background_method(value: str, *, option_name: str) -> str:
    key = normalise_key(value)
    if key in {"mc", "mconly", "simulation", "sim", "fullymc"}:
        return "mc"
    if key in {"", "data", "datadriven", "dd", "fit", "final", "datafit"}:
        return "data-driven"
    raise ValueError(f"Unknown {option_name}: {value}. Use 'mc' or 'data-driven'.")


def canonical_uncertainty(value: str) -> Tuple[str, bool]:
    key = normalise_key(value)
    if key in {"stat", "statonly", "statistical", "nosyst", "nosystematics"}:
        return "stat-only", False
    if key in {"syststat", "statandsyst", "systandstat", "full", "syst", "systematic", "systematics"}:
        return "syst+stat", True
    raise ValueError(f"Unknown uncertainty mode: {value}. Use 'stat-only' or 'syst+stat'.")


def canonical_jet_mode(value: str) -> str:
    key = normalise_key(value)
    if key in {"b", "bjet", "btag", "btagged"}:
        return "bjet"
    if key in {"light", "lightjet", "lj", "veto", "vetobjet"}:
        return "lightjet"
    raise ValueError(f"Unknown jet mode: {value}. Use 'bjet' or 'lightjet'.")


def canonical_dilepton_sign(value: str) -> str:
    key = normalise_key(value)
    if key in {"os", "oppositesign", "opposite", "opp", "oppositecharge"}:
        return "OS"
    if key in {"ss", "samesign", "same", "samecharge"}:
        return "SS"
    raise ValueError(f"Unknown dilepton sign: {value}. Use 'OS'/'opposite-sign' or 'SS'/'same-sign'.")


def canonical_data_mode(value: str) -> Tuple[bool, str, str]:
    key = normalise_key(value)
    if key in {"unblind", "observed", "datafull", "fulldata"}:
        return False, "data", "unblind"
    if key in {"blind", "blinddata", "blindlowmass", "lowmass", "lowmassdata", "data"}:
        return True, "data", "blind"
    if key in {"asimov", "blindasimov", "unblindasimov", "expected"}:
        return True, "asimov", "asimov"
    if key in {"toy", "blindtoy", "unblindtoy", "pseudodata", "pseudo"}:
        return True, "toy", "toy"
    raise ValueError(
        f"Unknown data mode: {value}. Use blind, blind_asimov, blind_toy, or unblind."
    )


def parse_float_list(value: str) -> List[float]:
    tokens = re.split(r"[,;:\s]+", value.strip())
    out: List[float] = []
    for token in tokens:
        if not token:
            continue
        cleaned = token.replace("GeV", "").replace("gev", "").replace("M-", "").replace("p", ".")
        out.append(float(cleaned))
    if not out:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return out


def parse_str_list(value: str) -> List[str]:
    tokens = [x.strip() for x in re.split(r"[,;:\s]+", value.strip()) if x.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return tokens


def parse_optional_float(value: str) -> Optional[float]:
    key = value.strip().lower()
    if key in {"auto", "none", "-1"}:
        return None
    return float(value)


def mass_label(mass: float) -> str:
    if abs(mass - round(mass)) < 1.0e-9:
        return str(int(round(mass)))
    return f"{mass:.8g}".replace(".", "p")


def base_region(cfg: Config) -> str:
    tag = "BJet" if cfg.jet_mode == "bjet" else "LightJet"
    sign = canonical_dilepton_sign(cfg.dilepton_sign)
    return f"{sign}_{cfg.muon_id}_{cfg.jet_id}_{tag}_NIsoDimuon"


def canonical_variable(value: str) -> str:
    key = value.strip().lower().replace("-", "_")
    key = VARIABLE_ALIASES.get(key, key)
    if key == "all":
        return key
    if key not in VARIABLE_SPECS:
        raise ValueError(
            f"Unknown variable: {value}. Use one of: "
            + ", ".join(["all"] + list(VARIABLE_SPECS))
        )
    return key


def variable_spec(cfg: Config) -> VariableSpec:
    return VARIABLE_SPECS[canonical_variable(cfg.variable)]


def hist_path(cfg: Config, region: str) -> str:
    spec = variable_spec(cfg)
    return f"{region}/{spec.hist_name}___{region}"


def syst_region(cfg: Config, suffix: str) -> str:
    region = base_region(cfg).replace("_NIsoDimuon", "")
    return f"{region}_Syst_{suffix}_NIsoDimuon"


def stack_draw_order(cfg: Config) -> Tuple[str, ...]:
    return LIGHT_JET_STACK_DRAW_ORDER if cfg.jet_mode == "lightjet" else B_TAG_STACK_DRAW_ORDER


def legend_bkg_order(cfg: Config) -> Tuple[str, ...]:
    return LIGHT_JET_LEGEND_BKG_ORDER if cfg.jet_mode == "lightjet" else B_TAG_LEGEND_BKG_ORDER


def is_mass_variable(cfg: Config) -> bool:
    return canonical_variable(cfg.variable) == "dimuon_mass"


def use_qcd_mc(cfg: Config) -> bool:
    # The fitted SS QCD template is a dimuon-mass estimator.  Object-level
    # validation plots therefore compare data with the full MC stack.
    return (not is_mass_variable(cfg)) or cfg.qcd_method == "mc"


def use_dy_mc(cfg: Config) -> bool:
    # The data-driven DY estimator is defined for the dimuon-mass template.
    # Use DY MC for pT/eta/phi validation distributions.
    return (not is_mass_variable(cfg)) or cfg.dy_method == "mc"


def root_file_by_process(cfg: Config) -> Dict[str, str]:
    return {
        "QCD": cfg.qcd_mc_file if use_qcd_mc(cfg) else cfg.qcd_data_driven_file,
        "tt": cfg.tt_file,
        "ST": cfg.st_file,
        "DY": cfg.dy_mc_file if use_dy_mc(cfg) else cfg.dy_data_driven_file,
        "Others": cfg.others_file,
    }


def process_label(cfg: Config) -> Dict[str, str]:
    return {
        "QCD": "QCD" if use_qcd_mc(cfg) else "QCD (SS data-driven)",
        "tt": "t#bar{t}",
        "ST": "single top",
        "DY": "DY" if use_dy_mc(cfg) else "DY (light-jet data-driven)",
        "Others": "Others",
    }


def process_fill_color(ROOT) -> Dict[str, int]:
    return {
        "QCD": ROOT.kAzure - 9,
        "tt": ROOT.kOrange - 2,
        "ST": ROOT.kViolet - 4,
        "DY": ROOT.kGray + 1,
        "Others": ROOT.kSpring - 9,
    }


def signal_colors(ROOT) -> List[int]:
    return [ROOT.kRed + 1, ROOT.kGreen + 2, ROOT.kBlue + 1, ROOT.kViolet + 1, ROOT.kMagenta + 2, ROOT.kCyan + 2]


def lumi_rel_syst(year: str) -> float:
    if year in {"2016preVFP", "2016postVFP"}:
        return 0.012
    if year == "2017":
        return 0.023
    if year == "2018":
        return 0.025
    if year in {"2022", "2022EE"}:
        return 0.014
    if year in {"2023", "2023BPix"}:
        return 0.013
    return 0.0


def lumi_group(year: str) -> str:
    if year.startswith("2016"):
        return "2016"
    if year in {"2022", "2022EE"}:
        return "2022"
    if year in {"2023", "2023BPix"}:
        return "2023"
    return year


def lumi_fb(year: str) -> float:
    values = {
        "2016preVFP": 19.52,
        "2016postVFP": 16.81,
        "2017": 41.48,
        "2018": 59.83,
        "2022": 7.9804,
        "2022EE": 26.6717,
        "2023": 18.064,
        "2023BPix": 9.693,
    }
    return values.get(year, 0.0)


def canonical_era(era_in: str) -> str:
    key = era_in.lower()
    aliases = {
        "run2": "Run2",
        "run3": "Run3",
        "full": "full",
        "run2+3": "Run2+3",
        "run23": "Run2+3",
    }
    return aliases.get(key, era_in)


def years_for_era(era_in: str) -> List[str]:
    era = canonical_era(era_in)
    if era in ERA_GROUPS:
        return list(ERA_GROUPS[era])
    raise ValueError(
        f"Unknown era/year: {era_in}. Use one of: " + ", ".join(ERA_GROUPS)
    )


def lumi_pb_for_years(years: Sequence[str]) -> float:
    return sum(lumi_fb(year) * 1000.0 for year in years)


def lumi_label(era: str, lumi_pb: float) -> str:
    canonical = canonical_era(era)
    if canonical in {"full", "Run2+3"}:
        return (
            f"{RUN2_LUMI_LABEL_FB} fb^{{-1}} (13 TeV) + "
            f"{RUN3_LUMI_LABEL_FB} fb^{{-1}} (13.6 TeV)"
        )
    if canonical == "Run2":
        return f"{RUN2_LUMI_LABEL_FB} fb^{{-1}} (13 TeV)"
    if canonical == "Run3":
        return f"{RUN3_LUMI_LABEL_FB} fb^{{-1}} (13.6 TeV)"
    energy = "13.6 TeV" if canonical in RUN3_ERAS else "13 TeV"
    return f"{lumi_pb / 1000.0:.1f} fb^{{-1}} ({energy})"


def root_dir_for_year(cfg: Config, year: str, collection: str = "") -> str:
    parts = [cfg.base_dir]
    if collection:
        parts.append(collection)
    parts.append(year)
    if cfg.trigger:
        parts.append(cfg.trigger)
    return os.path.join(*parts)


def exists(path: str) -> bool:
    return os.path.exists(path)


def add_warning(cfg: Config, warnings: List[str], msg: str) -> None:
    if cfg.strict:
        raise RuntimeError(msg)
    warnings.append(msg)


def parse_mass_from_signal_file(path: str) -> Optional[float]:
    base = os.path.basename(path)
    match = re.match(r"NIsoMuon_Zp_M-([^./]+(?:p[0-9]+)?(?:\.[0-9]+)?)\.root$", base)
    if not match:
        return None
    try:
        return float(match.group(1).replace("p", "."))
    except ValueError:
        return None


def locate_signal_file(root_dir: str, mass: float) -> str:
    """Locate a nominal signal file without applying warning or strict policy."""
    label = mass_label(mass)
    candidates = [
        os.path.join(root_dir, f"NIsoMuon_Zp_M-{label}.root"),
        os.path.join(root_dir, f"NIsoMuon_Zp_M-{mass:.0f}.root"),
        os.path.join(root_dir, f"NIsoMuon_Zp_M-{mass:.1f}.root"),
    ]
    for path in candidates:
        if exists(path):
            return path

    pattern = os.path.join(root_dir, "NIsoMuon_Zp_M-*.root")
    for path in sorted(glob.glob(pattern)):
        parsed = parse_mass_from_signal_file(path)
        if parsed is not None and abs(parsed - mass) < 1.0e-6:
            return path
    return ""


def find_signal_file(
    cfg: Config,
    root_dir: str,
    mass: float,
    warnings: List[str],
    *,
    report_missing: bool = True,
) -> str:
    path = locate_signal_file(root_dir, mass)
    if path:
        return path
    if report_missing:
        add_warning(cfg, warnings, f"Missing signal file for M-{mass_label(mass)} under {root_dir}")
    return ""


def process_file(
    cfg: Config,
    year: str,
    process: str,
    signal_mass: float,
    syst_suffix: str,
    warnings: List[str],
    *,
    syst_subdir: str = "RunSyst",
    report_missing: bool = True,
) -> str:
    nominal_dir = root_dir_for_year(cfg, year)
    is_syst = bool(syst_suffix)
    selected_dir = root_dir_for_year(cfg, year, syst_subdir) if is_syst else nominal_dir

    if process == "data":
        if is_syst:
            return ""
        return os.path.join(nominal_dir, "data.root")

    if process == "sig":
        sig = find_signal_file(
            cfg, nominal_dir, signal_mass, warnings, report_missing=report_missing
        )
        if not sig:
            return ""
        return os.path.join(selected_dir, os.path.basename(sig)) if is_syst else sig

    files = root_file_by_process(cfg)
    if process not in files:
        if report_missing:
            add_warning(cfg, warnings, f"Unknown process: {process}")
        return ""

    return os.path.join(selected_dir, files[process])


def read_hist(ROOT, filename: str, path: str):
    """Read and detach one histogram, returning (histogram, error message)."""
    if not filename:
        return None, "empty ROOT filename"
    if not exists(filename):
        return None, f"missing ROOT file {filename}"

    f = ROOT.TFile.Open(filename, "READ")
    if not f or f.IsZombie():
        if f:
            f.Close()
        return None, f"could not open ROOT file {filename}"

    h = f.Get(path)
    if not h:
        f.Close()
        return None, f"missing histogram {filename}:{path}"

    out = h.Clone(_NAMES.unique(h.GetName()))
    out.SetDirectory(0)
    f.Close()
    return out, None


def open_hist(
    ROOT,
    cfg: Config,
    filename: str,
    path: str,
    warnings: List[str],
    *,
    report_missing: bool = True,
):
    h, error = read_hist(ROOT, filename, path)
    if error:
        if report_missing:
            add_warning(cfg, warnings, error)
        return None
    return h


def load_year_hist(
    ROOT,
    cfg: Config,
    year: str,
    process: str,
    signal_mass: float,
    warnings: List[str],
    syst_suffix: str = "",
    *,
    syst_subdir: str = "RunSyst",
    report_missing: bool = True,
):
    filename = process_file(
        cfg,
        year,
        process,
        signal_mass,
        syst_suffix,
        warnings,
        syst_subdir=syst_subdir,
        report_missing=report_missing,
    )
    if not filename:
        return None
    region = syst_region(cfg, syst_suffix) if syst_suffix else base_region(cfg)
    return open_hist(
        ROOT,
        cfg,
        filename,
        hist_path(cfg, region),
        warnings,
        report_missing=report_missing,
    )

def sum_hists(hists: Iterable[object], name_prefix: str):
    out = None
    for h in hists:
        if not h:
            continue
        if out is None:
            out = h.Clone(_NAMES.unique(name_prefix))
            out.SetDirectory(0)
        else:
            out.Add(h)
    return out


def sum_background_hists(bkg: Dict[str, object], name_prefix: str):
    return sum_hists([bkg.get(proc) for proc in BKG_PROCESSES], name_prefix)


def combined_hist(
    ROOT,
    cfg: Config,
    years: Sequence[str],
    process: str,
    signal_mass: float,
    warnings: List[str],
    syst_suffix: str = "",
    *,
    syst_subdir: str = "RunSyst",
    report_missing: bool = True,
):
    parts = []
    for year in years:
        h = load_year_hist(
            ROOT,
            cfg,
            year,
            process,
            signal_mass,
            warnings,
            syst_suffix=syst_suffix,
            syst_subdir=syst_subdir,
            report_missing=report_missing,
        )
        if h:
            parts.append(h)
    return sum_hists(parts, f"{process}_combined")

def make_bin_edges(cfg: Config) -> List[float]:
    """Build final plotting edges using variable-specific defaults."""
    if cfg.rebin_factor < 1:
        raise ValueError("rebin_factor must be a positive integer.")

    spec = variable_spec(cfg)
    if cfg.bin_edges:
        if len(cfg.bin_edges) < 2:
            raise ValueError("At least two bin edges are required.")
        for left, right in zip(cfg.bin_edges, cfg.bin_edges[1:]):
            if left >= right:
                raise ValueError("Bin edges must be increasing.")
        base_edges = [float(x) for x in cfg.bin_edges]
    else:
        xmin = spec.xmin if cfg.xmin is None else float(cfg.xmin)
        xmax = spec.xmax if cfg.xmax is None else float(cfg.xmax)
        width = spec.bin_width if cfg.bin_width is None else float(cfg.bin_width)
        if width <= 0.0:
            raise ValueError("bin_width must be positive.")
        if xmax <= xmin:
            raise ValueError("xmax must be larger than xmin.")
        base_edges = []
        x = xmin
        while x < xmax - 1.0e-9:
            base_edges.append(float(x))
            x += width
        base_edges.append(float(xmax))

    if cfg.rebin_factor == 1:
        return base_edges
    n_base_bins = len(base_edges) - 1
    rebinned_edges = [base_edges[0]]
    idx = cfg.rebin_factor
    while idx < n_base_bins:
        rebinned_edges.append(base_edges[idx])
        idx += cfg.rebin_factor
    if rebinned_edges[-1] != base_edges[-1]:
        rebinned_edges.append(base_edges[-1])
    return rebinned_edges


def rebin_hist(h, edges: Sequence[float]):
    if not h:
        return None
    nbins = len(edges) - 1
    arr = array("d", [float(x) for x in edges])
    out = h.Rebin(nbins, _NAMES.unique(f"{h.GetName()}_rebin"), arr)
    out.SetDirectory(0)
    return out


def apply_scale(h, scale: float) -> None:
    if h and abs(scale - 1.0) > 0.0:
        h.Scale(scale)


def apply_bin_width_normalization(cfg: Config, h) -> None:
    if not h or not cfg.divide_by_bin_width or not variable_spec(cfg).divide_by_bin_width:
        return
    for ib in range(1, h.GetNbinsX() + 1):
        width = h.GetXaxis().GetBinWidth(ib)
        if width <= 0.0:
            continue
        h.SetBinContent(ib, h.GetBinContent(ib) / width)
        h.SetBinError(ib, h.GetBinError(ib) / width)


def plot_scale(cfg: Config, norm: str, lumi_pb: float) -> float:
    if norm == "events":
        return 1.0
    if norm == "xsec":
        if canonical_era(cfg.era) in {"full", "Run2+3"}:
            raise ValueError(
                "Run2+Run3 combines 13 and 13.6 TeV data. Use event yields, "
                "not one cross-section normalisation, for --era full/Run2+3."
            )
        return 1.0 / lumi_pb
    raise ValueError(f"Unknown normalisation: {norm}")


def signal_scale(cfg: Config) -> float:
    scale = cfg.signal_scale
    if cfg.signal_xsec_pb >= 0.0:
        scale *= cfg.signal_xsec_pb / cfg.signal_reference_xsec_pb
    return scale


def signal_scale_label(cfg: Config) -> str:
    if cfg.signal_xsec_pb >= 0.0:
        return f"#sigma={cfg.signal_xsec_pb:g} pb, #times {cfg.signal_scale:g}"
    if abs(cfg.signal_scale - 1.0) > 1.0e-12:
        return f"#alpha_{{qZ'}} = {cfg.signal_scale:g}"
    return ""


def y_axis_title(cfg: Config, norm: str) -> str:
    per_width = cfg.divide_by_bin_width and variable_spec(cfg).divide_by_bin_width
    if norm == "events":
        return "Events / GeV" if per_width else "Events / bin"
    if norm == "xsec":
        return "d#sigma/dx [pb/GeV]" if per_width else "#sigma [pb] / bin"
    return ""


def zero_uncertainty(nbins: int) -> Uncertainty:
    return Uncertainty(low=[0.0] * nbins, high=[0.0] * nbins)



def _hist_integral_for_qcd_normalisation(cfg: Config, h) -> float:
    """Return an integral proportional to event yield over the plotted bins.

    Histograms may already have been divided by bin width and/or luminosity.
    The luminosity factor cancels in the QCD normalisation ratio.  If a
    histogram has been divided by bin width, multiply each bin back by its
    width before summing so variable-width plotting does not bias the factor.
    """
    if not h:
        return 0.0

    total = 0.0
    per_width = cfg.divide_by_bin_width and variable_spec(cfg).divide_by_bin_width

    for ib in range(1, h.GetNbinsX() + 1):
        if cfg.blind and cfg.blind_point_mode == "data" and is_mass_variable(cfg):
            # Never use the blinded dimuon-mass interval to derive the QCD NF.
            # Use exactly the bins that remain visible in make_data_graph().
            high_edge = h.GetXaxis().GetBinUpEdge(ib)
            low_edge = h.GetXaxis().GetBinLowEdge(ib)
            if high_edge > cfg.blind_visible_data_max + 1.0e-9 and low_edge < 70.0:
                continue

        value = float(h.GetBinContent(ib))
        if per_width:
            value *= float(h.GetXaxis().GetBinWidth(ib))
        total += value

    return total


def qcd_normalisation_factor(
    cfg: Config,
    data,
    bkg: Dict[str, object],
    warnings: List[str],
) -> float:
    """Compute one QCD MC scale factor: (Data - non-QCD MC) / QCD MC."""
    if not cfg.qcd_normalise:
        return 1.0

    # Do not derive a QCD normalisation factor from observed data when the
    # displayed points are blinded background-only pseudo-data.  In Asimov/toy
    # modes the MC prediction therefore keeps its nominal QCD normalisation.
    if cfg.blind and cfg.blind_point_mode in {"asimov", "toy"}:
        print(
            "[INFO] QCD normalisation from data is disabled in "
            f"{cfg.blind_point_mode} mode; using QCD factor 1."
        )
        return 1.0

    if not use_qcd_mc(cfg):
        # The SS data-driven QCD template already has its own normalisation.
        return 1.0

    qcd = bkg.get("QCD")
    if not data or not qcd:
        add_warning(
            cfg,
            warnings,
            "QCD normalisation requested but data or QCD MC histogram is missing; using factor 1.",
        )
        return 1.0

    n_data = _hist_integral_for_qcd_normalisation(cfg, data)
    n_qcd = _hist_integral_for_qcd_normalisation(cfg, qcd)
    n_nonqcd = 0.0
    for proc in BKG_PROCESSES:
        if proc == "QCD":
            continue
        n_nonqcd += _hist_integral_for_qcd_normalisation(cfg, bkg.get(proc))

    if n_qcd <= 0.0:
        add_warning(
            cfg,
            warnings,
            f"QCD normalisation denominator is non-positive ({n_qcd:g}); using factor 1.",
        )
        return 1.0

    numerator = n_data - n_nonqcd
    if numerator < 0.0:
        add_warning(
            cfg,
            warnings,
            "QCD normalisation gives Data - non-QCD MC < 0. "
            "Using a non-negative QCD factor of 0.",
        )

    factor = max(0.0, numerator / n_qcd)
    print(
        "[INFO] QCD normalisation: "
        f"(Data - non-QCD MC) / QCD MC = {factor:.6g} "
        f"(Data={n_data:.6g}, non-QCD={n_nonqcd:.6g}, QCD={n_qcd:.6g})"
    )
    return factor


def apply_qcd_normalisation(
    cfg: Config,
    bkg: Dict[str, object],
    bkg_by_year: Dict[str, Dict[str, object]],
    data,
    warnings: List[str],
) -> float:
    """Scale nominal QCD MC by one common factor for the selected era group."""
    factor = qcd_normalisation_factor(cfg, data, bkg, warnings)
    if abs(factor - 1.0) < 1.0e-15:
        return factor

    qcd = bkg.get("QCD")
    if qcd:
        qcd.Scale(factor)

    for by_proc in bkg_by_year.values():
        h = by_proc.get("QCD")
        if h:
            h.Scale(factor)

    return factor


def bkg_stat_uncertainty(cfg: Config, bkg: Dict[str, object]) -> Uncertainty:
    if not bkg:
        return Uncertainty()
    first = next(iter(bkg.values()))
    n = first.GetNbinsX()
    e2 = [0.0] * n

    for proc, h in bkg.items():
        # The fitted data-driven QCD template is controlled by QCD_norm and
        # QCD_shape, not by its stored TH1 bin errors.  Likewise, the final
        # constant-NF DY model uses DY_NFStat and DY_LightJetStat and explicitly
        # has no generic DY_stat term.
        if proc == "QCD" and not use_qcd_mc(cfg):
            continue
        if proc == "DY" and not use_dy_mc(cfg):
            continue
        for ib in range(1, n + 1):
            err = h.GetBinError(ib)
            e2[ib - 1] += err * err

    err = [math.sqrt(x) for x in e2]
    return Uncertainty(low=err[:], high=err[:])


def add_delta_pair_shift(delta_down, delta_up, down2: List[float], up2: List[float]) -> None:
    """Add one correlated up/down source represented by varied-minus-nominal shifts."""
    if not delta_down or not delta_up:
        return
    n = min(delta_down.GetNbinsX(), delta_up.GetNbinsX(), len(down2), len(up2))
    for ib in range(1, n + 1):
        dd = float(delta_down.GetBinContent(ib))
        du = float(delta_up.GetBinContent(ib))
        up = max(0.0, du, dd)
        down = max(0.0, -du, -dd)
        up2[ib - 1] += up * up
        down2[ib - 1] += down * down


def add_symmetric_hist_shift(h, rel: float, down2: List[float], up2: List[float]) -> None:
    if not h or rel <= 0.0:
        return
    for ib in range(1, h.GetNbinsX() + 1):
        delta = abs(rel * h.GetBinContent(ib))
        up2[ib - 1] += delta * delta
        down2[ib - 1] += delta * delta


def add_variation_envelope_shift(
    nominal,
    variations: Sequence[object],
    down2: List[float],
    up2: List[float],
) -> None:
    if not nominal or not variations:
        return
    for ib in range(1, nominal.GetNbinsX() + 1):
        nom = float(nominal.GetBinContent(ib))
        deltas = [float(h.GetBinContent(ib)) - nom for h in variations if h]
        if not deltas:
            continue
        up = max([0.0] + deltas)
        down = max([0.0] + [-delta for delta in deltas])
        up2[ib - 1] += up * up
        down2[ib - 1] += down * down


def add_symmetric_delta_pair_shift(
    delta_down,
    delta_up,
    down2: List[float],
    up2: List[float],
) -> None:
    # Symmetric source using the largest absolute +/-1 sigma shift.
    if not delta_down or not delta_up:
        return
    n = min(delta_down.GetNbinsX(), delta_up.GetNbinsX(), len(down2), len(up2))
    for ib in range(1, n + 1):
        sigma = max(
            abs(float(delta_down.GetBinContent(ib))),
            abs(float(delta_up.GetBinContent(ib))),
        )
        down2[ib - 1] += sigma * sigma
        up2[ib - 1] += sigma * sigma


def hessian_sigma_hist(nominal, variations: Sequence[object], name_prefix: str):
    # sqrt(sum_i (variation_i - nominal)^2), bin by bin.
    if not nominal:
        return None
    valid = [h for h in variations if h]
    if not valid:
        return None
    out = nominal.Clone(_NAMES.unique(name_prefix))
    out.SetDirectory(0)
    out.Reset("ICESM")
    for ib in range(1, nominal.GetNbinsX() + 1):
        nom = float(nominal.GetBinContent(ib))
        sigma2 = 0.0
        for varied in valid:
            delta = float(varied.GetBinContent(ib)) - nom
            sigma2 += delta * delta
        out.SetBinContent(ib, math.sqrt(max(0.0, sigma2)))
        out.SetBinError(ib, 0.0)
    return out


def add_positive_sigma_hist_shift(
    sigma_hist,
    down2: List[float],
    up2: List[float],
) -> None:
    if not sigma_hist:
        return
    for ib in range(1, sigma_hist.GetNbinsX() + 1):
        sigma = abs(float(sigma_hist.GetBinContent(ib)))
        down2[ib - 1] += sigma * sigma
        up2[ib - 1] += sigma * sigma


def make_delta_hist(varied, nominal, name_prefix: str):
    if not varied or not nominal:
        return None
    out = varied.Clone(_NAMES.unique(name_prefix))
    out.SetDirectory(0)
    out.Add(nominal, -1.0)
    return out


def prepare_year_hist(
    ROOT,
    cfg: Config,
    year: str,
    proc: str,
    signal_mass: float,
    edges: Sequence[float],
    scale: float,
    warnings: List[str],
    *,
    syst_suffix: str = "",
    syst_subdir: str = "RunSyst",
    report_missing: bool = True,
):
    h = load_year_hist(
        ROOT,
        cfg,
        year,
        proc,
        signal_mass,
        warnings,
        syst_suffix=syst_suffix,
        syst_subdir=syst_subdir,
        report_missing=report_missing,
    )
    if not h:
        return None
    h = rebin_hist(h, edges)
    apply_scale(h, scale)
    apply_bin_width_normalization(cfg, h)
    return h


def _missing_pair_sides(h_down, h_up) -> List[str]:
    missing: List[str] = []
    if not h_down:
        missing.append("Down")
    if not h_up:
        missing.append("Up")
    return missing


def _format_missing_by_year(missing_by_year: Dict[str, List[str]]) -> str:
    return ", ".join(
        f"{year}:{'/'.join(sides)}" for year, sides in missing_by_year.items()
    )


def _record_pair_status(
    summary: List[str],
    errors: List[str],
    source: str,
    process: str,
    missing_by_year: Dict[str, List[str]],
) -> None:
    key = f"{source}/{process}"
    if not missing_by_year:
        summary.append(f"{key}: OK")
        return
    detail = _format_missing_by_year(missing_by_year)
    summary.append(f"{key}: MISSING ({detail})")
    errors.append(
        f"Required systematic templates are missing for {key}: {detail}. "
        "The nominal template is substituted for each missing variation side in non-strict mode."
    )


def _record_indexed_status(
    summary: List[str],
    errors: List[str],
    source: str,
    process: str,
    found: int,
    expected: int,
    missing: Sequence[str],
) -> None:
    key = f"{source}/{process}"
    if found == expected:
        summary.append(f"{key}: OK ({found}/{expected})")
        return
    preview = ", ".join(missing[:8])
    if len(missing) > 8:
        preview += f", ... +{len(missing) - 8}"
    summary.append(f"{key}: MISSING ({found}/{expected}; {preview})")
    errors.append(
        f"Required indexed systematic templates are missing for {key}: "
        f"found {found}/{expected}; missing {preview}. "
        "The nominal era template is substituted in non-strict mode."
    )


def _finish_syst_audit(
    cfg: Config,
    warnings: List[str],
    title: str,
    summary: Sequence[str],
    errors: Sequence[str],
) -> None:
    if cfg.verbose_systematics:
        print(title)
        for item in summary:
            print(f"  - {item}")
    warnings.extend(errors)
    if cfg.strict and errors:
        shown = list(errors[:20])
        details = "\n".join(f"  - {item}" for item in shown)
        if len(errors) > len(shown):
            details += f"\n  - ... {len(errors) - len(shown)} more"
        raise RuntimeError(
            f"Strict systematic check failed with {len(errors)} missing requirement(s):\n{details}"
        )


def background_detector_processes(cfg: Config) -> List[str]:
    return ["tt", "ST", "Others"]


def background_lumi_processes(cfg: Config) -> List[str]:
    processes = ["tt", "ST", "Others"]
    if use_dy_mc(cfg):
        processes.append("DY")
    if use_qcd_mc(cfg):
        processes.append("QCD")
    return processes


def run_group(year: str) -> str:
    return "Run2" if year in RUN2_ERAS else "Run3"



def _scaled_delta(nominal, factor: float, name_prefix: str):
    if not nominal:
        return None
    out = nominal.Clone(_NAMES.unique(name_prefix))
    out.SetDirectory(0)
    out.Scale(float(factor) - 1.0)
    return out


def _load_indexed_theory_variations(
    ROOT,
    cfg: Config,
    years: Sequence[str],
    proc: str,
    nominal_by_year: Dict[str, Dict[str, object]],
    edges: Sequence[float],
    scale: float,
    prefix: str,
    count: int,
    source_label: str,
    summary: List[str],
    errors: List[str],
) -> List[object]:
    variations: List[object] = []
    found = 0
    missing: List[str] = []

    for idx in range(count):
        parts: List[object] = []
        for year in years:
            nominal = nominal_by_year.get(year, {}).get(proc)
            h_var = prepare_year_hist(
                ROOT,
                cfg,
                year,
                proc,
                -1.0,
                edges,
                scale,
                [],
                syst_suffix=f"{prefix}{idx}",
                syst_subdir="RunXSecSyst",
                report_missing=False,
            )
            if h_var:
                parts.append(h_var)
                found += 1
            else:
                missing.append(f"{year}/{prefix}{idx}")
                if nominal:
                    fallback = nominal.Clone(
                        _NAMES.unique(f"{source_label}_{proc}_{year}_{idx}_nominal")
                    )
                    fallback.SetDirectory(0)
                    parts.append(fallback)

        combined = sum_hists(parts, f"{source_label}_{proc}_{idx}_combined")
        if combined:
            variations.append(combined)

    _record_indexed_status(
        summary,
        errors,
        source_label,
        proc,
        found,
        len(years) * count,
        missing,
    )
    return variations


def _generator_theory_uncertainty(
    ROOT,
    cfg: Config,
    years: Sequence[str],
    bkg_nom: Dict[str, object],
    nominal_by_year: Dict[str, Dict[str, object]],
    edges: Sequence[float],
    scale: float,
    down2: List[float],
    up2: List[float],
    summary: List[str],
    errors: List[str],
) -> None:
    # PDF: NNPDF31 symmetric-Hessian quadrature.  One PDF nuisance is shared
    # by tt and ST, so their positive 1-sigma responses add linearly first.
    pdf_sigma_parts: List[object] = []
    for proc in THEORY_PROCESSES:
        nominal = bkg_nom.get(proc)
        if not nominal:
            continue
        variations = _load_indexed_theory_variations(
            ROOT, cfg, years, proc, nominal_by_year, edges, scale,
            PDF_ERROR_PREFIX, PDF_ERROR_COUNT, "PDF_error", summary, errors,
        )
        sigma_hist = hessian_sigma_hist(
            nominal, variations, f"pdf_hessian_sigma_{proc}"
        )
        if sigma_hist:
            pdf_sigma_parts.append(sigma_hist)

    add_positive_sigma_hist_shift(
        sum_hists(pdf_sigma_parts, "pdf_hessian_shared_tt_ST"),
        down2,
        up2,
    )

    # alpha_s: one shared nuisance across tt and ST.
    alphas_down_parts: List[object] = []
    alphas_up_parts: List[object] = []
    for proc in THEORY_PROCESSES:
        nominal = bkg_nom.get(proc)
        if not nominal:
            continue
        variations = _load_indexed_theory_variations(
            ROOT, cfg, years, proc, nominal_by_year, edges, scale,
            ALPHAS_PREFIX, 2, "PDF_alphas", summary, errors,
        )
        if len(variations) < 2:
            continue
        alphas_down_parts.append(
            make_delta_hist(variations[ALPHAS_PAIR[0]], nominal, f"alphas_{proc}_down")
        )
        alphas_up_parts.append(
            make_delta_hist(variations[ALPHAS_PAIR[1]], nominal, f"alphas_{proc}_up")
        )

    add_delta_pair_shift(
        sum_hists(alphas_down_parts, "alphas_shared_down"),
        sum_hists(alphas_up_parts, "alphas_shared_up"),
        down2,
        up2,
    )

    # Scale: audit all nine members, use only process-specific muF/muR pairs.
    for proc in THEORY_PROCESSES:
        nominal = bkg_nom.get(proc)
        if not nominal:
            continue
        variations = _load_indexed_theory_variations(
            ROOT, cfg, years, proc, nominal_by_year, edges, scale,
            SCALE_PREFIX, SCALE_COUNT, "PDF_scale", summary, errors,
        )
        if len(variations) < SCALE_COUNT:
            continue
        for direction, (down_idx, up_idx) in SCALE_PAIRS.items():
            delta_down = make_delta_hist(
                variations[down_idx], nominal, f"scale_{direction}_{proc}_down"
            )
            delta_up = make_delta_hist(
                variations[up_idx], nominal, f"scale_{direction}_{proc}_up"
            )
            add_delta_pair_shift(delta_down, delta_up, down2, up2)
            summary.append(
                f"QCDscale_{direction}/{proc}: OK "
                f"(PDFScale{down_idx}/PDFScale{up_idx}; correlated across eras)"
            )


def bkg_syst_uncertainty(
    ROOT,
    cfg: Config,
    years: Sequence[str],
    bkg_nom: Dict[str, object],
    bkg_nom_by_year: Dict[str, Dict[str, object]],
    bkg_total,
    edges: Sequence[float],
    scale: float,
    warnings: List[str],
    qcd_norm_factor: float = 1.0,
) -> Uncertainty:
    n = bkg_total.GetNbinsX()
    down2 = [0.0] * n
    up2 = [0.0] * n

    if not cfg.draw_systematics:
        return zero_uncertainty(n)

    summary: List[str] = []
    errors: List[str] = []
    exp_processes = background_detector_processes(cfg)

    # JER/JES/PU/muon sources: coherent across affected processes inside one
    # era, independent between eras.
    for syst_name, (down_suffix, up_suffix) in EXP_SYST.items():
        missing_by_proc: Dict[str, Dict[str, List[str]]] = {proc: {} for proc in exp_processes}
        for year in years:
            delta_down_parts: List[object] = []
            delta_up_parts: List[object] = []
            for proc in exp_processes:
                nominal = bkg_nom_by_year.get(year, {}).get(proc)
                h_down = prepare_year_hist(
                    ROOT, cfg, year, proc, -1.0, edges, scale, warnings,
                    syst_suffix=down_suffix, report_missing=False,
                )
                h_up = prepare_year_hist(
                    ROOT, cfg, year, proc, -1.0, edges, scale, warnings,
                    syst_suffix=up_suffix, report_missing=False,
                )
                missing = _missing_pair_sides(h_down, h_up)
                if not nominal:
                    missing_by_proc[proc][year] = ["Nominal"]
                    continue
                if missing:
                    missing_by_proc[proc][year] = missing
                h_down = h_down or nominal
                h_up = h_up or nominal
                delta_down_parts.append(
                    make_delta_hist(h_down, nominal, f"{syst_name}_{proc}_{year}_down_delta")
                )
                delta_up_parts.append(
                    make_delta_hist(h_up, nominal, f"{syst_name}_{proc}_{year}_up_delta")
                )

            add_delta_pair_shift(
                sum_hists(delta_down_parts, f"{syst_name}_{year}_down_delta"),
                sum_hists(delta_up_parts, f"{syst_name}_{year}_up_delta"),
                down2,
                up2,
            )

        for proc in exp_processes:
            _record_pair_status(summary, errors, syst_name, proc, missing_by_proc[proc])

    # L1 ECAL prefiring exists only in 2016pre/postVFP and 2017 and is treated
    # as an era-specific source.
    for year in years:
        if year not in {"2016preVFP", "2016postVFP", "2017", "2018"}:
            continue
        down_suffix, up_suffix = L1_PREFIRE_SYST
        missing_by_proc: Dict[str, Dict[str, List[str]]] = {proc: {} for proc in exp_processes}
        delta_down_parts: List[object] = []
        delta_up_parts: List[object] = []
        for proc in exp_processes:
            nominal = bkg_nom_by_year.get(year, {}).get(proc)
            h_down = prepare_year_hist(
                ROOT, cfg, year, proc, -1.0, edges, scale, warnings,
                syst_suffix=down_suffix, report_missing=False,
            )
            h_up = prepare_year_hist(
                ROOT, cfg, year, proc, -1.0, edges, scale, warnings,
                syst_suffix=up_suffix, report_missing=False,
            )
            missing = _missing_pair_sides(h_down, h_up)
            if not nominal:
                missing_by_proc[proc][year] = ["Nominal"]
                continue
            if missing:
                missing_by_proc[proc][year] = missing
            h_down = h_down or nominal
            h_up = h_up or nominal
            delta_down_parts.append(make_delta_hist(h_down, nominal, f"prefire_{proc}_{year}_down"))
            delta_up_parts.append(make_delta_hist(h_up, nominal, f"prefire_{proc}_{year}_up"))

        add_delta_pair_shift(
            sum_hists(delta_down_parts, f"prefire_{year}_down"),
            sum_hists(delta_up_parts, f"prefire_{year}_up"),
            down2,
            up2,
        )
        for proc in exp_processes:
            _record_pair_status(summary, errors, f"l1prefire_{year}", proc, missing_by_proc[proc])

    # BTV uncorrelated components are independent for every era.
    for syst_name, (down_suffix, up_suffix) in BTAG_UNCORR_SYST.items():
        missing_by_proc: Dict[str, Dict[str, List[str]]] = {proc: {} for proc in exp_processes}
        for year in years:
            delta_down_parts: List[object] = []
            delta_up_parts: List[object] = []
            for proc in exp_processes:
                nominal = bkg_nom_by_year.get(year, {}).get(proc)
                h_down = prepare_year_hist(
                    ROOT, cfg, year, proc, -1.0, edges, scale, warnings,
                    syst_suffix=down_suffix, report_missing=False,
                )
                h_up = prepare_year_hist(
                    ROOT, cfg, year, proc, -1.0, edges, scale, warnings,
                    syst_suffix=up_suffix, report_missing=False,
                )
                missing = _missing_pair_sides(h_down, h_up)
                if not nominal:
                    missing_by_proc[proc][year] = ["Nominal"]
                    continue
                if missing:
                    missing_by_proc[proc][year] = missing
                h_down = h_down or nominal
                h_up = h_up or nominal
                delta_down_parts.append(make_delta_hist(h_down, nominal, f"{syst_name}_{proc}_{year}_down"))
                delta_up_parts.append(make_delta_hist(h_up, nominal, f"{syst_name}_{proc}_{year}_up"))
            add_delta_pair_shift(
                sum_hists(delta_down_parts, f"{syst_name}_{year}_down"),
                sum_hists(delta_up_parts, f"{syst_name}_{year}_up"),
                down2,
                up2,
            )
        for proc in exp_processes:
            _record_pair_status(summary, errors, syst_name, proc, missing_by_proc[proc])

    # BTV correlated components are shared within Run 2 and within Run 3, while
    # Run-2 and Run-3 components remain independent from each other.
    for syst_name, (down_suffix, up_suffix) in BTAG_CORR_SYST.items():
        missing_by_proc: Dict[str, Dict[str, List[str]]] = {proc: {} for proc in exp_processes}
        for group in ("Run2", "Run3"):
            group_years = [year for year in years if run_group(year) == group]
            if not group_years:
                continue
            delta_down_parts: List[object] = []
            delta_up_parts: List[object] = []
            for year in group_years:
                for proc in exp_processes:
                    nominal = bkg_nom_by_year.get(year, {}).get(proc)
                    h_down = prepare_year_hist(
                        ROOT, cfg, year, proc, -1.0, edges, scale, warnings,
                        syst_suffix=down_suffix, report_missing=False,
                    )
                    h_up = prepare_year_hist(
                        ROOT, cfg, year, proc, -1.0, edges, scale, warnings,
                        syst_suffix=up_suffix, report_missing=False,
                    )
                    missing = _missing_pair_sides(h_down, h_up)
                    if not nominal:
                        missing_by_proc[proc][year] = ["Nominal"]
                        continue
                    if missing:
                        missing_by_proc[proc][year] = missing
                    h_down = h_down or nominal
                    h_up = h_up or nominal
                    delta_down_parts.append(make_delta_hist(h_down, nominal, f"{syst_name}_{proc}_{year}_down"))
                    delta_up_parts.append(make_delta_hist(h_up, nominal, f"{syst_name}_{proc}_{year}_up"))
            add_delta_pair_shift(
                sum_hists(delta_down_parts, f"{syst_name}_{group}_down"),
                sum_hists(delta_up_parts, f"{syst_name}_{group}_up"),
                down2,
                up2,
            )
        for proc in exp_processes:
            _record_pair_status(summary, errors, syst_name, proc, missing_by_proc[proc])

    # Data-driven QCD nuisances are independent by era.
    # QCD_norm remains the ordinary pair.  QCD_shape follows the final additive
    # absolute-yield Gaussian: sigma=max(|down-nom|, |up-nom|), symmetrically.
    if not use_qcd_mc(cfg):
        for syst_name, (down_suffix, up_suffix) in QCD_SYST.items():
            missing_by_year: Dict[str, List[str]] = {}
            for year in years:
                nominal = bkg_nom_by_year.get(year, {}).get("QCD")
                h_down = prepare_year_hist(
                    ROOT, cfg, year, "QCD", -1.0, edges, scale, warnings,
                    syst_suffix=down_suffix, report_missing=False,
                )
                h_up = prepare_year_hist(
                    ROOT, cfg, year, "QCD", -1.0, edges, scale, warnings,
                    syst_suffix=up_suffix, report_missing=False,
                )
                missing = _missing_pair_sides(h_down, h_up)
                if not nominal:
                    missing_by_year[year] = ["Nominal"]
                    continue
                if missing:
                    missing_by_year[year] = missing
                h_down = h_down or nominal
                h_up = h_up or nominal

                delta_down = make_delta_hist(
                    h_down, nominal, f"{syst_name}_{year}_down_delta"
                )
                delta_up = make_delta_hist(
                    h_up, nominal, f"{syst_name}_{year}_up_delta"
                )
                if syst_name == "QCD_shape":
                    add_symmetric_delta_pair_shift(delta_down, delta_up, down2, up2)
                else:
                    add_delta_pair_shift(delta_down, delta_up, down2, up2)
            _record_pair_status(summary, errors, syst_name, "QCD", missing_by_year)

    # Final data-driven DY uses constant NF; TFDown/TFUp are DY_NFStat,\n    # together with the independent DY_LightJetStat pair.\n    if not use_dy_mc(cfg):
        for syst_name, (down_suffix, up_suffix) in DY_SYST.items():
            missing_by_year: Dict[str, List[str]] = {}
            for year in years:
                nominal = bkg_nom_by_year.get(year, {}).get("DY")
                h_down = prepare_year_hist(
                    ROOT, cfg, year, "DY", -1.0, edges, scale, warnings,
                    syst_suffix=down_suffix, report_missing=False,
                )
                h_up = prepare_year_hist(
                    ROOT, cfg, year, "DY", -1.0, edges, scale, warnings,
                    syst_suffix=up_suffix, report_missing=False,
                )
                missing = _missing_pair_sides(h_down, h_up)
                if not nominal:
                    missing_by_year[year] = ["Nominal"]
                    continue
                if missing:
                    missing_by_year[year] = missing
                h_down = h_down or nominal
                h_up = h_up or nominal
                add_delta_pair_shift(
                    make_delta_hist(h_down, nominal, f"{syst_name}_{year}_down_delta"),
                    make_delta_hist(h_up, nominal, f"{syst_name}_{year}_up_delta"),
                    down2,
                    up2,
                )
            _record_pair_status(summary, errors, syst_name, "DY", missing_by_year)

    # Final generator theory: tt/ST only.  No generic tt_xsec or ST_xsec.
    _generator_theory_uncertainty(
        ROOT,
        cfg,
        years,
        bkg_nom,
        bkg_nom_by_year,
        edges,
        scale,
        down2,
        up2,
        summary,
        errors,
    )

    # One common tt-mass nuisance; 13 and 13.6 TeV use their corresponding
    # Top++ response factors, but the nuisance direction is shared.
    tt_mass_down_parts: List[object] = []
    tt_mass_up_parts: List[object] = []
    for year in years:
        nominal = bkg_nom_by_year.get(year, {}).get("tt")
        if not nominal:
            continue
        down_factor, up_factor = TT_MASS_FACTORS[run_group(year)]
        tt_mass_down_parts.append(_scaled_delta(nominal, down_factor, f"tt_mass_{year}_down"))
        tt_mass_up_parts.append(_scaled_delta(nominal, up_factor, f"tt_mass_{year}_up"))
    add_delta_pair_shift(
        sum_hists(tt_mass_down_parts, "tt_mass_down_combined"),
        sum_hists(tt_mass_up_parts, "tt_mass_up_combined"),
        down2,
        up2,
    )
    summary.append("tt_mass/tt: OK (common nuisance, energy-dependent response)")

    # Luminosity: 2016pre/post share one source; 2017 and 2018 are separate;
    # 2022/2022EE share one source and 2023/2023BPix share one source.
    for group in sorted({lumi_group(year) for year in years}):
        parts: List[object] = []
        rel = 0.0
        for year in years:
            if lumi_group(year) != group:
                continue
            rel = max(rel, lumi_rel_syst(year))
            for proc in background_lumi_processes(cfg):
                h = bkg_nom_by_year.get(year, {}).get(proc)
                if h:
                    parts.append(h)
        h_group = sum_hists(parts, f"mc_lumi_{group}")
        if h_group:
            add_symmetric_hist_shift(h_group, rel, down2, up2)
            summary.append(f"lumi_{group}/MC: OK")

    _finish_syst_audit(
        cfg,
        warnings,
        "[syst-check] non-stat background uncertainty sources",
        summary,
        errors,
    )
    return Uncertainty(low=[math.sqrt(x) for x in down2], high=[math.sqrt(x) for x in up2])


def total_uncertainty(stat: Uncertainty, syst: Uncertainty) -> Uncertainty:
    n = len(stat.low)
    return Uncertainty(
        low=[math.sqrt(stat.low[i] ** 2 + syst.low[i] ** 2) for i in range(n)],
        high=[math.sqrt(stat.high[i] ** 2 + syst.high[i] ** 2) for i in range(n)],
    )


def build_plot_inputs(ROOT, cfg: Config, norm: str, edges: Sequence[float]) -> PlotInputs:
    years = years_for_era(cfg.era)
    lumi_pb = lumi_pb_for_years(years)
    scale = plot_scale(cfg, norm, lumi_pb)
    out = PlotInputs(years=years, lumi_pb=lumi_pb)

    bkg_by_year: Dict[str, Dict[str, object]] = {year: {} for year in years}
    for year in years:
        for proc in BKG_PROCESSES:
            h = prepare_year_hist(
                ROOT, cfg, year, proc, -1.0, edges, scale, out.warnings
            )
            if h:
                bkg_by_year[year][proc] = h
    out.bkg_by_year = bkg_by_year

    for proc in BKG_PROCESSES:
        h = sum_hists(
            [bkg_by_year.get(year, {}).get(proc) for year in years],
            f"{proc}_combined",
        )
        if not h:
            raise RuntimeError(f"No nominal histogram loaded for {proc}")
        out.bkg[proc] = h

    data_parts: List[object] = []
    for year in years:
        h_data = prepare_year_hist(
            ROOT, cfg, year, "data", -1.0, edges, scale, out.warnings
        )
        if h_data:
            data_parts.append(h_data)
    out.data = sum_hists(data_parts, "data_combined")

    # For QCD MC, derive one common normalisation factor for the complete
    # selected era group and apply it to both the combined and per-era QCD
    # templates.  This is the default for object-validation plots.
    out.qcd_normalisation_factor = apply_qcd_normalisation(
        cfg, out.bkg, bkg_by_year, out.data, out.warnings
    )
    out.bkg_total = sum_background_hists(out.bkg, "bkg_total")

    signal_draw_scale = signal_scale(cfg) * scale
    signal_masses = cfg.signal_masses if (cfg.draw_signal and is_mass_variable(cfg)) else []
    for mass in signal_masses:
        nominal_by_year: Dict[str, object] = {}
        for year in years:
            h_sig = prepare_year_hist(
                ROOT,
                cfg,
                year,
                "sig",
                mass,
                edges,
                signal_draw_scale,
                out.warnings,
            )
            if h_sig:
                nominal_by_year[year] = h_sig

        h_total = sum_hists(
            [nominal_by_year.get(year) for year in years],
            f"sig_M{mass_label(mass)}_combined",
        )
        if not h_total:
            continue
        out.signals.append((mass, h_total))
    out.signals.sort(key=lambda x: x[0])

    out.stat = bkg_stat_uncertainty(cfg, out.bkg)
    if cfg.draw_systematics:
        out.syst = bkg_syst_uncertainty(
            ROOT,
            cfg,
            years,
            out.bkg,
            bkg_by_year,
            out.bkg_total,
            edges,
            scale,
            out.warnings,
            qcd_norm_factor=out.qcd_normalisation_factor,
        )
        out.total = total_uncertainty(out.stat, out.syst)
    else:
        # Stat-only validation mode: do not inspect or open RunSyst/ or
        # RunXSecSyst/ at all.
        out.syst = zero_uncertainty(out.bkg_total.GetNbinsX())
        out.total = out.stat
    return out

def make_unc_graph(ROOT, h, unc: Uncertainty, name: str):
    n = h.GetNbinsX()
    g = ROOT.TGraphAsymmErrors(n)
    g.SetName(_NAMES.unique(name))
    for ib in range(1, n + 1):
        i = ib - 1
        x = h.GetXaxis().GetBinCenter(ib)
        ex = 0.5 * h.GetXaxis().GetBinWidth(ib)
        y = h.GetBinContent(ib)
        g.SetPoint(i, x, y)
        g.SetPointError(i, ex, ex, max(0.0, unc.low[i]), max(0.0, unc.high[i]))
    return g


def make_ratio_unc_graph(ROOT, h, unc: Uncertainty, name: str):
    n = h.GetNbinsX()
    g = ROOT.TGraphAsymmErrors(n)
    g.SetName(_NAMES.unique(name))
    for ib in range(1, n + 1):
        i = ib - 1
        x = h.GetXaxis().GetBinCenter(ib)
        ex = 0.5 * h.GetXaxis().GetBinWidth(ib)
        y = h.GetBinContent(ib)
        el = unc.low[i] / y if y > 0.0 else 0.0
        eh = unc.high[i] / y if y > 0.0 else 0.0
        g.SetPoint(i, x, 1.0)
        g.SetPointError(i, ex, ex, max(0.0, el), max(0.0, eh))
    return g


def make_band_outline_graph(ROOT, h, unc: Uncertainty, name: str, *, ratio: bool = False):
    if not h:
        return None
    n = h.GetNbinsX()
    if n <= 0:
        return None
    g = ROOT.TGraph(4 * n + 1)
    g.SetName(_NAMES.unique(name))
    ip = 0
    for ib in range(1, n + 1):
        i = ib - 1
        x_low = h.GetXaxis().GetBinLowEdge(ib)
        x_up = h.GetXaxis().GetBinUpEdge(ib)
        y = h.GetBinContent(ib)
        central = 1.0 if ratio else y
        err = (unc.high[i] / y if y > 0.0 else 0.0) if ratio else unc.high[i]
        y_up = central + max(0.0, err)
        g.SetPoint(ip, x_low, y_up)
        ip += 1
        g.SetPoint(ip, x_up, y_up)
        ip += 1
    for ib in range(n, 0, -1):
        i = ib - 1
        x_low = h.GetXaxis().GetBinLowEdge(ib)
        x_up = h.GetXaxis().GetBinUpEdge(ib)
        y = h.GetBinContent(ib)
        central = 1.0 if ratio else y
        err = (unc.low[i] / y if y > 0.0 else 0.0) if ratio else unc.low[i]
        y_low = max(0.0, central - max(0.0, err))
        g.SetPoint(ip, x_up, y_low)
        ip += 1
        g.SetPoint(ip, x_low, y_low)
        ip += 1
    # Close the polygon without calling TGraph::GetPoint.  Different PyROOT
    # versions expose GetPoint either as a tuple-returning method or as a
    # C++-style pass-by-reference call, and some CMSSW builds require
    # ctypes.c_double for the latter.  The first point is known directly from
    # the first bin, so storing it avoids the PyROOT API ambiguity entirely.
    first_x = h.GetXaxis().GetBinLowEdge(1)
    first_central = 1.0 if ratio else h.GetBinContent(1)
    if ratio:
        y_first = h.GetBinContent(1)
        first_err = unc.high[0] / y_first if y_first > 0.0 else 0.0
    else:
        first_err = unc.high[0]
    first_y = first_central + max(0.0, first_err)
    g.SetPoint(ip, first_x, first_y)
    return g


def make_data_graph(ROOT, cfg: Config, norm: str, lumi_pb: float, data, bkg_total):
    asimov = cfg.blind and cfg.blind_point_mode == "asimov"
    toy = cfg.blind and cfg.blind_point_mode == "toy"
    source = bkg_total if (asimov or toy) else data
    if not source:
        return None

    bins: List[int] = []
    for ib in range(1, source.GetNbinsX() + 1):
        if cfg.blind and not asimov and not toy and is_mass_variable(cfg):
            high_edge = source.GetXaxis().GetBinUpEdge(ib)
            low_edge = source.GetXaxis().GetBinLowEdge(ib)
            if high_edge > cfg.blind_low + 1.0e-9 and low_edge < cfg.blind_high:
                continue
        bins.append(ib)

    if not bins:
        return None

    rng = ROOT.TRandom3(int(cfg.toy_seed))
    scale = plot_scale(cfg, norm, lumi_pb)
    g = ROOT.TGraphAsymmErrors(len(bins))
    g.SetName(_NAMES.unique("data_graph"))

    for i, ib in enumerate(bins):
        x = source.GetXaxis().GetBinCenter(ib)
        ex = 0.5 * source.GetXaxis().GetBinWidth(ib)
        y = source.GetBinContent(ib)
        ey = 0.0 if asimov else source.GetBinError(ib)

        if toy:
            unit_scale = scale
            if cfg.divide_by_bin_width:
                unit_scale /= source.GetXaxis().GetBinWidth(ib)
            mean_events = max(0.0, y / unit_scale) if unit_scale > 0.0 else max(0.0, y)
            toy_events = rng.PoissonD(mean_events)
            y = toy_events * unit_scale if unit_scale > 0.0 else toy_events
            ey = math.sqrt(toy_events) * unit_scale if unit_scale > 0.0 else math.sqrt(toy_events)

        g.SetPoint(i, x, y)
        g.SetPointError(i, ex, ex, ey, ey)
    return g


def graph_point(g, i: int) -> Tuple[float, float]:
    """Return one TGraph point for both modern and legacy PyROOT bindings."""
    try:
        point = g.GetPoint(i)
        if isinstance(point, tuple) and len(point) >= 2:
            return float(point[0]), float(point[1])
    except TypeError:
        pass

    # CMSSW/PyROOT builds that expose only the C++ signature
    # GetPoint(Int_t, Double_t&, Double_t&) require ctypes.c_double for the
    # pass-by-reference Double_t arguments.  ROOT.Double and plain floats fail
    # in those builds.
    import ctypes
    x = ctypes.c_double(0.0)
    y = ctypes.c_double(0.0)
    g.GetPoint(int(i), x, y)
    return float(x.value), float(y.value)


def make_ratio_graph(ROOT, data_graph, bkg_total):
    if not data_graph or not bkg_total:
        return None
    n = data_graph.GetN()
    g = ROOT.TGraphAsymmErrors(n)
    g.SetName(_NAMES.unique("ratio_graph"))
    for i in range(n):
        x, y = graph_point(data_graph, i)
        ib = bkg_total.GetXaxis().FindFixBin(x)
        den = bkg_total.GetBinContent(ib)
        ratio = y / den if den > 0.0 else 0.0
        eyl = data_graph.GetErrorYlow(i) / den if den > 0.0 else 0.0
        eyh = data_graph.GetErrorYhigh(i) / den if den > 0.0 else 0.0
        g.SetPoint(i, x, ratio)
        g.SetPointError(i, data_graph.GetErrorXlow(i), data_graph.GetErrorXhigh(i), eyl, eyh)
    return g


def set_readable_cms_style(ROOT) -> None:
    style = ROOT.TStyle("ReadableCMSPy", "Readable CMS style for dimuon mass")
    style.SetCanvasBorderMode(0)
    style.SetCanvasColor(0)
    style.SetFrameBorderMode(0)
    style.SetFrameFillColor(0)
    style.SetPadBorderMode(0)
    style.SetPadColor(0)
    style.SetStatColor(0)
    style.SetOptStat(0)
    style.SetOptTitle(0)
    style.SetPadTopMargin(0.09)
    style.SetPadRightMargin(0.05)
    style.SetPadBottomMargin(0.13)
    style.SetPadLeftMargin(0.120)
    font = 42
    style.SetTextFont(font)
    style.SetLabelFont(font, "x")
    style.SetLabelFont(font, "y")
    style.SetTitleFont(font, "x")
    style.SetTitleFont(font, "y")
    style.SetLabelSize(0.040, "x")
    style.SetLabelSize(0.040, "y")
    style.SetTitleSize(0.045, "x")
    style.SetTitleSize(0.045, "y")
    style.SetTitleOffset(1.05, "x")
    style.SetTitleOffset(1.05, "y")
    style.SetPadTickX(1)
    style.SetPadTickY(1)
    style.SetEndErrorSize(0)
    style.SetErrorX(0.5)
    style.SetNdivisions(510, "x")
    style.SetNdivisions(510, "y")
    style.cd()
    ROOT.gROOT.SetStyle("ReadableCMSPy")
    ROOT.gROOT.ForceStyle()
    ROOT.TGaxis.SetMaxDigits(3)


def style_bkg_hist(ROOT, h, proc: str) -> None:
    colors = process_fill_color(ROOT)
    h.SetFillColor(colors[proc])
    h.SetLineColor(ROOT.kBlack)
    h.SetLineWidth(1)


def style_signal_hist(h, color: int, idx: int) -> None:
    h.SetFillStyle(0)
    h.SetLineColor(color)
    h.SetLineWidth(3)
    #h.SetLineStyle(1 + (idx % 4))
    h.SetLineStyle(1)


def style_data_graph(ROOT, g, cfg: Config) -> None:
    if not g:
        return
    g.SetMarkerStyle(24 if cfg.blind and cfg.blind_point_mode in {"asimov", "toy"} else 20)
    g.SetMarkerSize(0.95)
    g.SetMarkerColor(ROOT.kBlack)
    g.SetLineColor(ROOT.kBlack)
    g.SetLineWidth(1)


def style_unc_graph(ROOT, g, kind: str) -> None:
    if not g:
        return
    if kind == "total":
        g.SetFillColorAlpha(ROOT.kGray + 1, 0.30)
        g.SetLineColor(ROOT.kGray + 2)
        g.SetLineWidth(1)
        g.SetFillStyle(1001)
    elif kind == "stat":
        g.SetFillColor(ROOT.kGray + 3)
        g.SetLineColor(ROOT.kGray + 3)
        g.SetLineWidth(1)
        g.SetFillStyle(3354)


def style_band_outline(ROOT, g) -> None:
    if not g:
        return
    g.SetLineColor(ROOT.kGray + 1)
    g.SetLineWidth(1)
    g.SetLineStyle(1)
    g.SetFillStyle(0)


def hist_max_with_error(h, unc: Optional[Uncertainty] = None) -> float:
    if not h:
        return 0.0
    out = 0.0
    for ib in range(1, h.GetNbinsX() + 1):
        y = h.GetBinContent(ib)
        if unc and len(unc.high) >= ib:
            y += unc.high[ib - 1]
        else:
            y += h.GetBinError(ib)
        out = max(out, y)
    return out


def hist_min_positive(h) -> float:
    if not h:
        return 0.0
    out = 1.0e100
    for ib in range(1, h.GetNbinsX() + 1):
        y = h.GetBinContent(ib)
        if 0.0 < y < out:
            out = y
    return out if out < 1.0e99 else 0.0


def graph_max_y(g) -> float:
    if not g:
        return 0.0
    out = 0.0
    for i in range(g.GetN()):
        _, y = graph_point(g, i)
        out = max(out, y + g.GetErrorYhigh(i))
    return out


def graph_min_positive_y(g) -> float:
    if not g:
        return 0.0
    out = 1.0e100
    for i in range(g.GetN()):
        _, y = graph_point(g, i)
        if 0.0 < y < out:
            out = y
    return out if out < 1.0e99 else 0.0


def draw_manual_y_axis_title(ROOT, title: str, x_ndc: float, y_ndc: float, text_size: float) -> None:
    latex = ROOT.TLatex()
    latex.SetNDC(True)
    latex.SetTextFont(42)
    latex.SetTextSize(text_size)
    #latex.SetTextAlign(22)
    latex.SetTextAlign(12)
    latex.SetTextAngle(90.0)
    latex.DrawLatex(x_ndc, y_ndc, title)


def draw_cms_labels(ROOT, cfg: Config, lumi_pb: float) -> None:
    latex = ROOT.TLatex()
    latex.SetNDC(True)
    latex.SetTextAngle(0)
    latex.SetTextColor(ROOT.kBlack)

    latex.SetTextFont(61)
    latex.SetTextSize(0.055)
    latex.DrawLatex(0.145, 0.925, "CMS")

    latex.SetTextFont(52)
    latex.SetTextSize(0.040)
    latex.DrawLatex(0.240, 0.925, cfg.cms_label)

    latex.SetTextFont(42)
    latex.SetTextSize(0.040)
    latex.SetTextAlign(31)
    latex.DrawLatex(0.95, 0.925, lumi_label(cfg.era, lumi_pb))

    latex.SetTextAlign(13)
    latex.SetTextSize(0.040)
    jet_text = "b-jet" if cfg.jet_mode == "bjet" else "light-jet"
    sign_text = canonical_dilepton_sign(cfg.dilepton_sign)
    latex.DrawLatex(0.155, 0.85, f"{cfg.era}, {sign_text}, {jet_text}")


def print_warnings(cfg: Config, norm: str, warnings: Sequence[str]) -> None:
    if not cfg.verbose_warnings or not warnings:
        return
    print(f"[WARN] {cfg.era} {norm}: {len(warnings)} warning(s)", file=sys.stderr)
    nprint = min(len(warnings), 30)
    for msg in warnings[:nprint]:
        print(f"  - {msg}", file=sys.stderr)
    if len(warnings) > nprint:
        print(f"  - ... {len(warnings) - nprint} more", file=sys.stderr)



def sanitise_hist_for_log_draw(h) -> None:
    """Clip non-finite or negative draw-only bin contents before log-y painting."""
    if not h:
        return
    for ib in range(1, h.GetNbinsX() + 1):
        y = h.GetBinContent(ib)
        if (not math.isfinite(y)) or y < 0.0:
            h.SetBinContent(ib, 0.0)
            h.SetBinError(ib, 0.0)


def build_manual_stack_hists(ROOT, cfg: Config, bkg: Dict[str, object]) -> List[Tuple[str, object]]:
    """Return cumulative histograms that reproduce THStack without using THStack.

    THStack::BuildAndPaint segfaults in some ROOT 6.32/CMSSW 15 PyROOT setups when
    saving this canvas.  A manual stack is robust: for processes ordered bottom to
    top, make cumulative histograms and draw them in reverse order.  The last drawn
    histogram covers only the lower part, leaving the upper bands from the earlier
    cumulative histograms visible.
    """
    cumulative: List[Tuple[str, object]] = []
    running = None
    for proc in stack_draw_order(cfg):
        source = bkg.get(proc)
        if not source:
            continue
        if running is None:
            hcum = source.Clone(_NAMES.unique(f"manual_stack_{proc}"))
        else:
            hcum = running.Clone(_NAMES.unique(f"manual_stack_{proc}"))
            hcum.Add(source)
        hcum.SetDirectory(0)
        hcum.SetStats(False)
        style_bkg_hist(ROOT, hcum, proc)
        if cfg.logy:
            sanitise_hist_for_log_draw(hcum)
        cumulative.append((proc, hcum))
        running = hcum
    return cumulative

def draw_one_plot(ROOT, cfg: Config, norm: str, edges: Sequence[float]) -> str:
    inputs = build_plot_inputs(ROOT, cfg, norm, edges)

    for proc in BKG_PROCESSES:
        style_bkg_hist(ROOT, inputs.bkg[proc], proc)
    manual_stack_hists = build_manual_stack_hists(ROOT, cfg, inputs.bkg)

    g_total = make_unc_graph(ROOT, inputs.bkg_total, inputs.total, "bkg_total_unc") if cfg.draw_systematics else None
    g_stat = make_unc_graph(ROOT, inputs.bkg_total, inputs.stat, "bkg_stat_unc")
    g_stat_outline = make_band_outline_graph(ROOT, inputs.bkg_total, inputs.stat, "bkg_stat_unc_outline", ratio=False)
    style_unc_graph(ROOT, g_total, "total")
    style_unc_graph(ROOT, g_stat, "stat")
    style_band_outline(ROOT, g_stat_outline)

    data_graph = make_data_graph(ROOT, cfg, norm, inputs.lumi_pb, inputs.data, inputs.bkg_total)
    style_data_graph(ROOT, data_graph, cfg)

    colors = signal_colors(ROOT)
    for idx, (mass, h) in enumerate(inputs.signals):
        color = colors[idx % len(colors)]
        style_signal_hist(h, color, idx)

    use_ratio = cfg.draw_ratio and data_graph is not None
    canvas = ROOT.TCanvas(_NAMES.unique("c"), "", 900, 850 if use_ratio else 760)

    if use_ratio:
        pad1 = ROOT.TPad(_NAMES.unique("pad1"), "", 0.0, 0.30, 1.0, 1.0)
        pad2 = ROOT.TPad(_NAMES.unique("pad2"), "", 0.0, 0.00, 1.0, 0.30)
        pad1.SetLeftMargin(cfg.left_margin)
        pad1.SetRightMargin(cfg.right_margin)
        pad1.SetTopMargin(cfg.main_top_margin)
        pad1.SetBottomMargin(cfg.main_bottom_margin)
        pad2.SetLeftMargin(cfg.left_margin)
        pad2.SetRightMargin(cfg.right_margin)
        pad2.SetTopMargin(cfg.ratio_top_margin)
        pad2.SetBottomMargin(cfg.ratio_bottom_margin)
        pad1.Draw()
        pad2.Draw()
        pad1.cd()
    else:
        pad1 = canvas
        pad2 = None
        canvas.SetLeftMargin(cfg.left_margin)
        canvas.SetRightMargin(cfg.right_margin)
        canvas.SetTopMargin(cfg.main_top_margin)
        canvas.SetBottomMargin(cfg.no_ratio_bottom_margin)

    if cfg.logy:
        pad1.SetLogy(True)

    # Compute and fix the upper-panel axis range before drawing any object.
    #
    # In earlier versions the range was set through THStack.  Drawing an explicit
    # empty frame first makes --ymin and --ymax literal axis limits, independent
    # of ROOT histogram autoscaling.
    ymax = hist_max_with_error(inputs.bkg_total, inputs.total if cfg.draw_systematics else inputs.stat)
    ymax = max(ymax, graph_max_y(data_graph))
    for _, sig in inputs.signals:
        ymax = max(ymax, hist_max_with_error(sig))
    if ymax <= 0.0:
        ymax = 1.0

    if cfg.logy:
        positives = [hist_min_positive(inputs.bkg_total), graph_min_positive_y(data_graph)]
        positives.extend(hist_min_positive(sig) for _, sig in inputs.signals)
        ymin_auto = min([v for v in positives if v > 0.0], default=1.0e-5)
        y_min_plot = cfg.ymin if cfg.ymin is not None and cfg.ymin > 0.0 else max(1.0e-12, 0.5 * ymin_auto)
        y_max_plot = cfg.ymax if cfg.ymax is not None and cfg.ymax > 0.0 else ymax * 80.0

        # A single --ymin is applied to every requested normalisation.  A value
        # suitable for event yields can be too high after luminosity scaling, so
        # keep the log-frame maximum larger than the requested minimum.
        if y_max_plot <= y_min_plot:
            if cfg.ymax is not None:
                raise ValueError(
                    f"Invalid y-axis range for {norm}: ymin={y_min_plot:g}, "
                    f"ymax={y_max_plot:g}.  Use --ymax larger than --ymin, "
                    "or restrict --normalisations."
                )
            y_max_plot = y_min_plot * 10.0
            inputs.warnings.append(
                f"Auto ymax for {norm} was below --ymin ({y_min_plot:g}); "
                f"raised ymax to {y_max_plot:g} to keep a valid log-axis range.  "
                "For this normalisation, consider using --ymin auto or "
                "--normalisations events."
            )
    else:
        y_min_plot = cfg.ymin if cfg.ymin is not None and cfg.ymin >= 0.0 else 0.0
        y_max_plot = cfg.ymax if cfg.ymax is not None and cfg.ymax > 0.0 else ymax * 1.55
        if y_max_plot <= y_min_plot:
            if cfg.ymax is not None:
                raise ValueError(
                    f"Invalid y-axis range for {norm}: ymin={y_min_plot:g}, "
                    f"ymax={y_max_plot:g}."
                )
            y_max_plot = y_min_plot + max(1.0, abs(y_min_plot) * 0.10)
            inputs.warnings.append(
                f"Auto ymax for {norm} was not above --ymin ({y_min_plot:g}); "
                f"raised ymax to {y_max_plot:g}."
            )

    main_frame = ROOT.TH1D(
        _NAMES.unique("main_frame"),
        "",
        len(edges) - 1,
        array("d", [float(x) for x in edges]),
    )
    main_frame.SetDirectory(0)
    main_frame.SetStats(False)
    main_frame.SetMinimum(y_min_plot)
    main_frame.SetMaximum(y_max_plot)
    main_frame.GetXaxis().SetRangeUser(float(edges[0]), float(edges[-1]))
    main_frame.GetXaxis().SetTitle(variable_spec(cfg).x_title)
    main_frame.GetYaxis().SetTitle("")
    main_frame.GetYaxis().SetLabelSize(0.045)
    if use_ratio:
        main_frame.GetXaxis().SetLabelSize(0.0)
        main_frame.GetXaxis().SetTitleSize(0.0)
    main_frame.Draw("AXIS")

    # Draw the stacked backgrounds manually, avoiding THStack entirely.
    for _, h_stack in reversed(manual_stack_hists):
        h_stack.Draw("hist same")

    if g_total:
        g_total.Draw("E2 same")
    g_stat.Draw("E2 same")
    if g_stat_outline:
        g_stat_outline.Draw("L same")

    for _, sig in inputs.signals:
        sig.Draw("hist same")
    if data_graph:
        data_graph.Draw("PZ same")

    # Re-draw the explicit frame axes so that the displayed limits remain those
    # requested by --ymin/--ymax after all same-pad objects have been drawn.
    main_frame.Draw("AXIS SAME")

    labels = process_label(cfg)
    if use_qcd_mc(cfg) and cfg.qcd_normalise:
        labels["QCD"] = f"QCD (x {inputs.qcd_normalisation_factor:.3g})"

    leg = ROOT.TLegend(0.56, 0.53, 0.93, 0.86)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.040)

    if data_graph:
        if cfg.blind and cfg.blind_point_mode == "asimov":
            leg.AddEntry(data_graph, "Bkg Asimov", "pe")
        elif cfg.blind and cfg.blind_point_mode == "toy":
            leg.AddEntry(data_graph, "Bkg toy", "pe")
        else:
            leg.AddEntry(data_graph, "Data", "pe")

    for proc in legend_bkg_order(cfg):
        leg.AddEntry(inputs.bkg[proc], labels[proc], "f")

    if g_total:
        leg.AddEntry(g_total, "Bkg stat #oplus syst unc.", "f")
    leg.AddEntry(g_stat, "Bkg stat unc.", "f")

    for mass, sig in inputs.signals:
        scale_label = signal_scale_label(cfg)
        label = f"Z' (m={mass_label(mass).replace('p', '.')} GeV"
        if scale_label:
            label += f", {scale_label}"
        label += ")"
        leg.AddEntry(sig, label, "l")
    leg.Draw()

    draw_cms_labels(ROOT, cfg, inputs.lumi_pb)
    draw_manual_y_axis_title(ROOT, y_axis_title(cfg, norm), cfg.y_axis_title_ndc_x, 0.65, cfg.main_y_axis_title_size)

    if cfg.blind:
        note = ROOT.TLatex()
        note.SetNDC(True)
        note.SetTextFont(42)
        note.SetTextSize(0.040)
        if cfg.blind_point_mode == "data" and is_mass_variable(cfg):
            note.DrawLatex(0.155, 0.78, f"Data blinded for {cfg.blind_low:g} < m_{{#mu^{{+}}#mu^{{-}}}} < {cfg.blind_high:g} GeV")
        elif cfg.blind_point_mode == "asimov":
            note.DrawLatex(0.155, 0.78, "Blind: background-only Asimov points")
        elif cfg.blind_point_mode == "toy":
            note.DrawLatex(0.155, 0.78, "Blind: background-only toy points")

    pad1.RedrawAxis()

    if use_ratio and pad2 is not None:
        pad2.cd()
        ratio = make_ratio_graph(ROOT, data_graph, inputs.bkg_total)
        ratio_total = make_ratio_unc_graph(ROOT, inputs.bkg_total, inputs.total, "ratio_total_unc") if cfg.draw_systematics else None
        ratio_stat = make_ratio_unc_graph(ROOT, inputs.bkg_total, inputs.stat, "ratio_stat_unc")
        ratio_stat_outline = make_band_outline_graph(ROOT, inputs.bkg_total, inputs.stat, "ratio_stat_unc_outline", ratio=True)
        style_unc_graph(ROOT, ratio_total, "total")
        style_unc_graph(ROOT, ratio_stat, "stat")
        style_band_outline(ROOT, ratio_stat_outline)
        style_data_graph(ROOT, ratio, cfg)

        frame = inputs.bkg_total.Clone(_NAMES.unique("ratio_frame"))
        frame.Reset("ICESM")
        frame.SetMinimum(cfg.ratio_min)
        frame.SetMaximum(cfg.ratio_max)
        frame.GetXaxis().SetRangeUser(float(edges[0]), float(edges[-1]))
        frame.GetXaxis().SetTitle(variable_spec(cfg).x_title)
        frame.GetYaxis().SetTitle("")
        frame.GetXaxis().SetTitleSize(0.12)
        frame.GetXaxis().SetLabelSize(0.1)
        frame.GetYaxis().SetTitleSize(0.12)
        frame.GetYaxis().SetLabelSize(0.1)
        frame.GetYaxis().SetNdivisions(505)
        frame.Draw("axis")
        draw_manual_y_axis_title(ROOT, "Data / Bkg", cfg.y_axis_title_ndc_x, 0.5, cfg.ratio_y_axis_title_size)

        if ratio_total:
            ratio_total.Draw("E2 same")
        ratio_stat.Draw("E2 same")
        if ratio_stat_outline:
            ratio_stat_outline.Draw("L same")
        line = ROOT.TLine(float(edges[0]), 1.0, float(edges[-1]), 1.0)
        line.SetLineColor(ROOT.kBlack)
        line.SetLineStyle(2)
        line.SetLineWidth(1)
        line.Draw("same")
        if ratio:
            ratio.Draw("PZ same")
        pad2.RedrawAxis()

    os.makedirs(cfg.output_dir, exist_ok=True)

    if not cfg.blind:
        data_tag = "unblind"
    elif cfg.blind_point_mode == "asimov":
        data_tag = "asimov"
    elif cfg.blind_point_mode == "toy":
        data_tag = "toy"
    else:
        data_tag = "blind"

    qcd_tag = "_QCDMC" if use_qcd_mc(cfg) else "_QCDDD"
    if use_qcd_mc(cfg):
        qcd_tag += "_norm" if cfg.qcd_normalise else "_raw"
    dy_tag = "_DYMC" if use_dy_mc(cfg) else "_DYDD"
    syst_tag = "" if cfg.draw_systematics else "_statOnly"
    jet_tag = "_LightJet" if cfg.jet_mode == "lightjet" else "_BJet"
    sign_tag = "_SS" if canonical_dilepton_sign(cfg.dilepton_sign) == "SS" else ""
    rebin_tag = f"_rebin{cfg.rebin_factor}" if cfg.rebin_factor > 1 else ""
    x_range_tag = ""
    spec = variable_spec(cfg)
    plot_xmin = spec.xmin if cfg.xmin is None else cfg.xmin
    plot_xmax = spec.xmax if cfg.xmax is None else cfg.xmax
    if not cfg.bin_edges and (abs(plot_xmin - spec.xmin) > 1.0e-9 or abs(plot_xmax - spec.xmax) > 1.0e-9):
        x_range_tag = f"_x{mass_label(plot_xmin)}to{mass_label(plot_xmax)}"

    out_base = os.path.join(
        cfg.output_dir,
        f"{cfg.era}_{spec.key}_{norm}_{data_tag}{jet_tag}{sign_tag}{qcd_tag}{dy_tag}{syst_tag}{rebin_tag}{x_range_tag}",
    )
    for ext in cfg.extensions:
        canvas.SaveAs(f"{out_base}.{ext}")

    if cfg.save_root:
        fout = ROOT.TFile.Open(f"{out_base}.root", "RECREATE")
        for proc, hist in inputs.bkg.items():
            hist.Write(proc)
        inputs.bkg_total.Write("bkg_total")
        if inputs.data:
            inputs.data.Write("data")
        for mass, sig in inputs.signals:
            sig.Write(f"sig_M{mass_label(mass)}")
        g_stat.Write("bkg_stat_unc")
        if g_total:
            g_total.Write("bkg_stat_plus_syst_unc")
        fout.Close()

    print_warnings(cfg, norm, inputs.warnings)
    return out_base


def variables_for_request(cfg: Config) -> List[str]:
    requested = canonical_variable(cfg.variable)
    if requested != "all":
        return [requested]
    keys = list(VARIABLE_SPECS)
    if cfg.jet_mode == "lightjet":
        keys = [key for key in keys if not key.startswith("jet1_")]
    return keys


def apply_variable_blinding_mode(cfg: Config) -> Config:
    """Apply the requested blinding policy variable by variable.

    With --blind:
      * dimuon_mass keeps the ordinary mass-window blinding and therefore uses
        real data only outside the blinded mass interval;
      * every other validation variable is automatically converted to a
        background-only Asimov plot, because the one-dimensional object
        histogram does not retain m_mumu and cannot be mass-window blinded
        afterwards.

    Explicit --data-mode blind_asimov/blind_toy and --unblind are preserved.
    """
    if is_mass_variable(cfg):
        return cfg

    if cfg.blind and cfg.blind_point_mode == "data":
        print(
            f"[INFO] {cfg.variable}: --blind requested; using background-only "
            "Asimov points for this non-mass validation variable."
        )
        return replace(cfg, blind=True, blind_point_mode="asimov")

    return cfg


def run(cfg: Config) -> List[str]:
    ROOT = import_root()
    ROOT.gROOT.SetBatch(True)
    set_readable_cms_style(ROOT)

    cfg.era = canonical_era(cfg.era)
    cfg.jet_mode = canonical_jet_mode(cfg.jet_mode)
    cfg.dilepton_sign = canonical_dilepton_sign(cfg.dilepton_sign)
    cfg.qcd_method = canonical_background_method(cfg.qcd_method, option_name="qcd-method")
    cfg.dy_method = canonical_background_method(cfg.dy_method, option_name="dy-method")
    cfg.uncertainty, cfg.draw_systematics = canonical_uncertainty(cfg.uncertainty)
    years_for_era(cfg.era)

    if cfg.draw_systematics and (cfg.jet_mode != "bjet" or cfg.dilepton_sign != "OS"):
        raise ValueError(
            "The current RunSyst/RunXSecSyst production contains only OS BJet histograms. "
            "Use --uncertainty stat-only for BJet-SS or LightJet control-region plots."
        )

    outputs: List[str] = []
    requested_variables = variables_for_request(cfg)
    for variable in requested_variables:
        this_cfg = replace(cfg, variable=variable)
        this_cfg = apply_variable_blinding_mode(this_cfg)
        if not is_mass_variable(this_cfg):
            print(
                f"[INFO] {variable}: validation distribution uses QCD MC and DY MC; "
                "the data-driven QCD/DY estimators are mass-template methods."
            )
            if this_cfg.qcd_normalise:
                print("[INFO] QCD MC single-value normalisation is enabled.")
        edges = make_bin_edges(this_cfg)
        for norm in this_cfg.normalisations:
            norm_key = norm.lower().strip()
            if norm_key not in {"events", "xsec"}:
                raise ValueError(f"Unknown normalisation: {norm}. Use events or xsec.")
            print(f"[INFO] Drawing {variable}, normalisation: {norm_key}")
            outputs.append(draw_one_plot(ROOT, this_cfg, norm_key, edges))

    print("[DONE] wrote:")
    for out in outputs:
        for ext in cfg.extensions:
            print(f"  {out}.{ext}")
        if cfg.save_root:
            print(f"  {out}.root")
    return outputs


def make_parser() -> argparse.ArgumentParser:
    examples = r"""
Examples:
  # Default: unblinded, stat-only. Object-validation variables use QCD MC
  # with one (Data - non-QCD MC)/QCD MC normalisation factor.
  python3 plotter.py --era Run2

  # All Run-3 object-validation plots
  python3 plotter.py --era Run3 --variable all

  # With --blind, dimuon mass uses mass-window blinding and all other\n  # validation variables automatically use background-only Asimov points.\n  python3 plotter.py --era Run2 --variable all --blind

  # Draw stat+syst and require every applicable systematic template
  python3 plotter.py --era Run2 --uncertainty syst+stat --strict

  # Disable the default QCD MC normalisation
  python3 plotter.py --era Run3 --variable all --no-qcd-normalise

  # Explicitly draw signal.  Without --signal-scale or --draw-signal, no signal is drawn.
  python3 plotter.py --era Run2 --signal-scale 1000

  # Same-sign light-jet region using the new aliases
  python3 plotter.py --era 2018 --dimuon-sign SS --jet-flavour light-jet

Systematic treatment:
  * --uncertainty stat-only (default) never opens RunSyst/ or RunXSecSyst/
  * --uncertainty syst+stat follows the final limit_workflow.py nuisance model
  * data-driven QCD: QCD_norm + symmetric absolute QCD_shape; no QCD_stat
  * data-driven DY: constant NF + DY_NFStat(TFDown/Up) + DY_LightJetStat; no DY_stat
  * JER/JES/PU/muon and BTV-uncorrelated sources are independent between eras
  * L1 prefiring is era-specific for 2016pre/postVFP, 2017, and 2018
  * BTV correlated sources are shared within Run2 and within Run3
  * generator theory is tt/ST only: symmetric-Hessian PDF, shared alphaS,
    and separate process-specific muF/muR nuisances; no 7-point scale envelope
  * generic tt_xsec and ST_xsec nuisances are disabled; tt_mass is retained
  * 2016pre/post share luminosity; 2022/2022EE share; 2023/2023BPix share
  * signal is drawn nominal-only; signal RunXSecSyst is not produced

No-argument behaviour:
  Running `python3 plotter.py` prints this instruction and exits.
"""
    parser = argparse.ArgumentParser(
        prog="plotter.py",
        description=(
            "Draw NIsoMuon mass and object-validation distributions using PyROOT, with explicit "
            "per-process systematic checks and Run-2/Run-3 correlation propagation."
        ),
        epilog=examples,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--era", choices=list(ERA_GROUPS.keys()), required=True)
    parser.add_argument("--variable", default="dimuon_mass", help="dimuon_mass, jet0/1_{pt,eta,phi}, mu_lead/sub_{pt,eta,phi}, or all")
    parser.add_argument("--data-mode", default="unblind", help="blind, blind_asimov, blind_toy, or unblind; default: unblind")
    parser.add_argument("--blind", dest="data_mode", action="store_const", const="blind", help="alias for --data-mode blind")
    parser.add_argument("--unblind", dest="data_mode", action="store_const", const="unblind", help="alias for --data-mode unblind")

    parser.add_argument("--qcd-method", default="data-driven", help="data-driven or mc; non-mass validation variables always use QCD MC")
    parser.add_argument("--qcd-normalise", "--qcd-normalize", dest="qcd_normalise", action="store_true",
                        help="normalise QCD MC by one (Data - non-QCD MC)/QCD MC factor; default")
    parser.add_argument("--no-qcd-normalise", "--no-qcd-normalize", dest="qcd_normalise", action="store_false",
                        help="do not apply the single-value QCD MC normalisation")
    parser.set_defaults(qcd_normalise=True)

    parser.add_argument("--dy-method", default="data-driven", help="mc or data-driven; default: data-driven")
    parser.add_argument("--uncertainty", default="stat-only", help="stat-only or syst+stat; default: stat-only")
    parser.add_argument("--stat-only", dest="uncertainty", action="store_const", const="stat-only",
                        help="alias for --uncertainty stat-only")
    parser.add_argument("--stat-syst", "--syst-stat", dest="uncertainty", action="store_const", const="syst+stat",
                        help="alias for --uncertainty syst+stat")

    parser.add_argument("--jet-mode", "--jet-flavour", "--jet-flavor", dest="jet_mode", default="bjet",
                        help="bjet/b-jet or lightjet/light-jet; default: bjet")
    parser.add_argument("--dilepton-sign", "--dimuon-sign", "--sign", dest="dilepton_sign", default="OS",
                        help="OS/opposite-sign or SS/same-sign; default: OS")
    parser.add_argument("--ss", "--same-sign", dest="dilepton_sign", action="store_const", const="SS", help="alias for --dilepton-sign SS")
    parser.add_argument("--os", "--opposite-sign", dest="dilepton_sign", action="store_const", const="OS", help="alias for --dilepton-sign OS")
    parser.add_argument("--b-jet", dest="jet_mode", action="store_const", const="bjet", help="alias for --jet-flavour b-jet")
    parser.add_argument("--light-jet", dest="jet_mode", action="store_const", const="lightjet", help="alias for --jet-flavour light-jet")

    parser.add_argument("--xmin", type=float, default=None, help="override variable-specific default xmin")
    parser.add_argument("--xmax", type=float, default=None, help="override variable-specific default xmax")
    parser.add_argument("--ymin", type=parse_optional_float, default=0.5, help="number or auto; default: 10")
    parser.add_argument("--ymax", type=parse_optional_float, default=None, help="number or auto; default: auto")
    parser.add_argument("--bin-width", type=float, default=None, help="override variable-specific default bin width")
    parser.add_argument("--rebin-factor", "--rebin", dest="rebin_factor", type=int, default=1, help="merge this many neighbouring plotting bins; default: 1")
    parser.add_argument("--bin-edges", type=parse_float_list, default=None, help="comma/space separated edges; overrides xmin/xmax/bin-width before rebinning")

    parser.add_argument("--signal-scale", type=float, default=None,
                        help="draw signal with this extra multiplicative scale; if omitted, signal is not drawn")
    parser.add_argument("--draw-signal", action="store_true",
                        help="draw signal with scale 1 unless --signal-scale is also given")
    parser.add_argument("--signal-masses", type=parse_float_list, default=parse_float_list("20,50"), help="comma/space separated mass list; default: 20,50")
    parser.add_argument("--signal-reference-xsec-pb", type=float, default=1.0)
    parser.add_argument("--signal-xsec-pb", type=float, default=-1.0)

    parser.add_argument("--normalisations", type=parse_str_list, default=parse_str_list("events"), help="comma/space separated list among events,xsec; default: events")

    parser.add_argument("--base-dir", default="/data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13/NIsoMuon")
    parser.add_argument("--trigger", default="", help="legacy optional trigger subdirectory; leave empty for current SKOutput")
    parser.add_argument("--muon-id", default="POGMedium")
    parser.add_argument("--jet-id", default="tight")

    parser.add_argument("--qcd-data-driven-file", default="NIsoMuon_SS_fit.root")
    parser.add_argument("--qcd-mc-file", default="NIsoMuon_QCD_Inclusive.root")
    parser.add_argument("--dy-data-driven-file", default="NIsoMuon_DYJets_est.root")
    parser.add_argument("--dy-mc-file", default="NIsoMuon_DYJets_Inclusive.root")
    parser.add_argument("--tt-file", default="NIsoMuon_tt.root")
    parser.add_argument("--st-file", default="NIsoMuon_ST.root")
    parser.add_argument("--others-file", default="NIsoMuon_Others.root")

    parser.add_argument("--blind-low", type=float, default=10.4, help="lower edge of dimuon-mass blind interval; default: 10.4 GeV")
    parser.add_argument("--blind-high", type=float, default=80.0, help="upper edge of dimuon-mass blind interval; default: 80 GeV")
    parser.add_argument("--blind-visible-data-max", type=float, default=9.0)
    parser.add_argument("--toy-seed", type=int, default=37829)

    parser.add_argument("--output-dir", default="plots")
    parser.add_argument("--extensions", type=parse_str_list, default=parse_str_list("pdf,png"))
    parser.add_argument("--save-root", action="store_true")

    parser.add_argument("--linear", dest="logy", action="store_false", help="use linear y axis instead of log")
    parser.add_argument("--logy", dest="logy", action="store_true", help="use log y axis; default")
    parser.set_defaults(logy=True)
    parser.add_argument("--no-ratio", dest="draw_ratio", action="store_false", help="do not draw the Data/Bkg ratio panel")
    parser.add_argument("--ratio-min", "--ratio-ymin", "--ratio-y-min", dest="ratio_min", type=float, default=0.5, help="minimum of the Data/Bkg ratio-panel y axis; default: 0.5")
    parser.add_argument("--ratio-max", "--ratio-ymax", "--ratio-y-max", dest="ratio_max", type=float, default=1.5, help="maximum of the Data/Bkg ratio-panel y axis; default: 1.5")
    parser.add_argument("--cms-label", default="Preliminary")

    parser.add_argument("--no-bin-width-normalisation", dest="divide_by_bin_width", action="store_false")
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "abort if any nominal input or any required systematic variation is "
            "missing for an applicable process or era"
        ),
    )
    parser.add_argument("--quiet-warnings", dest="verbose_warnings", action="store_false")
    parser.add_argument("--quiet-systematics", dest="verbose_systematics", action="store_false")

    return parser

def config_from_args(args: argparse.Namespace) -> Config:
    blind, blind_point_mode, _ = canonical_data_mode(args.data_mode)
    uncertainty, draw_systematics = canonical_uncertainty(args.uncertainty)
    if args.blind_high <= args.blind_low:
        raise ValueError("blind_high must be larger than blind_low.")
    if args.draw_ratio and args.ratio_max <= args.ratio_min:
        raise ValueError(
            "ratio_max must be larger than ratio_min. "
            "Use --ratio-ymin/--ratio-ymax or --ratio-min/--ratio-max."
        )

    return Config(
        base_dir=args.base_dir,
        era=args.era,
        trigger=args.trigger,
        variable=canonical_variable(args.variable),
        muon_id=args.muon_id,
        jet_id=args.jet_id,
        jet_mode=canonical_jet_mode(args.jet_mode),
        dilepton_sign=canonical_dilepton_sign(args.dilepton_sign),
        blind=blind,
        blind_point_mode=blind_point_mode,
        toy_seed=args.toy_seed,
        blind_low=args.blind_low,
        blind_high=args.blind_high,
        blind_visible_data_max=args.blind_visible_data_max,
        draw_signal=bool(args.draw_signal or args.signal_scale is not None),
        signal_masses=list(args.signal_masses),
        signal_scale=1.0 if args.signal_scale is None else args.signal_scale,
        signal_reference_xsec_pb=args.signal_reference_xsec_pb,
        signal_xsec_pb=args.signal_xsec_pb,
        normalisations=[x.lower() for x in args.normalisations],
        xmin=args.xmin,
        xmax=args.xmax,
        bin_width=args.bin_width,
        rebin_factor=args.rebin_factor,
        bin_edges=list(args.bin_edges) if args.bin_edges else [],
        qcd_method=canonical_background_method(args.qcd_method, option_name="qcd-method"),
        qcd_normalise=bool(args.qcd_normalise),
        dy_method=canonical_background_method(args.dy_method, option_name="dy-method"),
        uncertainty=uncertainty,
        draw_systematics=draw_systematics,
        qcd_data_driven_file=args.qcd_data_driven_file,
        qcd_mc_file=args.qcd_mc_file,
        dy_data_driven_file=args.dy_data_driven_file,
        dy_mc_file=args.dy_mc_file,
        tt_file=args.tt_file,
        st_file=args.st_file,
        others_file=args.others_file,
        divide_by_bin_width=args.divide_by_bin_width,
        output_dir=args.output_dir,
        extensions=args.extensions,
        save_root=args.save_root,
        logy=args.logy,
        draw_ratio=args.draw_ratio,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
        ymin=args.ymin,
        ymax=args.ymax,
        cms_label=args.cms_label,
        strict=args.strict,
        verbose_warnings=args.verbose_warnings,
        verbose_systematics=args.verbose_systematics,
    )

def print_instructions() -> None:
    parser = make_parser()
    parser.print_help()


def normalise_cli_hyphen_typos(argv: Sequence[str]) -> List[str]:
    """Accept accidental long-option prefixes such as ----normalisations.

    The canonical form remains two hyphens.  This only collapses tokens that look
    like long options with three or more leading hyphens; negative numbers are not
    affected.
    """
    out: List[str] = []
    for token in argv:
        if token.startswith("---") and len(token) > 3:
            stripped = token.lstrip("-")
            if stripped:
                out.append("--" + stripped)
                continue
        out.append(token)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = normalise_cli_hyphen_typos(list(sys.argv[1:] if argv is None else argv))
    if not argv:
        print_instructions()
        return 0

    parser = make_parser()
    args = parser.parse_args(argv)
    if args.draw_ratio and args.ratio_max <= args.ratio_min:
        parser.error("ratio-panel ymax must be larger than ymin. Use --ratio-ymin/--ratio-ymax or --ratio-min/--ratio-max.")
    cfg = config_from_args(args)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
