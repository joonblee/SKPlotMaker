#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Unified same-sign-data and QCD-MC shape fits for the NIsoMuon Run-2/Run-3
analysis.

Fixed filesystem layout
-----------------------
Input and produced QCD ROOT templates use
  /data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13/NIsoMuon
All fit plots are always written to
  /data6/Users/joonblee/PlotMaker/plots
All dynamically fitted SS anchors are always stored in
  /data6/Users/joonblee/PlotMaker/NIsoMuon_SS_fit_anchors.json
The plot and anchor locations are fixed and cannot be changed by options.
The script is self-contained for plotting and does not require PlotHelper,
tdrstyle.C, CMS_lumi.C, or PlotterCore.h.

Supported periods
-----------------
  2016preVFP, 2016postVFP, 2016, 2017, 2018,
  2022, 2022EE, 2023, 2023BPix,
  Run2, Run3, Run2+3
An SS fit to an individual era writes its QCD templates into that era.  An SS
fit to a combined period saves only the combined anchor and does not overwrite
per-era templates.

Anchor priority in QCD-MC mode
------------------------------
For an individual Run-2 era, an available Run2 JSON anchor is tried first,
followed by the same-era JSON anchor and then the built-in Run-2 fallback.  For
an individual Run-3 era, Run3 is tried before the same-era JSON anchor.  A
Run2+3 fit uses only its own Run2+3 anchor.  Use
--no-prefer-full-run-anchor to reverse the full-run/per-era JSON priority.
The SS shape is used as an anchor; its raw amplitude is re-normalised to the
QCD-MC histogram before amplitude constraints are constructed.

Fit modes and objectives
------------------------
  --mode ss-data
      Bin-integrated statistical chi-square; produces the central SS-based QCD
      template plus Norm and analytic-function-envelope Shape variations.
  --mode qcd-mc
      Fits the QCD MC shape.  --fit-objective auto selects ROOT weighted
      likelihood; chi2 and log-chi2 are available as cross-checks.
The interval 9 < m(mumu) < 11 GeV is excluded from every QCD-MC fit objective
and diagnostic.

Required option
---------------
  --mode {ss-data,qcd-mc}

Main optional controls
----------------------
  --year/--era PERIOD
  --fit-objective {auto,chi2,weighted-likelihood,log-chi2}
  --log-relative-error-floor VALUE
  --rebin FACTOR
  --fit-max-attempts N
  --fit-attempt-details
  --initial-values-only
  --no-prefer-full-run-anchor
  --allow-invalid-fit-output
  --print-shape-syst-bin-info
  --print-raw-shape-debug
  --trigger SUBDIRECTORY      optional legacy input subdirectory
  --analyzer NAME             default: NIsoMuon
  --base-dir PATH
  --debug                     re-raise errors through the normal Python trace

Luminosity labels [fb^-1]
-------------------------
  2016preVFP 19.5, 2016postVFP 16.8, 2016 36.31,
  2017 42.07, 2018 59.56, Run2 137.94,
  2022 7.98, 2022EE 26.67, 2022 total 34.65,
  2023 17.7, 2023BPix 9.5, 2023 total 27.20, Run3 61.85.
The luminosities affect labels only; the fit uses event yields already stored
in the input ROOT files.

Examples
--------
  python3 qcd_bkg_estimation.py --mode ss-data --year 2018
  python3 qcd_bkg_estimation.py --mode ss-data --year Run2
  python3 qcd_bkg_estimation.py --mode ss-data --year Run3
  python3 qcd_bkg_estimation.py --mode qcd-mc --year 2023
  python3 qcd_bkg_estimation.py --mode qcd-mc --year 2023 \
      --fit-objective chi2
  python3 qcd_bkg_estimation.py --mode qcd-mc --year Run3 \
      --fit-objective log-chi2 --log-relative-error-floor 0.10
  python3 qcd_bkg_estimation.py --mode qcd-mc --year 2018 \
      --initial-values-only

Build every per-era and full-run anchor:
  for era in 2016preVFP 2016postVFP 2017 2018 2022 2022EE 2023 2023BPix; do
    python3 qcd_bkg_estimation.py --mode ss-data --year "$era"
  done
  python3 qcd_bkg_estimation.py --mode ss-data --year Run2
  python3 qcd_bkg_estimation.py --mode ss-data --year Run3
  python3 qcd_bkg_estimation.py --mode ss-data --year 'Run2+3'

Running with no arguments prints this guide and every command-line option, then
exits without importing ROOT.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import shutil
import sys
import tempfile
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# =============================================================================
# GENERAL CONFIGURATION
# =============================================================================

DEFAULT_BASE_DIR = "/data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13"
PLOT_DIR = Path("/data6/Users/joonblee/PlotMaker/plots")
ANCHOR_FILE = Path("/data6/Users/joonblee/PlotMaker/NIsoMuon_SS_fit_anchors.json")

RUN2_ERAS: Tuple[str, ...] = (
    "2016preVFP", "2016postVFP", "2017", "2018",
)
RUN3_ERAS: Tuple[str, ...] = (
    "2022", "2022EE", "2023", "2023BPix",
)
ALL_ERAS: Tuple[str, ...] = RUN2_ERAS + RUN3_ERAS
PERIOD_ERAS: Dict[str, Tuple[str, ...]] = {
    **{era: (era,) for era in ALL_ERAS},
    "2016": ("2016preVFP", "2016postVFP"),
    "Run2": RUN2_ERAS,
    "Run3": RUN3_ERAS,
    "Run2+3": ALL_ERAS,
}
VALID_YEARS = tuple(PERIOD_ERAS.keys())

LUMI_FB: Dict[str, float] = {
    "2016preVFP": 19.5,
    "2016postVFP": 16.8,
    "2017": 42.07,
    "2018": 59.56,
    "2022": 7.98,
    "2022EE": 26.67,
    "2023": 17.7,
    "2023BPix": 9.5,
}
PERIOD_LUMI_FB: Dict[str, float] = {
    "2016": 36.31,
    "Run2": 137.94,
    "2022total": 34.65,
    "2023total": 27.20,
    "Run3": 61.85,
    "Run2+3": 199.79,
}

SS_REGION = "SS_POGMedium_tight_BJet_NIsoDimuon"
OS_REGION = "OS_POGMedium_tight_BJet_NIsoDimuon"
HIST_NAME = "Dilepton_Mass"

QCD_FIT_EXCLUDED_RANGES: Tuple[Tuple[float, float], ...] = ((9.0, 11.),)

# OS/SS transfer-factor windows.  The transfer factor measured in data at low
# mass is transported to the high-mass search region with the QCD-MC double ratio.
QCD_TRANSFER_LOW_WINDOW: Tuple[float, float] = (5.0, 9.0)
QCD_TRANSFER_HIGH_WINDOW: Tuple[float, float] = (11.0, 80.0)

TAIL_DIAGNOSTIC_MIN = 20.0


@dataclass(frozen=True)
class ModeConfig:
    key: str
    display_name: str
    file_name: str
    region: str
    fit_min: float
    fit_max: float
    plot_tag: str
    legend_label: str
    components: Tuple[Tuple[str, float], ...]


@dataclass(frozen=True)
class FitModelConfig:
    key: str
    cpp_id: int
    label: str
    npar: int


@dataclass(frozen=True)
class FitAnchor:
    amplitude: float
    shape: Tuple[float, ...]


@dataclass(frozen=True)
class ResolvedAnchor:
    anchor: FitAnchor
    source_period: str
    source_kind: str

    @property
    def label(self) -> str:
        return f"SS-{self.source_period} ({self.source_kind})"


@dataclass(frozen=True)
class FitConstraint:
    source_label: str
    anchor: FitAnchor
    shape_bounds: Tuple[Tuple[str, float, float], ...]
    amplitude_reference: float
    amplitude_bounds: Tuple[float, float]


@dataclass
class PreparedHistograms:
    counts: object
    density: object
    edges: List[float]


@dataclass
class FitData:
    objective: str
    data: object
    options: str
    n_points: int
    keepalive: List[object] = field(default_factory=list)


@dataclass
class FitDiagnostics:
    status: int = 999
    covariance_status: int = -1
    edm: float = math.inf
    objective_value: float = math.inf
    ndf: int = 0
    result_valid: bool = False
    finite_parameters: bool = False
    boundary_parameters: List[str] = field(default_factory=list)

    @property
    def objective_ndf(self) -> float:
        if self.ndf <= 0 or not math.isfinite(self.objective_value):
            return math.inf
        return self.objective_value / self.ndf

    @property
    def covariance_warning_only(self) -> bool:
        # For Minuit2, status=1 means that the minimum was found but the
        # covariance matrix had to be made positive definite.  This affects
        # parameter-error/correlation reliability, not necessarily the central
        # fitted shape.  CovMatrixStatus=2 carries the same warning.
        return self.status == 1 and self.covariance_status >= 2

    @property
    def accepted(self) -> bool:
        result_ok = self.result_valid or self.covariance_warning_only
        return (
            self.status in (0, 1)
            and self.covariance_status >= 2
            and result_ok
            and self.finite_parameters
            and self.ndf > 0
            and math.isfinite(self.edm)
            and self.edm < 1.0e-2
            and math.isfinite(self.objective_value)
        )


@dataclass
class FitMetrics:
    stat_chi2: float = 0.0
    stat_ndf: int = 0
    log_chi2: float = 0.0
    log_ndf: int = 0
    tail_chi2: float = 0.0
    tail_n_points: int = 0

    @staticmethod
    def ratio(value: float, ndf: int) -> float:
        return value / ndf if ndf > 0 and math.isfinite(value) else math.inf

    @property
    def stat_chi2_ndf(self) -> float:
        return self.ratio(self.stat_chi2, self.stat_ndf)

    @property
    def log_chi2_ndf(self) -> float:
        return self.ratio(self.log_chi2, self.log_ndf)

    @property
    def tail_chi2_per_bin(self) -> float:
        return self.tail_chi2 / self.tail_n_points if self.tail_n_points > 0 else math.inf


@dataclass
class FitCandidate:
    fit_function: object
    result: object
    diagnostics: FitDiagnostics
    attempt_index: int
    seed: Tuple[float, ...]
    amplitude_seed: float


@dataclass
class SelectedFit:
    model: FitModelConfig
    function: object                 # smooth density function used for drawing/output
    average_function: object         # numerically bin-averaged function for diagnostics/ratio
    fit_function: object             # objective-specific function used by ROOT
    result: object
    diagnostics: FitDiagnostics
    metrics: FitMetrics
    attempt_index: int
    seed: Tuple[float, ...]
    amplitude_seed: float

    @property
    def accepted(self) -> bool:
        return self.diagnostics.accepted


SS_MODE = ModeConfig(
    key="ss-data",
    display_name="SS data",
    file_name="SS_fit",
    region=SS_REGION,
    fit_min=10.0,
    fit_max=80.0,
    plot_tag="SS",
    legend_label="Data - MC^{Top, Others}",
    components=(
        ("data", +1.0),
        ("NIsoMuon_Top", -1.0),
        ("NIsoMuon_Others", -1.0),
    ),
)

QCD_MODE = ModeConfig(
    key="qcd-mc",
    display_name="QCD MC",
    file_name="QCDMC_fit",
    region=OS_REGION,
    fit_min=6.0,
    fit_max=70.0,
    plot_tag="QCD",
    legend_label="QCD MC",
    components=(("NIsoMuon_QCD_Inclusive", +1.0),),
)

FIT_MODELS: Tuple[FitModelConfig, ...] = (
    FitModelConfig("power_erf", 0, "Power#times ERF", 4),
    FitModelConfig("power_logistic", 1, "Power#times Logistic", 4),
    FitModelConfig("power_exp_erf", 2, "Power#times Exp#times ERF", 5),
    FitModelConfig("power_exp_logistic", 3, "Power#times Exp#times Logistic", 5),
    FitModelConfig("exp_erf", 4, "Exp#times ERF", 4),
    FitModelConfig("exp_logistic", 5, "Exp#times Logistic", 4),
)

# Same-sign central prediction and function-choice envelope.
# The central shape is Power x Exp x ERF.  Every other fitted function enters
# the per-bin envelope, so all six fits must be usable before the ROOT templates
# are written.
SS_NOMINAL_MODEL = "power_exp_logistic"
SS_SHAPE_ALTERNATIVES = (
    "power_erf",
    "power_logistic",
    "power_exp_erf",
    "exp_erf",
    "exp_logistic",
)
SS_REQUIRED_ROOT_MODELS = (SS_NOMINAL_MODEL,) + SS_SHAPE_ALTERNATIVES


# =============================================================================
# FIT INITIAL VALUES AND CONSTRAINTS
# =============================================================================
# Shape tuples exclude the overall amplitude A.  Parameter order is given by
# MODEL_SHAPE_PARAMETER_NAMES.  This is the only place that needs editing when
# initial values or hard constraints are changed.

MODEL_SHAPE_PARAMETER_NAMES: Dict[str, Tuple[str, ...]] = {
    "power_erf": ("n", "m0", "w"),
    "power_logistic": ("n", "m0", "w"),
    "power_exp_erf": ("n", "k", "m0", "w"),
    "power_exp_logistic": ("n", "k", "m0", "w"),
    "exp_erf": ("k", "m0", "w"),
    "exp_logistic": ("k", "m0", "w"),
}

# One adopted set of global hard bounds.  n>=0.15 prevents Power x Exp from
# collapsing exactly to a pure exponential.  QCD-MC anchor bounds below can be
# narrower than these global bounds.
BASE_SHAPE_BOUNDS: Dict[str, Tuple[Tuple[str, float, float], ...]] = {
    "power_erf": (("n", 2.0, 12.0), ("m0", 2.0, 30.0), ("w", 0.20, 8.0)),
    "power_logistic": (("n", 2.0, 12.0), ("m0", 2.0, 30.0), ("w", 0.20, 8.0)),
    "power_exp_erf": (
        ("n", 0.15, 15.0), ("k", 0.0, 5.0), ("m0", 2.0, 35.0), ("w", 0.20, 8.0)
    ),
    "power_exp_logistic": (
        ("n", 0.15, 15.0), ("k", 0.0, 5.0), ("m0", 2.0, 35.0), ("w", 0.20, 8.0)
    ),
    "exp_erf": (("k", 0.0, 5.0), ("m0", 6.0, 35.0), ("w", 0.20, 15.0)),
    "exp_logistic": (("k", 0.0, 5.0), ("m0", 6.0, 35.0), ("w", 0.20, 15.0)),
}

# Static multi-start shape seeds.  Partner-fit and same-era SS-anchor seeds are
# inserted dynamically before these values.  The Power x Exp x Logistic list
# deliberately scans n through 10.
STATIC_SHAPE_SEEDS: Dict[str, Tuple[Tuple[float, ...], ...]] = {
    "power_erf": (
        (4.5, 6.0, 1.5), (4.2, 7.0, 0.8), (4.5, 8.5, 1.2),
        (4.9, 9.5, 1.8), (5.4, 10.5, 2.5), (5.8, 12.0, 2.0),
        (4.4, 8.0, 4.0), (5.5, 13.0, 4.5), (6.5, 15.0, 3.0),
        (7.5, 17.0, 5.0), (4.1, 10.0, 7.0),
    ),
    "power_logistic": (
        (4.5, 6.0, 1.5), (4.2, 7.0, 0.8), (4.5, 8.5, 1.2),
        (4.9, 9.5, 1.8), (5.4, 10.5, 2.5), (5.8, 12.0, 2.0),
        (4.4, 8.0, 4.0), (5.5, 13.0, 4.5), (6.5, 15.0, 3.0),
        (7.5, 17.0, 5.0), (4.1, 10.0, 7.0),
    ),
    "power_exp_erf": (
        (4.0, 0.10, 8.0, 1.5), (3.0, 0.08, 8.5, 1.0),
        (3.0, 0.12, 9.0, 1.8), (2.0, 0.20, 9.2, 2.2),
        (1.0, 0.32, 9.5, 2.8), (0.20, 0.45, 9.7, 3.5),
        (5.0, 0.05, 9.0, 2.0), (6.0, 0.03, 9.8, 3.0),
        (3.5, 0.20, 7.0, 1.0), (0.15, 0.42, 9.8, 2.5),
        (4.0, 0.06, 11.0, 3.0), (3.0, 0.10, 13.0, 4.0),
        (1.0, 0.35, 12.0, 4.0), (0.20, 0.45, 15.0, 6.0),
    ),
    "power_exp_logistic": (
        (1.58, 0.31, 8.97, 1.66),
        (0.20, 0.45, 9.7, 3.5), (0.50, 0.40, 9.5, 3.2),
        (1.0, 0.35, 9.5, 2.8), (2.0, 0.28, 9.2, 2.4),
        (3.0, 0.22, 9.0, 2.0), (4.0, 0.18, 8.8, 1.6),
        (5.0, 0.14, 9.0, 1.8), (6.0, 0.10, 9.5, 2.2),
        (7.0, 0.08, 10.0, 2.6), (8.0, 0.06, 11.0, 3.0),
        (9.0, 0.04, 12.0, 3.5), (10.0, 0.02, 13.0, 4.0),
        (7.0, 0.20, 7.0, 1.0), (8.0, 0.10, 15.0, 5.0),
        (9.0, 0.08, 17.0, 5.5), (10.0, 0.05, 20.0, 6.0),
    ),
    "exp_erf": (
        (0.35, 8.0, 3.0), (0.15, 6.0, 1.0), (0.22, 7.5, 1.5),
        (0.28, 8.5, 2.0), (0.36, 9.3, 2.5), (0.45, 9.7, 3.5),
        (0.18, 7.0, 4.5), (0.30, 5.5, 1.0), (0.50, 9.9, 4.5),
        (0.25, 2.0, 2.0), (0.30, 12.0, 4.0), (0.45, 15.0, 6.0),
    ),
    "exp_logistic": (
        (0.35, 8.0, 3.0), (0.15, 6.0, 1.0), (0.22, 7.5, 1.5),
        (0.28, 8.5, 2.0), (0.36, 9.3, 2.5), (0.45, 9.7, 3.5),
        (0.18, 7.0, 4.5), (0.30, 5.5, 1.0), (0.50, 9.9, 4.5),
        (0.25, 2.0, 2.0), (0.30, 12.0, 4.0), (0.45, 15.0, 6.0),
    ),
}

# Same-sign fit results read from the four user-supplied plots.  They are used
# only to initialise and constrain the corresponding QCD-MC model.
SS_DATA_FIT_ANCHORS: Dict[str, Dict[str, FitAnchor]] = {
    "2016preVFP": {
        "power_erf": FitAnchor(4.53e8, (6.40, 10.73, 2.45)),
        "power_logistic": FitAnchor(3.46e7, (5.48, 9.35, 1.26)),
        "power_exp_erf": FitAnchor(2.24e8, (6.02, 0.02, 10.64, 2.49)),
        "power_exp_logistic": FitAnchor(2.25e6, (3.75, 0.13, 9.29, 1.49)),
        "exp_erf": FitAnchor(4.24e3, (0.38, 8.10, 3.93)),
        "exp_logistic": FitAnchor(6.76e3, (0.41, 9.15, 2.50)),
    },
    "2016postVFP": {
        "power_erf": FitAnchor(1.76e10, (7.79, 11.97, 2.45)),
        "power_logistic": FitAnchor(4.93e7, (5.70, 9.46, 1.26)),
        "power_exp_erf": FitAnchor(1.17e10, (0.73, 0.92, 26.80, 4.86)),
        "power_exp_logistic": FitAnchor(7.96e4, (0.39, 0.50, 12.26, 2.25)),
        "exp_erf": FitAnchor(4.70e8, (0.87, 27.07, 5.39)),
        "exp_logistic": FitAnchor(9.65e4, (0.58, 13.26, 2.24)),
    },
    "2017": {
        "power_erf": FitAnchor(1.79e10, (7.48, 11.63, 2.42)),
        "power_logistic": FitAnchor(1.55e8, (5.79, 9.47, 1.22)),
        "power_exp_erf": FitAnchor(1.09e6, (0.44, 0.56, 15.98, 4.68)),
        "power_exp_logistic": FitAnchor(4.76e4, (0.34, 0.42, 10.39, 2.38)),
        "exp_erf": FitAnchor(5.84e6, (0.69, 20.40, 5.23)),
        "exp_logistic": FitAnchor(3.33e4, (0.46, 10.78, 2.53)),
    },
    "2018": {
        "power_erf": FitAnchor(1.21e10, (7.21, 10.92, 2.26)),
        "power_logistic": FitAnchor(1.70e8, (5.66, 9.06, 1.14)),
        "power_exp_erf": FitAnchor(6.17e7, (4.18, 0.19, 10.54, 2.59)),
        "power_exp_logistic": FitAnchor(3.40e5, (1.58, 0.31, 8.97, 1.66)),
        "exp_erf": FitAnchor(4.90e4, (0.46, 9.81, 3.61)),
        "exp_logistic": FitAnchor(3.37e4, (0.44, 9.04, 2.05)),
    },
}

# QCD constraints around the resolved SS anchors.  The raw SS amplitude is not
# imposed.  The SS shape is first re-normalised to QCD MC; A is then allowed to
# move within the multiplicative factor below.
QCD_ANCHOR_AMPLITUDE_FACTOR = 10.0
QCD_ANCHOR_RELATIVE_HALF_WIDTH = {"n": 0.50, "k": 0.50, "w": 0.50}
QCD_ANCHOR_MIN_ABSOLUTE_HALF_WIDTH = {"n": 0.50, "k": 0.05, "m0": 4.0, "w": 0.75}
QCD_ANCHOR_ABSOLUTE_LIMITS = {
    "n": (0.15, 15.0), "k": (0.0, 5.0), "m0": (0.0, 35.0), "w": (0.20, 12.0)
}

ERF_TO_LOGISTIC_SLOPE_WIDTH = math.sqrt(2.0 * math.pi) / 4.0
ERF_TO_LOGISTIC_VARIANCE_WIDTH = math.sqrt(3.0) / math.pi


# =============================================================================
# C++ FIT FUNCTIONS AND BIN-INTEGRATED OBJECTIVES
# =============================================================================

CPP_FIT_FUNCTIONS = r"""
#include <TF1.h>
#include <TMath.h>
#include <algorithm>
#include <cmath>
#include <vector>

namespace BkgFitFnVariationPy {

  enum Model {
    kPowerErf = 0,
    kPowerLogistic = 1,
    kPowerExpErf = 2,
    kPowerExpLogistic = 3,
    kExpErf = 4,
    kExpLogistic = 5
  };

  enum Objective {
    kBinAverage = 0,
    kBinIntegral = 1,
    kLogBinAverage = 2
  };

  std::vector<double> gBinEdges;
  std::vector<double> gExcludedLow;
  std::vector<double> gExcludedHigh;

  void SetObjectiveBinning(const std::vector<double>& edges,
                           const std::vector<double>& excludedLow,
                           const std::vector<double>& excludedHigh) {
    gBinEdges = edges;
    gExcludedLow = excludedLow;
    gExcludedHigh = excludedHigh;
  }

  bool IsExcluded(double x) {
    for (std::size_t i = 0; i < gExcludedLow.size() && i < gExcludedHigh.size(); ++i) {
      if (x > gExcludedLow[i] && x < gExcludedHigh[i]) return true;
    }
    return false;
  }

  int FindObjectiveBin(double x) {
    if (gBinEdges.size() < 2) return -1;
    if (x < gBinEdges.front() || x > gBinEdges.back()) return -1;
    if (x == gBinEdges.back()) return static_cast<int>(gBinEdges.size()) - 2;
    auto it = std::upper_bound(gBinEdges.begin(), gBinEdges.end(), x);
    int index = static_cast<int>(it - gBinEdges.begin()) - 1;
    if (index < 0 || index + 1 >= static_cast<int>(gBinEdges.size())) return -1;
    return index;
  }

  Double_t PowerLogisticTurnOn(Double_t *x, Double_t *par) {
    const double xx = x[0];
    if (xx <= 0.) return 1e-300;
    const double A = par[0];
    const double n = par[1];
    const double m0 = par[2];
    const double w  = TMath::Abs(par[3]);
    const double turnon = 1. / (1. + TMath::Exp(-(xx - m0) / w));
    return TMath::Max(A * TMath::Power(xx, -n) * turnon, 1e-300);
  }

  Double_t PowerErfTurnOn(Double_t *x, Double_t *par) {
    const double xx = x[0];
    if (xx <= 0.) return 1e-300;
    const double A = par[0];
    const double n = par[1];
    const double m0 = par[2];
    const double w  = TMath::Abs(par[3]);
    const double arg = (xx - m0) / (TMath::Sqrt2() * w);
    const double turnon = 0.5 * (1. + TMath::Erf(arg));
    return TMath::Max(A * TMath::Power(xx, -n) * turnon, 1e-300);
  }

  Double_t PowerExpErfTurnOn(Double_t *x, Double_t *par) {
    const double xx = x[0];
    if (xx <= 0.) return 1e-300;
    const double A = par[0];
    const double n = par[1];
    const double k = par[2];
    const double m0 = par[3];
    const double w  = TMath::Abs(par[4]);
    const double arg = (xx - m0) / (TMath::Sqrt2() * w);
    const double turnon = 0.5 * (1. + TMath::Erf(arg));
    return TMath::Max(A * TMath::Power(xx, -n) * TMath::Exp(-k * xx) * turnon, 1e-300);
  }

  Double_t PowerExpLogisticTurnOn(Double_t *x, Double_t *par) {
    const double xx = x[0];
    if (xx <= 0.) return 1e-300;
    const double A = par[0];
    const double n = par[1];
    const double k = par[2];
    const double m0 = par[3];
    const double w  = TMath::Abs(par[4]);
    const double turnon = 1. / (1. + TMath::Exp(-(xx - m0) / w));
    return TMath::Max(A * TMath::Power(xx, -n) * TMath::Exp(-k * xx) * turnon, 1e-300);
  }

  Double_t ExpErfTurnOn(Double_t *x, Double_t *par) {
    const double xx = x[0];
    if (xx <= 0.) return 1e-300;
    const double A = par[0];
    const double k = par[1];
    const double m0 = par[2];
    const double w  = TMath::Abs(par[3]);
    const double arg = (xx - m0) / (TMath::Sqrt2() * w);
    const double turnon = 0.5 * (1. + TMath::Erf(arg));
    return TMath::Max(A * TMath::Exp(-k * xx) * turnon, 1e-300);
  }

  Double_t ExpLogisticTurnOn(Double_t *x, Double_t *par) {
    const double xx = x[0];
    if (xx <= 0.) return 1e-300;
    const double A = par[0];
    const double k = par[1];
    const double m0 = par[2];
    const double w  = TMath::Abs(par[3]);
    const double turnon = 1. / (1. + TMath::Exp(-(xx - m0) / w));
    return TMath::Max(A * TMath::Exp(-k * xx) * turnon, 1e-300);
  }

  double EvalDensity(int model, double x, double *par) {
    double xx[1] = {x};
    if (model == kPowerErf) return PowerErfTurnOn(xx, par);
    if (model == kPowerLogistic) return PowerLogisticTurnOn(xx, par);
    if (model == kPowerExpErf) return PowerExpErfTurnOn(xx, par);
    if (model == kPowerExpLogistic) return PowerExpLogisticTurnOn(xx, par);
    if (model == kExpErf) return ExpErfTurnOn(xx, par);
    return ExpLogisticTurnOn(xx, par);
  }

  double IntegrateDensity(int model, double low, double high, double *par) {
    // Fixed 8-point Gauss-Legendre integration.  This removes the adaptive GSL
    // failures seen for narrow turn-ons while retaining bin-integrated fits.
    static const double nodes[4] = {
      0.1834346424956498, 0.5255324099163290,
      0.7966664774136267, 0.9602898564975363
    };
    static const double weights[4] = {
      0.3626837833783620, 0.3137066458778873,
      0.2223810344533745, 0.1012285362903763
    };
    const double mid = 0.5 * (low + high);
    const double half = 0.5 * (high - low);
    double sum = 0.;
    for (int i = 0; i < 4; ++i) {
      const double dx = half * nodes[i];
      sum += weights[i] * (EvalDensity(model, mid - dx, par) + EvalDensity(model, mid + dx, par));
    }
    return half * sum;
  }

  double EvalObjective(int model, int objective, double x, double *par) {
    if (IsExcluded(x)) {
      TF1::RejectPoint();
      return 0.;
    }
    const int ibin = FindObjectiveBin(x);
    if (ibin < 0) {
      TF1::RejectPoint();
      return 0.;
    }
    const double low = gBinEdges[ibin];
    const double high = gBinEdges[ibin + 1];
    const double integral = IntegrateDensity(model, low, high, par);
    if (objective == kBinIntegral) return TMath::Max(integral, 1e-300);
    const double average = integral / (high - low);
    if (objective == kLogBinAverage) return TMath::Log(TMath::Max(average, 1e-300));
    return TMath::Max(average, 1e-300);
  }

  TF1 *MakeFitFunction(const char *name, int model, double xmin, double xmax) {
    if (model == kPowerErf) return new TF1(name, PowerErfTurnOn, xmin, xmax, 4);
    if (model == kPowerLogistic) return new TF1(name, PowerLogisticTurnOn, xmin, xmax, 4);
    if (model == kPowerExpErf) return new TF1(name, PowerExpErfTurnOn, xmin, xmax, 5);
    if (model == kPowerExpLogistic) return new TF1(name, PowerExpLogisticTurnOn, xmin, xmax, 5);
    if (model == kExpErf) return new TF1(name, ExpErfTurnOn, xmin, xmax, 4);
    return new TF1(name, ExpLogisticTurnOn, xmin, xmax, 4);
  }

  TF1 *MakeObjectiveFunction(const char *name, int model, int objective,
                             double xmin, double xmax, int npar) {
    return new TF1(name,
      [model, objective](double *x, double *par) {
        return EvalObjective(model, objective, x[0], par);
      }, xmin, xmax, npar);
  }
}
"""


HOW_TO_RUN = __doc__ or ""


# =============================================================================
# GENERAL UTILITIES AND ROOT SETUP
# =============================================================================


class NameFactory:
    def __init__(self) -> None:
        self.counter = 0

    def unique(self, prefix: str) -> str:
        self.counter += 1
        clean = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in prefix)
        return f"{clean}_{self.counter}"


_NAMES = NameFactory()


def positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value!r}.") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {parsed}.")
    return parsed


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a positive number, got {value!r}.") from exc
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError(f"Expected a positive number, got {parsed}.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qcd_bkg_estimation.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Fit SS data or QCD MC with the six background functions.",
        epilog=HOW_TO_RUN,
    )
    parser.add_argument("--mode", required=True, help="ss-data or qcd-mc")
    parser.add_argument(
        "--year", "--era", dest="year", default="2018", choices=VALID_YEARS,
        help="Individual era, 2016, Run2, Run3, or Run2+3",
    )
    parser.add_argument(
        "--trigger", default="",
        help="Optional legacy trigger subdirectory; empty for the SKOutput layout",
    )
    parser.add_argument("--analyzer", default="NIsoMuon")
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument(
        "--no-prefer-full-run-anchor", dest="prefer_full_run_anchor",
        action="store_false",
        help=("For an individual era, prefer its own SS anchor over the full "
              "Run2 or Run3 anchor. By default the full-run anchor is tried first."),
    )
    parser.set_defaults(prefer_full_run_anchor=True)
    parser.add_argument("--rebin", "--rebin-factor", dest="rebin", type=positive_integer, default=1)
    parser.add_argument(
        "--fit-objective",
        choices=("auto", "chi2", "weighted-likelihood", "log-chi2"),
        default="auto",
        help=(
            "auto: chi2 for SS data and weighted-likelihood for QCD MC; "
            "chi2: bin-error chi-square; weighted-likelihood: ROOT WL on weighted "
            "bin yields; log-chi2: fit log density. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--log-relative-error-floor",
        type=positive_float,
        default=0.10,
        help=(
            "Minimum relative uncertainty in log-chi2. A 0.10 floor caps the "
            "per-bin weight at 1/0.1^2 and reduces low-mass domination. Default: %(default)s"
        ),
    )
    parser.add_argument("--fit-max-attempts", type=positive_integer, default=30)
    parser.add_argument("--fit-attempt-details", action="store_true")
    parser.add_argument(
        "--initial-values-only",
        "--qcd-anchor-initial-only",
        dest="initial_values_only",
        action="store_true",
        help=(
            "In qcd-mc mode, use the resolved SS-data anchor values as the first "
            "shape seeds, but do not apply their shape or amplitude hard "
            "constraints. The common BASE_SHAPE_BOUNDS and broad adaptive A "
            "bounds remain active."
        ),
    )
    parser.add_argument("--allow-invalid-fit-output", action="store_true")
    parser.add_argument("--print-shape-syst-bin-info", action="store_true")
    parser.add_argument("--print-raw-shape-debug", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def canonical_mode(value: str) -> ModeConfig:
    key = "".join(ch for ch in value.lower() if ch.isalnum())
    if key in {"ss", "ssdata", "datass", "samesign", "samesigndata"}:
        return SS_MODE
    if key in {"qcd", "qcdmc", "mcqcd", "qcdsimulation"}:
        return QCD_MODE
    raise ValueError(f"Unknown --mode {value!r}; use ss-data or qcd-mc.")


def resolve_objective(requested: str, mode: ModeConfig) -> str:
    objective = requested
    if objective == "auto":
        objective = "chi2" if mode.key == SS_MODE.key else "weighted-likelihood"
    if mode.key == SS_MODE.key and objective != "chi2":
        raise ValueError(
            "SS data is a data-minus-background histogram and may contain non-Poisson or "
            "negative bins; only --fit-objective chi2 is allowed in ss-data mode."
        )
    return objective


def import_root():
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        import ROOT  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Could not import PyROOT. Run inside a ROOT/CMSSW environment, e.g. after cmsenv. "
            f"Original error: {exc}"
        ) from exc
    finally:
        sys.argv = original_argv

    ROOT.PyConfig.IgnoreCommandLineOptions = True
    ROOT.gROOT.SetBatch(True)
    try:
        ROOT.TH1.AddDirectory(False)
    except Exception:
        pass
    return ROOT


def declare_fit_functions(ROOT) -> None:
    if ROOT.gInterpreter.Declare(CPP_FIT_FUNCTIONS) is False:
        raise RuntimeError("Failed to declare fit functions.")


def configure_minimizer(ROOT) -> None:
    try:
        ROOT.Math.MinimizerOptions.SetDefaultMinimizer("Minuit2", "Migrad")
        ROOT.Math.MinimizerOptions.SetDefaultStrategy(1)
        ROOT.Math.MinimizerOptions.SetDefaultMaxFunctionCalls(200000)
        ROOT.Math.MinimizerOptions.SetDefaultMaxIterations(200000)
    except Exception as exc:
        print(f"[WARNING] Could not configure ROOT minimizer: {exc}")


def cpp_float(value: float) -> str:
    return format(float(value), ".6g")


def selected_eras(period: str) -> Tuple[str, ...]:
    try:
        return PERIOD_ERAS[period]
    except KeyError as exc:
        raise ValueError(
            f"Unknown period {period!r}; allowed values: {', '.join(VALID_YEARS)}"
        ) from exc


def analyzer_base_dir(args: argparse.Namespace) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(args.base_dir))) / args.analyzer


def era_dir(args: argparse.Namespace, era: str, collection: str = "") -> Path:
    directory = analyzer_base_dir(args)
    if collection:
        directory = directory / collection
    directory = directory / era
    if args.trigger:
        directory = directory / args.trigger
    return directory


def input_dirs(args: argparse.Namespace) -> List[Tuple[str, Path]]:
    return [(era, era_dir(args, era)) for era in selected_eras(args.year)]


def anchor_file_path(args: argparse.Namespace) -> Path:
    # Fixed by analysis convention; retained as a function to keep callers simple.
    return ANCHOR_FILE


def hist_path(region: str, hist_name: str = HIST_NAME) -> str:
    return f"{region}/{hist_name}___{region}"


def clone_detached(hist, name: str):
    if not hist:
        return None
    out = hist.Clone(name)
    out.SetDirectory(0)
    return out


def open_root_file(ROOT, path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing ROOT file: {path}")
    root_file = ROOT.TFile.Open(str(path), "READ")
    if not root_file or root_file.IsZombie():
        if root_file:
            root_file.Close()
        raise OSError(f"Could not open ROOT file: {path}")
    return root_file


def get_required_histogram(root_file, file_label: str, path: str):
    print(f"[Call hist] {file_label}.root:{path} -> ", end="", flush=True)
    hist = root_file.Get(path)
    if not hist:
        print("FAIL")
        raise KeyError(f"Missing histogram: {file_label}.root:{path}")
    print("SUCCESS")
    print(
        f"[MC hist info] Nbins={hist.GetNbinsX()} "
        f"XRangeMin={hist.GetXaxis().GetXmin():g} XRangeMax={hist.GetXaxis().GetXmax():g}"
    )
    return hist


def get_histogram_clone(root_file, path: str, name: str):
    if not root_file or root_file.IsZombie():
        return None
    hist = root_file.Get(path)
    return clone_detached(hist, name) if hist else None


def build_input_histogram(
    ROOT,
    mode: ModeConfig,
    directories: Sequence[Tuple[str, Path]],
):
    """Build one fit histogram, summing the requested component files over eras."""
    files: Dict[str, object] = {}
    handles: List[object] = []
    combined = None
    path = hist_path(mode.region)

    try:
        for era, directory in directories:
            for stem, scale in mode.components:
                root_file = open_root_file(ROOT, directory / f"{stem}.root")
                files[f"{era}:{stem}"] = root_file
                if len(directories) == 1:
                    # Preserve the legacy keys used by the per-era ROOT writer.
                    files[stem] = root_file
                handles.append(root_file)

                label = f"{era}/{stem}"
                source = get_required_histogram(root_file, label, path)
                piece = source.Clone(_NAMES.unique(f"input_{era}_{stem}"))
                piece.SetDirectory(0)
                if int(piece.GetSumw2N()) == 0:
                    piece.Sumw2()
                if scale != 1.0:
                    piece.Scale(scale)

                if combined is None:
                    combined = piece
                    combined.SetName("Hist_0")
                else:
                    combined.Add(piece)

        if combined is None:
            raise RuntimeError("No input histograms were loaded.")
        print(
            f"[SYSTEM] eras={len(directories)}, components/era={len(mode.components)}, "
            "combined histograms=1"
        )
        return combined, files, handles
    except Exception:
        for handle in handles:
            handle.Close()
        raise


# =============================================================================
# BINNING AND FIT DATA
# =============================================================================


def validate_bin_edges(edges: Sequence[float], label: str) -> None:
    if len(edges) < 2:
        raise ValueError(f"{label} contains fewer than two edges.")
    if any(not right > left for left, right in zip(edges, edges[1:])):
        raise ValueError(f"{label} is not strictly increasing.")


def make_variable_binning(mode: ModeConfig) -> List[float]:
    edges: List[float] = []
    if mode.key == SS_MODE.key:
        segments = ((0.0, 8.0, 0.1), (8.0, 11.0, 0.2), (11.0, 15.0, 0.5))
        tail_edges = (15.0, 16.0, 17.0, 19.0, 21.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0)
    else:
        segments = (
            (0.0, 9.0, 0.1), (9.0, 11., 0.2),
            (11.0, 20.0, 0.5), (20.0, 40.0, 2.0), (40.0, 70.0, 5.0),
        )
        tail_edges = (70.0, 80.0, 90.0, 100.0)

    for xmin, xmax, step in segments:
        n_bins = int(round((xmax - xmin) / step))
        for index in range(n_bins):
            edges.append(round(xmin + index * step, 12))
    edges.extend(tail_edges)
    validate_bin_edges(edges, f"{mode.key} binning")
    return edges


def merge_adjacent_bin_edges(base_edges: Sequence[float], factor: int, protected_edges: Sequence[float]) -> List[float]:
    edges = [float(x) for x in base_edges]
    if factor == 1:
        return edges

    protected_indices = []
    for protected in protected_edges:
        found = [i for i, edge in enumerate(edges) if math.isclose(edge, protected, abs_tol=1e-10, rel_tol=0.0)]
        if not found:
            raise ValueError(f"Protected edge {protected:g} GeV is absent.")
        protected_indices.append(found[0])

    splits = sorted(set([0, len(edges) - 1, *protected_indices]))
    merged = [edges[0]]
    for start, stop in zip(splits, splits[1:]):
        index = start + factor
        while index < stop:
            merged.append(edges[index])
            index += factor
        if not math.isclose(merged[-1], edges[stop], abs_tol=1e-12, rel_tol=0.0):
            merged.append(edges[stop])
    validate_bin_edges(merged, "rebinned edges")
    return merged


def clone_to_th1d(ROOT, source, name: str, edges: Sequence[float]):
    """Copy a rebinned TH1 into a double-precision TH1D without changing yields.

    ROOT preserves the original histogram precision in TH1::Rebin.  The input
    NIsoMuon histograms are TH1F, so a large Run2/Run3 sum can accumulate
    visible float-rounding after division by variable bin widths.  The fit
    itself does not need single precision, therefore the rebinned histogram is
    copied once into TH1D before the Events -> Events/GeV conversion.
    """
    out = ROOT.TH1D(
        _NAMES.unique(name),
        "",
        len(edges) - 1,
        array("d", [float(x) for x in edges]),
    )
    out.SetDirectory(0)
    out.Sumw2()

    if source.GetNbinsX() != out.GetNbinsX():
        raise RuntimeError(
            f"TH1D conversion bin-count mismatch: "
            f"{source.GetNbinsX()} versus {out.GetNbinsX()}"
        )

    # Preserve regular bins and under/overflow.  Only regular bins enter the fit
    # yield checks, but keeping the flow bins makes the copy fully transparent.
    for ibin in range(0, source.GetNbinsX() + 2):
        out.SetBinContent(ibin, float(source.GetBinContent(ibin)))
        out.SetBinError(ibin, float(source.GetBinError(ibin)))
    return out


def regular_bin_yield(hist) -> float:
    return float(hist.Integral(1, hist.GetNbinsX()))


def density_yield(hist) -> float:
    return sum(
        float(hist.GetBinContent(i)) * float(hist.GetXaxis().GetBinWidth(i))
        for i in range(1, hist.GetNbinsX() + 1)
    )


def yields_agree(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-7 * max(1.0, abs(a), abs(b))


def prepare_histograms(ROOT, source_hist, mode: ModeConfig, rebin_factor: int) -> PreparedHistograms:
    base_edges = make_variable_binning(mode)
    protected = (9.0, 11.) if mode.key == QCD_MODE.key else (10.0, 80.0)
    final_edges = merge_adjacent_bin_edges(base_edges, rebin_factor, protected)

    base_source = source_hist.Clone(_NAMES.unique("base_rebin_source"))
    base_source.SetDirectory(0)
    base_hist_raw = base_source.Rebin(
        len(base_edges) - 1, _NAMES.unique("base_rebin_raw"), array("d", base_edges)
    )
    base_hist_raw.SetDirectory(0)
    base_hist = clone_to_th1d(ROOT, base_hist_raw, "base_rebin", base_edges)
    base_yield = regular_bin_yield(base_hist)

    final_source = source_hist.Clone(_NAMES.unique("final_rebin_source"))
    final_source.SetDirectory(0)
    counts_raw = final_source.Rebin(
        len(final_edges) - 1, _NAMES.unique("fit_counts_raw"), array("d", final_edges)
    )
    counts_raw.SetDirectory(0)

    # From this point onward use double precision.  This is important for the
    # combined Run2/Run3 SS histograms, which can have large bin contents.
    counts = clone_to_th1d(ROOT, counts_raw, "fit_counts", final_edges)
    count_yield = regular_bin_yield(counts)

    if not yields_agree(base_yield, count_yield):
        delta = count_yield - base_yield
        rel = delta / base_yield if base_yield != 0.0 else math.inf
        raise RuntimeError(
            "Rebinning changed the yield: "
            f"base={base_yield:.12g}, final={count_yield:.12g}, "
            f"delta={delta:.12g}, relative={rel:.6g}"
        )

    density = clone_to_th1d(ROOT, counts, "fit_density", final_edges)
    for ibin in range(1, density.GetNbinsX() + 1):
        width = float(density.GetXaxis().GetBinWidth(ibin))
        if width <= 0.0:
            raise RuntimeError(f"Invalid bin width in bin {ibin}: {width}")
        density.SetBinContent(ibin, float(density.GetBinContent(ibin)) / width)
        density.SetBinError(ibin, float(density.GetBinError(ibin)) / width)

    recovered_yield = density_yield(density)
    if not yields_agree(count_yield, recovered_yield):
        delta = recovered_yield - count_yield
        rel = delta / count_yield if count_yield != 0.0 else math.inf
        raise RuntimeError(
            "Events/GeV conversion changed the yield: "
            f"before={count_yield:.12g}, after={recovered_yield:.12g}, "
            f"delta={delta:.12g}, relative={rel:.6g}"
        )

    print(
        "[YIELD CHECK] Events -> Events/GeV conversion preserved the yield: "
        f"{count_yield:.12g} -> {recovered_yield:.12g}"
    )

    density.SetMarkerColor(ROOT.kBlack)
    density.SetMarkerStyle(20)
    density.SetMarkerSize(0.4)
    density.SetStats(0)

    widths = [density.GetBinWidth(i) for i in range(1, density.GetNbinsX() + 1)]
    print(
        f"[BINNING] mode={mode.key} baseBins={len(base_edges)-1} finalBins={len(final_edges)-1} "
        f"minWidth={min(widths):g} GeV maxWidth={max(widths):g} GeV"
    )
    print(f"[REBIN] factor={rebin_factor} yield={count_yield:.12g} (preserved)")
    return PreparedHistograms(counts=counts, density=density, edges=final_edges)


def is_excluded(mode: ModeConfig, low: float, high: float) -> bool:
    if mode.key != QCD_MODE.key:
        return False
    tolerance = 1e-10
    for rlow, rhigh in QCD_FIT_EXCLUDED_RANGES:
        overlap = min(high, rhigh) - max(low, rlow)
        if overlap <= tolerance:
            continue
        if low < rlow - tolerance or high > rhigh + tolerance:
            raise RuntimeError(
                f"Fit bin [{low:g},{high:g}] crosses excluded QCD interval [{rlow:g},{rhigh:g}]."
            )
        return True
    return False


def configure_objective_binning(ROOT, prepared: PreparedHistograms, mode: ModeConfig) -> None:
    edges = ROOT.std.vector("double")()
    lows = ROOT.std.vector("double")()
    highs = ROOT.std.vector("double")()
    for value in prepared.edges:
        edges.push_back(float(value))
    if mode.key == QCD_MODE.key:
        for low, high in QCD_FIT_EXCLUDED_RANGES:
            lows.push_back(low)
            highs.push_back(high)
    ROOT.BkgFitFnVariationPy.SetObjectiveBinning(edges, lows, highs)


def make_log_graph(ROOT, density, mode: ModeConfig, relative_floor: float):
    graph = ROOT.TGraphErrors()
    graph.SetName(_NAMES.unique("log_fit_graph"))
    for ibin in range(1, density.GetNbinsX() + 1):
        low = float(density.GetXaxis().GetBinLowEdge(ibin))
        high = float(density.GetXaxis().GetBinUpEdge(ibin))
        x = float(density.GetBinCenter(ibin))
        if x < mode.fit_min or x > mode.fit_max or is_excluded(mode, low, high):
            continue
        y = float(density.GetBinContent(ibin))
        error = float(density.GetBinError(ibin))
        if y <= 0.0 or error <= 0.0 or not (math.isfinite(y) and math.isfinite(error)):
            continue
        index = graph.GetN()
        graph.SetPoint(index, x, math.log(y))
        graph.SetPointError(index, 0.0, max(error / y, relative_floor))
    if graph.GetN() < 6:
        raise RuntimeError(f"Only {graph.GetN()} positive bins are available for log-chi2.")
    return graph


def build_fit_data(ROOT, prepared: PreparedHistograms, mode: ModeConfig, objective: str, log_floor: float) -> FitData:
    configure_objective_binning(ROOT, prepared, mode)

    if objective == "chi2":
        n_points = 0
        for ibin in range(1, prepared.density.GetNbinsX() + 1):
            low = prepared.density.GetXaxis().GetBinLowEdge(ibin)
            high = prepared.density.GetXaxis().GetBinUpEdge(ibin)
            x = prepared.density.GetBinCenter(ibin)
            if mode.fit_min <= x <= mode.fit_max and not is_excluded(mode, low, high) and prepared.density.GetBinError(ibin) > 0:
                n_points += 1

        # Preserve the original SS-data fitting path exactly: fit the smooth
        # density function and let ROOT's I option perform the bin integration.
        # The explicit bin-average wrapper remains useful for QCD MC, where the
        # 9--11. GeV veto interval must be rejected bin by bin.
        options = "S R I Q 0 N" if mode.key == SS_MODE.key else "S R Q 0 N"
        return FitData(objective, prepared.density, options, n_points)

    if objective == "weighted-likelihood":
        negative = []
        n_points = 0
        for ibin in range(1, prepared.counts.GetNbinsX() + 1):
            low = prepared.counts.GetXaxis().GetBinLowEdge(ibin)
            high = prepared.counts.GetXaxis().GetBinUpEdge(ibin)
            x = prepared.counts.GetBinCenter(ibin)
            if x < mode.fit_min or x > mode.fit_max or is_excluded(mode, low, high):
                continue
            n_points += 1
            if prepared.counts.GetBinContent(ibin) < 0.0:
                negative.append(ibin)
        if negative:
            raise RuntimeError(
                "weighted-likelihood cannot be used with negative weighted bins: "
                + ", ".join(map(str, negative))
                + ". Use --fit-objective chi2."
            )
        return FitData(objective, prepared.counts, "S R WL Q 0 N", n_points)

    graph = make_log_graph(ROOT, prepared.density, mode, log_floor)
    return FitData(objective, graph, "S R Q 0 N", int(graph.GetN()), [graph])


# =============================================================================
# INITIALISATION, CONSTRAINTS, AND MINIMISATION
# =============================================================================


def shape_bounds(model: FitModelConfig, constraint: Optional[FitConstraint]) -> Tuple[Tuple[str, float, float], ...]:
    return constraint.shape_bounds if constraint is not None else BASE_SHAPE_BOUNDS[model.key]


def clip_seed(seed: Sequence[float], bounds: Sequence[Tuple[str, float, float]]) -> Tuple[float, ...]:
    output = []
    for value, (_, low, high) in zip(seed, bounds):
        epsilon = max(1e-6 * (high - low), 1e-8)
        output.append(min(max(float(value), low + epsilon), high - epsilon))
    return tuple(output)


def deduplicate_seeds(seeds: Iterable[Sequence[float]]) -> List[Tuple[float, ...]]:
    output: List[Tuple[float, ...]] = []
    seen = set()
    for seed in seeds:
        key = tuple(round(float(value), 8) for value in seed)
        if key in seen:
            continue
        seen.add(key)
        output.append(tuple(float(value) for value in seed))
    return output


def erf_width_to_logistic_width(width: float, method: str) -> float:
    factor = ERF_TO_LOGISTIC_SLOPE_WIDTH if method == "slope" else ERF_TO_LOGISTIC_VARIANCE_WIDTH
    return max(float(width) * factor, 1e-6)


def build_seed_list(
    model: FitModelConfig,
    selected_by_key: Dict[str, SelectedFit],
    anchor: Optional[FitAnchor],
    constraint: Optional[FitConstraint],
) -> List[Tuple[float, ...]]:
    seeds: List[Tuple[float, ...]] = list(STATIC_SHAPE_SEEDS[model.key])
    if anchor is not None:
        seeds.insert(0, anchor.shape)

    if model.key in {"power_erf", "power_logistic"}:
        partner = "power_logistic" if model.key == "power_erf" else "power_erf"
        if partner in selected_by_key:
            fn = selected_by_key[partner].function
            seeds.insert(1, (fn.GetParameter(1), fn.GetParameter(2), fn.GetParameter(3)))

    elif model.key in {"exp_erf", "exp_logistic"}:
        partner = "exp_logistic" if model.key == "exp_erf" else "exp_erf"
        if partner in selected_by_key:
            fn = selected_by_key[partner].function
            seeds.insert(1, (fn.GetParameter(1), fn.GetParameter(2), fn.GetParameter(3)))

    elif model.key in {"power_exp_erf", "power_exp_logistic"}:
        partner = "power_exp_logistic" if model.key == "power_exp_erf" else "power_exp_erf"
        if partner in selected_by_key:
            fn = selected_by_key[partner].function
            n, k, m0, w = (float(fn.GetParameter(i)) for i in range(1, 5))
            if model.key == "power_exp_logistic":
                seeds[0:0] = [
                    (n, k, m0, erf_width_to_logistic_width(w, "slope")),
                    (n, k, m0, erf_width_to_logistic_width(w, "variance")),
                    (n, k, m0, w),
                ]
            else:
                seeds.insert(0, (n, k, m0, w))

        simple = "exp_erf" if model.key == "power_exp_erf" else "exp_logistic"
        if simple in selected_by_key:
            fn = selected_by_key[simple].function
            seeds.insert(0, (0.15, fn.GetParameter(1), fn.GetParameter(2), fn.GetParameter(3)))

    bounds = shape_bounds(model, constraint)
    return deduplicate_seeds(clip_seed(seed, bounds) for seed in seeds)


def parameter_step(name: str, value: float, low: float, high: float, scale: float = 1.0) -> float:
    span = max(high - low, 1e-12)
    value = abs(value)
    if name == "n":
        step = min(max(0.03 * max(value, 1.0), 0.02), 0.15)
    elif name == "k":
        step = min(max(0.05 * max(value, 0.02), 0.001), 0.02)
    elif name == "m0":
        step = min(max(0.01 * max(value, 2.0), 0.02), 0.15)
    elif name == "w":
        step = min(max(0.03 * max(value, 0.20), 0.01), 0.10)
    else:
        step = 0.01 * span
    return min(max(step * scale, 1e-5), 0.25 * span)


def set_shape_parameters(function, seed: Sequence[float]) -> None:
    for index, value in enumerate(seed, start=1):
        function.SetParameter(index, float(value))


def adaptive_amplitude_seed(function, density, mode: ModeConfig) -> float:
    function.SetParameter(0, 1.0)
    numerator = 0.0
    denominator = 0.0
    fallbacks: List[float] = []
    for ibin in range(1, density.GetNbinsX() + 1):
        low = max(float(density.GetXaxis().GetBinLowEdge(ibin)), mode.fit_min)
        high = min(float(density.GetXaxis().GetBinUpEdge(ibin)), mode.fit_max)
        if high <= low or is_excluded(mode, low, high):
            continue
        y = float(density.GetBinContent(ibin))
        error = float(density.GetBinError(ibin))
        if error <= 0.0 or not (math.isfinite(y) and math.isfinite(error)):
            continue
        unit = float(function.Integral(low, high)) / (high - low)
        if unit <= 0.0 or not math.isfinite(unit):
            continue
        weight = 1.0 / (error * error)
        numerator += y * unit * weight
        denominator += unit * unit * weight
        if y > 0.0:
            fallbacks.append(y / unit)
    if denominator > 0.0:
        amplitude = numerator / denominator
        if amplitude > 0.0 and math.isfinite(amplitude):
            return amplitude
    fallbacks = sorted(x for x in fallbacks if x > 0.0 and math.isfinite(x))
    return fallbacks[len(fallbacks) // 2] if fallbacks else 1.0


def anchor_shape_bounds(model: FitModelConfig, anchor: FitAnchor) -> Tuple[Tuple[str, float, float], ...]:
    names = MODEL_SHAPE_PARAMETER_NAMES[model.key]
    bounds = []
    for name, value in zip(names, anchor.shape):
        half = max(
            QCD_ANCHOR_RELATIVE_HALF_WIDTH.get(name, 0.0) * abs(value),
            QCD_ANCHOR_MIN_ABSOLUTE_HALF_WIDTH[name],
        )
        abs_low, abs_high = QCD_ANCHOR_ABSOLUTE_LIMITS[name]
        low = max(abs_low, value - half)
        high = min(abs_high, value + half)
        if not high > low:
            raise ValueError(f"Invalid QCD anchor bounds for {model.key} {name}: [{low},{high}]")
        bounds.append((name, low, high))
    return tuple(bounds)


def _anchor_from_json_entry(entry: object) -> Optional[FitAnchor]:
    if not isinstance(entry, dict):
        return None
    if entry.get("accepted") is False:
        return None
    try:
        amplitude = float(entry["amplitude"])
        shape = tuple(float(value) for value in entry["shape"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(amplitude) or any(not math.isfinite(value) for value in shape):
        return None
    return FitAnchor(amplitude=amplitude, shape=shape)


def load_dynamic_anchor_catalog(args: argparse.Namespace) -> Dict[str, Dict[str, FitAnchor]]:
    path = anchor_file_path(args)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read SS anchor file {path}: {exc}") from exc

    raw_catalog = payload.get("anchors", payload)
    catalog: Dict[str, Dict[str, FitAnchor]] = {}
    if not isinstance(raw_catalog, dict):
        return catalog
    for period, models in raw_catalog.items():
        if not isinstance(models, dict):
            continue
        parsed: Dict[str, FitAnchor] = {}
        for model_key, entry in models.items():
            anchor = _anchor_from_json_entry(entry)
            if anchor is not None:
                parsed[str(model_key)] = anchor
        if parsed:
            catalog[str(period)] = parsed
    return catalog


def full_run_period(period: str) -> Optional[str]:
    if period in RUN2_ERAS or period == "2016":
        return "Run2"
    if period in RUN3_ERAS:
        return "Run3"
    if period in {"Run2", "Run3", "Run2+3"}:
        return period
    return None


def anchor_priority(period: str, prefer_full_run: bool) -> List[str]:
    if period == "Run2+3":
        # A 13 + 13.6 TeV combined QCD fit should be constrained by its own
        # combined SS fit, not by arbitrarily choosing only the Run2 or Run3 shape.
        return ["Run2+3"]
    full = full_run_period(period)
    if full is None or full == period:
        return [period]
    return [full, period] if prefer_full_run else [period, full]


def get_qcd_anchor(
    mode: ModeConfig,
    period: str,
    model: FitModelConfig,
    dynamic_catalog: Dict[str, Dict[str, FitAnchor]],
    prefer_full_run: bool,
) -> Optional[ResolvedAnchor]:
    if mode.key != QCD_MODE.key:
        return None

    tried: List[str] = []
    for candidate in anchor_priority(period, prefer_full_run):
        tried.append(candidate)
        dynamic = dynamic_catalog.get(candidate, {}).get(model.key)
        if dynamic is not None:
            return ResolvedAnchor(dynamic, candidate, "JSON")
        static = SS_DATA_FIT_ANCHORS.get(candidate, {}).get(model.key)
        if static is not None:
            return ResolvedAnchor(static, candidate, "built-in")

    raise KeyError(
        f"No SS-data anchor for period={period}, model={model.key}; "
        f"tried {', '.join(tried)} in {anchor_file_path_placeholder()} and built-ins"
    )


def anchor_file_path_placeholder() -> str:
    return str(ANCHOR_FILE)


def qcd_anchor_amplitude_reference(
    ROOT, density, mode: ModeConfig, model: FitModelConfig, anchor: FitAnchor
) -> float:
    display = ROOT.BkgFitFnVariationPy.MakeFitFunction(
        _NAMES.unique(f"anchor_{model.key}"), model.cpp_id, mode.fit_min, mode.fit_max
    )
    if not display:
        raise RuntimeError(f"Could not construct anchor function for {model.label}")
    set_shape_parameters(display, anchor.shape)
    return max(adaptive_amplitude_seed(display, density, mode), 1e-20)


def build_qcd_constraint(
    ROOT,
    density,
    mode: ModeConfig,
    period: str,
    model: FitModelConfig,
    resolved: Optional[ResolvedAnchor],
) -> Optional[FitConstraint]:
    if resolved is None:
        return None
    anchor = resolved.anchor
    amplitude_reference = qcd_anchor_amplitude_reference(
        ROOT, density, mode, model, anchor
    )
    amplitude_bounds = (
        amplitude_reference / QCD_ANCHOR_AMPLITUDE_FACTOR,
        amplitude_reference * QCD_ANCHOR_AMPLITUDE_FACTOR,
    )
    constraint = FitConstraint(
        source_label=resolved.label,
        anchor=anchor,
        shape_bounds=anchor_shape_bounds(model, anchor),
        amplitude_reference=amplitude_reference,
        amplitude_bounds=amplitude_bounds,
    )
    shape_text = ", ".join(
        f"{name}={value:g}->[{low:g},{high:g}]"
        for (name, low, high), value in zip(constraint.shape_bounds, anchor.shape)
    )
    print(
        f"[INITIAL/CONSTRAINT] period={period} anchor={resolved.label} "
        f"model={model.label.replace('#times','x')} "
        f"SS_A={anchor.amplitude:.6g} QCD_Aref={amplitude_reference:.6g} "
        f"A=[{amplitude_bounds[0]:.6g},{amplitude_bounds[1]:.6g}] {shape_text}"
    )
    return constraint


def print_qcd_initial_only(
    ROOT,
    density,
    mode: ModeConfig,
    period: str,
    model: FitModelConfig,
    resolved: Optional[ResolvedAnchor],
) -> None:
    if resolved is None:
        return
    anchor = resolved.anchor
    amplitude_reference = qcd_anchor_amplitude_reference(
        ROOT, density, mode, model, anchor
    )
    shape_text = ", ".join(
        f"{name}={value:g}"
        for name, value in zip(MODEL_SHAPE_PARAMETER_NAMES[model.key], anchor.shape)
    )
    bounds_text = ", ".join(
        f"{name}=[{low:g},{high:g}]"
        for name, low, high in BASE_SHAPE_BOUNDS[model.key]
    )
    print(
        f"[INITIAL-ONLY] period={period} anchor={resolved.label} "
        f"model={model.label.replace('#times','x')} "
        f"SS_A={anchor.amplitude:.6g} QCD_Aseed={amplitude_reference:.6g} "
        f"seed=({shape_text}); no SS-derived constraints; {bounds_text}"
    )


def configure_parameters(function, model: FitModelConfig, seed: Sequence[float], amplitude: float, constraint: Optional[FitConstraint]) -> None:
    set_shape_parameters(function, seed)
    function.SetParameter(0, amplitude)
    for index, (name, low, high) in enumerate(shape_bounds(model, constraint), start=1):
        function.SetParLimits(index, low, high)
        try:
            function.SetParError(index, parameter_step(name, function.GetParameter(index), low, high))
        except Exception:
            pass

    if constraint is not None:
        low, high = constraint.amplitude_bounds
        function.SetParLimits(0, low, high)
        function.SetParameter(0, min(max(amplitude, low), high))
        amp_step = 0.05 * constraint.amplitude_reference
    else:
        amplitude = max(float(amplitude), 1e-20)
        low = max(amplitude / 1e6, 1e-20)
        high = min(amplitude * 1e6, 1e30)
        function.SetParLimits(0, low, high)
        amp_step = 0.05 * amplitude
    try:
        function.SetParError(0, max(amp_step, 1e-10))
    except Exception:
        pass


def set_refinement_steps(function, model: FitModelConfig, constraint: Optional[FitConstraint], scale: float) -> None:
    try:
        function.SetParError(0, max(abs(function.GetParameter(0)) * 0.02 * scale, 1e-10))
    except Exception:
        pass
    for index, (name, low, high) in enumerate(shape_bounds(model, constraint), start=1):
        try:
            function.SetParError(index, parameter_step(name, function.GetParameter(index), low, high, scale))
        except Exception:
            pass


def make_objective_function(ROOT, model: FitModelConfig, mode: ModeConfig, objective: str, name: str):
    if mode.key == SS_MODE.key and objective == "chi2":
        # SS data should retain the original TH1::Fit(..., "S R I", ...)
        # behaviour.  In this branch the TF1 is the smooth density itself.
        function = ROOT.BkgFitFnVariationPy.MakeFitFunction(
            name, model.cpp_id, mode.fit_min, mode.fit_max
        )
    else:
        kind = {"chi2": 0, "weighted-likelihood": 1, "log-chi2": 2}[objective]
        function = ROOT.BkgFitFnVariationPy.MakeObjectiveFunction(
            name, model.cpp_id, kind, mode.fit_min, mode.fit_max, model.npar
        )
    if not function:
        raise RuntimeError(f"Could not construct objective function for {model.label}")
    function.SetNpx(1000)
    return function


def result_object(fit_result):
    try:
        return fit_result.Get()
    except Exception:
        return fit_result


def extract_diagnostics(
    fit_result,
    function,
    model: FitModelConfig,
    fit_data: FitData,
    constraint: Optional[FitConstraint],
) -> FitDiagnostics:
    diag = FitDiagnostics()
    try:
        diag.status = int(fit_result)
    except Exception:
        pass
    result = result_object(fit_result)
    if result:
        try:
            diag.covariance_status = int(result.CovMatrixStatus())
        except Exception:
            pass
        try:
            diag.edm = float(result.Edm())
        except Exception:
            pass
        try:
            diag.objective_value = float(result.MinFcnValue())
        except Exception:
            try:
                diag.objective_value = float(result.Chi2())
            except Exception:
                pass
        try:
            diag.ndf = int(result.Ndf())
        except Exception:
            pass
        try:
            diag.result_valid = bool(result.IsValid())
        except Exception:
            diag.result_valid = diag.status == 0
    if diag.ndf <= 0:
        diag.ndf = fit_data.n_points - model.npar

    parameters = [float(function.GetParameter(i)) for i in range(model.npar)]
    diag.finite_parameters = all(math.isfinite(x) for x in parameters)
    for index, (name, low, high) in enumerate(shape_bounds(model, constraint), start=1):
        value = float(function.GetParameter(index))
        tolerance = max(1e-4 * (high - low), 1e-5)
        if abs(value - low) <= tolerance:
            diag.boundary_parameters.append(f"{name}=lower")
        elif abs(value - high) <= tolerance:
            diag.boundary_parameters.append(f"{name}=upper")
    return diag


def diagnostics_score(diag: FitDiagnostics) -> Tuple[float, ...]:
    """Rank fit candidates by the fitted objective before covariance quality.

    A Minuit2 status=1/covariance-status=2 result can have a reliable central
    minimum even though its parameter covariance had to be regularised.  The
    previous ordering always preferred status=0, which allowed a catastrophically
    worse local minimum (for example the 2017 Power x Exp x Logistic fit) to beat
    the much better status=1 solution.  Among candidates that pass the central-fit
    validity requirements, select the lowest objective first and use status,
    covariance quality, and EDM only as tie-breakers.
    """
    return (
        0.0 if diag.accepted else 1.0,
        diag.objective_ndf if math.isfinite(diag.objective_ndf) else 1e99,
        0.0 if diag.status == 0 else 1.0,
        0.0 if diag.covariance_status >= 3 else 1.0,
        abs(diag.edm) if math.isfinite(diag.edm) else 1e99,
    )


def run_fit_attempt(
    ROOT,
    density,
    fit_data: FitData,
    model: FitModelConfig,
    mode: ModeConfig,
    objective: str,
    seed: Tuple[float, ...],
    attempt_index: int,
    constraint: Optional[FitConstraint],
) -> FitCandidate:
    function = make_objective_function(
        ROOT, model, mode, objective, _NAMES.unique(f"fit_{model.key}_attempt")
    )

    # A smooth density function is used only to calculate the amplitude seed.
    seed_density = ROOT.BkgFitFnVariationPy.MakeFitFunction(
        _NAMES.unique(f"seed_density_{model.key}"), model.cpp_id, mode.fit_min, mode.fit_max
    )
    set_shape_parameters(seed_density, seed)
    amplitude = adaptive_amplitude_seed(seed_density, density, mode)
    configure_parameters(function, model, seed, amplitude, constraint)

    records = []
    stages = ((1, 1.0), (2, 0.5), (2, 0.2))
    try:
        for stage_index, (strategy, step_scale) in enumerate(stages):
            try:
                ROOT.Math.MinimizerOptions.SetDefaultStrategy(strategy)
            except Exception:
                pass
            if stage_index > 0:
                set_refinement_steps(function, model, constraint, step_scale)
            result = fit_data.data.Fit(function, fit_data.options, "", mode.fit_min, mode.fit_max)
            diag = extract_diagnostics(result, function, model, fit_data, constraint)
            state = [(float(function.GetParameter(i)), float(function.GetParError(i))) for i in range(model.npar)]
            records.append((result, diag, state))
            if diag.accepted:
                break
    finally:
        try:
            ROOT.Math.MinimizerOptions.SetDefaultStrategy(1)
        except Exception:
            pass

    result, diag, state = min(records, key=lambda item: diagnostics_score(item[1]))
    for index, (value, error) in enumerate(state):
        function.SetParameter(index, value)
        try:
            function.SetParError(index, error)
        except Exception:
            pass
    function.SetRange(mode.fit_min, mode.fit_max)
    return FitCandidate(function, result, diag, attempt_index, seed, amplitude)


def release_parameter(function, model: FitModelConfig, parameter_index: int, constraint: Optional[FitConstraint]) -> None:
    try:
        function.ReleaseParameter(parameter_index)
    except Exception:
        pass
    name, low, high = shape_bounds(model, constraint)[parameter_index - 1]
    function.SetParLimits(parameter_index, low, high)
    try:
        function.SetParError(parameter_index, parameter_step(name, function.GetParameter(parameter_index), low, high, 0.5))
    except Exception:
        pass


def run_erf_to_logistic_transfer(
    ROOT,
    density,
    fit_data: FitData,
    mode: ModeConfig,
    objective: str,
    selected_by_key: Dict[str, SelectedFit],
    constraint: Optional[FitConstraint],
) -> Optional[FitCandidate]:
    source = selected_by_key.get("power_exp_erf")
    if source is None or not source.accepted:
        return None
    model = next(m for m in FIT_MODELS if m.key == "power_exp_logistic")
    fn = source.function
    seed = clip_seed(
        (
            float(fn.GetParameter(1)), float(fn.GetParameter(2)), float(fn.GetParameter(3)),
            erf_width_to_logistic_width(float(fn.GetParameter(4)), "slope"),
        ),
        shape_bounds(model, constraint),
    )
    print(f"[FIT TRANSFER] PowerxExpxERF -> PowerxExpxLogistic seed={seed}")

    function = make_objective_function(ROOT, model, mode, objective, _NAMES.unique("pexplog_transfer"))
    seed_density = ROOT.BkgFitFnVariationPy.MakeFitFunction(
        _NAMES.unique("pexplog_transfer_seed"), model.cpp_id, mode.fit_min, mode.fit_max
    )
    set_shape_parameters(seed_density, seed)
    amplitude = adaptive_amplitude_seed(seed_density, density, mode)
    configure_parameters(function, model, seed, amplitude, constraint)

    records = []
    try:
        ROOT.Math.MinimizerOptions.SetDefaultStrategy(2)
        function.FixParameter(1, seed[0])
        function.FixParameter(2, seed[1])
        for _ in range(2):
            fit_data.data.Fit(function, fit_data.options, "", mode.fit_min, mode.fit_max)
        release_parameter(function, model, 2, constraint)
        for _ in range(2):
            fit_data.data.Fit(function, fit_data.options, "", mode.fit_min, mode.fit_max)
        release_parameter(function, model, 1, constraint)
        for index in range(4):
            set_refinement_steps(function, model, constraint, max(0.08, 0.35 / (index + 1)))
            result = fit_data.data.Fit(function, fit_data.options, "", mode.fit_min, mode.fit_max)
            diag = extract_diagnostics(result, function, model, fit_data, constraint)
            state = [(float(function.GetParameter(i)), float(function.GetParError(i))) for i in range(model.npar)]
            records.append((result, diag, state))
            if diag.accepted:
                break
    finally:
        try:
            ROOT.Math.MinimizerOptions.SetDefaultStrategy(1)
        except Exception:
            pass

    if not records:
        return None
    result, diag, state = min(records, key=lambda item: diagnostics_score(item[1]))
    for index, (value, error) in enumerate(state):
        function.SetParameter(index, value)
        try:
            function.SetParError(index, error)
        except Exception:
            pass
    return FitCandidate(function, result, diag, 0, seed, amplitude)


def copy_parameters(source, target, npar: int) -> None:
    for index in range(npar):
        target.SetParameter(index, source.GetParameter(index))
        try:
            target.SetParError(index, source.GetParError(index))
        except Exception:
            pass


def compute_metrics(
    density,
    average_function,
    model: FitModelConfig,
    mode: ModeConfig,
    log_floor: float,
) -> FitMetrics:
    metrics = FitMetrics()
    n_stat = n_log = n_tail = 0
    for ibin in range(1, density.GetNbinsX() + 1):
        low = float(density.GetXaxis().GetBinLowEdge(ibin))
        high = float(density.GetXaxis().GetBinUpEdge(ibin))
        x = float(density.GetBinCenter(ibin))
        if x < mode.fit_min or x > mode.fit_max or is_excluded(mode, low, high):
            continue
        y = float(density.GetBinContent(ibin))
        error = float(density.GetBinError(ibin))
        if error <= 0.0 or not (math.isfinite(y) and math.isfinite(error)):
            continue
        prediction = max(float(average_function.Eval(x)), 1e-300)
        pull = (y - prediction) / error
        metrics.stat_chi2 += pull * pull
        n_stat += 1
        if x >= TAIL_DIAGNOSTIC_MIN:
            metrics.tail_chi2 += pull * pull
            n_tail += 1
        if y > 0.0:
            log_sigma = max(error / y, log_floor)
            log_pull = math.log(y / prediction) / log_sigma
            metrics.log_chi2 += log_pull * log_pull
            n_log += 1

    metrics.stat_ndf = n_stat - model.npar
    metrics.log_ndf = n_log - model.npar
    # This is a local tail diagnostic, not a separate tail-only fit.  Divide by
    # the number of contributing bins rather than subtracting all global fit
    # parameters and calling the result an ndf.
    metrics.tail_n_points = n_tail
    return metrics


def make_selected_fit(
    ROOT,
    density,
    model: FitModelConfig,
    mode: ModeConfig,
    candidate: FitCandidate,
    log_floor: float,
) -> SelectedFit:
    display = ROOT.BkgFitFnVariationPy.MakeFitFunction(
        _NAMES.unique(f"display_{model.key}"), model.cpp_id, mode.fit_min, mode.fit_max
    )
    average = ROOT.BkgFitFnVariationPy.MakeObjectiveFunction(
        _NAMES.unique(f"average_{model.key}"), model.cpp_id, 0,
        mode.fit_min, mode.fit_max, model.npar
    )
    copy_parameters(candidate.fit_function, display, model.npar)
    copy_parameters(candidate.fit_function, average, model.npar)
    display.SetNpx(1000)
    average.SetNpx(1000)
    metrics = compute_metrics(density, average, model, mode, log_floor)
    return SelectedFit(
        model, display, average, candidate.fit_function, candidate.result,
        candidate.diagnostics, metrics, candidate.attempt_index,
        candidate.seed, candidate.amplitude_seed,
    )


def format_parameters(function, model: FitModelConfig) -> str:
    labels = ("A",) + MODEL_SHAPE_PARAMETER_NAMES[model.key]
    return ", ".join(
        f"{label}={float(function.GetParameter(index)):.8g}"
        for index, label in enumerate(labels)
    )


def print_candidate(model: FitModelConfig, candidate: FitCandidate, selected: bool) -> None:
    diag = candidate.diagnostics
    boundary = ",".join(diag.boundary_parameters) if diag.boundary_parameters else "none"
    prefix = "[FIT SELECTED]" if selected else "[FIT ATTEMPT]"
    print(
        f"{prefix} model={model.label.replace('#times','x')} attempt={candidate.attempt_index} "
        f"status={diag.status} cov={diag.covariance_status} valid={int(diag.accepted)} "
        f"edm={diag.edm:.6g} minFCN/ndf={diag.objective_ndf:.6g} "
        f"boundary={boundary} seed={candidate.seed} Aseed={candidate.amplitude_seed:.6g}"
    )
    print(f"               {format_parameters(candidate.fit_function, model)}")


def fit_one_model(
    ROOT,
    density,
    fit_data: FitData,
    model: FitModelConfig,
    mode: ModeConfig,
    objective: str,
    max_attempts: int,
    log_floor: float,
    details: bool,
    selected_by_key: Dict[str, SelectedFit],
    anchor: Optional[FitAnchor],
    constraint: Optional[FitConstraint],
) -> SelectedFit:
    seeds = build_seed_list(model, selected_by_key, anchor, constraint)[:max_attempts]
    candidates: List[FitCandidate] = []

    if model.key == "power_exp_logistic":
        transferred = run_erf_to_logistic_transfer(
            ROOT, density, fit_data, mode, objective, selected_by_key, constraint
        )
        if transferred is not None:
            candidates.append(transferred)
            if details:
                print_candidate(model, transferred, False)

    for attempt, seed in enumerate(seeds, start=1):
        candidate = run_fit_attempt(
            ROOT, density, fit_data, model, mode, objective, seed, attempt, constraint
        )
        candidates.append(candidate)
        if details:
            print_candidate(model, candidate, False)

    best = min(candidates, key=lambda c: diagnostics_score(c.diagnostics))
    print_candidate(model, best, True)
    selected = make_selected_fit(ROOT, density, model, mode, best, log_floor)
    objective_name, objective_value = objective_metric_for_display(selected, objective)
    quality_parts = [
        f"[FIT QUALITY] model={model.label.replace('#times','x')}",
        f"objective={objective}",
        f"{objective_name}={objective_value:.6g}",
    ]
    # Always retain ordinary statistical chi2/ndf in the log as a common
    # comparison across all fit objectives.  Avoid printing it twice for the
    # ordinary chi2 fit, where it is already the objective metric.
    if objective != "chi2":
        quality_parts.append(f"chi2/ndf={selected.metrics.stat_chi2_ndf:.6g}")
    if objective != "log-chi2":
        quality_parts.append(f"logChi2/ndf={selected.metrics.log_chi2_ndf:.6g}")
    quality_parts.append(
        f"tailChi2/nBin(m>={TAIL_DIAGNOSTIC_MIN:g})={selected.metrics.tail_chi2_per_bin:.6g}"
    )
    print(" ".join(quality_parts))
    if selected.diagnostics.covariance_warning_only:
        print(
            f"[WARNING] {model.label}: Minuit2 status=1 and covariance status=2. "
            "The central minimum is retained, but the parameter errors and "
            "correlations should not be interpreted as accurate."
        )
    elif not selected.accepted:
        print(f"[WARNING] No accepted fit was found for {model.label}.")
    return selected


def fit_all_models(
    ROOT,
    density,
    fit_data: FitData,
    mode: ModeConfig,
    objective: str,
    args: argparse.Namespace,
):
    colours = (
        ROOT.kBlue + 1, ROOT.kRed + 1, ROOT.kGreen + 2,
        ROOT.kOrange + 7, ROOT.kCyan + 2, ROOT.kViolet + 1,
    )
    line_styles = (1, 1, 1, 1, 1, 1)
    model_by_key = {m.key: m for m in FIT_MODELS}

    dynamic_catalog = load_dynamic_anchor_catalog(args)
    resolved_anchors = {
        model.key: get_qcd_anchor(
            mode,
            args.year,
            model,
            dynamic_catalog,
            args.prefer_full_run_anchor,
        )
        for model in FIT_MODELS
    }
    anchors = {
        key: (resolved.anchor if resolved is not None else None)
        for key, resolved in resolved_anchors.items()
    }

    if mode.key == QCD_MODE.key and args.initial_values_only:
        constraints = {model.key: None for model in FIT_MODELS}
        for model in FIT_MODELS:
            print_qcd_initial_only(
                ROOT, density, mode, args.year, model,
                resolved_anchors[model.key],
            )
    else:
        constraints = {
            model.key: build_qcd_constraint(
                ROOT, density, mode, args.year, model,
                resolved_anchors[model.key],
            )
            for model in FIT_MODELS
        }

    selected_by_key: Dict[str, SelectedFit] = {}
    internal_order = (
        "power_erf", "power_logistic", "exp_erf", "exp_logistic",
        "power_exp_erf", "power_exp_logistic",
    )
    for key in internal_order:
        selected_by_key[key] = fit_one_model(
            ROOT, density, fit_data, model_by_key[key], mode, objective,
            args.fit_max_attempts, args.log_relative_error_floor,
            args.fit_attempt_details, selected_by_key, anchors[key], constraints[key],
        )

    # One compact recovery pass for invalid models, now with all partner results available.
    for key in internal_order:
        if selected_by_key[key].accepted:
            continue
        print(f"[FIT RECOVERY] model={model_by_key[key].label.replace('#times','x')}")
        recovered = fit_one_model(
            ROOT, density, fit_data, model_by_key[key], mode, objective,
            args.fit_max_attempts, args.log_relative_error_floor,
            args.fit_attempt_details, selected_by_key, anchors[key], constraints[key],
        )
        if diagnostics_score(recovered.diagnostics) < diagnostics_score(
            selected_by_key[key].diagnostics
        ):
            selected_by_key[key] = recovered

    selected = []
    for index, model in enumerate(FIT_MODELS):
        outcome = selected_by_key[model.key]
        outcome.function.SetName(f"fit_{index}")
        outcome.function.SetLineColor(colours[index])
        outcome.function.SetLineWidth(2)
        outcome.function.SetLineStyle(line_styles[index])
        selected.append(outcome)
    return selected, colours, line_styles


# =============================================================================
# PLOTTING
# =============================================================================


def objective_metric_for_display(selected: SelectedFit, objective: str) -> Tuple[str, float]:
    """Return the objective-specific label and value used in logs.

    ROOT minimises NLL for the weighted-likelihood option, so the displayed
    likelihood statistic is 2*NLL_w/ndf.  The chi2 and log-chi2 objectives are
    displayed using the corresponding independently recomputed diagnostics.
    """
    if objective == "chi2":
        return "chi2/ndf", selected.metrics.stat_chi2_ndf
    if objective == "weighted-likelihood":
        value = 2.0 * selected.diagnostics.objective_ndf
        return "2NLLw/ndf", value
    if objective == "log-chi2":
        return "logChi2/ndf", selected.metrics.log_chi2_ndf
    raise ValueError(f"Unknown fit objective: {objective}")


def objective_metric_for_plot(selected: SelectedFit, objective: str) -> str:
    name, value = objective_metric_for_display(selected, objective)
    if not math.isfinite(value):
        if objective == "weighted-likelihood":
            return "2NLL_{w}/ndf = n/a"
        if objective == "log-chi2":
            return "#chi^{2}_{log}/ndf = n/a"
        return "#chi^{2}/ndf = n/a"
    if objective == "weighted-likelihood":
        return f"2NLL_{{w}}/ndf = {value:.3f}"
    if objective == "log-chi2":
        return f"#chi^{{2}}_{{log}}/ndf = {value:.3f}"
    return f"#chi^{{2}}/ndf = {value:.3f}"


def fit_summary_line(selected: SelectedFit, objective: str) -> str:
    f = selected.function
    model = selected.model
    metric = objective_metric_for_plot(selected, objective)
    if model.key == "power_erf":
        a, n, m0, w = (f.GetParameter(i) for i in range(4))
        return "y = %.2E x^{-%.2f}#times [1 + erf((x-%.2f)/(#sqrt{2}#times%.2f))]/2   [%s]" % (a, n, m0, w, metric)
    if model.key == "power_logistic":
        a, n, m0, w = (f.GetParameter(i) for i in range(4))
        return "y = %.2E x^{-%.2f} / [1 + e^{-(x-%.2f)/%.2f}]   [%s]" % (a, n, m0, w, metric)
    if model.key == "power_exp_erf":
        a, n, k, m0, w = (f.GetParameter(i) for i in range(5))
        return "y = %.2E x^{-%.2f} e^{-%.2f x}#times [1 + erf((x-%.2f)/(#sqrt{2}#times%.2f))]/2   [%s]" % (a, n, k, m0, w, metric)
    if model.key == "power_exp_logistic":
        a, n, k, m0, w = (f.GetParameter(i) for i in range(5))
        return "y = %.2E x^{-%.2f} e^{-%.2f x} / [1 + e^{-(x-%.2f)/%.2f}]   [%s]" % (a, n, k, m0, w, metric)
    if model.key == "exp_erf":
        a, k, m0, w = (f.GetParameter(i) for i in range(4))
        return "y = %.2E e^{-%.2f x}#times [1 + erf((x-%.2f)/(#sqrt{2}#times%.2f))]/2   [%s]" % (a, k, m0, w, metric)
    a, k, m0, w = (f.GetParameter(i) for i in range(4))
    return "y = %.2E e^{-%.2f x} / [1 + e^{-(x-%.2f)/%.2f}]   [%s]" % (a, k, m0, w, metric)


def period_lumi_fb(period: str) -> float:
    if period in LUMI_FB:
        return LUMI_FB[period]
    if period in PERIOD_LUMI_FB:
        return PERIOD_LUMI_FB[period]
    return sum(LUMI_FB[era] for era in selected_eras(period))


def format_lumi_fb(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def cms_lumi_label(period: str) -> str:
    if period == "Run2+3":
        return (
            f"{format_lumi_fb(PERIOD_LUMI_FB['Run2'])} fb^{{-1}} (13 TeV) + "
            f"{format_lumi_fb(PERIOD_LUMI_FB['Run3'])} fb^{{-1}} (13.6 TeV)"
        )
    energy = "13.6 TeV" if period in {"Run3", *RUN3_ERAS} else "13 TeV"
    return f"{format_lumi_fb(period_lumi_fb(period))} fb^{{-1}} ({energy})"


def set_local_cms_style(ROOT) -> None:
    """Small self-contained CMS-like ROOT style; no PlotHelper files required."""
    style = ROOT.TStyle("NIsoMuonQCDStyle", "NIsoMuon QCD fit style")
    style.SetCanvasBorderMode(0)
    style.SetCanvasColor(0)
    style.SetFrameBorderMode(0)
    style.SetFrameFillColor(0)
    style.SetPadBorderMode(0)
    style.SetPadColor(0)
    style.SetOptStat(0)
    style.SetOptTitle(0)
    style.SetTextFont(42)
    style.SetLabelFont(42, "XYZ")
    style.SetTitleFont(42, "XYZ")
    style.SetLabelSize(0.040, "XYZ")
    style.SetTitleSize(0.045, "XYZ")
    style.SetPadTickX(1)
    style.SetPadTickY(1)
    style.SetEndErrorSize(0)
    style.cd()
    ROOT.gROOT.SetStyle("NIsoMuonQCDStyle")
    ROOT.gROOT.ForceStyle()


def draw_cms_header(ROOT, pad, period: str) -> object:
    pad.cd()
    latex = ROOT.TLatex()
    latex.SetNDC(True)
    latex.SetTextColor(ROOT.kBlack)

    latex.SetTextAlign(11)
    latex.SetTextFont(61)
    latex.SetTextSize(0.050)
    latex.DrawLatex(0.105, 0.935, "CMS")

    latex.SetTextFont(52)
    latex.SetTextSize(0.036)
    latex.DrawLatex(0.205, 0.935, "Preliminary")

    latex.SetTextAlign(31)
    latex.SetTextFont(42)
    latex.SetTextSize(0.034)
    latex.DrawLatex(0.955, 0.935, cms_lumi_label(period))
    return latex


def x_axis_title(ROOT, mode: ModeConfig) -> str:
    # The QCD fitter only handles the dimuon-mass histogram, so no external
    # PlotterCore axis-title lookup is necessary.
    if mode.key == SS_MODE.key:
        return "m_{#mu^{#pm}#mu^{#pm}} [GeV]"
    return "m_{#mu^{+}#mu^{-}} [GeV]"


def draw_fit_plot(
    ROOT,
    args: argparse.Namespace,
    mode: ModeConfig,
    objective: str,
    density,
    selected_fits: Sequence[SelectedFit],
    colours: Sequence[int],
    line_styles: Sequence[int],
):
    plot_dir = PLOT_DIR
    plot_dir.mkdir(parents=True, exist_ok=True)
    keepalive: List[object] = []

    canvas = ROOT.TCanvas("myCanvas", "", 1000, 1000)
    upper = ROOT.TPad("upper_pad", "", 0.0, 0.26, 1.0, 1.0, 0)
    upper.SetLeftMargin(0.1)
    upper.SetLogx(True)
    upper.SetLogy(True)
    upper.Draw()
    lower = ROOT.TPad("lower_pad", "", 0.0, 0.0, 1.0, 0.33, 0)
    lower.SetLeftMargin(0.1)
    lower.SetBottomMargin(0.3)
    lower.SetLogx(True)
    lower.Draw()

    upper.cd()
    set_local_cms_style(ROOT)
    density.Draw("PE")
    for selected in selected_fits:
        selected.function.Draw("LSAME")

    title_x = x_axis_title(ROOT, mode)
    density.GetYaxis().SetTitle("Events (log)")
    density.GetYaxis().SetTitleSize(0.045)
    density.GetYaxis().SetLabelSize(0.040)
    density.GetYaxis().SetTitleOffset(1.0)
    density.GetXaxis().SetTitle(title_x)
    density.GetXaxis().SetTitleSize(0.0)
    density.GetXaxis().SetLabelSize(0.025)

    legend = ROOT.TLegend(0.64, 0.56, 0.90, 0.89)
    legend.AddEntry(density, mode.legend_label, "lep")
    for selected in selected_fits:
        legend.AddEntry(selected.function, selected.model.label, "l")
    legend.SetTextSize(0.032)
    legend.SetTextFont(42)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    legend.Draw()

    density.Draw("PE SAME")
    for selected in selected_fits:
        selected.function.Draw("LSAME")
    upper.SetGridx()
    upper.SetGridy()
    cms_header = draw_cms_header(ROOT, upper, args.year)

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.032)
    latex.SetTextFont(42)
    latex.SetTextAlign(13)
    y_text = 0.4
    for selected, colour in zip(selected_fits, colours):
        latex.SetTextColor(colour)
        latex.DrawLatex(0.12, y_text, fit_summary_line(selected, objective))
        y_text -= 0.05
    latex.SetTextColor(ROOT.kBlack)

    lower.cd()
    frame = density.Clone("RatioFrame")
    frame.SetDirectory(0)
    frame.Reset("ICES")
    frame.GetYaxis().SetRangeUser(0.5, 1.5)
    frame.GetXaxis().SetRangeUser(mode.fit_min, mode.fit_max)
    frame.GetXaxis().SetTitle(title_x)
    frame.GetXaxis().SetTitleOffset(1.1)
    frame.GetXaxis().SetTitleSize(0.1)
    frame.GetXaxis().SetLabelSize(0.09)
    frame.GetYaxis().SetTitle("fit / QCD MC" if mode.key == QCD_MODE.key else "fit / (Data-MC^{Top,Others})")
    frame.GetYaxis().SetTitleSize(0.095)
    frame.GetYaxis().SetTitleOffset(0.40)
    frame.GetYaxis().SetLabelSize(0.075)
    frame.Draw()

    x, y_unity, ex, ey = [], [], [], []
    ratio_values = [[] for _ in selected_fits]
    for ibin in range(1, density.GetNbinsX() + 1):
        low = float(density.GetXaxis().GetBinLowEdge(ibin))
        high = float(density.GetXaxis().GetBinUpEdge(ibin))
        centre = float(density.GetBinCenter(ibin))
        if centre < mode.fit_min or centre > mode.fit_max or is_excluded(mode, low, high):
            continue
        value = float(density.GetBinContent(ibin))
        error = float(density.GetBinError(ibin))
        if value <= 0.0:
            continue
        x.append(centre)
        y_unity.append(1.0)
        ex.append(0.5 * (high - low))
        ey.append(error / value)
        for index, selected in enumerate(selected_fits):
            ratio_values[index].append(float(selected.average_function.Eval(centre)) / value)

    band = ROOT.TGraphErrors(len(x), array("d", x), array("d", y_unity), array("d", ex), array("d", ey))
    band.SetFillColor(18)
    band.SetFillStyle(1001)
    band.SetLineColor(0)
    band.SetMarkerSize(0)
    band.Draw("E2 SAME")

    ratio_graphs = []
    for index, (values, colour, style) in enumerate(zip(ratio_values, colours, line_styles)):
        graph = ROOT.TGraph(len(x), array("d", x), array("d", values))
        graph.SetName(f"RatioGraph_{index}")
        graph.SetLineColor(colour)
        graph.SetLineWidth(2)
        graph.SetLineStyle(style)
        graph.Draw("L SAME")
        ratio_graphs.append(graph)

    line = ROOT.TF1("hline", "1", mode.fit_min, mode.fit_max)
    line.SetLineColor(ROOT.kBlack)
    line.SetLineWidth(2)
    line.Draw("SAME")
    lower.SetGridx()
    lower.SetGridy()
    frame.Draw("AXIS SAME")

    objective_suffix = ""
    if mode.key == QCD_MODE.key and args.fit_objective != "auto":
        objective_suffix = {
            "chi2": "-Chi2",
            "weighted-likelihood": "-WeightedLikelihood",
            "log-chi2": "-LogChi2",
        }[objective]
    basename = (
        f"FitPlot_FnVariation-{mode.file_name}_{mode.plot_tag}_Dimuon_Mass-"
        f"{args.year}-AllFits{objective_suffix}"
    )
    pdf = plot_dir / f"{basename}.pdf"
    png = plot_dir / f"{basename}.png"
    canvas.SaveAs(str(pdf))
    canvas.SaveAs(str(png))
    keepalive.extend([canvas, upper, lower, legend, cms_header, latex, frame, band, line, *ratio_graphs])
    return pdf, png, keepalive


# =============================================================================
# SS-DATA ROOT OUTPUT
# =============================================================================


def integral_in_window(hist, low: float, high: float) -> float:
    first = int(hist.GetXaxis().FindBin(low + 1e-6))
    last = int(hist.GetXaxis().FindBin(high - 1e-6))
    return float(hist.Integral(first, last))


def fit_integral_in_output_bin(function, hist, index: int, allowed_ranges: Sequence[Tuple[float, float]]) -> float:
    bin_low = float(hist.GetXaxis().GetBinLowEdge(index))
    bin_high = float(hist.GetXaxis().GetBinUpEdge(index))
    total = 0.0
    for range_low, range_high in allowed_ranges:
        low = max(bin_low, range_low)
        high = min(bin_high, range_high)
        if high > low:
            total += float(function.Integral(low, high))
    return total


def make_output_histogram(out_file, source_hist, directory_name: str, histogram_name: str):
    directory = out_file.mkdir(directory_name)
    if not directory:
        raise RuntimeError(f"Could not create ROOT directory {directory_name}")
    directory.cd()
    hist = source_hist.Clone(histogram_name)
    hist.SetName(histogram_name)
    hist.SetTitle(histogram_name)
    hist.Reset("ICES")
    hist.SetDirectory(directory)
    return hist, directory


def print_saved_histogram(hist, directory_name: str, histogram_name: str) -> None:
    print(
        f"[fit.root] Saved {directory_name}/{histogram_name}: Nbins={hist.GetNbinsX()}, "
        f"Xmin={hist.GetXaxis().GetXmin():g} Xmax={hist.GetXaxis().GetXmax():g}"
    )


def verify_ss_fit_validity(args: argparse.Namespace, selected_by_key: Dict[str, SelectedFit]) -> None:
    invalid = [key for key in SS_REQUIRED_ROOT_MODELS if key not in selected_by_key or not selected_by_key[key].accepted]
    if not invalid:
        return
    message = "Refusing to write NIsoMuon_SS_fit.root because required fits are invalid: " + ", ".join(invalid)
    if args.allow_invalid_fit_output:
        print(f"[DANGER] {message}; continuing because --allow-invalid-fit-output was supplied.")
        return
    raise RuntimeError(message + ". Existing ROOT files were not overwritten.")


def write_ss_background_root(ROOT, args: argparse.Namespace, directory: Path, files: Dict[str, object], selected_fits: Sequence[SelectedFit]):
    selected_by_key = {selected.model.key: selected for selected in selected_fits}
    verify_ss_fit_validity(args, selected_by_key)

    ss_path = hist_path(SS_REGION)
    os_path = hist_path(OS_REGION)
    f_data = files["data"]
    f_top = files["NIsoMuon_Top"]
    f_others = files["NIsoMuon_Others"]

    h_data_ss = get_histogram_clone(f_data, ss_path, "hDataSSForNorm")
    h_top_ss = get_histogram_clone(f_top, ss_path, "hTopSSForNorm")
    h_others_ss = get_histogram_clone(f_others, ss_path, "hOthersSSForNorm")
    h_data_os = get_histogram_clone(f_data, os_path, "hDataOSForNorm")
    h_top_os = get_histogram_clone(f_top, os_path, "hTopOSForNorm")
    h_others_os = get_histogram_clone(f_others, os_path, "hOthersOSForNorm")

    f_dy_est = open_root_file(ROOT, directory / "NIsoMuon_DYJets_est.root")
    f_qcd = open_root_file(ROOT, directory / "NIsoMuon_QCD_Inclusive.root")
    try:
        h_dy_os = get_histogram_clone(f_dy_est, os_path, "hDYOSForNorm")
        h_qcd_ss = get_histogram_clone(f_qcd, ss_path, "hQCDSSForTransfer")
        h_qcd_os = get_histogram_clone(f_qcd, os_path, "hQCDOSForTransfer")
        required = [
            h_data_ss, h_top_ss, h_others_ss,
            h_data_os, h_top_os, h_dy_os, h_others_os,
            h_qcd_ss, h_qcd_os,
        ]
        if any(hist is None for hist in required):
            raise KeyError("A required SS/OS transfer-factor histogram is missing.")

        # Match os_ss_comparison.py: no DY subtraction in SS; use the
        # data-driven DY estimate in OS.
        h_ss = clone_detached(h_data_ss, "hSSForNorm_DataMinusBG")
        h_os = clone_detached(h_data_os, "hOSForNorm_DataMinusBG")
        for hist in (h_top_ss, h_others_ss):
            h_ss.Add(hist, -1.0)
        for hist in (h_top_os, h_dy_os, h_others_os):
            h_os.Add(hist, -1.0)

        low_min, low_max = QCD_TRANSFER_LOW_WINDOW
        high_min, high_max = QCD_TRANSFER_HIGH_WINDOW
        dt_ss_low = integral_in_window(h_ss, low_min, low_max)
        dt_os_low = integral_in_window(h_os, low_min, low_max)
        mc_ss_low = integral_in_window(h_qcd_ss, low_min, low_max)
        mc_os_low = integral_in_window(h_qcd_os, low_min, low_max)
        mc_ss_high = integral_in_window(h_qcd_ss, high_min, high_max)
        mc_os_high = integral_in_window(h_qcd_os, high_min, high_max)

        for label, value in (
            ("DT SS low", dt_ss_low),
            ("DT OS low", dt_os_low),
            ("QCD MC SS low", mc_ss_low),
            ("QCD MC OS low", mc_os_low),
            ("QCD MC SS high", mc_ss_high),
            ("QCD MC OS high", mc_os_high),
        ):
            if value <= 0.0:
                raise RuntimeError(f"Non-positive transfer-factor yield for {label}: {value}")

        dt_low_ratio = dt_os_low / dt_ss_low
        mc_low_ratio = mc_os_low / mc_ss_low
        mc_high_ratio = mc_os_high / mc_ss_high
        normalisation = dt_low_ratio * mc_high_ratio / mc_low_ratio
        if normalisation <= 0.0 or not math.isfinite(normalisation):
            raise RuntimeError(f"Invalid corrected QCD OS/SS transfer factor: {normalisation}")

        # QCD_norm is a multiplicative lnN nuisance.  Define the modelling
        # uncertainty symmetrically in log space, with one 1-sigma variation
        # reaching the uncorrected high-mass QCD-MC transfer factor.
        log_kappa = abs(math.log(normalisation / mc_high_ratio))
        norm_kappa = math.exp(log_kappa)
        norm_down = 1.0 / norm_kappa
        norm_up = norm_kappa
        transfer_down = normalisation * norm_down
        transfer_up = normalisation * norm_up

        print(
            f"[fit.root] DT(Data-nonQCD) OS/SS, {low_min:g}<m<{low_max:g} = "
            f"{dt_low_ratio:g}"
        )
        print(
            f"[fit.root] QCD MC OS/SS, {low_min:g}<m<{low_max:g} = "
            f"{mc_low_ratio:g}"
        )
        print(
            f"[fit.root] QCD MC OS/SS, {high_min:g}<m<{high_max:g} = "
            f"{mc_high_ratio:g}"
        )
        print(
            "[fit.root] Corrected OS/SS = DT(low) * MC(high) / MC(low) = "
            f"{normalisation:g}"
        )
        print(
            "[fit.root] Norm lnN: kappa=exp(|ln(corrected/MC(high))|)="
            f"{norm_kappa:g}, Down={transfer_down:g} ({norm_down:g}), "
            f"Up={transfer_up:g} ({norm_up:g})"
        )

        main = selected_by_key[SS_NOMINAL_MODEL].function
        alternatives = [selected_by_key[key].function for key in SS_SHAPE_ALTERNATIVES]
        model_labels = {model.key: model.label.replace("#times", "x") for model in FIT_MODELS}
        print(f"[fit.root] Central shape model = {model_labels[SS_NOMINAL_MODEL]}")
        print(
            "[fit.root] Shape-envelope models = "
            + ", ".join(model_labels[key] for key in SS_SHAPE_ALTERNATIVES)
        )
        allowed_ranges = ((5.0, 9.0), (11., 80.0))

        run_syst = era_dir(args, args.year, "RunSyst")
        run_syst.mkdir(parents=True, exist_ok=True)
        output = run_syst / "NIsoMuon_SS_fit.root"
        copied = directory / "NIsoMuon_SS_fit.root"
        root_file = ROOT.TFile.Open(str(output), "RECREATE")
        if not root_file or root_file.IsZombie():
            raise OSError(f"Could not create {output}")
        try:
            central_name = f"{HIST_NAME}___{OS_REGION}"
            h_central, d_central = make_output_histogram(root_file, h_data_os, OS_REGION, central_name)
            for ibin in range(1, h_central.GetNbinsX() + 1):
                value = normalisation * fit_integral_in_output_bin(main, h_central, ibin, allowed_ranges)
                h_central.SetBinContent(ibin, value)
                h_central.SetBinError(ibin, 0.0)
            d_central.cd()
            h_central.Write(central_name, ROOT.TObject.kOverwrite)
            print_saved_histogram(h_central, OS_REGION, central_name)

            up_dir = "OS_POGMedium_tight_BJet_Syst_ShapeUp_NIsoDimuon"
            down_dir = "OS_POGMedium_tight_BJet_Syst_ShapeDown_NIsoDimuon"
            up_name = f"Dilepton_Mass___{up_dir}"
            down_name = f"Dilepton_Mass___{down_dir}"
            h_up, d_up = make_output_histogram(root_file, h_data_os, up_dir, up_name)
            h_down, d_down = make_output_histogram(root_file, h_data_os, down_dir, down_name)
            for ibin in range(1, h_up.GetNbinsX() + 1):
                low = h_up.GetXaxis().GetBinLowEdge(ibin)
                high = h_up.GetXaxis().GetBinUpEdge(ibin)
                central_ss = fit_integral_in_output_bin(main, h_up, ibin, allowed_ranges)
                central_os = normalisation * central_ss
                max_dev = 0.0
                details = []
                if central_ss != 0.0:
                    for alt in alternatives:
                        alt_ss = fit_integral_in_output_bin(alt, h_up, ibin, allowed_ranges)
                        dev = abs(alt_ss / central_ss - 1.0)
                        max_dev = max(max_dev, dev)
                        details.append((alt_ss, dev))
                ### multiplicative approach ###
                #scale = 1.0 + max_dev
                #value_up = central_os * scale
                #value_down = central_os / scale if scale else 0.0
                ### additive approach ###
                delta = central_os * max_dev
                value_up = central_os + delta
                value_down = max(0.0, central_os - delta)
                ###
                h_up.SetBinContent(ibin, value_up)
                h_up.SetBinError(ibin, 0.0)
                h_down.SetBinContent(ibin, value_down)
                h_down.SetBinError(ibin, 0.0)
                if args.print_shape_syst_bin_info and central_ss != 0.0:
                    print(
                        f"[ShapeSystBin] bin={ibin} ({low:g},{high:g}) centralSS={central_ss:g} "
                        f"centralOS={central_os:g} r={max_dev:g} up={value_up:g} down={value_down:g}"
                    )
                if args.print_raw_shape_debug:
                    print(f"[{low:g}] centralSS={central_ss:g}, alternatives={details}, maxDev={max_dev:g}")
            d_up.cd(); h_up.Write(up_name, ROOT.TObject.kOverwrite)
            d_down.cd(); h_down.Write(down_name, ROOT.TObject.kOverwrite)
            print_saved_histogram(h_up, up_dir, up_name)
            print_saved_histogram(h_down, down_dir, down_name)

            norm_up_dir = "OS_POGMedium_tight_BJet_Syst_NormUp_NIsoDimuon"
            norm_down_dir = "OS_POGMedium_tight_BJet_Syst_NormDown_NIsoDimuon"
            norm_up_name = f"Dilepton_Mass___{norm_up_dir}"
            norm_down_name = f"Dilepton_Mass___{norm_down_dir}"
            h_norm_up, d_norm_up = make_output_histogram(root_file, h_data_os, norm_up_dir, norm_up_name)
            h_norm_down, d_norm_down = make_output_histogram(root_file, h_data_os, norm_down_dir, norm_down_name)
            for ibin in range(1, h_central.GetNbinsX() + 1):
                value = h_central.GetBinContent(ibin)
                h_norm_up.SetBinContent(ibin, value * norm_up)
                h_norm_down.SetBinContent(ibin, value * norm_down)
                h_norm_up.SetBinError(ibin, 0.0)
                h_norm_down.SetBinError(ibin, 0.0)
            d_norm_up.cd(); h_norm_up.Write(norm_up_name, ROOT.TObject.kOverwrite)
            d_norm_down.cd(); h_norm_down.Write(norm_down_name, ROOT.TObject.kOverwrite)
            print_saved_histogram(h_norm_up, norm_up_dir, norm_up_name)
            print_saved_histogram(h_norm_down, norm_down_dir, norm_down_name)
        finally:
            root_file.Close()

        shutil.copyfile(output, copied)
        print(f"[copy] {output} -> {copied}")
        return output, copied
    finally:
        f_qcd.Close()
        f_dy_est.Close()


def save_ss_fit_anchors(
    args: argparse.Namespace,
    selected_fits: Sequence[SelectedFit],
    objective: str,
) -> Path:
    """Atomically update the SS-anchor JSON for one era or combined period."""
    invalid = [fit.model.key for fit in selected_fits if not fit.accepted]
    if invalid and not args.allow_invalid_fit_output:
        raise RuntimeError(
            "Refusing to save SS anchors because the following fits are invalid: "
            + ", ".join(invalid)
        )

    path = anchor_file_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {"version": 1, "anchors": {}}
    if path.is_file():
        try:
            existing = json.loads(path.read_text())
            if isinstance(existing, dict):
                payload = existing
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not update anchor file {path}: {exc}") from exc

    anchors = payload.setdefault("anchors", {})
    if not isinstance(anchors, dict):
        anchors = {}
        payload["anchors"] = anchors

    model_payload: Dict[str, object] = {}
    for selected in selected_fits:
        fn = selected.function
        model_payload[selected.model.key] = {
            "amplitude": float(fn.GetParameter(0)),
            "shape": [
                float(fn.GetParameter(index))
                for index in range(1, selected.model.npar)
            ],
            "accepted": bool(selected.accepted),
            "status": int(selected.diagnostics.status),
            "covariance_status": int(selected.diagnostics.covariance_status),
            "objective": objective,
        }

    anchors[args.year] = model_payload
    payload["updated_utc"] = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()
    payload["period_eras"] = {
        period: list(eras) for period, eras in PERIOD_ERAS.items()
    }

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)
    print(f"[SAVE] SS anchor ({args.year}) -> {path}")
    return path


# =============================================================================
# MAIN
# =============================================================================


def close_files(files: Sequence[object]) -> None:
    for root_file in files:
        try:
            root_file.Close()
        except Exception:
            pass


def run(args: argparse.Namespace) -> int:
    mode = canonical_mode(args.mode)
    objective = resolve_objective(args.fit_objective, mode)
    ROOT = import_root()
    declare_fit_functions(ROOT)
    configure_minimizer(ROOT)

    directories = input_dirs(args)
    missing_dirs = [str(directory) for _, directory in directories if not directory.is_dir()]
    if missing_dirs:
        raise FileNotFoundError(
            "Input directories do not exist:\n  " + "\n  ".join(missing_dirs)
        )

    print(f"[INFO] mode            = {mode.display_name}")
    print(f"[INFO] period          = {args.year}")
    print(f"[INFO] eras            = {', '.join(era for era, _ in directories)}")
    print(f"[INFO] trigger subdir  = {args.trigger or '<none>'}")
    print(f"[INFO] analyzer        = {args.analyzer}")
    for era, directory in directories:
        print(f"[INFO] input[{era}]     = {directory}")
    print(f"[INFO] anchor file     = {anchor_file_path(args)}")
    print(f"[INFO] fit range       = [{mode.fit_min:g}, {mode.fit_max:g}] GeV")
    print(f"[INFO] fit objective   = {objective}")
    if mode.key == QCD_MODE.key:
        anchor_mode = (
            "initial-values-only"
            if args.initial_values_only
            else "initial-values+constraints"
        )
        priority = anchor_priority(args.year, args.prefer_full_run_anchor)
        print(f"[INFO] QCD anchor mode = {anchor_mode}")
        print(f"[INFO] anchor priority = {' -> '.join(priority)}")
    else:
        model_labels = {
            model.key: model.label.replace("#times", "x") for model in FIT_MODELS
        }
        print(f"[INFO] SS central model = {model_labels[SS_NOMINAL_MODEL]}")
        print(
            "[INFO] SS shape envelope = "
            + ", ".join(model_labels[key] for key in SS_SHAPE_ALTERNATIVES)
        )
        if args.initial_values_only:
            print("[WARNING] --initial-values-only has no effect in ss-data mode.")
    print(f"[INFO] rebin factor    = {args.rebin}")
    if objective == "log-chi2":
        print(f"[INFO] log relative-error floor = {args.log_relative_error_floor:g}")
    for model in FIT_MODELS:
        print(
            f"[BOUNDARIES] {model.label.replace('#times','x')}: "
            + ", ".join(
                f"{name}=[{low:g},{high:g}]"
                for name, low, high in BASE_SHAPE_BOUNDS[model.key]
            )
        )

    source, files, handles = build_input_histogram(ROOT, mode, directories)
    keepalive: List[object] = []
    try:
        prepared = prepare_histograms(ROOT, source, mode, args.rebin)
        density = prepared.density
        density.GetXaxis().SetRangeUser(mode.fit_min, mode.fit_max)
        density.GetXaxis().SetMoreLogLabels()
        y_min = 0.01
        y_max = 3.0 * float(density.GetBinContent(density.GetMaximumBin()))
        if not math.isfinite(y_max) or y_max <= y_min:
            raise RuntimeError(f"Invalid log-y range [{y_min},{y_max}]")
        density.GetYaxis().SetRangeUser(y_min, y_max)
        if y_max < 3000.0:
            density.GetYaxis().SetMoreLogLabels()
            density.GetYaxis().SetNoExponent()

        fit_data = build_fit_data(
            ROOT, prepared, mode, objective, args.log_relative_error_floor
        )
        selected, colours, styles = fit_all_models(
            ROOT, density, fit_data, mode, objective, args
        )
        pdf, png, plot_objects = draw_fit_plot(
            ROOT, args, mode, objective, density, selected, colours, styles
        )
        print(f"[SAVE] {pdf}")
        print(f"[SAVE] {png}")

        if mode.key == SS_MODE.key:
            save_ss_fit_anchors(args, selected, objective)
            if len(directories) == 1:
                directory = directories[0][1]
                output, copied = write_ss_background_root(
                    ROOT, args, directory, files, selected
                )
                print(f"[SAVE] {output}")
                print(f"[SAVE] {copied}")
            else:
                print(
                    "[INFO] Combined SS fit saved its anchor only. Per-era "
                    "NIsoMuon_SS_fit.root templates were not overwritten."
                )

        keepalive.extend(
            [source, prepared.counts, density, fit_data.data, *fit_data.keepalive,
             *[x.function for x in selected], *[x.average_function for x in selected],
             *[x.fit_function for x in selected], *[x.result for x in selected],
             *plot_objects]
        )
        n_valid = sum(fit.accepted for fit in selected)
        print(
            f"[DONE] mode={mode.display_name}, period={args.year}, "
            f"objective={objective}, accepted={n_valid}/{len(selected)}"
        )
        return 0
    finally:
        close_files(handles)


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    if args.debug:
        return run(args)
    try:
        return run(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
