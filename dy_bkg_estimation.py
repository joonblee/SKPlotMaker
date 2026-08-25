#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""DY transfer-factor/normalisation-factor estimator and closure-test plotter for
NIsoMuon Run 2 and Run 3.

Fixed filesystem layout
-----------------------
Input and produced DY ROOT templates use
  /data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13/NIsoMuon
All diagnostic plots are always written to
  /data6/Users/joonblee/PlotMaker/plots
There is deliberately no plot-output-directory command-line option.

Supported periods
-----------------
  2016preVFP, 2016postVFP, 2017, 2018,
  2022, 2022EE, 2023, 2023BPix,
  Run2, Run3, Run2+3
For a combined period, the script sums corresponding nominal input histograms
from all component eras before deriving the TF or NF.

Luminosity labels [fb^-1]
-------------------------
  2016preVFP 19.5, 2016postVFP 16.8, 2017 42.07, 2018 59.56,
  Run2 137.94, 2022 7.98, 2022EE 26.67, 2022 total 34.65,
  2023 17.7, 2023BPix 9.5, 2023 total 27.20, Run3 61.85.
These numbers label the plots; changing them here does not rescale event yields
already stored in the ROOT files.

Estimation modes
----------------
  --method nf  constant normalisation factor (default)
  --method tf  parameter-dependent transfer factor
In NF mode only the one-dimensional Dilepton_Mass histogram is required.
The Dilepton_Mass_<parameter> two-dimensional histogram is used only when
--method tf is explicitly requested.

The fixed NF windows are 5 < m < 9 GeV for background-subtracted data and
10.4 < m < 70 GeV for the DY-MC closure test.  The script writes the central
NIsoMuon_DYJets_est.root and TF/LightJetStat templates under RunSyst/.

Required option
---------------
  --era PERIOD      one supported era or Run2/Run3/Run2+3

Main optional controls
----------------------
  --method {nf,tf}          default: nf
  --tf-param NAME             used only for --method tf; default: Dilepton_pT
  --xmin/--xmax/--rebin
  --no-variable-binning
  --ratio-min/--ratio-max
  --logx, --linear-y
  --blind                     blind B-jet data for 9 < m < 70 GeV
  --inject-signal
  --signal-mass MASS
  --signal-scale FACTOR
  --trigger SUBDIRECTORY      optional legacy input subdirectory
  --base-dir PATH             input/ROOT-template base directory

Examples
--------
  python3 dy_bkg_estimation.py --era 2018 --blind
  python3 dy_bkg_estimation.py --era Run3
  python3 dy_bkg_estimation.py --era 2023 --method tf --blind
  python3 dy_bkg_estimation.py --era Run2 --method tf --ratio-max 2
  python3 dy_bkg_estimation.py --era 2023BPix --inject-signal \
      --signal-mass 20 --signal-scale 0.002 --blind

Running with no arguments prints this guide and all command-line options, then
exits without opening ROOT files or writing templates.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from array import array
from dataclasses import dataclass
from typing import Sequence, Tuple, List, Any

RUN2_ERAS: Tuple[str, ...] = (
    "2016preVFP", "2016postVFP", "2017", "2018",
)
RUN3_ERAS: Tuple[str, ...] = (
    "2022", "2022EE", "2023", "2023BPix",
)
ALL_ERAS: Tuple[str, ...] = RUN2_ERAS + RUN3_ERAS
ERA_GROUPS = {
    **{era: (era,) for era in ALL_ERAS},
    "Run2": RUN2_ERAS,
    "Run3": RUN3_ERAS,
    "Run2+3": ALL_ERAS,
}
VALID_ERAS = list(ERA_GROUPS.keys())
DEFAULT_BASE_DIR = "/data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13/NIsoMuon"
PLOT_DIR = "/data6/Users/joonblee/PlotMaker/plots"

# Edit these values directly if either constant-NF normalisation window is changed.
# No command-line option is intentionally provided for these windows.
NF_MASS_WINDOW = (5.0, 9.0)          # Data - background MC
DY_MC_NF_MASS_WINDOW = (10.4, 70.0)  # DY MC and DY + Z' MC closure tests

# CMS-style nuisance labels used in the output ROOT directory names.
TF_SYST_NAME = "TF"
NF_SYST_NAME = "TF"
LIGHTJET_STAT_SYST_NAME = "LightJetStat"

DEFAULT_MASS_BINS = [
    0.0, 0.5, 1.0, 1.5, 2.0, 2.5,
    3.0, 3.5, 4.0, 4.5,
    5.0, 6.0, 7.0, 8.0, 9.0,
    10.0, 15., 20.0, 25.,
    30.0, 35., 40., 45., 50., 55., 60., 65., 70., 75., 80., 85., 90., 95., 100., 105., 110., 120., 130., 150.
]

LUMI_FB = {
    "2016preVFP": 19.5,
    "2016postVFP": 16.8,
    "2017": 42.07,
    "2018": 59.56,
    "2022": 7.98,
    "2022EE": 26.67,
    "2023": 17.7,
    "2023BPix": 9.5,
}
PERIOD_LUMI_FB = {
    "2016": 36.31,
    "Run2": 137.94,
    "2022total": 34.65,
    "2023total": 27.20,
    "Run3": 61.85,
    "Run2+3": 199.79,
}

def selected_eras(period: str) -> Tuple[str, ...]:
    try:
        return ERA_GROUPS[period]
    except KeyError as exc:
        raise ValueError(
            f"Unknown era/period {period!r}; use one of: {', '.join(VALID_ERAS)}"
        ) from exc


def era_dir(args, era: str, collection: str = "") -> str:
    parts = [args.base_dir]
    if collection:
        parts.append(collection)
    parts.append(era)
    if args.trigger:
        parts.append(args.trigger)
    return os.path.join(*parts)


def period_output_dir(args, collection: str = "") -> str:
    parts = [args.base_dir]
    if collection:
        parts.append(collection)
    parts.append(args.era)
    if args.trigger:
        parts.append(args.trigger)
    return os.path.join(*parts)


def period_lumi_fb(period: str) -> float:
    if period in LUMI_FB:
        return LUMI_FB[period]
    if period in PERIOD_LUMI_FB:
        return PERIOD_LUMI_FB[period]
    return sum(LUMI_FB[era] for era in selected_eras(period))


def format_lumi_fb(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def lumi_label(period: str) -> str:
    if period == "Run2+3":
        return (
            f"{format_lumi_fb(PERIOD_LUMI_FB['Run2'])} fb^{{-1}} (13 TeV) + "
            f"{format_lumi_fb(PERIOD_LUMI_FB['Run3'])} fb^{{-1}} (13.6 TeV)"
        )
    energy = "13.6 TeV" if period == "Run3" or period in RUN3_ERAS else "13 TeV"
    return f"{format_lumi_fb(period_lumi_fb(period))} fb^{{-1}} ({energy})"


def get_tf_param_bins(param_name: str) -> array:
    if "pTratio" in param_name:
        edges = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0]
        return array("d", edges)
    else:
        edges = [50., 100., 120., 150., 200., 300., 600., 1000.]
        #edges = [50., 400., 1000.]
        return array("d", edges)

def import_root():
    import ROOT # type: ignore
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)
    return ROOT

def clone_hist(ROOT, hist, name: str):
    out = hist.Clone(name)
    out.SetDirectory(0)
    out.Sumw2()
    return out

# ==============================================================================
# Style Functions 
# ==============================================================================
def apply_style(hist, is_ratio=False):
    font = 42
    hist.GetXaxis().SetTitleFont(font)
    hist.GetXaxis().SetLabelFont(font)
    hist.GetYaxis().SetTitleFont(font)
    hist.GetYaxis().SetLabelFont(font)
    
    if is_ratio:
        hist.GetYaxis().SetTitleSize(0.11)
        hist.GetYaxis().SetTitleOffset(0.52) 
        hist.GetYaxis().SetLabelSize(0.10)
        
        hist.GetXaxis().SetTitleSize(0.13)
        hist.GetXaxis().SetTitleOffset(1.15)
        hist.GetXaxis().SetLabelSize(0.11)
        hist.GetYaxis().SetNdivisions(505)
    else:
        hist.GetYaxis().SetTitleSize(0.055)
        hist.GetYaxis().SetTitleOffset(1.10)
        hist.GetYaxis().SetLabelSize(0.045)
        
        hist.GetXaxis().SetTitleSize(0.0)
        hist.GetXaxis().SetLabelSize(0.0)

# ==============================================================================
# Binning & Scaling Functions
# ==============================================================================
def make_variable_edges(hist, requested_edges: Sequence[float]) -> array:
    xmin = float(hist.GetXaxis().GetXmin())
    xmax = float(hist.GetXaxis().GetXmax())
    edges = [x for x in requested_edges if xmin <= x <= xmax]
    if not edges or edges[0] > xmin:
        edges.insert(0, xmin)
    if edges[-1] < xmax:
        edges.append(xmax)
    unique = []
    for x in edges:
        if not unique or abs(x - unique[-1]) > 1e-9:
            unique.append(float(x))
    return array("d", unique)

def rebin_hist(ROOT, hist, name: str, use_variable_binning: bool, rebin: int):
    hist.Sumw2()
    if use_variable_binning:
        edges = make_variable_edges(hist, DEFAULT_MASS_BINS)
        if len(edges) >= 2:
            out = hist.Rebin(len(edges) - 1, name, edges)
            out.SetDirectory(0)
            out.Sumw2()
            return out
    if rebin > 1:
        out = hist.Rebin(int(rebin), name)
        out.SetDirectory(0)
        out.Sumw2()
        return out
    return clone_hist(ROOT, hist, name)

def scale_to_yield_per_gev(hist) -> None:
    axis = hist.GetXaxis()
    for ibin in range(1, hist.GetNbinsX() + 1):
        width = float(axis.GetBinWidth(ibin))
        if width <= 0.0 or not math.isfinite(width):
            continue
        hist.SetBinContent(ibin, float(hist.GetBinContent(ibin)) / width)
        hist.SetBinError(ibin, float(hist.GetBinError(ibin)) / width)
    hist.GetYaxis().SetTitle("Yield / GeV")

def visible_max(hist, x_min: float, x_max: float) -> float:
    if not hist: return 0.0
    axis = hist.GetXaxis()
    first = max(1, axis.FindFixBin(x_min))
    last = min(hist.GetNbinsX(), axis.FindFixBin(x_max))
    max_y = 0.0
    for ibin in range(first, last + 1):
        max_y = max(max_y, float(hist.GetBinContent(ibin) + hist.GetBinError(ibin)))
    return max_y

def apply_blinding(hist, x_min=9.0, x_max=70.0):
    if not hist: return
    for ibin in range(1, hist.GetNbinsX() + 1):
        center = hist.GetXaxis().GetBinCenter(ibin)
        if x_min < center < x_max:
            hist.SetBinContent(ibin, -9999.0)
            hist.SetBinError(ibin, 0.0)


def zero_hist_range(hist, x_min: float, x_max: float) -> None:
    """Set an analyser-excluded mass interval to zero before rebinning."""
    if not hist:
        return
    for ibin in range(1, hist.GetNbinsX() + 1):
        centre = float(hist.GetXaxis().GetBinCenter(ibin))
        if x_min < centre < x_max:
            hist.SetBinContent(ibin, 0.0)
            hist.SetBinError(ibin, 0.0)

def load_hist_direct(ROOT, path, folder, hist_name, project_x=False):
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        if f:
            f.Close()
        return None

    h = f.Get(f"{folder}/{hist_name}")
    if not h:
        f.Close()
        return None

    if project_x and h.InheritsFrom("TH2"):
        h_proj = h.ProjectionX(f"proj_{folder}_{hist_name}_{os.path.basename(path).replace('.root', '')}")
        h_clone = clone_hist(ROOT, h_proj, f"raw_proj_{folder}_{hist_name}_{os.path.basename(path).replace('.root', '')}")
    else:
        h_clone = clone_hist(ROOT, h, f"raw_{folder}_{hist_name}_{os.path.basename(path).replace('.root', '')}")

    f.Close()
    return h_clone


def load_required_histogram(ROOT, path: str, folder: str, hist_name: str, clone_name: str) -> Any:
    """Load and detach a required ROOT histogram, raising a useful error if absent."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] ROOT file not found: {path}")

    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        if f:
            f.Close()
        raise OSError(f"[ERROR] Failed to open ROOT file: {path}")

    full_path = f"{folder}/{hist_name}"
    h = f.Get(full_path)
    if not h:
        f.Close()
        raise KeyError(f"[ERROR] Histogram not found: {path}:{full_path}")

    out = clone_hist(ROOT, h, clone_name)
    f.Close()
    return out


def load_histogram_across_eras(
    ROOT,
    args,
    relative_path: str,
    folder: str,
    hist_name: str,
    clone_name: str,
    *,
    required: bool = True,
):
    combined = None
    loaded = 0
    missing = []
    for era in selected_eras(args.era):
        path = os.path.join(era_dir(args, era), relative_path)
        hist = load_hist_direct(ROOT, path, folder, hist_name, project_x=False)
        if hist is None:
            missing.append(f"{era}:{path}:{folder}/{hist_name}")
            continue
        if combined is None:
            combined = clone_hist(ROOT, hist, clone_name)
        else:
            if hist.InheritsFrom("TH2") and combined.InheritsFrom("TH2"):
                assert_same_2d_binning(combined, hist, clone_name, f"{era}/{relative_path}")
            elif (not hist.InheritsFrom("TH2")) and (not combined.InheritsFrom("TH2")):
                assert_same_1d_binning(combined, hist, clone_name, f"{era}/{relative_path}")
            else:
                raise ValueError(
                    f"[ERROR] Histogram dimensionality mismatch while combining {relative_path}: "
                    f"{clone_name} versus {era}/{relative_path}"
                )
            combined.Add(hist)
        loaded += 1
    if required and missing:
        raise FileNotFoundError(
            "[ERROR] Required histogram input(s) missing:\n  " + "\n  ".join(missing)
        )
    if required and combined is None:
        raise RuntimeError(f"[ERROR] No histogram loaded for {relative_path}:{folder}/{hist_name}")
    if combined is not None:
        print(
            f"[INFO] Combined {loaded}/{len(selected_eras(args.era))} era(s): "
            f"{relative_path}:{folder}/{hist_name}"
        )
    return combined


def load_histogram_from_paths(
    ROOT,
    labelled_paths: Sequence[Tuple[str, str]],
    folder: str,
    hist_name: str,
    clone_name: str,
):
    combined = None
    for label, path in labelled_paths:
        hist = load_required_histogram(ROOT, path, folder, hist_name, f"{clone_name}_{label}")
        if combined is None:
            combined = clone_hist(ROOT, hist, clone_name)
        else:
            if hist.InheritsFrom("TH2") and combined.InheritsFrom("TH2"):
                assert_same_2d_binning(combined, hist, clone_name, label)
            elif (not hist.InheritsFrom("TH2")) and (not combined.InheritsFrom("TH2")):
                assert_same_1d_binning(combined, hist, clone_name, label)
            else:
                raise ValueError(
                    f"[ERROR] Histogram dimensionality mismatch for {clone_name}: {label}"
                )
            combined.Add(hist)
    if combined is None:
        raise RuntimeError(f"[ERROR] No input histogram loaded for {clone_name}")
    return combined


def assert_same_2d_binning(reference, candidate, reference_label: str, candidate_label: str) -> None:
    """Require identical X and Y bin edges before adding two TH2 histograms."""
    if not reference or not candidate:
        raise ValueError("Cannot compare the binning of missing histograms.")

    for axis_name, ref_axis, cand_axis, n_ref, n_cand in [
        (
            "X",
            reference.GetXaxis(),
            candidate.GetXaxis(),
            reference.GetNbinsX(),
            candidate.GetNbinsX(),
        ),
        (
            "Y",
            reference.GetYaxis(),
            candidate.GetYaxis(),
            reference.GetNbinsY(),
            candidate.GetNbinsY(),
        ),
    ]:
        if n_ref != n_cand:
            raise ValueError(
                f"[ERROR] {reference_label} and {candidate_label} have different "
                f"numbers of {axis_name} bins: {n_ref} versus {n_cand}."
            )

        for ibin in range(1, n_ref + 1):
            ref_low = float(ref_axis.GetBinLowEdge(ibin))
            cand_low = float(cand_axis.GetBinLowEdge(ibin))
            if not math.isclose(ref_low, cand_low, rel_tol=0.0, abs_tol=1.0e-9):
                raise ValueError(
                    f"[ERROR] {reference_label} and {candidate_label} have different "
                    f"{axis_name}-axis binning at bin {ibin}: "
                    f"low edges {ref_low} versus {cand_low}."
                )

        ref_high = float(ref_axis.GetBinUpEdge(n_ref))
        cand_high = float(cand_axis.GetBinUpEdge(n_cand))
        if not math.isclose(ref_high, cand_high, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(
                f"[ERROR] {reference_label} and {candidate_label} have different "
                f"{axis_name}-axis upper edges: {ref_high} versus {cand_high}."
            )



def assert_same_1d_binning(reference, candidate, reference_label: str, candidate_label: str) -> None:
    """Require identical one-dimensional binning before combining/using histograms."""
    if not reference or not candidate:
        raise ValueError("Cannot compare the binning of missing histograms.")
    if reference.GetNbinsX() != candidate.GetNbinsX():
        raise ValueError(
            f"[ERROR] {reference_label} and {candidate_label} have different numbers "
            f"of bins: {reference.GetNbinsX()} versus {candidate.GetNbinsX()}."
        )
    ref_axis = reference.GetXaxis()
    cand_axis = candidate.GetXaxis()
    for ibin in range(1, reference.GetNbinsX() + 1):
        ref_low = float(ref_axis.GetBinLowEdge(ibin))
        cand_low = float(cand_axis.GetBinLowEdge(ibin))
        if not math.isclose(ref_low, cand_low, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(
                f"[ERROR] {reference_label} and {candidate_label} have different "
                f"X-axis binning at bin {ibin}: {ref_low} versus {cand_low}."
            )
    ref_high = float(ref_axis.GetBinUpEdge(reference.GetNbinsX()))
    cand_high = float(cand_axis.GetBinUpEdge(candidate.GetNbinsX()))
    if not math.isclose(ref_high, cand_high, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(
            f"[ERROR] {reference_label} and {candidate_label} have different "
            f"X-axis upper edges: {ref_high} versus {cand_high}."
        )


def load_subtracted_histogram(ROOT, args, folder: str, hist_name: str) -> Any:
    """Return the selected-period sum of Data - QCD - Top - Others."""
    h_sub = load_histogram_across_eras(
        ROOT,
        args,
        os.path.join("DATA", "data.root"),
        folder,
        hist_name,
        f"sub_{folder}_{hist_name}",
        required=True,
    )

    bg_files = [
        "NIsoMuon_QCD_Inclusive.root",
        "NIsoMuon_Top.root",
        "NIsoMuon_Others.root",
    ]
    for bg in bg_files:
        h_bg = load_histogram_across_eras(
            ROOT,
            args,
            bg,
            folder,
            hist_name,
            f"subtract_{bg}_{folder}_{hist_name}",
            required=True,
        )
        if h_bg is None:
            print(f"[WARNING] Background missing during subtraction: {bg}")
            continue
        h_sub.Add(h_bg, -1.0)

    return h_sub

# ==============================================================================
# Parameter-Dependent 2D TF Application Helper
# ==============================================================================
@dataclass
class EstimateResult:
    """Central prediction and statistically separated template variations."""

    central: Any
    factor_down: Any
    factor_up: Any
    lightjet_stat_down: Any
    lightjet_stat_up: Any
    factor_value: float = math.nan
    factor_error: float = math.nan


def _empty_mass_histogram(h2_source, name: str):
    out = h2_source.ProjectionX(name)
    out.Reset()
    out.SetDirectory(0)
    out.Sumw2()
    return out


def _make_binwise_variations(ROOT, central, sigma_hist, name: str) -> Tuple[Any, Any]:
    """Return central-sigma and central+sigma templates, evaluated bin by bin."""
    down = clone_hist(ROOT, central, name + "_down")
    up = clone_hist(ROOT, central, name + "_up")
    for ix in range(1, central.GetNbinsX() + 1):
        value = float(central.GetBinContent(ix))
        sigma = abs(float(sigma_hist.GetBinContent(ix)))
        down.SetBinContent(ix, max(0.0, value - sigma))
        up.SetBinContent(ix, value + sigma)
        down.SetBinError(ix, 0.0)
        up.SetBinError(ix, 0.0)
    return down, up


def _integral_and_error_in_open_range(hist, x_min: float, x_max: float) -> Tuple[float, float]:
    """Integral and quadrature error for x_min < x < x_max."""
    total = 0.0
    variance = 0.0
    for ibin in range(1, hist.GetNbinsX() + 1):
        centre = float(hist.GetXaxis().GetBinCenter(ibin))
        if not (x_min < centre < x_max):
            continue
        total += float(hist.GetBinContent(ibin))
        error = float(hist.GetBinError(ibin))
        variance += error * error
    return total, math.sqrt(max(0.0, variance))


def apply_2d_tf(ROOT, h2_source, h_tf, name: str) -> Any:
    """Backward-compatible central TF application with the original error propagation.

    The production code below uses ``build_tf_estimate`` instead, because that
    function separates the TF numerator statistics from the correlated
    light-jet source/denominator statistics and therefore avoids counting the
    light-jet statistical component twice.
    """
    h_mass_pred_raw = _empty_mass_histogram(h2_source, name + "_raw")

    for iy in range(1, h2_source.GetNbinsY() + 1):
        y_center = h2_source.GetYaxis().GetBinCenter(iy)
        tf_bin = h_tf.FindFixBin(y_center)

        if tf_bin < 1 or tf_bin > h_tf.GetNbinsX():
            tf_val, tf_err = 0.0, 0.0
        else:
            tf_val = float(h_tf.GetBinContent(tf_bin))
            tf_err = float(h_tf.GetBinError(tf_bin))

        if tf_val <= 0.0:
            continue

        for ix in range(1, h2_source.GetNbinsX() + 1):
            bin_content = float(h2_source.GetBinContent(ix, iy))
            bin_error = float(h2_source.GetBinError(ix, iy))
            if bin_content == 0.0 and bin_error == 0.0:
                continue

            pred_content = bin_content * tf_val
            pred_error = math.sqrt(
                (bin_error * tf_val) ** 2 + (bin_content * tf_err) ** 2
            )

            current_content = float(h_mass_pred_raw.GetBinContent(ix))
            current_error = float(h_mass_pred_raw.GetBinError(ix))
            h_mass_pred_raw.SetBinContent(ix, current_content + pred_content)
            h_mass_pred_raw.SetBinError(
                ix, math.sqrt(current_error * current_error + pred_error * pred_error)
            )

    return h_mass_pred_raw


def build_tf_estimate(
    ROOT,
    h2_source,
    h1_b,
    h1_l,
    h_tf,
    name: str,
) -> EstimateResult:
    """Build ``light-jet x TF`` and separate its two statistical components.

    For one transfer-factor bin ``j`` and mass bin ``i``,

        P_ij = L_ij * B_j / L_j.

    ``TF`` variations contain only the statistically independent B-jet
    numerator uncertainty.  The ``DY_LightJetStat`` variation contains the
    direct light-jet mass-bin uncertainty *and* the correlated change of the
    light-jet denominator ``L_j``.  This decomposition prevents the same
    light-jet events from entering both uncertainties independently.

    The light-jet term is obtained by first-order covariance propagation,

        Var_L(P_ij) = TF_j^2 [v_ij + f_ij^2 V_j - 2 f_ij v_ij],

    where ``v_ij = Var(L_ij)``, ``V_j = Var(L_j)``, and ``f_ij=L_ij/L_j``.
    Cross-mass-bin covariance induced by the common denominator cannot be
    stored in a TH1; the saved Up/Down templates retain the resulting
    bin-by-bin one-standard-deviation shifts.
    """
    central = _empty_mass_histogram(h2_source, name + "_central")
    factor_down = _empty_mass_histogram(h2_source, name + "_factor_down")
    factor_up = _empty_mass_histogram(h2_source, name + "_factor_up")
    light_sigma = _empty_mass_histogram(h2_source, name + "_light_sigma")

    n_mass = h2_source.GetNbinsX()
    n_tf = h_tf.GetNbinsX()

    # Aggregate the original Y bins into the coarser TF bins used by h_tf.
    source_yield = [[0.0] * (n_mass + 1) for _ in range(n_tf + 1)]
    source_variance = [[0.0] * (n_mass + 1) for _ in range(n_tf + 1)]
    for iy in range(1, h2_source.GetNbinsY() + 1):
        tf_bin = int(h_tf.FindFixBin(h2_source.GetYaxis().GetBinCenter(iy)))
        if tf_bin < 1 or tf_bin > n_tf:
            continue
        for ix in range(1, n_mass + 1):
            value = float(h2_source.GetBinContent(ix, iy))
            error = float(h2_source.GetBinError(ix, iy))
            source_yield[tf_bin][ix] += value
            source_variance[tf_bin][ix] += error * error

    for ix in range(1, n_mass + 1):
        pred = 0.0
        pred_factor_down = 0.0
        pred_factor_up = 0.0
        light_variance = 0.0

        for jt in range(1, n_tf + 1):
            b_yield = float(h1_b.GetBinContent(jt))
            b_error = abs(float(h1_b.GetBinError(jt)))
            l_total = float(h1_l.GetBinContent(jt))
            l_total_variance = float(h1_l.GetBinError(jt)) ** 2
            l_mass = source_yield[jt][ix]
            l_mass_variance = source_variance[jt][ix]

            if l_total <= 0.0:
                continue

            tf_value = b_yield / l_total
            if tf_value <= 0.0 or not math.isfinite(tf_value):
                continue

            tf_down = max(0.0, b_yield - b_error) / l_total
            tf_up = (b_yield + b_error) / l_total

            pred += l_mass * tf_value
            pred_factor_down += l_mass * tf_down
            pred_factor_up += l_mass * tf_up

            fraction = l_mass / l_total
            this_light_variance = tf_value * tf_value * (
                l_mass_variance
                + fraction * fraction * l_total_variance
                - 2.0 * fraction * l_mass_variance
            )
            light_variance += max(0.0, this_light_variance)

        light_error = math.sqrt(max(0.0, light_variance))
        factor_error = max(abs(pred_factor_down - pred), abs(pred_factor_up - pred))

        central.SetBinContent(ix, pred)
        central.SetBinError(ix, math.sqrt(light_error ** 2 + factor_error ** 2))
        factor_down.SetBinContent(ix, pred_factor_down)
        factor_up.SetBinContent(ix, pred_factor_up)
        factor_down.SetBinError(ix, 0.0)
        factor_up.SetBinError(ix, 0.0)
        light_sigma.SetBinContent(ix, light_error)
        light_sigma.SetBinError(ix, 0.0)

    light_down, light_up = _make_binwise_variations(
        ROOT, central, light_sigma, name + "_lightjet_stat"
    )
    return EstimateResult(
        central=central,
        factor_down=factor_down,
        factor_up=factor_up,
        lightjet_stat_down=light_down,
        lightjet_stat_up=light_up,
    )


def build_nf_estimate(
    ROOT,
    h_b_mass_input,
    h_l_mass_input,
    name: str,
    norm_window: Tuple[float, float],
) -> EstimateResult:
    """Build a constant-normalisation-factor prediction from 1D mass histograms.

    NF mode does not require any mass-versus-parameter TH2.  The B-jet and
    light-jet Dilepton_Mass histograms are used directly.

    The factor is evaluated in ``norm_window``.  The factor nuisance contains
    only the B-jet numerator statistics, while ``LightJetStat`` contains the
    direct light-jet bin statistics and the correlated light-jet denominator
    contribution.
    """
    if h_b_mass_input.InheritsFrom("TH2") or h_l_mass_input.InheritsFrom("TH2"):
        raise TypeError(
            "[ERROR] build_nf_estimate expects one-dimensional Dilepton_Mass "
            "histograms. Use --method tf for the 2D parameter-dependent method."
        )

    assert_same_1d_binning(
        h_b_mass_input,
        h_l_mass_input,
        f"{name} B-jet",
        f"{name} Light-jet",
    )

    h_b_mass = clone_hist(ROOT, h_b_mass_input, name + "_b_mass")
    h_l_mass = clone_hist(ROOT, h_l_mass_input, name + "_l_mass")
    h_b_mass.Sumw2()
    h_l_mass.Sumw2()

    norm_low, norm_high = norm_window
    b_yield, b_error = _integral_and_error_in_open_range(
        h_b_mass, norm_low, norm_high
    )
    l_yield, l_error = _integral_and_error_in_open_range(
        h_l_mass, norm_low, norm_high
    )
    if l_yield <= 0.0 or b_yield <= 0.0:
        raise ValueError(
            f"[ERROR] Cannot derive NF in {norm_low:g} < m < {norm_high:g} GeV: "
            f"B-jet yield={b_yield:g}, Light-jet yield={l_yield:g}."
        )

    nf = b_yield / l_yield
    nf_num_error = b_error / l_yield
    nf_full_error = math.sqrt(
        (b_error / l_yield) ** 2
        + (b_yield * l_error / (l_yield * l_yield)) ** 2
    )

    central = clone_hist(ROOT, h_l_mass, name + "_central")
    factor_down = clone_hist(ROOT, h_l_mass, name + "_factor_down")
    factor_up = clone_hist(ROOT, h_l_mass, name + "_factor_up")
    light_sigma = clone_hist(ROOT, h_l_mass, name + "_light_sigma")
    central.Reset()
    factor_down.Reset()
    factor_up.Reset()
    light_sigma.Reset()
    central.Sumw2()
    factor_down.Sumw2()
    factor_up.Sumw2()
    light_sigma.Sumw2()

    nf_down = max(0.0, nf - nf_num_error)
    nf_up = nf + nf_num_error
    l_window_variance = l_error * l_error

    for ix in range(1, h_l_mass.GetNbinsX() + 1):
        l_mass = float(h_l_mass.GetBinContent(ix))
        l_mass_variance = float(h_l_mass.GetBinError(ix)) ** 2
        centre = float(h_l_mass.GetXaxis().GetBinCenter(ix))
        in_norm_window = norm_low < centre < norm_high

        pred = l_mass * nf
        factor_pred_down = l_mass * nf_down
        factor_pred_up = l_mass * nf_up

        fraction = l_mass / l_yield
        light_variance = nf * nf * (
            l_mass_variance
            + fraction * fraction * l_window_variance
            - (2.0 * fraction * l_mass_variance if in_norm_window else 0.0)
        )
        light_error = math.sqrt(max(0.0, light_variance))
        factor_error = max(abs(factor_pred_down - pred), abs(factor_pred_up - pred))

        central.SetBinContent(ix, pred)
        central.SetBinError(ix, math.sqrt(light_error ** 2 + factor_error ** 2))
        factor_down.SetBinContent(ix, factor_pred_down)
        factor_up.SetBinContent(ix, factor_pred_up)
        factor_down.SetBinError(ix, 0.0)
        factor_up.SetBinError(ix, 0.0)
        light_sigma.SetBinContent(ix, light_error)
        light_sigma.SetBinError(ix, 0.0)

    light_down, light_up = _make_binwise_variations(
        ROOT, central, light_sigma, name + "_lightjet_stat"
    )

    return EstimateResult(
        central=central,
        factor_down=factor_down,
        factor_up=factor_up,
        lightjet_stat_down=light_down,
        lightjet_stat_up=light_up,
        factor_value=nf,
        factor_error=nf_full_error,
    )


def copy_to_full_mass_axis(ROOT, source, name: str, variation: str = "central"):
    """Copy a 0.02 GeV-binned mass histogram to the 7500-bin [0,150] axis.

    In NF mode the source is the one-dimensional Dilepton_Mass histogram
    (already 7500 bins over [0,150] GeV).  In TF mode the source is the mass
    projection of the two-dimensional histogram (7250 bins over [5,150] GeV).
    Both have a bin width of 0.02 GeV.  This function verifies that the source bins align one-to-one
    with the destination bins before copying their contents.  Destination bins
    below the source range remain zero.

    variation may be "central", "up", or "down".  The latter two reproduce
    the original dy_bkg_estimation.py convention content +/- propagated error.
    """
    if variation not in {"central", "up", "down"}:
        raise ValueError(f"Unknown variation: {variation}")

    target = ROOT.TH1D(name, "", 7500, 0.0, 150.0)
    target.SetDirectory(0)
    target.Sumw2()

    target_width = float(target.GetXaxis().GetBinWidth(1))
    used_target_bins = set()

    for src_bin in range(1, source.GetNbinsX() + 1):
        src_width = float(source.GetXaxis().GetBinWidth(src_bin))
        if not math.isclose(src_width, target_width, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(
                "The projected 2D mass histogram is not compatible with the "
                f"7500-bin output axis: source bin {src_bin} has width "
                f"{src_width:.12g} GeV, expected {target_width:.12g} GeV."
            )

        centre = float(source.GetXaxis().GetBinCenter(src_bin))
        target_bin = int(target.FindFixBin(centre))
        if target_bin < 1 or target_bin > target.GetNbinsX():
            continue

        target_centre = float(target.GetXaxis().GetBinCenter(target_bin))
        if not math.isclose(centre, target_centre, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(
                "The projected 2D mass bins are shifted relative to the 7500-bin "
                f"output axis: source centre={centre:.12g}, destination "
                f"centre={target_centre:.12g}."
            )
        if target_bin in used_target_bins:
            raise ValueError(
                f"Multiple source bins map to destination bin {target_bin}; "
                "the mapping is not one-to-one."
            )
        used_target_bins.add(target_bin)

        content = float(source.GetBinContent(src_bin))
        error = float(source.GetBinError(src_bin))
        if variation == "up":
            content += error
        elif variation == "down":
            content -= error

        target.SetBinContent(target_bin, content)
        target.SetBinError(target_bin, error)

    if variation == "central":
        source_integral = float(source.Integral(1, source.GetNbinsX()))
        target_integral = float(target.Integral(1, target.GetNbinsX()))
        if not math.isclose(source_integral, target_integral, rel_tol=1.0e-12, abs_tol=1.0e-10):
            raise ValueError(
                "The yield changed during the one-to-one 7250-bin to 7500-bin "
                f"mapping: source={source_integral:.12g}, target={target_integral:.12g}."
            )
        print(
            "[INFO] Verified one-to-one mass-bin mapping: "
            f"{source.GetNbinsX()} bins over "
            f"[{source.GetXaxis().GetXmin():g}, {source.GetXaxis().GetXmax():g}] GeV "
            f"-> 7500 bins over [0, 150] GeV, bin width={target_width:g} GeV; "
            f"integral preserved ({source_integral:g})."
        )

    return target


def filename_token(value: Any) -> str:
    """Return a compact filename-safe representation of a mass or scale."""
    return str(value).strip().replace("-", "m").replace("+", "p").replace(".", "p")


def build_parameter_tf(ROOT, h2_b, h2_l, tf_bins, prefix: str):
    """Project the parameter axis and form TF(parameter) = B-jet / Light-jet."""
    h1_b_raw = h2_b.ProjectionY(f"{prefix}_b_param_raw")
    h1_l_raw = h2_l.ProjectionY(f"{prefix}_l_param_raw")

    h1_b = h1_b_raw.Rebin(len(tf_bins) - 1, f"{prefix}_b_param_rebin", tf_bins)
    h1_l = h1_l_raw.Rebin(len(tf_bins) - 1, f"{prefix}_l_param_rebin", tf_bins)
    h1_b.SetDirectory(0)
    h1_l.SetDirectory(0)
    h1_b.Sumw2()
    h1_l.Sumw2()

    h_tf = clone_hist(ROOT, h1_b, f"{prefix}_tf")
    h_tf.Divide(h1_l)
    return h1_b, h1_l, h_tf


def signal_output_tag(signal_mass: str, signal_scale: float) -> str:
    return f"_withZpM{filename_token(signal_mass)}_x{filename_token(f'{signal_scale:g}')}"


def signal_mass_display(signal_mass: str) -> str:
    """Human-readable mass label, accepting ROOT-style decimal labels such as 20p5."""
    text = str(signal_mass).strip().replace("p", ".")
    try:
        return f"{float(text):g}"
    except ValueError:
        return text


def tf_parameter_display(param_name: str) -> str:
    """Return the ROOT-TLatex label used for a TF parameter in plots."""
    if param_name == "Dilepton_pT":
        return "p_{T}(#mu#mu)"
    return param_name


def find_signal_files(args) -> List[Tuple[str, str]]:
    """Find one nominal signal ROOT file per selected era."""
    raw_label = str(args.signal_mass).strip()
    raw_label = raw_label.replace("GeV", "").replace("gev", "")
    raw_label = raw_label.replace("M-", "").strip()
    numeric_mass = None
    try:
        numeric_mass = float(raw_label.replace("p", "."))
    except ValueError:
        pass

    labels = [raw_label, raw_label.replace(".", "p")]
    if numeric_mass is not None:
        labels.extend([f"{numeric_mass:g}", f"{numeric_mass:g}".replace(".", "p")])

    outputs: List[Tuple[str, str]] = []
    missing: List[str] = []
    for era in selected_eras(args.era):
        signal_dir = era_dir(args, era)
        found = ""
        seen = set()
        for label in labels:
            if label in seen:
                continue
            seen.add(label)
            candidate = os.path.join(signal_dir, f"NIsoMuon_Zp_M-{label}.root")
            if os.path.exists(candidate):
                found = candidate
                break
        if not found and os.path.isdir(signal_dir) and numeric_mass is not None:
            prefix = "NIsoMuon_Zp_M-"
            suffix = ".root"
            for filename in sorted(os.listdir(signal_dir)):
                if not (filename.startswith(prefix) and filename.endswith(suffix)):
                    continue
                label = filename[len(prefix):-len(suffix)]
                try:
                    file_mass = float(label.replace("p", "."))
                except ValueError:
                    continue
                if math.isclose(file_mass, numeric_mass, rel_tol=0.0, abs_tol=1.0e-9):
                    found = os.path.join(signal_dir, filename)
                    break
        if found:
            outputs.append((era, found))
        else:
            missing.append(era)
    if missing:
        raise FileNotFoundError(
            f"[ERROR] Signal M={args.signal_mass} was not found for: "
            + ", ".join(missing)
        )
    return outputs


def _clear_hist_errors(hist) -> None:
    for ibin in range(0, hist.GetNbinsX() + 2):
        hist.SetBinError(ibin, 0.0)


def _systematic_region(reg_b: str, syst_name: str, direction: str) -> str:
    return reg_b.replace(
        "_NIsoDimuon", f"_Syst_{syst_name}{direction}_NIsoDimuon"
    )


def write_data_driven_root_outputs(
    ROOT,
    args,
    estimate: EstimateResult,
    reg_b: str,
) -> None:
    """Write central, factor-stat, and light-jet-stat templates.

    For backward compatibility with the existing counting workflow, the
    nominal histogram errors retain only the light-jet statistical component.
    The TF/NF factor component is stored exclusively in its Up/Down templates.
    An explicit ``DY_LightJetStat`` pair is also written for shape-based uses;
    it must not be enabled simultaneously with a separate DY statistical
    nuisance derived from the nominal histogram errors.
    """
    factor_syst_name = TF_SYST_NAME if args.method == "tf" else NF_SYST_NAME
    variations = [
        (factor_syst_name, estimate.factor_down, estimate.factor_up),
        (
            LIGHTJET_STAT_SYST_NAME,
            estimate.lightjet_stat_down,
            estimate.lightjet_stat_up,
        ),
    ]

    print("[INFO] Exporting 7500-bin DY estimate and separated systematic templates...")
    print(
        "[INFO] Output nuisance templates: "
        f"{factor_syst_name}Down/Up, "
        f"{LIGHTJET_STAT_SYST_NAME}Down/Up"
    )
    print(
        "[INFO] Nominal TH1 errors contain the same light-jet statistical component; "
        f"use either those errors (e.g. DY_stat/autoMCStats) or {LIGHTJET_STAT_SYST_NAME}, not both."
    )

    def make_output(source, directory_name: str, clear_errors: bool = True):
        hist_name = f"Dilepton_Mass___{directory_name}"
        out = copy_to_full_mass_axis(ROOT, source, hist_name, "central")
        if clear_errors:
            _clear_hist_errors(out)
        return out

    dir_name_cent = reg_b
    h_central_7500 = make_output(estimate.central, dir_name_cent, clear_errors=True)

    output_variations = []
    for syst_name, h_down, h_up in variations:
        dir_down = _systematic_region(reg_b, syst_name, "Down")
        dir_up = _systematic_region(reg_b, syst_name, "Up")
        output_variations.append(
            (
                dir_down,
                make_output(h_down, dir_down),
                dir_up,
                make_output(h_up, dir_up),
            )
        )

    # Retain only the light-jet statistical component in nominal TH1 errors.
    # This preserves the existing DY_stat/IntegralAndError workflow while the
    # factor uncertainty remains exclusive to TF or DY_NFStat templates.
    light_down_7500 = next(
        item[1] for item in output_variations
        if f"_Syst_{LIGHTJET_STAT_SYST_NAME}Down_" in item[0]
    )
    light_up_7500 = next(
        item[3] for item in output_variations
        if f"_Syst_{LIGHTJET_STAT_SYST_NAME}Down_" in item[0]
    )
    for ibin in range(0, h_central_7500.GetNbinsX() + 2):
        central_value = float(h_central_7500.GetBinContent(ibin))
        down_shift = abs(central_value - float(light_down_7500.GetBinContent(ibin)))
        up_shift = abs(float(light_up_7500.GetBinContent(ibin)) - central_value)
        h_central_7500.SetBinError(ibin, max(down_shift, up_shift))

    out_dir_nominal = period_output_dir(args)
    os.makedirs(out_dir_nominal, exist_ok=True)
    out_path_nominal = os.path.join(out_dir_nominal, "NIsoMuon_DYJets_est.root")

    f_out_nominal = ROOT.TFile.Open(out_path_nominal, "RECREATE")
    if not f_out_nominal or f_out_nominal.IsZombie():
        raise OSError(f"[ERROR] Could not create ROOT output: {out_path_nominal}")
    f_out_nominal.mkdir(dir_name_cent).cd()
    h_central_7500.Write()
    f_out_nominal.Close()
    print(f"[SAVED] Exported 7500-bin central estimation to: {out_path_nominal}")

    out_dir_syst = period_output_dir(args, "RunSyst")
    os.makedirs(out_dir_syst, exist_ok=True)
    out_path_syst = os.path.join(out_dir_syst, "NIsoMuon_DYJets_est.root")

    f_out_syst = ROOT.TFile.Open(out_path_syst, "RECREATE")
    if not f_out_syst or f_out_syst.IsZombie():
        raise OSError(f"[ERROR] Could not create ROOT output: {out_path_syst}")

    f_out_syst.mkdir(dir_name_cent).cd()
    h_central_7500.Write()
    for dir_down, h_down, dir_up, h_up in output_variations:
        f_out_syst.cd()
        f_out_syst.mkdir(dir_down).cd()
        h_down.Write()
        f_out_syst.cd()
        f_out_syst.mkdir(dir_up).cd()
        h_up.Write()
    f_out_syst.Close()
    print(f"[SAVED] Exported separated systematic templates to: {out_path_syst}")

# ==============================================================================
# Plotting Routines
# ==============================================================================
def draw_cms_text(ROOT, era: str, text_right: str, extra_lines: List[str] = None) -> None:
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextSize(0.045)
    latex.SetTextAlign(11)
    latex.DrawLatex(0.120, 0.925, "#bf{CMS} #it{Preliminary}")
    latex.SetTextSize(0.038)
    latex.SetTextAlign(31)
    latex.DrawLatex(0.950, 0.925, lumi_label(era))
    
    latex.SetTextSize(0.034)
    latex.SetTextAlign(11)
    latex.DrawLatex(0.16, 0.84, text_right)
    
    if extra_lines:
        y_pos = 0.79
        latex.SetTextFont(42)
        latex.SetTextSize(0.030)
        latex.SetTextColor(ROOT.kGray + 2)
        for line in extra_lines:
            latex.DrawLatex(0.16, y_pos, line)
            y_pos -= 0.038

def draw_styled_plot(ROOT, targets: List[Tuple], preds: List[Tuple], ratios: List[Tuple], args, out_name, title_x, ratio_title, is_tf=False, tf_bins=None, extra_info: List[str] = None):
    ROOT.gSystem.mkdir(PLOT_DIR, True)

    canvas = ROOT.TCanvas(f"c_{out_name}", "", 900, 900)
    upper = ROOT.TPad("upper", "", 0.0, 0.30, 1.0, 1.0)
    lower = ROOT.TPad("lower", "", 0.0, 0.00, 1.0, 0.30)

    upper.SetLeftMargin(0.120); upper.SetRightMargin(0.050)
    upper.SetTopMargin(0.100); upper.SetBottomMargin(0.030)
    lower.SetLeftMargin(0.120); lower.SetRightMargin(0.050)
    lower.SetTopMargin(0.040); lower.SetBottomMargin(0.350)

    if is_tf and args.logx:
        upper.SetLogx(True); lower.SetLogx(True)
    if args.logy:
        upper.SetLogy(True)

    upper.Draw(); lower.Draw()
    upper.cd()

    xmin = tf_bins[0] if is_tf else args.xmin
    xmax = tf_bins[-1] if is_tf else args.xmax

    ymax = 0.0
    for t_hist, _, _, _ in targets: ymax = max(ymax, visible_max(t_hist, xmin, xmax))
    for p_hist, _, _, _ in preds: ymax = max(ymax, visible_max(p_hist, xmin, xmax))

    if args.logy:
        ymin = 1e-2
        ymax = max(ymax * 100.0, ymin * 10.0)
    else:
        ymin = 0.0
        ymax = ymax * 1.45 if ymax > 0.0 else 1.0

    h_frame = targets[0][0]
    h_frame.GetXaxis().SetRangeUser(xmin, xmax)
    h_frame.GetYaxis().SetRangeUser(ymin, ymax)
    h_frame.GetYaxis().SetTitle("Yield / GeV")
    
    apply_style(h_frame, is_ratio=False)

    h_frame.SetLineColor(targets[0][1])
    h_frame.SetMarkerColor(targets[0][1])
    h_frame.SetMarkerStyle(targets[0][2])
    h_frame.Draw("PE")

    legend = ROOT.TLegend(0.48, 0.60, 0.94, 0.89)
    legend.SetBorderSize(0); legend.SetFillStyle(0)
    legend.SetTextSize(0.028); legend.SetTextFont(42)

    cloned_lines_keep_alive = []
    pred_legend_entries = []
    for idx, (h_pred, col, style, label) in enumerate(preds):
        # For light-jet estimates, the tuple's style field is the line style.
        # This keeps the nominal DY curve solid in every plot and allows the
        # DY + Z' MC curve to be dashed, so an almost overlapping nominal curve
        # remains visible underneath it.
        line_style = int(style) if int(style) > 0 else 1
        fill_style = 3004 if line_style == 1 else 3005

        h_pred.SetLineColor(col); h_pred.SetMarkerColor(col)
        h_pred.SetMarkerSize(0.0)
        h_pred.SetLineStyle(line_style)
        h_pred.SetLineWidth(0)
        h_pred.SetFillColor(col)
        h_pred.SetFillStyle(fill_style)
        h_pred.GetXaxis().SetRangeUser(xmin, xmax)
        h_pred.Draw("E2 SAME")

        h_line = h_pred.Clone(f"{h_pred.GetName()}_line")
        h_line.SetFillStyle(0)
        h_line.SetLineColor(col)
        h_line.SetLineStyle(line_style)
        h_line.SetLineWidth(2)
        cloned_lines_keep_alive.append(h_line)
        pred_legend_entries.append((h_line, label))

    # Draw all central lines after all uncertainty bands.  In particular, this
    # prevents the DY + Z' MC band from hiding the nominal light-jet DY line.
    for h_line in cloned_lines_keep_alive:
        h_line.Draw("HIST SAME")
        
    for h_target, col, style, label in targets:
        h_target.SetLineColor(col); h_target.SetMarkerColor(col)
        h_target.SetMarkerStyle(style); h_target.SetMarkerSize(0.85)
        h_target.SetLineWidth(1)
        h_target.GetXaxis().SetRangeUser(xmin, xmax)
        h_target.Draw("PE SAME")

    for h_target, _, _, label in targets: legend.AddEntry(h_target, label, "p")
    for h_line, label in pred_legend_entries: legend.AddEntry(h_line, label, "l")
    legend.Draw()

    draw_text = f"TF Extraction ({tf_parameter_display(args.tf_param)})" if is_tf else "Kinematic Reweighting Closure Test"
    draw_cms_text(ROOT, args.era, draw_text, extra_info)

    upper.SetTickx(); upper.SetTicky(); upper.RedrawAxis()
    lower.cd()

    first_ratio = True
    for idx, (h_ratio, col, style, label) in enumerate(ratios):
        h_ratio.SetLineColor(col); h_ratio.SetMarkerColor(col)
        h_ratio.SetMarkerStyle(style); h_ratio.SetMarkerSize(0.85)
        h_ratio.SetLineWidth(1)
        h_ratio.GetXaxis().SetRangeUser(xmin, xmax)

        if first_ratio:
            if is_tf:
                tf_max = max([h.GetMaximum() for h, _, _, _ in ratios])
                h_ratio.GetYaxis().SetRangeUser(0.0, max(tf_max * 1.5, 0.5))
            else:
                h_ratio.GetYaxis().SetRangeUser(args.ratio_min, args.ratio_max)
                
            h_ratio.SetTitle("")
            apply_style(h_ratio, is_ratio=True)
            h_ratio.GetXaxis().SetTitle(title_x)
            
            # 분수식(#frac) 렌더링을 위한 Y축 타이틀 크기 및 여백 동적 최적화
            if "#frac" in ratio_title:
                h_ratio.GetYaxis().SetTitleSize(0.12)
                h_ratio.GetYaxis().SetTitleOffset(0.38)
            
            h_ratio.GetYaxis().SetTitle(ratio_title)
            h_ratio.Draw("PE")
            first_ratio = False
        else:
            h_ratio.Draw("PE SAME")

    if not is_tf:
        line = ROOT.TF1("line_one", "1.0", xmin, xmax)
        line.SetLineColor(ROOT.kBlack); line.SetLineStyle(2)
        line.Draw("SAME")

    lower.SetTickx(); lower.SetTicky(); lower.SetGridy(); lower.RedrawAxis()

    output_pdf = os.path.join(PLOT_DIR, f"{out_name}.pdf")
    canvas.SaveAs(output_pdf)
    print(f"[SAVED] {output_pdf}")


# ==============================================================================
# Validation Plot Function (Data vs Full MC Stack from 2D Histograms)
# ==============================================================================
def draw_validation_plot(ROOT, args, folder, var_name, tf_bins, out_name, title_x, is_mass=False, use_1d_mass=False):
    ROOT.gSystem.mkdir(PLOT_DIR, True)

    if use_1d_mass:
        mass_hist_name = f"Dilepton_Mass___{folder}"
        h_data_raw = load_histogram_across_eras(
            ROOT, args, os.path.join("DATA", "data.root"),
            folder, mass_hist_name, f"validation_data_{folder}_{out_name}",
            required=True,
        )
        if not h_data_raw:
            print(f"[WARNING] Validation Data missing for {folder} ({mass_hist_name})")
            return
        h_data = rebin_hist(
            ROOT, h_data_raw, f"data_rebin_val_{folder}_{out_name}",
            args.variable_binning, args.rebin
        )
        xmin, xmax = args.xmin, args.xmax
    else:
        h_data_2d = load_histogram_across_eras(
            ROOT, args, os.path.join("DATA", "data.root"),
            folder, var_name, f"validation_data_{folder}_{out_name}",
            required=True,
        )
        if not h_data_2d:
            print(f"[WARNING] Validation Data missing for {folder} ({var_name})")
            return

        if is_mass:
            h_data_raw = h_data_2d.ProjectionX(f"proj_x_data_{folder}_{out_name}")
            h_data = rebin_hist(
                ROOT, h_data_raw, f"data_rebin_val_{folder}_{out_name}",
                args.variable_binning, args.rebin
            )
            xmin, xmax = args.xmin, args.xmax
        else:
            h_data_raw = h_data_2d.ProjectionY(f"proj_y_data_{folder}_{out_name}")
            h_data = h_data_raw.Rebin(
                len(tf_bins) - 1, f"data_rebin_val_{folder}_{out_name}", tf_bins
            )
            xmin, xmax = tf_bins[0], tf_bins[-1]
        
    scale_to_yield_per_gev(h_data)

    if args.blind and is_mass and "BJet" in folder:
        apply_blinding(h_data, 9.0, 70.0)

    color_DY = ROOT.TColor.GetColor("#FFCC66")   
    color_Top = ROOT.TColor.GetColor("#669966")  
    color_QCD = ROOT.TColor.GetColor("#99CCFF")  
    color_Others = ROOT.TColor.GetColor("#CCCCCC") 

    mc_files = [
        ("NIsoMuon_Others.root", color_Others, "Others"),
        ("NIsoMuon_Top.root", color_Top, "Top"),
        ("NIsoMuon_QCD_Inclusive.root", color_QCD, "QCD"),
        ("NIsoMuon_DYJets_Inclusive.root", color_DY, "DY")
    ]

    stack = ROOT.THStack(f"stack_val_{folder}_{out_name}", "")
    h_total_mc = h_data.Clone(f"total_mc_val_{folder}_{out_name}")
    h_total_mc.Reset()

    mc_hists = []
    for bg_file, color, label in mc_files:
        if use_1d_mass:
            mass_hist_name = f"Dilepton_Mass___{folder}"
            h_mc_raw = load_histogram_across_eras(
                ROOT, args, bg_file, folder, mass_hist_name,
                f"validation_{bg_file}_{folder}_{out_name}", required=False,
            )
            if h_mc_raw is None:
                continue
            h_mc = rebin_hist(
                ROOT, h_mc_raw, f"{bg_file}_rebin_val_{folder}_{out_name}",
                args.variable_binning, args.rebin
            )
        else:
            h_mc_2d = load_histogram_across_eras(
                ROOT, args, bg_file, folder, var_name,
                f"validation_{bg_file}_{folder}_{out_name}", required=False,
            )
            if h_mc_2d is None:
                continue

            if is_mass:
                h_mc_raw = h_mc_2d.ProjectionX(f"proj_x_{bg_file}_{folder}_{out_name}")
                h_mc = rebin_hist(
                    ROOT, h_mc_raw, f"{bg_file}_rebin_val_{folder}_{out_name}",
                    args.variable_binning, args.rebin
                )
            else:
                h_mc_raw = h_mc_2d.ProjectionY(f"proj_y_{bg_file}_{folder}_{out_name}")
                h_mc = h_mc_raw.Rebin(
                    len(tf_bins) - 1, f"{bg_file}_rebin_val_{folder}_{out_name}", tf_bins
                )
            
        scale_to_yield_per_gev(h_mc) 

        h_mc.SetFillColor(color)
        h_mc.SetLineColor(ROOT.kBlack)
        h_mc.SetLineWidth(1)

        stack.Add(h_mc)
        h_total_mc.Add(h_mc)
        mc_hists.append((h_mc, label))

    c = ROOT.TCanvas(f"c_val_{folder}_{out_name}", "", 900, 900)
    upper = ROOT.TPad("upper", "", 0.0, 0.30, 1.0, 1.0)
    lower = ROOT.TPad("lower", "", 0.0, 0.00, 1.0, 0.30)

    upper.SetLeftMargin(0.120); upper.SetRightMargin(0.050)
    upper.SetTopMargin(0.100); upper.SetBottomMargin(0.030)
    lower.SetLeftMargin(0.120); lower.SetRightMargin(0.050)
    lower.SetTopMargin(0.040); lower.SetBottomMargin(0.350)

    if args.logy: upper.SetLogy(True)
    #if is_mass and args.logx:
    #    upper.SetLogx(True)
    #    lower.SetLogx(True)
        
    upper.Draw(); lower.Draw()
    upper.cd()

    max_val = max(visible_max(h_data, xmin, xmax), visible_max(h_total_mc, xmin, xmax))
    
    h_data.GetXaxis().SetRangeUser(xmin, xmax)
    h_data.GetYaxis().SetTitle("Yield / GeV")
    apply_style(h_data, is_ratio=False)
    
    if args.logy:
        h_data.SetMaximum(max_val * 100.0)
        h_data.SetMinimum(1e-3)
    else:
        h_data.SetMaximum(max_val * 1.45)
        h_data.SetMinimum(0.0)

    h_data.SetMarkerStyle(20); h_data.SetMarkerColor(ROOT.kBlack); h_data.SetLineColor(ROOT.kBlack)

    h_data.Draw("PE")
    stack.Draw("HIST SAME")
    h_data.Draw("PE SAME")

    legend = ROOT.TLegend(0.60, 0.60, 0.94, 0.89)
    legend.SetBorderSize(0); legend.SetFillStyle(0)
    legend.SetTextSize(0.030); legend.SetTextFont(42)
    legend.AddEntry(h_data, "Data", "lep")
    for h, name in reversed(mc_hists): legend.AddEntry(h, name, "f")
    legend.Draw()

    region_name = "B-Jet" if "BJet" in folder else "Light-Jet"
    val_title = "Dimuon Mass Validation" if is_mass else f"{tf_parameter_display(args.tf_param)} Validation"
    
    extra_val_info = []
    if args.blind and is_mass and "BJet" in folder:
        extra_val_info.append("9 < m(#mu#mu) < 70 GeV Blinded")
        
    draw_cms_text(ROOT, era=args.era, text_right=val_title, extra_lines=[f"Region: {region_name}"] + extra_val_info)
    upper.SetTickx(); upper.SetTicky(); upper.RedrawAxis()

    lower.cd()
    h_ratio = h_data.Clone(f"ratio_val_{folder}_{out_name}")
    h_ratio.Divide(h_total_mc)
    
    if args.blind and is_mass and "BJet" in folder:
        apply_blinding(h_ratio, 9.0, 70.0)
        
    h_ratio.GetXaxis().SetRangeUser(xmin, xmax)
    apply_style(h_ratio, is_ratio=True)
    
    h_ratio.GetYaxis().SetTitle("Data / MC") 
    h_ratio.GetYaxis().SetRangeUser(0.0, 2.0)
    h_ratio.GetXaxis().SetTitle(title_x)

    h_ratio.Draw("PE")

    line = ROOT.TF1(f"line_one_val_{out_name}", "1.0", xmin, xmax)
    line.SetLineColor(ROOT.kBlack); line.SetLineStyle(2); line.Draw("SAME")

    lower.SetTickx(); lower.SetTicky(); lower.SetGridy(); lower.RedrawAxis()

    output_pdf = os.path.join(PLOT_DIR, f"{out_name}.pdf")
    c.SaveAs(output_pdf)
    print(f"[SAVED] {output_pdf}")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dy_bkg_estimation.py",
        description="Derive and validate the NIsoMuon DY TF/NF estimate.",
        epilog=__doc__ or "",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--era", required=True, choices=VALID_ERAS,
        help="Individual era, Run2, Run3, or Run2+3",
    )
    parser.add_argument(
        "--method",
        "--estimation-method",
        choices=("nf", "tf"),
        default="nf",
        help="DY estimation method: constant normalisation factor (nf, default) or parameter-dependent transfer factor (tf)",
    )
    parser.add_argument(
        "--tf-param",
        default="Dilepton_pT",
        help="2D input parameter name used only with --method tf; ignored in nf mode",
    )
    parser.add_argument(
        "--trigger", default="",
        help="Optional legacy trigger subdirectory; empty for SKOutput",
    )
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR, help="Base directory")
    parser.add_argument("--xmin", type=float, default=5.0, help="Minimum mass value shown")
    parser.add_argument("--xmax", type=float, default=150.0, help="Maximum mass value shown")
    parser.add_argument("--ratio-min", type=float, default=0.0, help="Mass Ratio-panel y minimum")
    parser.add_argument("--ratio-max", type=float, default=2.0, help="Mass Ratio-panel y maximum")
    parser.add_argument("--rebin", type=int, default=1, help="Integer rebinning for final mass plot")
    parser.add_argument("--no-variable-binning", dest="variable_binning", action="store_false", help="Disable variable mass binning")
    parser.set_defaults(variable_binning=True)
    parser.add_argument("--logx", action="store_true", help="Use log x axis for Mass plot")
    parser.add_argument("--linear-y", dest="logy", action="store_false", help="Use linear y axis")
    parser.set_defaults(logy=True)
    parser.add_argument("--blind", action="store_true", help="Blind Data in B-Jet dimuon mass region (9 < mass < 70)")
    parser.add_argument(
        "--inject-signal",
        action="store_true",
        help=(
            "Add Z' MC to the DY samples and compare the nominal and injected "
            "factor and closure distributions for the selected tf or nf method"
        ),
    )
    parser.add_argument(
        "--signal-mass",
        default="20",
        help="Signal mass label used in NIsoMuon_Zp_M-<mass>.root, e.g. 20 or 20p5",
    )
    parser.add_argument(
        "--signal-scale",
        type=float,
        default=1.0,
        help="Multiplicative scale applied to the signal MC before adding it to DY MC",
    )
    return parser

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)

    if args.signal_scale < 0.0:
        print("[ERROR] --signal-scale must be non-negative.")
        sys.exit(1)

    ROOT = import_root()

    reg_b = "OS_POGMedium_tight_BJet_NIsoDimuon"
    reg_l = "OS_POGMedium_tight_LightJet_NIsoDimuon"
    tf_param_bins = get_tf_param_bins(args.tf_param)
    tf_param_plot_label = tf_parameter_display(args.tf_param)
    title_x = (
        tf_param_plot_label
        if "pTratio" in args.tf_param
        else f"{tf_param_plot_label} [GeV]"
    )

    hist_2d_b = f"Dilepton_Mass_{args.tf_param}___{reg_b}"
    hist_2d_l = f"Dilepton_Mass_{args.tf_param}___{reg_l}"
    injection_tag = (
        signal_output_tag(args.signal_mass, args.signal_scale)
        if args.inject_signal
        else ""
    )
    method_tag = "" if args.method == "tf" else "_NFMethod"
    factor_word = "TF" if args.method == "tf" else "NF"

    print("=" * 80)
    print(f"[INFO] Running DY MC vs Data (BG Subtracted) closure test for {args.era}")
    print(f"[INFO] Estimation method: {args.method.upper()}")
    if args.method == "nf":
        print("[INFO] NF mode does not use --tf-param or any 2D mass-parameter histogram.")
    if args.method == "tf":
        print(f"[INFO] TF parameter: {args.tf_param}")
        print(
            "[INFO] Statistical decomposition: TF = B-jet numerator statistics; "
            f"{LIGHTJET_STAT_SYST_NAME} = light-jet bin statistics plus the correlated TF denominator effect"
        )
    else:
        print(
            "[INFO] Constant NF windows: "
            f"Data - Bkg MC: {NF_MASS_WINDOW[0]:g} < m(mumu) < {NF_MASS_WINDOW[1]:g} GeV; "
            f"DY MC: {DY_MC_NF_MASS_WINDOW[0]:g} < m(mumu) < {DY_MC_NF_MASS_WINDOW[1]:g} GeV"
        )
        print(
            "[INFO] Statistical decomposition: "
            f"{NF_SYST_NAME} = B-jet normalisation-window statistics; "
            f"{LIGHTJET_STAT_SYST_NAME} = light-jet bin statistics plus the correlated NF denominator effect"
        )
    if args.blind:
        print("[INFO] B-jet Signal Region Blinding is ENABLED (9 < m(mumu) < 70 GeV)")
    if args.inject_signal:
        print(
            "[INFO] Signal injection is ENABLED: "
            f"NIsoMuon_Zp_M-{args.signal_mass}.root, scale={args.signal_scale:g}"
        )

    # --------------------------------------------------------------------------
    # 1--2a. Load inputs.
    #
    # NF mode uses ONLY the one-dimensional Dilepton_Mass histograms.
    # The 2D Dilepton_Mass_<parameter> histograms are loaded only for TF mode.
    # --------------------------------------------------------------------------
    hist_1d_b = f"Dilepton_Mass___{reg_b}"
    hist_1d_l = f"Dilepton_Mass___{reg_l}"

    h2_b_dy_raw = h2_l_dy_raw = None
    h2_b_data_sub_raw = h2_l_data_sub_raw = None
    h2_b_dy_signal_raw = h2_l_dy_signal_raw = None

    h1_b_dy_raw = h1_l_dy_raw = None
    h1_b_data_sub_raw = h1_l_data_sub_raw = None
    h1_b_dy_signal_raw = h1_l_dy_signal_raw = None

    h1_b_dy_rebin = h1_l_dy_rebin = h_tf_dy = None
    h1_b_data_sub_rebin = h1_l_data_sub_rebin = h_tf_data_sub = None
    h_tf_dy_signal = None
    h1_b_dy_signal_rebin = h1_l_dy_signal_rebin = None

    if args.method == "tf":
        print("[INFO] TF mode: loading Dilepton_Mass_<parameter> 2D histograms.")

        h2_b_dy_raw = load_histogram_across_eras(
            ROOT, args, "NIsoMuon_DYJets_Inclusive.root",
            reg_b, hist_2d_b, "h2_b_dy_raw", required=True,
        )
        h2_l_dy_raw = load_histogram_across_eras(
            ROOT, args, "NIsoMuon_DYJets_Inclusive.root",
            reg_l, hist_2d_l, "h2_l_dy_raw", required=True,
        )
        assert_same_2d_binning(
            h2_b_dy_raw, h2_l_dy_raw, "DY B-jet", "DY Light-jet"
        )

        print("[INFO] Loading background-subtracted data from 2D histograms...")
        h2_b_data_sub_raw = load_subtracted_histogram(
            ROOT, args, reg_b, hist_2d_b
        )
        h2_l_data_sub_raw = load_subtracted_histogram(
            ROOT, args, reg_l, hist_2d_l
        )
        assert_same_2d_binning(
            h2_b_data_sub_raw,
            h2_l_data_sub_raw,
            "Background-subtracted Data B-jet",
            "Background-subtracted Data Light-jet",
        )

        h1_b_dy_rebin, h1_l_dy_rebin, h_tf_dy = build_parameter_tf(
            ROOT, h2_b_dy_raw, h2_l_dy_raw, tf_param_bins, "dy"
        )
        (
            h1_b_data_sub_rebin,
            h1_l_data_sub_rebin,
            h_tf_data_sub,
        ) = build_parameter_tf(
            ROOT, h2_b_data_sub_raw, h2_l_data_sub_raw, tf_param_bins, "data_sub"
        )

    else:
        print("[INFO] NF mode: loading one-dimensional Dilepton_Mass histograms only.")

        h1_b_dy_raw = load_histogram_across_eras(
            ROOT, args, "NIsoMuon_DYJets_Inclusive.root",
            reg_b, hist_1d_b, "h1_b_dy_raw", required=True,
        )
        h1_l_dy_raw = load_histogram_across_eras(
            ROOT, args, "NIsoMuon_DYJets_Inclusive.root",
            reg_l, hist_1d_l, "h1_l_dy_raw", required=True,
        )
        assert_same_1d_binning(
            h1_b_dy_raw, h1_l_dy_raw, "DY B-jet", "DY Light-jet"
        )

        print("[INFO] Loading background-subtracted data from 1D mass histograms...")
        h1_b_data_sub_raw = load_subtracted_histogram(
            ROOT, args, reg_b, hist_1d_b
        )
        h1_l_data_sub_raw = load_subtracted_histogram(
            ROOT, args, reg_l, hist_1d_l
        )
        assert_same_1d_binning(
            h1_b_data_sub_raw,
            h1_l_data_sub_raw,
            "Background-subtracted Data B-jet",
            "Background-subtracted Data Light-jet",
        )

    # Optional DY + Z' MC input.
    if args.inject_signal:
        try:
            signal_paths = find_signal_files(args)
        except FileNotFoundError as exc:
            print(str(exc))
            return 1

        for era, signal_path in signal_paths:
            print(f"[INFO] Signal file used [{era}]: {signal_path}")

        if args.method == "tf":
            h2_b_signal_raw = load_histogram_from_paths(
                ROOT, signal_paths, reg_b, hist_2d_b, "h2_b_signal_raw"
            )
            h2_l_signal_raw = load_histogram_from_paths(
                ROOT, signal_paths, reg_l, hist_2d_l, "h2_l_signal_raw"
            )
            assert_same_2d_binning(
                h2_b_dy_raw, h2_b_signal_raw, "DY B-jet", "Signal B-jet"
            )
            assert_same_2d_binning(
                h2_l_dy_raw, h2_l_signal_raw, "DY Light-jet", "Signal Light-jet"
            )

            h2_b_signal_raw.Scale(args.signal_scale)
            h2_l_signal_raw.Scale(args.signal_scale)
            h2_b_dy_signal_raw = clone_hist(
                ROOT, h2_b_dy_raw, "h2_b_dy_signal_raw"
            )
            h2_l_dy_signal_raw = clone_hist(
                ROOT, h2_l_dy_raw, "h2_l_dy_signal_raw"
            )
            h2_b_dy_signal_raw.Add(h2_b_signal_raw)
            h2_l_dy_signal_raw.Add(h2_l_signal_raw)

            (
                h1_b_dy_signal_rebin,
                h1_l_dy_signal_rebin,
                h_tf_dy_signal,
            ) = build_parameter_tf(
                ROOT,
                h2_b_dy_signal_raw,
                h2_l_dy_signal_raw,
                tf_param_bins,
                "dy_signal",
            )

        else:
            h1_b_signal_raw = load_histogram_from_paths(
                ROOT, signal_paths, reg_b, hist_1d_b, "h1_b_signal_raw"
            )
            h1_l_signal_raw = load_histogram_from_paths(
                ROOT, signal_paths, reg_l, hist_1d_l, "h1_l_signal_raw"
            )
            assert_same_1d_binning(
                h1_b_dy_raw, h1_b_signal_raw, "DY B-jet", "Signal B-jet"
            )
            assert_same_1d_binning(
                h1_l_dy_raw, h1_l_signal_raw, "DY Light-jet", "Signal Light-jet"
            )

            h1_b_signal_raw.Scale(args.signal_scale)
            h1_l_signal_raw.Scale(args.signal_scale)
            h1_b_dy_signal_raw = clone_hist(
                ROOT, h1_b_dy_raw, "h1_b_dy_signal_raw"
            )
            h1_l_dy_signal_raw = clone_hist(
                ROOT, h1_l_dy_raw, "h1_l_dy_signal_raw"
            )
            h1_b_dy_signal_raw.Add(h1_b_signal_raw)
            h1_l_dy_signal_raw.Add(h1_l_signal_raw)

    # --------------------------------------------------------------------------
    # 2b. Retain the existing TF extraction plot in the default TF mode.
    # --------------------------------------------------------------------------
    if args.method == "tf":
        h1_b_dy_draw = clone_hist(ROOT, h1_b_dy_rebin, "b_dy_draw")
        h1_l_dy_draw = clone_hist(ROOT, h1_l_dy_rebin, "l_dy_draw")
        h1_b_data_draw = clone_hist(ROOT, h1_b_data_sub_rebin, "b_data_draw")
        h1_l_data_draw = clone_hist(ROOT, h1_l_data_sub_rebin, "l_data_draw")
        for hist in [h1_b_dy_draw, h1_l_dy_draw, h1_b_data_draw, h1_l_data_draw]:
            scale_to_yield_per_gev(hist)

        if args.inject_signal:
            h1_b_dy_signal_draw = clone_hist(
                ROOT, h1_b_dy_signal_rebin, "b_dy_signal_draw"
            )
            h1_l_dy_signal_draw = clone_hist(
                ROOT, h1_l_dy_signal_rebin, "l_dy_signal_draw"
            )
            scale_to_yield_per_gev(h1_b_dy_signal_draw)
            scale_to_yield_per_gev(h1_l_dy_signal_draw)
            tf_targets = [
                (h1_b_dy_draw, ROOT.kBlack, 20, "b-jet (DY MC)"),
                (h1_b_dy_signal_draw, ROOT.kBlue + 1, 21, "b-jet (DY + Z' MC)"),
            ]
            tf_preds = [
                (h1_l_dy_draw, ROOT.kBlack, 1, "light-jet (DY MC)"),
                (h1_l_dy_signal_draw, ROOT.kBlue + 1, 2, "light-jet (DY + Z' MC)"),
            ]
            tf_ratios = [
                (h_tf_dy, ROOT.kBlack, 20, "TF (DY MC)"),
                (h_tf_dy_signal, ROOT.kBlue + 1, 21, "TF (DY + Z' MC)"),
            ]
            tf_extra_info = [
                f"Signal: m(Z') = {signal_mass_display(args.signal_mass)} GeV, #alpha_{{qZ'}} = {args.signal_scale:g}"
            ]
        else:
            tf_targets = [
                (h1_b_dy_draw, ROOT.kBlack, 20, "b-jet (DY MC)"),
                (h1_b_data_draw, ROOT.kRed, 20, "b-jet (Data - Bkg MC)"),
            ]
            tf_preds = [
                (h1_l_dy_draw, ROOT.kBlack, 1, "light-jet (DY MC)"),
                (h1_l_data_draw, ROOT.kRed, 1, "light-jet (Data - Bkg MC)"),
            ]
            tf_ratios = [
                (h_tf_dy, ROOT.kBlack, 20, "TF (DY MC)"),
                (h_tf_data_sub, ROOT.kRed, 20, "TF (Data - Bkg MC)"),
            ]
            tf_extra_info = ["Data TF: Data - Bkg MC"]

        tf_out_name = f"TransferFactor_{args.era}_{args.tf_param}_DataOverlay{injection_tag}"
        draw_styled_plot(
            ROOT,
            tf_targets,
            tf_preds,
            tf_ratios,
            args,
            tf_out_name,
            title_x,
            ratio_title="b-jet/light-jet",
            is_tf=True,
            tf_bins=tf_param_bins,
            extra_info=tf_extra_info,
        )

    # --------------------------------------------------------------------------
    # 3. Build the selected TF or NF mass estimates.
    # --------------------------------------------------------------------------
    print(f"[INFO] Running closure tests with the {factor_word} method...")

    if args.method == "tf":
        h1_mass_actual_dy_raw = h2_b_dy_raw.ProjectionX("mass_actual_dy_raw")
        h1_mass_actual_data_raw = h2_b_data_sub_raw.ProjectionX(
            "mass_actual_data_raw"
        )
        h1_mass_actual_dy_raw.SetDirectory(0)
        h1_mass_actual_data_raw.SetDirectory(0)

        dy_estimate = build_tf_estimate(
            ROOT,
            h2_l_dy_raw,
            h1_b_dy_rebin,
            h1_l_dy_rebin,
            h_tf_dy,
            "mass_pred_dy",
        )
        data_estimate = build_tf_estimate(
            ROOT,
            h2_l_data_sub_raw,
            h1_b_data_sub_rebin,
            h1_l_data_sub_rebin,
            h_tf_data_sub,
            "mass_pred_data",
        )
    else:
        h1_mass_actual_dy_raw = clone_hist(
            ROOT, h1_b_dy_raw, "mass_actual_dy_raw"
        )
        h1_mass_actual_data_raw = clone_hist(
            ROOT, h1_b_data_sub_raw, "mass_actual_data_raw"
        )

        dy_estimate = build_nf_estimate(
            ROOT,
            h1_b_dy_raw,
            h1_l_dy_raw,
            "mass_pred_dy",
            DY_MC_NF_MASS_WINDOW,
        )
        data_estimate = build_nf_estimate(
            ROOT,
            h1_b_data_sub_raw,
            h1_l_data_sub_raw,
            "mass_pred_data",
            NF_MASS_WINDOW,
        )
        print(
            f"[NF] DY MC: {dy_estimate.factor_value:.8g} +/- "
            f"{dy_estimate.factor_error:.8g}"
        )
        print(
            "[NF] Data - Bkg MC: "
            f"{data_estimate.factor_value:.8g} +/- "
            f"{data_estimate.factor_error:.8g}"
        )

    h1_mass_pred_dy_raw = dy_estimate.central
    h1_mass_pred_data_raw = data_estimate.central

    signal_estimate = None
    h1_mass_actual_dy_signal_raw = None
    if args.inject_signal:
        if args.method == "tf":
            h1_mass_actual_dy_signal_raw = h2_b_dy_signal_raw.ProjectionX(
                "mass_actual_dy_signal_raw"
            )
            h1_mass_actual_dy_signal_raw.SetDirectory(0)
            signal_estimate = build_tf_estimate(
                ROOT,
                h2_l_dy_signal_raw,
                h1_b_dy_signal_rebin,
                h1_l_dy_signal_rebin,
                h_tf_dy_signal,
                "mass_pred_dy_signal_closure",
            )
        else:
            h1_mass_actual_dy_signal_raw = clone_hist(
                ROOT, h1_b_dy_signal_raw, "mass_actual_dy_signal_raw"
            )
            signal_estimate = build_nf_estimate(
                ROOT,
                h1_b_dy_signal_raw,
                h1_l_dy_signal_raw,
                "mass_pred_dy_signal_closure",
                DY_MC_NF_MASS_WINDOW,
            )
        h1_mass_pred_dy_signal_closure_raw = signal_estimate.central

    # Production always uses the nominal data-driven estimate for the selected method.
    write_data_driven_root_outputs(ROOT, args, data_estimate, reg_b)

    # The analyser does not fill 9--10.4 GeV.  Keep this excluded interval at zero.
    raw_mass_hists = [
        h1_mass_actual_dy_raw,
        h1_mass_pred_dy_raw,
        h1_mass_actual_data_raw,
        h1_mass_pred_data_raw,
    ]
    if args.inject_signal:
        raw_mass_hists.extend(
            [h1_mass_actual_dy_signal_raw, h1_mass_pred_dy_signal_closure_raw]
        )
    for hist in raw_mass_hists:
        zero_hist_range(hist, 9.0, 10.4)

    h1_mass_actual_dy = rebin_hist(
        ROOT, h1_mass_actual_dy_raw, "mass_actual_dy", args.variable_binning, args.rebin
    )
    h1_mass_pred_dy = rebin_hist(
        ROOT, h1_mass_pred_dy_raw, "mass_pred_dy", args.variable_binning, args.rebin
    )
    h1_mass_actual_data = rebin_hist(
        ROOT, h1_mass_actual_data_raw, "mass_actual_data", args.variable_binning, args.rebin
    )
    h1_mass_pred_data = rebin_hist(
        ROOT, h1_mass_pred_data_raw, "mass_pred_data", args.variable_binning, args.rebin
    )

    main_mass_hists = [
        h1_mass_actual_dy,
        h1_mass_pred_dy,
        h1_mass_actual_data,
        h1_mass_pred_data,
    ]

    h1_mass_actual_dy_signal = None
    h1_mass_pred_dy_signal_closure = None
    if args.inject_signal:
        h1_mass_actual_dy_signal = rebin_hist(
            ROOT,
            h1_mass_actual_dy_signal_raw,
            "mass_actual_dy_signal",
            args.variable_binning,
            args.rebin,
        )
        h1_mass_pred_dy_signal_closure = rebin_hist(
            ROOT,
            h1_mass_pred_dy_signal_closure_raw,
            "mass_pred_dy_signal_closure",
            args.variable_binning,
            args.rebin,
        )
        main_mass_hists.extend(
            [h1_mass_actual_dy_signal, h1_mass_pred_dy_signal_closure]
        )

    for hist in main_mass_hists:
        scale_to_yield_per_gev(hist)

    if args.blind:
        apply_blinding(h1_mass_actual_data, 9.0, 70.0)

    h_closure_ratio_dy = clone_hist(ROOT, h1_mass_actual_dy, "closure_ratio_dy")
    h_closure_ratio_dy.Divide(h1_mass_pred_dy)

    if args.method == "tf":
        extra_info_closure = [
            f"Transfer factor function: {tf_param_plot_label}",
            "Upsilon region discarded (9.0-10.4 GeV)",
        ]
    else:
        extra_info_closure = [
            f"Data: NF from {NF_MASS_WINDOW[0]:g} < m < {NF_MASS_WINDOW[1]:g} GeV",
            f"DY MC: NF from {DY_MC_NF_MASS_WINDOW[0]:g} < m < {DY_MC_NF_MASS_WINDOW[1]:g} GeV",
            "Upsilon region discarded (9.0-10.4 GeV)",
        ]
    if args.inject_signal:
        extra_info_closure.append(
            f"Signal: m(Z') = {signal_mass_display(args.signal_mass)} GeV, #alpha_{{qZ'}} = {args.signal_scale:g}"
        )

    # Keep every TF legend string unchanged.  Numeric factors are inserted only
    # for the constant-NF method.
    if args.method == "tf":
        dy_prediction_label = "light-jet #times TF(DY MC)"
        data_prediction_label = "light-jet #times TF(Data - Bkg MC)"
        injected_prediction_label = "light-jet #times TF(DY + Z' MC)"
    else:
        dy_prediction_label = (
            f"light-jet #times {dy_estimate.factor_value:.6g} (DY MC)"
        )
        data_prediction_label = (
            f"light-jet #times {data_estimate.factor_value:.6g} (Data - Bkg MC)"
        )
        injected_prediction_label = (
            f"light-jet #times {signal_estimate.factor_value:.6g} (DY + Z' MC)"
            if signal_estimate is not None
            else "light-jet (DY + Z' MC)"
        )

    if args.inject_signal:
        closure_targets = [
            (h1_mass_actual_dy, ROOT.kBlack, 20, "b-jet (DY MC)"),
            (h1_mass_actual_dy_signal, ROOT.kBlue + 1, 21, "b-jet (DY + Z' MC)"),
        ]
        closure_preds = [
            (h1_mass_pred_dy, ROOT.kBlack, 1, dy_prediction_label),
            (
                h1_mass_pred_dy_signal_closure,
                ROOT.kBlue + 1,
                2,
                injected_prediction_label,
            ),
        ]
        h_closure_ratio_dy_signal = clone_hist(
            ROOT, h1_mass_actual_dy_signal, "closure_ratio_dy_signal"
        )
        h_closure_ratio_dy_signal.Divide(h1_mass_pred_dy_signal_closure)
        closure_ratios = [
            (h_closure_ratio_dy, ROOT.kBlack, 24, "Ratio (DY MC)"),
            (h_closure_ratio_dy_signal, ROOT.kBlue + 1, 21, "Ratio (DY + Z' MC)"),
        ]
    else:
        closure_targets = [
            (h1_mass_actual_dy, ROOT.kBlack, 20, "b-jet (DY MC)"),
            (h1_mass_actual_data, ROOT.kRed, 20, "b-jet (Data - Bkg MC)"),
        ]
        closure_preds = [
            (h1_mass_pred_dy, ROOT.kBlack, 1, dy_prediction_label),
            (
                h1_mass_pred_data,
                ROOT.kRed,
                1,
                data_prediction_label,
            ),
        ]
        h_closure_ratio_data = clone_hist(
            ROOT, h1_mass_actual_data, "closure_ratio_data"
        )
        h_closure_ratio_data.Divide(h1_mass_pred_data)
        if args.blind:
            apply_blinding(h_closure_ratio_data, 9.0, 70.0)
        closure_ratios = [
            (h_closure_ratio_dy, ROOT.kBlack, 24, "Ratio (DY MC)"),
            (h_closure_ratio_data, ROOT.kRed + 1, 20, "Ratio (Data - Bkg MC)"),
        ]
        if args.blind:
            extra_info_closure.append("9 < m(#mu#mu) < 70 GeV blinded")

    if args.method == "tf":
        closure_out_name = (
            f"ClosureTest_{args.era}_{args.tf_param}_DataOverlay{injection_tag}"
        )
    else:
        closure_out_name = (
            f"ClosureTest_{args.era}_DataOverlay_NFMethod{injection_tag}"
        )
    if args.variable_binning:
        closure_out_name += "_varBin"
    else:
        closure_out_name += f"_rebin{args.rebin}"

    draw_styled_plot(
        ROOT,
        closure_targets,
        closure_preds,
        closure_ratios,
        args,
        closure_out_name,
        title_x="m(#mu#mu) [GeV]",
        ratio_title=f"#frac{{b-jet}}{{light-jet #times {factor_word}}}",
        is_tf=False,
        extra_info=extra_info_closure,
    )

    if args.inject_signal:
        h_signal_closure_ratio = clone_hist(
            ROOT, h1_mass_actual_dy_signal, "signal_injected_closure_ratio"
        )
        h_signal_closure_ratio.Divide(h1_mass_pred_dy_signal_closure)
        if args.method == "tf":
            signal_closure_name = (
                f"SignalInjectedClosure_{args.era}_{args.tf_param}{injection_tag}"
            )
        else:
            signal_closure_name = (
                f"SignalInjectedClosure_{args.era}_NFMethod{injection_tag}"
            )
        if args.variable_binning:
            signal_closure_name += "_varBin"
        else:
            signal_closure_name += f"_rebin{args.rebin}"
        draw_styled_plot(
            ROOT,
            [(h1_mass_actual_dy_signal, ROOT.kBlue + 1, 21, "b-jet (DY + Z' MC)")],
            [
                (
                    h1_mass_pred_dy_signal_closure,
                    ROOT.kBlue + 1,
                    2,
                    f"light-jet #times {factor_word}(DY + Z' MC)",
                )
            ],
            [(h_signal_closure_ratio, ROOT.kBlue + 1, 21, "DY + Z' MC closure ratio")],
            args,
            signal_closure_name,
            title_x="m(#mu#mu) [GeV]",
            ratio_title=f"#frac{{b-jet}}{{light-jet #times {factor_word}}}",
            is_tf=False,
            extra_info=[
                f"m(Z') = {signal_mass_display(args.signal_mass)} GeV",
                f"Signal scale = {args.signal_scale:g}",
                "Upsilon region discarded (9.0-10.4 GeV)",
            ],
        )

    # --------------------------------------------------------------------------
    # 4. Validation plots.
    #
    # TF mode keeps the parameter and mass validation plots based on TH2.
    # NF mode makes only the B-jet and light-jet mass validation plots and
    # reads the one-dimensional Dilepton_Mass histograms directly.
    # --------------------------------------------------------------------------
    if args.method == "tf":
        print("[INFO] Generating TF validation plots from 2D projections...")
        draw_validation_plot(
            ROOT,
            args,
            reg_b,
            hist_2d_b,
            tf_param_bins,
            f"Validation_{args.era}_{args.tf_param}_BJet",
            title_x,
            is_mass=False,
        )
        draw_validation_plot(
            ROOT,
            args,
            reg_l,
            hist_2d_l,
            tf_param_bins,
            f"Validation_{args.era}_{args.tf_param}_LightJet",
            title_x,
            is_mass=False,
        )
        draw_validation_plot(
            ROOT,
            args,
            reg_b,
            hist_2d_b,
            tf_param_bins,
            f"Validation_{args.era}_DimuonMass_BJet",
            "m(#mu#mu) [GeV]",
            is_mass=True,
        )
        draw_validation_plot(
            ROOT,
            args,
            reg_l,
            hist_2d_l,
            tf_param_bins,
            f"Validation_{args.era}_DimuonMass_LightJet",
            "m(#mu#mu) [GeV]",
            is_mass=True,
        )
    else:
        print("[INFO] Generating NF mass validation plots from 1D histograms...")
        draw_validation_plot(
            ROOT,
            args,
            reg_b,
            "",
            tf_param_bins,
            f"Validation_{args.era}_DimuonMass_BJet",
            "m(#mu#mu) [GeV]",
            is_mass=True,
            use_1d_mass=True,
        )
        draw_validation_plot(
            ROOT,
            args,
            reg_l,
            "",
            tf_param_bins,
            f"Validation_{args.era}_DimuonMass_LightJet",
            "m(#mu#mu) [GeV]",
            is_mass=True,
            use_1d_mass=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
