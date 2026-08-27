#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit NIsoMuon Z' signal dimuon-mass distributions with double crystal ball or
Gaussian functions and parameterise the relative mass resolution sigma/m as a
linear function of mass.

Current analysis defaults
-------------------------
Input base:
  /data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13/NIsoMuon

Signal region:
  OS_POGMedium_tight_BJet_NIsoDimuon

Histogram:
  Dilepton_Mass

Final mass hypotheses:
  12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70 GeV

The default double-crystal-ball fit uses
  m(Z') - 0.07*m(Z') < m(mumu) < m(Z') + 0.07*m(Z'),
with a Gaussian core covering at least
  m(Z') - 0.02*m(Z') < m(mumu) < m(Z') + 0.02*m(Z').

The optional Gaussian fit keeps the original range
  m(Z') - 0.04*m(Z') < m(mumu) < m(Z') + 0.04*m(Z').

The default fit function is a double crystal ball.  A Gaussian can be selected
with --fit-function gaussian.

For each era, sigma/m is then fitted as
  sigma/m = a*m + b.

The script supports all four Run-2 and all four Run-3 detector configurations.

Typical usage
-------------
  python3 sigFit.py
  python3 sigFit.py --era Run2
  python3 sigFit.py --era Run3
  python3 sigFit.py --era 2023
  python3 sigFit.py --fit-function gaussian
  python3 sigFit.py --masses final
  python3 sigFit.py --masses auto
  python3 sigFit.py --masses 12,20,40,70

Outputs
-------
  plots/sigFit/sigFit_<era>_M-<mass>.pdf/png
  plots/sigFit/sigFit_resolution_<era>.pdf/png
  plots/sigFit/sigFit_results.csv
  plots/sigFit/sigFit_results.json
  plots/sigFit/resolution_coefficients.csv
  plots/sigFit/resolution_coefficients.json
  plots/sigFit/resolution_coefficients.py

The generated resolution_coefficients.py contains a RESOLUTION dictionary in
the same convention used by the counting-limit workflow:
  sigma_m / m = a*m + b.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


RUN2_ERAS: Tuple[str, ...] = (
    "2016preVFP",
    "2016postVFP",
    "2017",
    "2018",
)
RUN3_ERAS: Tuple[str, ...] = (
    "2022",
    "2022EE",
    "2023",
    "2023BPix",
)
ALL_ERAS: Tuple[str, ...] = RUN2_ERAS + RUN3_ERAS

FINAL_MASSES: Tuple[float, ...] = (
    12.0,
    15.0,
    20.0,
    25.0,
    30.0,
    35.0,
    40.0,
    45.0,
    50.0,
    55.0,
    60.0,
    65.0,
    70.0,
)

DEFAULT_BASE_DIR = (
    "/data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13/NIsoMuon"
)
DEFAULT_REGION = "OS_POGMedium_tight_BJet_NIsoDimuon"
DEFAULT_HIST_NAME = "Dilepton_Mass"
DEFAULT_OUTPUT_DIR = "plots/sigFit"
MIN_DCB_FIT_FRACTION = 0.1
MIN_DCB_CORE_FRACTION = 0.03

LUMI_FB: Dict[str, float] = {
    "2016preVFP": 19.5,
    "2016postVFP": 16.8,
    "2017": 41.5,
    "2018": 59.8,
    "2022": 7.98,
    "2022EE": 26.67,
    "2023": 17.7,
    "2023BPix": 9.5,
}


@dataclass
class FitResult:
    era: str
    mass: float
    file: str
    hist_path: str
    fit_min: float
    fit_max: float
    plot_min: float
    plot_max: float
    original_bin_width: float
    rebin_factor: int
    rebinned_bin_width: float
    fit_status: int
    covariance_status: int
    amplitude: float
    amplitude_error: float
    mean: float
    mean_error: float
    sigma: float
    sigma_error: float
    sigma_over_mass: float
    sigma_over_mass_error: float
    chi2: float
    ndf: int
    chi2_ndf: Optional[float]
    plot_files: List[str]


@dataclass
class ResolutionResult:
    era: str
    n_points: int
    mass_min: float
    mass_max: float
    intercept_b: float
    intercept_b_error: float
    slope_a: float
    slope_a_error: float
    chi2: float
    ndf: int
    chi2_ndf: Optional[float]
    plot_files: List[str]


class NameFactory:
    def __init__(self) -> None:
        self.counter = 0

    def unique(self, prefix: str) -> str:
        self.counter += 1
        clean = re.sub(r"[^A-Za-z0-9_]+", "_", str(prefix))
        return f"{clean}_{self.counter}"


NAMES = NameFactory()


def import_root():
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        import ROOT  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Could not import PyROOT. Run in a ROOT/CMSSW environment "
            "(for example after cmsenv). "
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
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(0)
    ROOT.gStyle.SetOptTitle(0)
    try:
        ROOT.TGaxis.SetMaxDigits(4)
    except Exception:
        pass
    return ROOT


def parse_fit_function(value: str) -> str:
    key = re.sub(r"[\s_-]+", "", value.strip().lower())
    aliases = {
        "gaus": "gaussian",
        "gaussian": "gaussian",
        "dcb": "double-crystal-ball",
        "dscb": "double-crystal-ball",
        "doublecb": "double-crystal-ball",
        "doublecrystalball": "double-crystal-ball",
    }
    if key not in aliases:
        raise argparse.ArgumentTypeError(
            "fit function must be 'gaussian' or 'double-crystal-ball'"
        )
    return aliases[key]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sigFit.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--era",
        default="all",
        choices=(*ALL_ERAS, "Run2", "Run3", "all"),
        help="era selection; Run2/Run3 mean all component eras separately",
    )
    parser.add_argument(
        "--base-dir",
        default=DEFAULT_BASE_DIR,
        help="NIsoMuon base directory",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help="ROOT region/directory name",
    )
    parser.add_argument(
        "--hist-name",
        default=DEFAULT_HIST_NAME,
        help="histogram name inside the selected region",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="output directory",
    )
    parser.add_argument(
        "--masses",
        default="final",
        help=(
            "'final' for the 13 final mass hypotheses, 'auto' for every discovered "
            "nominal signal file, or a comma-separated list"
        ),
    )
    parser.add_argument(
        "--fit-function",
        type=parse_fit_function,
        choices=("gaussian", "double-crystal-ball"),
        default="double-crystal-ball",
        help="signal fit function; default: %(default)s",
    )
    parser.add_argument(
        "--fit-fraction",
        type=float,
        default=0.04,
        help=(
            "requested fit half-width as a fraction of the mass; default: 0.04. "
            "The double-crystal-ball fit always uses at least 0.07"
        ),
    )
    parser.add_argument(
        "--plot-fraction",
        type=float,
        default=0.20,
        help="display half-width as a fraction of the mass; default: 0.20",
    )
    parser.add_argument(
        "--min-fit-half-width",
        type=float,
        default=0.05,
        help="minimum fit half-width in GeV",
    )
    parser.add_argument(
        "--rebin-fraction",
        type=float,
        default=0.005,
        help=(
            "target mass-bin width as a fraction of m(Z'); the actual target is "
            "max(rebin_fraction*m, min_rebin_width)"
        ),
    )
    parser.add_argument(
        "--min-rebin-width",
        type=float,
        default=0.05,
        help="minimum target mass-bin width in GeV",
    )
    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="use linear y scale for the per-mass fit plots",
    )
    parser.add_argument(
        "--extensions",
        default="pdf,png",
        help="comma-separated output formats; default: pdf,png",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="abort on a missing signal file, histogram, or failed fit",
    )
    parser.add_argument(
        "--no-resolution-fit",
        action="store_true",
        help="skip the linear sigma/m versus mass fit",
    )
    return parser


def selected_eras(value: str) -> Tuple[str, ...]:
    if value == "Run2":
        return RUN2_ERAS
    if value == "Run3":
        return RUN3_ERAS
    if value == "all":
        return ALL_ERAS
    return (value,)


def parse_masses(value: str) -> Optional[List[float]]:
    key = value.strip().lower()
    if key == "final":
        return list(FINAL_MASSES)
    if key == "auto":
        return None

    out: List[float] = []
    for token in re.split(r"[,;:\s]+", value.strip()):
        if not token:
            continue
        cleaned = token.replace("GeV", "").replace("gev", "")
        cleaned = cleaned.replace("M-", "").replace("p", ".")
        out.append(float(cleaned))
    if not out:
        raise ValueError("--masses did not contain any valid mass values.")
    return sorted(set(out))


def mass_label(mass: float) -> str:
    if abs(mass - round(mass)) < 1.0e-9:
        return str(int(round(mass)))
    return f"{mass:.8g}".replace(".", "p")


def hist_path(region: str, hist_name: str) -> str:
    return f"{region}/{hist_name}___{region}"


def signal_mass_from_filename(path: str) -> Optional[float]:
    base = os.path.basename(path)

    # Nominal files end directly after the mass label.  pT-binned generation
    # fragments such as ..._M-20_Pt-... are intentionally rejected here.
    match = re.search(
        r"Zp_M-([0-9]+(?:p[0-9]+|\.[0-9]+)?)\.root$",
        base,
    )
    if not match:
        return None
    try:
        return float(match.group(1).replace("p", "."))
    except ValueError:
        return None


def signal_file_rank(path: Path) -> Tuple[int, int, str]:
    base = path.name
    if base.startswith("NIsoMuon_Zp_M-"):
        priority = 0
    elif base.startswith("Skim_NIsoMuon_Zp_M-"):
        priority = 1
    elif base.startswith("NIsoMuon_SkimTree_NIsoMuon_Zp_M-"):
        priority = 2
    else:
        priority = 3
    return (priority, len(base), base)


def discover_signal_files(base_dir: str, era: str) -> Dict[float, str]:
    directory = Path(base_dir) / era
    if not directory.is_dir():
        return {}

    candidates: List[Path] = []
    for pattern in (
        "NIsoMuon_Zp_M-*.root",
        "Skim_NIsoMuon_Zp_M-*.root",
        "NIsoMuon_SkimTree_NIsoMuon_Zp_M-*.root",
    ):
        candidates.extend(directory.glob(pattern))

    found: Dict[float, str] = {}
    ranks: Dict[float, Tuple[int, int, str]] = {}
    for path in candidates:
        mass = signal_mass_from_filename(str(path))
        if mass is None:
            continue
        rank = signal_file_rank(path)
        if mass not in found or rank < ranks[mass]:
            found[mass] = str(path)
            ranks[mass] = rank
    return found


def find_requested_file(
    files: Dict[float, str],
    requested_mass: float,
) -> Optional[str]:
    for mass, path in files.items():
        if abs(mass - requested_mass) < 1.0e-6:
            return path
    return None


def open_histogram(ROOT, filename: str, path: str):
    root_file = ROOT.TFile.Open(filename, "READ")
    if not root_file or root_file.IsZombie():
        if root_file:
            root_file.Close()
        raise OSError(f"Could not open ROOT file: {filename}")

    hist = root_file.Get(path)
    if not hist:
        root_file.Close()
        raise KeyError(f"Histogram not found: {filename}:{path}")

    out = hist.Clone(NAMES.unique("signal_mass"))
    out.SetDirectory(0)
    if out.GetSumw2N() == 0:
        out.Sumw2()
    root_file.Close()
    return out


def is_uniform_binning(hist, tol: float = 1.0e-9) -> bool:
    if hist.GetNbinsX() < 2:
        return True
    width0 = float(hist.GetXaxis().GetBinWidth(1))
    for ibin in range(2, hist.GetNbinsX() + 1):
        width = float(hist.GetXaxis().GetBinWidth(ibin))
        if not math.isclose(width, width0, rel_tol=0.0, abs_tol=tol):
            return False
    return True


def get_rebin_factor(
    hist,
    mass: float,
    rebin_fraction: float,
    min_rebin_width: float,
) -> int:
    if not is_uniform_binning(hist):
        return 1

    original_width = float(hist.GetXaxis().GetBinWidth(1))
    if original_width <= 0.0:
        return 1

    target_width = max(rebin_fraction * mass, min_rebin_width)
    factor = max(1, int(round(target_width / original_width)))

    # ROOT's integer Rebin is cleanest when the factor divides the original
    # number of bins.  Reduce the factor until that is true.
    n_bins = int(hist.GetNbinsX())
    while factor > 1 and n_bins % factor != 0:
        factor -= 1
    return max(1, factor)


def compute_chi2(hist, fit_function, fit_min: float, fit_max: float) -> Tuple[float, int]:
    chi2 = 0.0
    n_points = 0
    for ibin in range(1, hist.GetNbinsX() + 1):
        x = float(hist.GetBinCenter(ibin))
        if x < fit_min or x > fit_max:
            continue
        y = float(hist.GetBinContent(ibin))
        ey = float(hist.GetBinError(ibin))
        if ey <= 0.0 or not math.isfinite(ey):
            continue
        f = float(fit_function.Eval(x))
        pull = (y - f) / ey
        chi2 += pull * pull
        n_points += 1
    ndf = n_points - int(fit_function.GetNumberFreeParameters())
    return chi2, ndf


def double_crystal_ball(x, parameters) -> float:
    amplitude = max(0.0, float(parameters[0]))
    mean = float(parameters[1])
    sigma = max(abs(float(parameters[2])), 1.0e-12)
    alpha_left = max(abs(float(parameters[3])), 1.0e-12)
    n_left = max(float(parameters[4]), 1.0001)
    alpha_right = max(abs(float(parameters[5])), 1.0e-12)
    n_right = max(float(parameters[6]), 1.0001)
    t = (float(x[0]) - mean) / sigma

    if t < -alpha_left:
        a_left = (n_left / alpha_left) ** n_left
        a_left *= math.exp(-0.5 * alpha_left * alpha_left)
        b_left = n_left / alpha_left - alpha_left
        return amplitude * a_left * (b_left - t) ** (-n_left)

    if t > alpha_right:
        a_right = (n_right / alpha_right) ** n_right
        a_right *= math.exp(-0.5 * alpha_right * alpha_right)
        b_right = n_right / alpha_right - alpha_right
        return amplitude * a_right * (b_right + t) ** (-n_right)

    return amplitude * math.exp(-0.5 * t * t)


def constrained_double_crystal_ball(
    mass: float,
    core_fraction: float = MIN_DCB_CORE_FRACTION,
):
    core_min = mass * (1.0 - core_fraction)
    core_max = mass * (1.0 + core_fraction)

    def evaluate(x, parameters) -> float:
        mean = float(parameters[1])
        sigma = max(abs(float(parameters[2])), 1.0e-12)
        left_transition = core_min - max(0.0, float(parameters[3]))
        right_transition = core_max + max(0.0, float(parameters[5]))
        alpha_left = max((mean - left_transition) / sigma, 1.0e-12)
        alpha_right = max((right_transition - mean) / sigma, 1.0e-12)
        dcb_parameters = (
            parameters[0],
            mean,
            sigma,
            alpha_left,
            parameters[4],
            alpha_right,
            parameters[6],
        )
        return double_crystal_ball(x, dcb_parameters)

    return evaluate


def fit_function_label(fit_function_name: str) -> str:
    if fit_function_name == "gaussian":
        return "Gaussian"
    return "Double crystal ball"


def create_signal_fit_function(
    ROOT,
    *,
    fit_function_name: str,
    era: str,
    mass: float,
    fit_min: float,
    fit_max: float,
    fit_half_width: float,
    peak_y: float,
    sigma_init: float,
):
    label = mass_label(mass)
    if fit_function_name == "gaussian":
        fit_function = ROOT.TF1(
            NAMES.unique(f"gaus_{era}_{label}"),
            "gaus",
            fit_min,
            fit_max,
        )
        fit_function.SetParameter(0, peak_y)
        fit_function.SetParameter(1, mass)
        fit_function.SetParameter(2, sigma_init)
    else:
        core_min = mass * (1.0 - MIN_DCB_CORE_FRACTION)
        core_max = mass * (1.0 + MIN_DCB_CORE_FRACTION)
        dcb_callable = constrained_double_crystal_ball(mass)
        fit_function = ROOT.TF1(
            NAMES.unique(f"dcb_{era}_{label}"),
            dcb_callable,
            fit_min,
            fit_max,
            7,
        )
        fit_function.SetParameters(
            peak_y,
            mass,
            sigma_init,
            0.0,
            5.0,
            0.0,
            5.0,
        )
        fit_function.SetParName(3, "Left core extension")
        fit_function.SetParName(4, "n_{L}")
        fit_function.SetParName(5, "Right core extension")
        fit_function.SetParName(6, "n_{R}")
        fit_function.SetParLimits(3, 0.0, max(core_min - fit_min, 1.0e-6))
        fit_function.SetParLimits(4, 1.05, 60.0)
        fit_function.SetParLimits(5, 0.0, max(fit_max - core_max, 1.0e-6))
        fit_function.SetParLimits(6, 1.05, 60.0)
        # Keep PyROOT's Python callback alive for as long as the TF1 is used.
        fit_function._python_callable = dcb_callable

    fit_function.SetParName(0, "Amplitude")
    fit_function.SetParName(1, "Mean")
    fit_function.SetParName(2, "#sigma")
    fit_function.SetParLimits(0, 0.0, peak_y * 100.0 + 1.0)
    if fit_function_name == "gaussian":
        fit_function.SetParLimits(1, fit_min, fit_max)
    else:
        fit_function.SetParLimits(1, core_min, core_max)
    fit_function.SetParLimits(2, 0.001, fit_half_width)
    return fit_function


def gaussian_uncertainty(ROOT, fit_function, fit_result, x: float) -> float:
    if not fit_result:
        return 0.0

    amplitude = float(fit_function.GetParameter(0))
    mean = float(fit_function.GetParameter(1))
    sigma = float(fit_function.GetParameter(2))
    if amplitude == 0.0 or sigma == 0.0:
        return 0.0

    fx = float(fit_function.Eval(x))
    dx = x - mean
    grad = (
        fx / amplitude,
        fx * dx / (sigma * sigma),
        fx * dx * dx / (sigma * sigma * sigma),
    )

    variance = 0.0
    for i in range(3):
        for j in range(3):
            variance += (
                grad[i]
                * float(fit_result.CovMatrix(i, j))
                * grad[j]
            )
    return math.sqrt(max(0.0, variance))


def fit_uncertainty(
    ROOT,
    fit_function,
    fit_result,
    x: float,
    fit_function_name: str,
) -> float:
    if fit_function_name == "gaussian":
        return gaussian_uncertainty(ROOT, fit_function, fit_result, x)
    if not fit_result:
        return 0.0

    n_parameters = int(fit_function.GetNpar())
    parameters = [
        float(fit_function.GetParameter(i)) for i in range(n_parameters)
    ]
    gradient: List[float] = []
    for i, value in enumerate(parameters):
        step = max(abs(value) * 1.0e-5, 1.0e-6)
        fit_function.SetParameter(i, value + step)
        upper = float(fit_function.Eval(x))
        fit_function.SetParameter(i, value - step)
        lower = float(fit_function.Eval(x))
        fit_function.SetParameter(i, value)
        gradient.append((upper - lower) / (2.0 * step))

    variance = 0.0
    for i in range(n_parameters):
        for j in range(n_parameters):
            variance += (
                gradient[i]
                * float(fit_result.CovMatrix(i, j))
                * gradient[j]
            )
    return math.sqrt(max(0.0, variance))


def positive_hist_min(hist, xmin: float, xmax: float) -> float:
    best = math.inf
    for ibin in range(1, hist.GetNbinsX() + 1):
        x = float(hist.GetBinCenter(ibin))
        if x < xmin or x > xmax:
            continue
        y = float(hist.GetBinContent(ibin))
        if y > 0.0 and math.isfinite(y):
            best = min(best, y)
    return best if math.isfinite(best) else 1.0


def lumi_label(era: str) -> str:
    energy = "13.6 TeV" if era in RUN3_ERAS else "13 TeV"
    lumi = LUMI_FB.get(era)
    if lumi is None:
        return f"{era}, {energy}"
    return f"{lumi:g} fb^{{-1}} ({energy})"


def style_hist(ROOT, hist) -> None:
    hist.SetStats(0)
    hist.SetMarkerStyle(20)
    hist.SetMarkerSize(0.8)
    hist.SetMarkerColor(ROOT.kBlack)
    hist.SetLineColor(ROOT.kBlack)
    hist.SetLineWidth(1)


def draw_mass_fit(
    ROOT,
    *,
    hist,
    fit_function,
    fit_result,
    era: str,
    mass: float,
    fit_min: float,
    fit_max: float,
    plot_min: float,
    plot_max: float,
    chi2: float,
    ndf: int,
    output_dir: Path,
    extensions: Sequence[str],
    log_y: bool,
    fit_function_name: str = "double-crystal-ball",
) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    label = mass_label(mass)

    canvas = ROOT.TCanvas(NAMES.unique("c_sigfit"), "", 950, 950)
    upper = ROOT.TPad(NAMES.unique("upper"), "", 0.0, 0.30, 1.0, 1.0)
    lower = ROOT.TPad(NAMES.unique("lower"), "", 0.0, 0.00, 1.0, 0.30)

    upper.SetLeftMargin(0.13)
    upper.SetRightMargin(0.04)
    upper.SetTopMargin(0.08)
    upper.SetBottomMargin(0.03)
    lower.SetLeftMargin(0.13)
    lower.SetRightMargin(0.04)
    lower.SetTopMargin(0.04)
    lower.SetBottomMargin(0.34)

    if log_y:
        upper.SetLogy(True)

    upper.Draw()
    lower.Draw()

    upper.cd()
    style_hist(ROOT, hist)
    fit_function.SetLineColor(ROOT.kRed + 1)
    fit_function.SetLineWidth(2)
    fit_function.SetNpx(1000)

    hist.GetXaxis().SetRangeUser(plot_min, plot_max)
    hist.GetXaxis().SetLabelSize(0.0)
    hist.GetXaxis().SetTitleSize(0.0)
    hist.GetYaxis().SetTitle("Events")
    hist.GetYaxis().SetTitleSize(0.055)
    hist.GetYaxis().SetTitleOffset(1.05)
    hist.GetYaxis().SetLabelSize(0.045)

    ymax = max(float(hist.GetMaximum()), 1.0)
    if log_y:
        ymin = max(1.0e-4, positive_hist_min(hist, plot_min, plot_max) * 0.25)
        hist.SetMinimum(ymin)
        hist.SetMaximum(ymax * 20.0)
    else:
        hist.SetMinimum(0.0)
        hist.SetMaximum(ymax * 1.55)

    hist.Draw("E1")
    fit_function.Draw("L SAME")
    hist.Draw("E1 SAME")

    legend = ROOT.TLegend(0.62, 0.72, 0.91, 0.86)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.034)
    legend.AddEntry(hist, "Signal MC", "pe")
    legend.AddEntry(
        fit_function,
        f"{fit_function_label(fit_function_name)} fit",
        "l",
    )
    legend.Draw()

    latex = ROOT.TLatex()
    latex.SetNDC(True)
    latex.SetTextFont(42)

    latex.SetTextAlign(13)
    latex.SetTextSize(0.043)
    latex.DrawLatex(0.14, 0.965, "#bf{CMS} #it{Simulation Preliminary}")

    latex.SetTextAlign(33)
    latex.SetTextSize(0.037)
    latex.DrawLatex(0.96, 0.965, lumi_label(era))

    amplitude = float(fit_function.GetParameter(0))
    mean = float(fit_function.GetParameter(1))
    sigma = float(fit_function.GetParameter(2))
    sigma_error = float(fit_function.GetParError(2))

    latex.SetTextAlign(13)
    latex.SetTextSize(0.032)
    latex.DrawLatex(0.16, 0.84, f"m_{{Z'}} = {mass:g} GeV")
    latex.DrawLatex(0.16, 0.79, f"#mu = {mean:.4g} GeV")
    latex.DrawLatex(0.16, 0.74, f"#sigma = {sigma:.4g} #pm {sigma_error:.2g} GeV")
    latex.DrawLatex(
        0.16,
        0.69,
        f"#sigma/m_{{Z'}} = {sigma / mass:.4g}",
    )
    latex.DrawLatex(
        0.16,
        0.64,
        f"#chi^{{2}}/ndf = {chi2 / ndf:.3g}" if ndf > 0 else "#chi^{2}/ndf = n/a",
    )
    latex.DrawLatex(
        0.16,
        0.59,
        f"fit: [{fit_min:.3g}, {fit_max:.3g}] GeV",
    )

    # Keep the amplitude available in the plot only through the fit curve.  It is
    # saved numerically in the CSV/JSON output.
    _ = amplitude

    upper.RedrawAxis()

    lower.cd()

    frame = hist.Clone(NAMES.unique("ratio_frame"))
    frame.Reset("ICES")
    frame.SetStats(0)
    frame.GetXaxis().SetRangeUser(plot_min, plot_max)
    frame.GetXaxis().SetTitle("m_{#mu#mu} [GeV]")
    frame.GetXaxis().SetTitleSize(0.12)
    frame.GetXaxis().SetTitleOffset(1.05)
    frame.GetXaxis().SetLabelSize(0.095)
    frame.GetYaxis().SetTitle("MC / fit")
    frame.GetYaxis().SetTitleSize(0.105)
    frame.GetYaxis().SetTitleOffset(0.48)
    frame.GetYaxis().SetLabelSize(0.085)
    frame.GetYaxis().SetNdivisions(505)
    frame.SetMinimum(0.0)
    frame.SetMaximum(2.0)
    frame.Draw("AXIS")

    x_ratio = array("d")
    y_ratio = array("d")
    ex_ratio = array("d")
    ey_ratio = array("d")
    x_band = array("d")
    y_band = array("d")
    ex_band = array("d")
    ey_band = array("d")

    for ibin in range(1, hist.GetNbinsX() + 1):
        x = float(hist.GetBinCenter(ibin))
        if x < fit_min or x > fit_max:
            continue

        ex = 0.5 * float(hist.GetBinWidth(ibin))
        y = float(hist.GetBinContent(ibin))
        ey = float(hist.GetBinError(ibin))
        f = float(fit_function.Eval(x))
        if f <= 0.0 or not math.isfinite(f):
            continue

        fit_unc = fit_uncertainty(
            ROOT,
            fit_function,
            fit_result,
            x,
            fit_function_name,
        )
        x_band.append(x)
        y_band.append(1.0)
        ex_band.append(ex)
        ey_band.append(fit_unc / f)

        if y <= 0.0:
            continue
        x_ratio.append(x)
        y_ratio.append(y / f)
        ex_ratio.append(ex)
        ey_ratio.append(ey / f)

    fit_band = ROOT.TGraphErrors(
        len(x_band),
        x_band,
        y_band,
        ex_band,
        ey_band,
    )
    fit_band.SetFillColor(ROOT.kGray)
    fit_band.SetFillStyle(3004)
    fit_band.SetLineColor(ROOT.kGray)
    fit_band.Draw("E2 SAME")

    ratio = ROOT.TGraphErrors(
        len(x_ratio),
        x_ratio,
        y_ratio,
        ex_ratio,
        ey_ratio,
    )
    ratio.SetMarkerStyle(20)
    ratio.SetMarkerSize(0.8)
    ratio.SetMarkerColor(ROOT.kBlack)
    ratio.SetLineColor(ROOT.kBlack)
    ratio.Draw("PE SAME")

    unity = ROOT.TLine(plot_min, 1.0, plot_max, 1.0)
    unity.SetLineStyle(2)
    unity.SetLineWidth(2)
    unity.SetLineColor(ROOT.kBlack)
    unity.Draw("SAME")
    frame.Draw("AXIS SAME")

    canvas._keepalive = [
        upper,
        lower,
        legend,
        latex,
        frame,
        fit_band,
        ratio,
        unity,
    ]

    outputs: List[str] = []
    base = output_dir / f"sigFit_{era}_M-{label}"
    for ext in extensions:
        path = f"{base}.{ext}"
        canvas.SaveAs(path)
        outputs.append(path)
        print(f"[SAVE] {path}")

    return outputs


def fit_one_mass(
    ROOT,
    *,
    era: str,
    mass: float,
    filename: str,
    region: str,
    hist_name: str,
    output_dir: Path,
    extensions: Sequence[str],
    fit_fraction: float,
    plot_fraction: float,
    min_fit_half_width: float,
    rebin_fraction: float,
    min_rebin_width: float,
    log_y: bool,
    fit_function_name: str = "double-crystal-ball",
) -> FitResult:
    path = hist_path(region, hist_name)
    hist = open_histogram(ROOT, filename, path)

    original_bin_width = float(hist.GetXaxis().GetBinWidth(1))
    rebin_factor = get_rebin_factor(
        hist,
        mass,
        rebin_fraction,
        min_rebin_width,
    )

    if rebin_factor > 1:
        rebinned = hist.Rebin(
            rebin_factor,
            NAMES.unique(f"sig_mass_{era}_{mass_label(mass)}"),
        )
        rebinned.SetDirectory(0)
        if rebinned.GetSumw2N() == 0:
            rebinned.Sumw2()
        hist = rebinned

    rebinned_bin_width = float(hist.GetXaxis().GetBinWidth(1))

    minimum_fit_fraction = (
        MIN_DCB_FIT_FRACTION
        if fit_function_name == "double-crystal-ball"
        else 0.0
    )
    fit_half_width = max(
        fit_fraction * mass,
        minimum_fit_fraction * mass,
        min_fit_half_width,
    )
    fit_min = mass - fit_half_width
    fit_max = mass + fit_half_width
    plot_half_width = plot_fraction * mass
    plot_min = mass - plot_half_width
    plot_max = mass + plot_half_width

    peak_y = float(hist.GetBinContent(hist.GetMaximumBin()))
    if peak_y <= 0.0:
        peak_y = max(1.0, float(hist.GetMaximum()))

    sigma_init = max(0.01 * mass, 0.02)

    fit_function = create_signal_fit_function(
        ROOT,
        fit_function_name=fit_function_name,
        era=era,
        mass=mass,
        fit_min=fit_min,
        fit_max=fit_max,
        fit_half_width=fit_half_width,
        peak_y=peak_y,
        sigma_init=sigma_init,
    )

    fit_result_ptr = hist.Fit(
        fit_function,
        "SRQ0",
        "",
        fit_min,
        fit_max,
    )
    fit_status = int(fit_result_ptr)
    fit_result = fit_result_ptr.Get()
    covariance_status = (
        int(fit_result.CovMatrixStatus()) if fit_result else -1
    )

    chi2, ndf = compute_chi2(hist, fit_function, fit_min, fit_max)
    chi2_ndf = chi2 / ndf if ndf > 0 else None

    amplitude = float(fit_function.GetParameter(0))
    amplitude_error = float(fit_function.GetParError(0))
    mean = float(fit_function.GetParameter(1))
    mean_error = float(fit_function.GetParError(1))
    sigma = abs(float(fit_function.GetParameter(2)))
    sigma_error = float(fit_function.GetParError(2))

    print(
        f"[FIT] {era:12s} M={mass:g} GeV "
        f"status={fit_status} cov={covariance_status} "
        f"mean={mean:.6g} sigma={sigma:.6g} "
        f"sigma/m={sigma / mass:.6g} "
        + (f"chi2/ndf={chi2_ndf:.4g}" if chi2_ndf is not None else "chi2/ndf=n/a")
    )

    plot_files = draw_mass_fit(
        ROOT,
        hist=hist,
        fit_function=fit_function,
        fit_result=fit_result,
        era=era,
        mass=mass,
        fit_min=fit_min,
        fit_max=fit_max,
        plot_min=plot_min,
        plot_max=plot_max,
        chi2=chi2,
        ndf=ndf,
        output_dir=output_dir,
        extensions=extensions,
        log_y=log_y,
        fit_function_name=fit_function_name,
    )

    return FitResult(
        era=era,
        mass=mass,
        file=filename,
        hist_path=path,
        fit_min=fit_min,
        fit_max=fit_max,
        plot_min=plot_min,
        plot_max=plot_max,
        original_bin_width=original_bin_width,
        rebin_factor=rebin_factor,
        rebinned_bin_width=rebinned_bin_width,
        fit_status=fit_status,
        covariance_status=covariance_status,
        amplitude=amplitude,
        amplitude_error=amplitude_error,
        mean=mean,
        mean_error=mean_error,
        sigma=sigma,
        sigma_error=sigma_error,
        sigma_over_mass=sigma / mass,
        sigma_over_mass_error=sigma_error / mass,
        chi2=chi2,
        ndf=ndf,
        chi2_ndf=chi2_ndf,
        plot_files=plot_files,
    )


def draw_resolution_fit(
    ROOT,
    *,
    era: str,
    results: Sequence[FitResult],
    output_dir: Path,
    extensions: Sequence[str],
) -> ResolutionResult:
    ordered = sorted(results, key=lambda x: x.mass)
    masses = array("d", [r.mass for r in ordered])
    values = array("d", [r.sigma_over_mass for r in ordered])
    xerrors = array("d", [0.0 for _ in ordered])
    yerrors = array("d", [r.sigma_over_mass_error for r in ordered])

    graph = ROOT.TGraphErrors(
        len(ordered),
        masses,
        values,
        xerrors,
        yerrors,
    )
    graph.SetName(NAMES.unique(f"resolution_{era}"))
    graph.SetTitle("")
    graph.SetMarkerStyle(20)
    graph.SetMarkerSize(1.0)
    graph.SetMarkerColor(ROOT.kBlack)
    graph.SetLineColor(ROOT.kBlack)

    xmin = min(r.mass for r in ordered)
    xmax = max(r.mass for r in ordered)

    linear = ROOT.TF1(
        NAMES.unique(f"resolution_pol1_{era}"),
        "pol1",
        xmin,
        xmax,
    )
    linear.SetLineColor(ROOT.kRed + 1)
    linear.SetLineWidth(2)

    fit_result_ptr = graph.Fit(linear, "SRQ0")
    fit_result = fit_result_ptr.Get()
    fit_status = int(fit_result_ptr)

    intercept = float(linear.GetParameter(0))
    intercept_error = float(linear.GetParError(0))
    slope = float(linear.GetParameter(1))
    slope_error = float(linear.GetParError(1))
    chi2 = float(linear.GetChisquare())
    ndf = int(linear.GetNDF())
    chi2_ndf = chi2 / ndf if ndf > 0 else None

    if fit_status != 0:
        print(f"[WARNING] Resolution fit status for {era}: {fit_status}")

    canvas = ROOT.TCanvas(NAMES.unique("c_resolution"), "", 850, 700)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.04)
    canvas.SetTopMargin(0.08)
    canvas.SetBottomMargin(0.13)

    graph.Draw("AP")
    graph.GetXaxis().SetTitle("m_{Z'} [GeV]")
    graph.GetYaxis().SetTitle("#sigma / m_{Z'}")
    graph.GetXaxis().SetTitleSize(0.050)
    graph.GetYaxis().SetTitleSize(0.050)
    graph.GetXaxis().SetLabelSize(0.043)
    graph.GetYaxis().SetLabelSize(0.043)
    graph.GetYaxis().SetTitleOffset(1.15)
    linear.Draw("L SAME")

    latex = ROOT.TLatex()
    latex.SetNDC(True)
    latex.SetTextFont(42)

    latex.SetTextAlign(13)
    latex.SetTextSize(0.043)
    latex.DrawLatex(0.14, 0.965, "#bf{CMS} #it{Simulation Preliminary}")

    latex.SetTextAlign(33)
    latex.SetTextSize(0.037)
    latex.DrawLatex(0.96, 0.965, lumi_label(era))

    latex.SetTextAlign(13)
    latex.SetTextSize(0.035)
    latex.DrawLatex(
        0.17,
        0.86,
        f"#sigma/m_{{Z'}} = ({slope:.4g}) m_{{Z'}} + ({intercept:.4g})",
    )
    latex.DrawLatex(
        0.17,
        0.80,
        f"#chi^{{2}}/ndf = {chi2_ndf:.3g}" if chi2_ndf is not None else "#chi^{2}/ndf = n/a",
    )

    canvas._keepalive = [graph, linear, latex]

    outputs: List[str] = []
    base = output_dir / f"sigFit_resolution_{era}"
    for ext in extensions:
        path = f"{base}.{ext}"
        canvas.SaveAs(path)
        outputs.append(path)
        print(f"[SAVE] {path}")

    print(
        f"[RESOLUTION] {era}: sigma/m = "
        f"({slope:.8g})*m + ({intercept:.8g})"
    )

    return ResolutionResult(
        era=era,
        n_points=len(ordered),
        mass_min=xmin,
        mass_max=xmax,
        intercept_b=intercept,
        intercept_b_error=intercept_error,
        slope_a=slope,
        slope_a_error=slope_error,
        chi2=chi2,
        ndf=ndf,
        chi2_ndf=chi2_ndf,
        plot_files=outputs,
    )


def write_fit_outputs(output_dir: Path, results: Sequence[FitResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "sigFit_results.json"
    json_path.write_text(
        json.dumps([asdict(r) for r in results], indent=2, sort_keys=True) + "\n"
    )

    csv_path = output_dir / "sigFit_results.csv"
    fieldnames = [
        "era",
        "mass",
        "file",
        "hist_path",
        "fit_min",
        "fit_max",
        "plot_min",
        "plot_max",
        "original_bin_width",
        "rebin_factor",
        "rebinned_bin_width",
        "fit_status",
        "covariance_status",
        "amplitude",
        "amplitude_error",
        "mean",
        "mean_error",
        "sigma",
        "sigma_error",
        "sigma_over_mass",
        "sigma_over_mass_error",
        "chi2",
        "ndf",
        "chi2_ndf",
        "plot_files",
    ]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["plot_files"] = ";".join(result.plot_files)
            writer.writerow(row)

    print(f"[SAVE] {json_path}")
    print(f"[SAVE] {csv_path}")


def write_resolution_outputs(
    output_dir: Path,
    results: Sequence[ResolutionResult],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "resolution_coefficients.json"
    payload = {
        r.era: {
            "a": r.slope_a,
            "a_error": r.slope_a_error,
            "b": r.intercept_b,
            "b_error": r.intercept_b_error,
            "formula": "sigma_m / m = a*m + b",
            "n_points": r.n_points,
            "mass_min": r.mass_min,
            "mass_max": r.mass_max,
            "chi2": r.chi2,
            "ndf": r.ndf,
            "chi2_ndf": r.chi2_ndf,
            "plot_files": r.plot_files,
        }
        for r in results
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    csv_path = output_dir / "resolution_coefficients.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "era",
                "a_slope",
                "a_error",
                "b_intercept",
                "b_error",
                "n_points",
                "mass_min",
                "mass_max",
                "chi2",
                "ndf",
                "chi2_ndf",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.era,
                    r.slope_a,
                    r.slope_a_error,
                    r.intercept_b,
                    r.intercept_b_error,
                    r.n_points,
                    r.mass_min,
                    r.mass_max,
                    r.chi2,
                    r.ndf,
                    r.chi2_ndf,
                ]
            )

    py_path = output_dir / "resolution_coefficients.py"
    lines = [
        "# Auto-generated by sigFit.py",
        "# Convention: sigma_m / m = a*m + b",
        "",
        "RESOLUTION = {",
    ]
    for r in results:
        lines.append(
            f'    "{r.era}": ({r.slope_a:.12g}, {r.intercept_b:.12g}),'
        )
    lines.extend(["}", ""])
    py_path.write_text("\n".join(lines))

    print(f"[SAVE] {json_path}")
    print(f"[SAVE] {csv_path}")
    print(f"[SAVE] {py_path}")


def validate_args(args: argparse.Namespace) -> None:
    if args.fit_fraction <= 0.0:
        raise ValueError("--fit-fraction must be positive.")
    if args.plot_fraction <= 0.0:
        raise ValueError("--plot-fraction must be positive.")
    if args.min_fit_half_width <= 0.0:
        raise ValueError("--min-fit-half-width must be positive.")
    if args.rebin_fraction < 0.0:
        raise ValueError("--rebin-fraction must be non-negative.")
    if args.min_rebin_width <= 0.0:
        raise ValueError("--min-rebin-width must be positive.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)

    extensions = [
        token.strip().lstrip(".")
        for token in args.extensions.split(",")
        if token.strip()
    ]
    if not extensions:
        extensions = ["pdf"]

    requested_masses = parse_masses(args.masses)
    eras = selected_eras(args.era)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ROOT = import_root()

    fit_results: List[FitResult] = []
    resolution_results: List[ResolutionResult] = []

    for era in eras:
        discovered = discover_signal_files(args.base_dir, era)
        if not discovered:
            message = (
                f"No nominal signal files were found for {era} under "
                f"{Path(args.base_dir) / era}"
            )
            if args.strict:
                raise FileNotFoundError(message)
            print(f"[WARNING] {message}")
            continue

        masses = (
            sorted(discovered)
            if requested_masses is None
            else list(requested_masses)
        )

        era_results: List[FitResult] = []
        for mass in masses:
            filename = find_requested_file(discovered, mass)
            if not filename:
                message = f"Missing nominal signal file: era={era}, M={mass:g} GeV"
                if args.strict:
                    raise FileNotFoundError(message)
                print(f"[WARNING] {message}")
                continue

            try:
                result = fit_one_mass(
                    ROOT,
                    era=era,
                    mass=mass,
                    filename=filename,
                    region=args.region,
                    hist_name=args.hist_name,
                    output_dir=output_dir,
                    extensions=extensions,
                    fit_fraction=args.fit_fraction,
                    plot_fraction=args.plot_fraction,
                    min_fit_half_width=args.min_fit_half_width,
                    rebin_fraction=args.rebin_fraction,
                    min_rebin_width=args.min_rebin_width,
                    log_y=not args.linear_y,
                    fit_function_name=args.fit_function,
                )
            except (OSError, KeyError, RuntimeError, ValueError) as exc:
                if args.strict:
                    raise
                print(f"[WARNING] {era}, M={mass:g}: {exc}")
                continue

            if result.fit_status != 0:
                message = (
                    f"{fit_function_label(args.fit_function)} fit did not "
                    f"converge cleanly: "
                    f"era={era}, M={mass:g}, status={result.fit_status}"
                )
                if args.strict:
                    raise RuntimeError(message)
                print(f"[WARNING] {message}")

            if result.covariance_status < 2:
                message = (
                    f"{fit_function_label(args.fit_function)} fit covariance is "
                    f"weak: era={era}, M={mass:g}, "
                    f"covariance_status={result.covariance_status}"
                )
                if args.strict:
                    raise RuntimeError(message)
                print(f"[WARNING] {message}")

            era_results.append(result)
            fit_results.append(result)

        if args.no_resolution_fit:
            continue

        good_for_resolution = [
            r
            for r in era_results
            if r.fit_status == 0
            and r.covariance_status >= 2
            and r.sigma > 0.0
            and r.sigma_error >= 0.0
            and math.isfinite(r.sigma_over_mass)
        ]

        if len(good_for_resolution) < 2:
            message = (
                f"Need at least two valid {fit_function_label(args.fit_function)} "
                f"fits for the resolution fit in {era}; "
                f"found {len(good_for_resolution)}."
            )
            if args.strict:
                raise RuntimeError(message)
            print(f"[WARNING] {message}")
            continue

        resolution_results.append(
            draw_resolution_fit(
                ROOT,
                era=era,
                results=good_for_resolution,
                output_dir=output_dir,
                extensions=extensions,
            )
        )

    if not fit_results:
        print("[ERROR] No signal fits were produced.", file=sys.stderr)
        return 2

    write_fit_outputs(output_dir, fit_results)
    if resolution_results:
        write_resolution_outputs(output_dir, resolution_results)

    print("\n[DONE]")
    print(
        f"  {fit_function_label(args.fit_function)} fits: "
        f"{len(fit_results)}"
    )
    print(f"  Resolution fits: {len(resolution_results)}")
    print(f"  Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
