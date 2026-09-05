#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
OS-vs-SS dimuon comparison for the current NIsoMuon Run-2/Run-3 SKOutput.

PyROOT replacement for the historical OS_SS_ComparisonPlot.cc:
  * OS: black data points
  * SS: red line with a statistical uncertainty band
  * lower panel: OS / SS
  * optional --apply-scale rescales SS by the OS/SS integral ratio
  * optional --overlay-lightjet adds LightJet OS/SS and its OS/SS ratio
  * with --overlay-lightjet, primary and LightJet SS are scaled independently
  * default mass binning is identical to dy_bkg_estimation.py
  * OS and SS jet categories can be selected independently

Current input layout:
  /data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13/NIsoMuon/<era>/

The script supports three comparison modes:
  * raw-data: raw OS and SS data, matching the historical C++ macro
  * subtract-nonqcd: Data - Top - DY - Others
  * qcd-mc: OS and SS QCD MC only from NIsoMuon_QCD_Inclusive.root

Available categories:
  * bjet:       standard BJet category (at least one independent Medium b-tagged jet)
  * lightjet:   standard LightJet category
  * onebjet:    test category with exactly one Medium b-tagged analysis jet
  * inclusive:  test category with no b-tag requirement or veto

Examples:
  python3 os_ss_comparison.py --era Run2 --mode raw-data
  python3 os_ss_comparison.py --era Run2 --mode subtract-nonqcd
  python3 os_ss_comparison.py --era Run2 --mode qcd-mc
  python3 os_ss_comparison.py --era Run2 --mode qcd-mc --apply-scale
  python3 os_ss_comparison.py --era Run2 --os-category bjet --ss-category onebjet
  python3 os_ss_comparison.py --era Run2 --os-category bjet --ss-category inclusive
  python3 os_ss_comparison.py --era Run2 --mode raw-data --overlay-lightjet
  python3 os_ss_comparison.py --era Run2 --mode qcd-mc \
      --no-variable-binning --bin-width 1

The historical --jet-flavour/--jet-flavor option is retained as a shorthand
for setting both OS and SS categories when --os-category/--ss-category are not
specified.
'''

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from array import array
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

RUN2_ERAS: Tuple[str, ...] = ("2016preVFP", "2016postVFP", "2017", "2018")
RUN3_ERAS: Tuple[str, ...] = ("2022", "2022EE", "2023", "2023BPix")
YEARS: Tuple[str, ...] = RUN2_ERAS + RUN3_ERAS

ERA_GROUPS: Dict[str, Tuple[str, ...]] = {
    **{era: (era,) for era in YEARS},
    "Run2": RUN2_ERAS,
    "Run3": RUN3_ERAS,
    "Run2+3": YEARS,
    "full": YEARS,
}

DEFAULT_BASE_DIR = "/data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13/NIsoMuon"
DEFAULT_OUTPUT_DIR = "/data6/Users/joonblee/PlotMaker/plots"

LUMI_FB: Dict[str, float] = {
    "2016preVFP": 19.52,
    "2016postVFP": 16.81,
    "2017": 41.48,
    "2018": 59.83,
    "2022": 7.9804,
    "2022EE": 26.6717,
    "2023": 18.064,
    "2023BPix": 9.693,
}
RUN2_LUMI_LABEL_FB = 138
RUN3_LUMI_LABEL_FB = 62

DEFAULT_MUON_ID = "POGMedium"
DEFAULT_JET_ID = "tight"
DEFAULT_JET_FLAVOUR = "bjet"
DEFAULT_HIST_NAME = "Dilepton_Mass"

DEFAULT_NORM_MIN = 6.0
DEFAULT_NORM_MAX = 9.0

# Keep the default mass binning identical to dy_bkg_estimation.py.
DEFAULT_MASS_BINS = [
    0.0, 0.5, 1.0, 1.5, 2.0, 2.5,
    3.0, 3.5, 4.0, 4.5,
    5.0, 6.0, 7.0, 8.0, 9.0,
    10.0, 12.0, 14.0, 16.0, 18.0,
    20.0, 25.0, 30.0, 40.0, 60.0, 80.0,
    85.0, 90.0, 95.0, 100.0, 105.0, 110.0,
    120.0, 130.0, 150.0,
]


class PlotError(RuntimeError):
    pass


def import_root():
    try:
        import ROOT  # type: ignore
    except Exception as exc:
        raise PlotError(
            "Could not import PyROOT. Run inside ROOT/CMSSW. "
            f"Original error: {exc}"
        )
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)
    try:
        ROOT.TH1.AddDirectory(False)
    except Exception:
        pass
    return ROOT


def canonical_era(value: str) -> str:
    key = value.strip().lower().replace(" ", "")
    aliases = {
        "run2": "Run2",
        "run3": "Run3",
        "run2+3": "Run2+3",
        "run23": "Run2+3",
        "full": "full",
    }
    return aliases.get(key, value)


def years_for_era(value: str) -> Tuple[str, ...]:
    era = canonical_era(value)
    try:
        return ERA_GROUPS[era]
    except KeyError as exc:
        raise PlotError(
            f"Unknown era {value!r}. Use one of: {', '.join(ERA_GROUPS)}"
        ) from exc


def canonical_jet_category(value: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", value.lower())

    if key in {"b", "bjet", "btag", "btagged", "standardbjet"}:
        return "bjet"
    if key in {"light", "lightjet", "lj"}:
        return "lightjet"
    if key in {
        "oneb",
        "onebjet",
        "1b",
        "1bjet",
        "exactlyoneb",
        "exactlyonebjet",
        "singleb",
        "singlebjet",
    }:
        return "onebjet"
    if key in {
        "inclusive",
        "inclusivejet",
        "incl",
        "incljet",
        "alljet",
        "alljets",
    }:
        return "inclusive"

    raise PlotError(
        f"Unknown jet category: {value}. "
        "Use bjet, lightjet, onebjet, or inclusive."
    )


def category_root_tag(category: str) -> str:
    category = canonical_jet_category(category)
    return {
        "bjet": "BJet",
        "lightjet": "LightJet",
        "onebjet": "OneBJet",
        "inclusive": "InclusiveJet",
    }[category]


def category_label(category: str) -> str:
    category = canonical_jet_category(category)
    return {
        "bjet": "b-jet category",
        "lightjet": "light-jet category",
        "onebjet": "exactly-one-b-jet category",
        "inclusive": "inclusive-jet category",
    }[category]


def region(sign: str, muon_id: str, jet_id: str, category: str) -> str:
    return (
        f"{sign}_{muon_id}_{jet_id}_"
        f"{category_root_tag(category)}_NIsoDimuon"
    )


def hist_path(reg: str, hist_name: str) -> str:
    return f"{reg}/{hist_name}___{reg}"


def era_dir(args, year: str) -> str:
    parts = [args.base_dir, year]
    if args.trigger:
        parts.append(args.trigger)
    return os.path.join(*parts)


def open_hist(ROOT, filename: str, path: str, clone_name: str):
    if not os.path.isfile(filename):
        raise PlotError(f"Missing ROOT file: {filename}")
    f = ROOT.TFile.Open(filename, "READ")
    if not f or f.IsZombie():
        if f:
            f.Close()
        raise PlotError(f"Could not open ROOT file: {filename}")
    h = f.Get(path)
    if not h:
        f.Close()
        raise PlotError(f"Missing histogram: {filename}:{path}")
    out = h.Clone(clone_name)
    out.SetDirectory(0)
    out.Sumw2()
    f.Close()
    return out


def sum_hists(hists: Sequence[object], name: str):
    if not hists:
        return None
    out = hists[0].Clone(name)
    out.SetDirectory(0)
    out.Sumw2()
    for h in hists[1:]:
        out.Add(h)
    return out


def load_across_eras(
    ROOT,
    args,
    relative_file: str,
    reg: str,
    hist_name: str,
    prefix: str,
):
    pieces = []
    path = hist_path(reg, hist_name)
    for year in years_for_era(args.era):
        filename = os.path.join(era_dir(args, year), relative_file)
        pieces.append(
            open_hist(ROOT, filename, path, f"{prefix}_{year}_{reg}")
        )
    return sum_hists(pieces, f"{prefix}_{canonical_era(args.era)}_{reg}")


def load_data(ROOT, args, sign: str, category: str):
    reg = region(sign, args.muon_id, args.jet_id, category)
    return load_across_eras(
        ROOT, args, "data.root", reg, args.hist_name, f"data_{sign}"
    )


def load_qcd_mc(ROOT, args, sign: str, category: str):
    reg = region(sign, args.muon_id, args.jet_id, category)
    return load_across_eras(
        ROOT,
        args,
        "NIsoMuon_QCD_Inclusive.root",
        reg,
        args.hist_name,
        f"qcdmc_{sign}",
    )


def load_qcd_enriched_data(ROOT, args, sign: str, category: str):
    # Data - Top - DY - Others, matching the QCD normalization ingredients.
    reg = region(sign, args.muon_id, args.jet_id, category)
    out = load_across_eras(
        ROOT, args, "data.root", reg, args.hist_name, f"dataSub_{sign}"
    )

    for bg_file in ("NIsoMuon_Top.root", "NIsoMuon_Others.root"):
        h = load_across_eras(
            ROOT,
            args,
            bg_file,
            reg,
            args.hist_name,
            f"subtract_{sign}_{bg_file}",
        )
        out.Add(h, -1.0)

    # The final data-driven DY prediction exists for the nominal OS BJet
    # category. Other categories use DY MC, including the new SS test regions.
    if (
        sign == "OS"
        and args.dy_method == "data-driven"
        and canonical_jet_category(category) == "bjet"
    ):
        dy_file = "NIsoMuon_DYJets_est.root"
    else:
        dy_file = "NIsoMuon_DYJets_Inclusive.root"

    h_dy = load_across_eras(
        ROOT, args, dy_file, reg, args.hist_name, f"subtract_{sign}_DY"
    )
    out.Add(h_dy, -1.0)
    return out


def make_variable_edges(
    xmin: float,
    xmax: float,
    requested_edges: Sequence[float] = DEFAULT_MASS_BINS,
) -> List[float]:
    if xmax <= xmin:
        raise PlotError("--xmax must be larger than --xmin")

    edges = [float(x) for x in requested_edges if xmin <= float(x) <= xmax]
    if not edges or not math.isclose(edges[0], xmin, rel_tol=0.0, abs_tol=1.0e-9):
        edges.insert(0, float(xmin))
    if not math.isclose(edges[-1], xmax, rel_tol=0.0, abs_tol=1.0e-9):
        edges.append(float(xmax))

    unique = []
    for value in edges:
        if not unique or not math.isclose(
            value, unique[-1], rel_tol=0.0, abs_tol=1.0e-9
        ):
            unique.append(value)

    if len(unique) < 2:
        raise PlotError("Could not construct variable mass binning")
    return unique


def make_uniform_edges(xmin: float, xmax: float, width: float) -> List[float]:
    if width <= 0:
        raise PlotError("--bin-width must be positive")
    if xmax <= xmin:
        raise PlotError("--xmax must be larger than --xmin")

    n = int(round((xmax - xmin) / width))
    if n <= 0 or not math.isclose(
        xmin + n * width, xmax, rel_tol=0.0, abs_tol=1.0e-8
    ):
        raise PlotError(
            f"[{xmin:g},{xmax:g}] is not an integer number of {width:g}-GeV bins"
        )
    return [xmin + i * width for i in range(n + 1)]


def make_edges(
    xmin: float,
    xmax: float,
    width: float,
    variable_binning: bool,
) -> List[float]:
    if variable_binning:
        return make_variable_edges(xmin, xmax)
    return make_uniform_edges(xmin, xmax, width)


def rebin_hist(hist, edges: Sequence[float], name: str):
    arr = array("d", [float(x) for x in edges])
    out = hist.Rebin(len(edges) - 1, name, arr)
    out.SetDirectory(0)
    out.Sumw2()
    return out


def integral_open(hist, xmin: float, xmax: float) -> Tuple[float, float]:
    value = 0.0
    variance = 0.0
    for ib in range(1, hist.GetNbinsX() + 1):
        x = float(hist.GetXaxis().GetBinCenter(ib))
        if not (xmin < x < xmax):
            continue
        value += float(hist.GetBinContent(ib))
        error = float(hist.GetBinError(ib))
        variance += error * error
    return value, math.sqrt(max(0.0, variance))


def divide_by_bin_width(hist) -> None:
    for ib in range(1, hist.GetNbinsX() + 1):
        width = float(hist.GetXaxis().GetBinWidth(ib))
        if width <= 0.0:
            continue
        hist.SetBinContent(ib, float(hist.GetBinContent(ib)) / width)
        hist.SetBinError(ib, float(hist.GetBinError(ib)) / width)


def cms_lumi_label(era: str) -> str:
    era = canonical_era(era)
    if era in {"Run2+3", "full"}:
        return (
            f"{RUN2_LUMI_LABEL_FB} fb^{{-1}} (13 TeV) + "
            f"{RUN3_LUMI_LABEL_FB} fb^{{-1}} (13.6 TeV)"
        )
    if era == "Run2":
        return f"{RUN2_LUMI_LABEL_FB} fb^{{-1}} (13 TeV)"
    if era == "Run3":
        return f"{RUN3_LUMI_LABEL_FB} fb^{{-1}} (13.6 TeV)"
    energy = "13.6 TeV" if era in RUN3_ERAS else "13 TeV"
    return f"{LUMI_FB[era]:.1f} fb^{{-1}} ({energy})"


def draw_cms_header(ROOT, pad, args, extra_lines: Sequence[str]) -> List[object]:
    keep = []
    pad.cd()

    latex = ROOT.TLatex()
    latex.SetNDC(True)
    latex.SetTextFont(42)
    latex.SetTextColor(ROOT.kBlack)

    latex.SetTextAlign(11)
    latex.SetTextSize(0.047)
    latex.DrawLatex(0.120, 0.925, "#bf{CMS} #it{Preliminary}")

    latex.SetTextAlign(31)
    latex.SetTextSize(0.038)
    latex.DrawLatex(0.950, 0.925, cms_lumi_label(args.era))

    latex.SetTextAlign(11)
    latex.SetTextSize(0.033)
    latex.DrawLatex(0.155, 0.855, "Opposite-sign vs same-sign")

    y = 0.810
    latex.SetTextSize(0.029)
    latex.SetTextColor(ROOT.kGray + 2)
    for line in extra_lines:
        latex.DrawLatex(0.155, y, line)
        y -= 0.038

    keep.append(latex)
    return keep


def positive_minimum(*hists) -> float:
    values = []
    for h in hists:
        for ib in range(1, h.GetNbinsX() + 1):
            value = float(h.GetBinContent(ib))
            if value > 0.0:
                values.append(value)
    return min(values) if values else 1.0


def resolve_categories(args) -> Tuple[str, str]:
    legacy = canonical_jet_category(args.jet_flavour)
    os_category = canonical_jet_category(args.os_category or legacy)
    ss_category = canonical_jet_category(args.ss_category or legacy)
    return os_category, ss_category


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Draw current NIsoMuon OS-vs-SS dimuon distributions."
    )
    parser.add_argument("--era", required=True, choices=list(ERA_GROUPS.keys()))
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trigger", default="")

    parser.add_argument("--muon-id", default=DEFAULT_MUON_ID)
    parser.add_argument("--jet-id", default=DEFAULT_JET_ID)
    parser.add_argument(
        "--jet-flavour",
        "--jet-flavor",
        default=DEFAULT_JET_FLAVOUR,
        help=(
            "Backward-compatible shorthand for setting both OS and SS jet "
            "categories. Default: bjet. --os-category/--ss-category override it."
        ),
    )
    parser.add_argument(
        "--os-category",
        default=None,
        help=(
            "OS jet category: bjet, lightjet, onebjet, or inclusive. "
            "Default: value of --jet-flavour."
        ),
    )
    parser.add_argument(
        "--ss-category",
        default=None,
        help=(
            "SS jet category: bjet, lightjet, onebjet, or inclusive. "
            "Use onebjet/inclusive for the new SS test regions. "
            "Default: value of --jet-flavour."
        ),
    )
    parser.add_argument(
        "--overlay-lightjet",
        action="store_true",
        help=(
            "Overlay LightJet OS and SS on the primary comparison and draw "
            "the LightJet OS/SS ratio in the lower panel. If --apply-scale "
            "is used, the LightJet SS scale is calculated independently."
        ),
    )
    parser.add_argument("--hist-name", default=DEFAULT_HIST_NAME)

    parser.add_argument(
        "--mode",
        choices=("raw-data", "subtract-nonqcd", "qcd-mc"),
        default="raw-data",
        help=(
            "comparison content: raw-data (default), "
            "subtract-nonqcd = Data - Top - DY - Others, "
            "or qcd-mc = QCD MC only"
        ),
    )
    parser.add_argument(
        "--subtract-nonqcd",
        dest="mode",
        action="store_const",
        const="subtract-nonqcd",
        help="backward-compatible alias for --mode subtract-nonqcd",
    )
    parser.add_argument(
        "--qcd-mc",
        dest="mode",
        action="store_const",
        const="qcd-mc",
        help="alias for --mode qcd-mc",
    )
    parser.add_argument(
        "--dy-method",
        choices=("data-driven", "mc"),
        default="data-driven",
        help="DY subtraction method used only with --mode subtract-nonqcd.",
    )

    parser.add_argument("--xmin", type=float, default=6.0)
    parser.add_argument("--xmax", type=float, default=80.0)
    parser.add_argument(
        "--bin-width",
        type=float,
        default=1.0,
        help=(
            "uniform bin width used only with --no-variable-binning; "
            "default: 1 GeV"
        ),
    )
    parser.add_argument(
        "--no-variable-binning",
        dest="variable_binning",
        action="store_false",
        help=(
            "disable the default dy_bkg_estimation.py-style variable mass "
            "binning and use --bin-width instead"
        ),
    )
    parser.set_defaults(variable_binning=True)
    parser.add_argument("--logy", action="store_true")
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)

    parser.add_argument(
        "--apply-scale",
        action="store_true",
        help="Scale SS by OS/SS in --norm-min/--norm-max.",
    )
    parser.add_argument("--norm-min", type=float, default=DEFAULT_NORM_MIN)
    parser.add_argument("--norm-max", type=float, default=DEFAULT_NORM_MAX)

    parser.add_argument("--ratio-min", type=float, default=0.0)
    parser.add_argument("--ratio-max", type=float, default=2.0)

    parser.add_argument(
        "--no-bin-width",
        dest="divide_by_width",
        action="store_false",
        help="Do not divide rebinned distributions by bin width.",
    )
    parser.set_defaults(divide_by_width=True)

    parser.add_argument(
        "--extensions",
        default="pdf,png",
        help="Comma-separated output extensions; default: pdf,png",
    )

    args = parser.parse_args(argv)

    if args.norm_max <= args.norm_min:
        raise PlotError("--norm-max must be larger than --norm-min")
    if args.ratio_max <= args.ratio_min:
        raise PlotError("--ratio-max must be larger than --ratio-min")

    os_category, ss_category = resolve_categories(args)
    args.os_category = os_category
    args.ss_category = ss_category

    if args.overlay_lightjet and (
        os_category == "lightjet" or ss_category == "lightjet"
    ):
        raise PlotError(
            "--overlay-lightjet adds a separate LightJet OS/SS pair. "
            "Use a non-LightJet primary OS/SS category (for example the "
            "default BJet/BJet comparison)."
        )

    print(
        f"[CATEGORY] OS = {category_root_tag(os_category)} "
        f"({category_label(os_category)})"
    )
    print(
        f"[CATEGORY] SS = {category_root_tag(ss_category)} "
        f"({category_label(ss_category)})"
    )

    ROOT = import_root()

    if args.mode == "subtract-nonqcd":
        h_os_raw = load_qcd_enriched_data(ROOT, args, "OS", os_category)
        h_ss_raw = load_qcd_enriched_data(ROOT, args, "SS", ss_category)
        content_label = "Data - non-QCD"
    elif args.mode == "qcd-mc":
        h_os_raw = load_qcd_mc(ROOT, args, "OS", os_category)
        h_ss_raw = load_qcd_mc(ROOT, args, "SS", ss_category)
        content_label = "QCD MC"
    else:
        h_os_raw = load_data(ROOT, args, "OS", os_category)
        h_ss_raw = load_data(ROOT, args, "SS", ss_category)
        content_label = "Data"

    h_light_os_raw = None
    h_light_ss_raw = None
    if args.overlay_lightjet:
        if args.mode == "subtract-nonqcd":
            h_light_os_raw = load_qcd_enriched_data(
                ROOT, args, "OS", "lightjet"
            )
            h_light_ss_raw = load_qcd_enriched_data(
                ROOT, args, "SS", "lightjet"
            )
        elif args.mode == "qcd-mc":
            h_light_os_raw = load_qcd_mc(ROOT, args, "OS", "lightjet")
            h_light_ss_raw = load_qcd_mc(ROOT, args, "SS", "lightjet")
        else:
            h_light_os_raw = load_data(ROOT, args, "OS", "lightjet")
            h_light_ss_raw = load_data(ROOT, args, "SS", "lightjet")

    os_norm, os_norm_err = integral_open(h_os_raw, args.norm_min, args.norm_max)
    ss_norm, ss_norm_err = integral_open(h_ss_raw, args.norm_min, args.norm_max)

    if ss_norm <= 0.0:
        raise PlotError(
            f"SS integral is non-positive in "
            f"{args.norm_min:g}<m<{args.norm_max:g}: {ss_norm:g}"
        )

    scale_factor = os_norm / ss_norm

    print(
        f"[OS/SS] normalization window: "
        f"{args.norm_min:g} < m(mumu) < {args.norm_max:g} GeV"
    )
    print(
        f"[OS/SS][{category_root_tag(os_category)}/"
        f"{category_root_tag(ss_category)}] OS integral = "
        f"{os_norm:.10g} +/- {os_norm_err:.10g}"
    )
    print(
        f"[OS/SS][{category_root_tag(os_category)}/"
        f"{category_root_tag(ss_category)}] SS integral = "
        f"{ss_norm:.10g} +/- {ss_norm_err:.10g}"
    )
    print(
        f"[OS/SS][{category_root_tag(os_category)}/"
        f"{category_root_tag(ss_category)}] OS/SS scale = {scale_factor:.10g}"
    )

    light_scale_factor = None
    if args.overlay_lightjet:
        light_os_norm, light_os_norm_err = integral_open(
            h_light_os_raw, args.norm_min, args.norm_max
        )
        light_ss_norm, light_ss_norm_err = integral_open(
            h_light_ss_raw, args.norm_min, args.norm_max
        )
        if light_ss_norm <= 0.0:
            raise PlotError(
                f"LightJet SS integral is non-positive in "
                f"{args.norm_min:g}<m<{args.norm_max:g}: {light_ss_norm:g}"
            )
        light_scale_factor = light_os_norm / light_ss_norm
        print(
            f"[OS/SS][LightJet] OS integral = "
            f"{light_os_norm:.10g} +/- {light_os_norm_err:.10g}"
        )
        print(
            f"[OS/SS][LightJet] SS integral = "
            f"{light_ss_norm:.10g} +/- {light_ss_norm_err:.10g}"
        )
        print(
            f"[OS/SS][LightJet] OS/SS scale = {light_scale_factor:.10g}"
        )

    edges = make_edges(
        args.xmin,
        args.xmax,
        args.bin_width,
        args.variable_binning,
    )
    print(
        "[BINNING] "
        + ("variable: " if args.variable_binning else "uniform: ")
        + ", ".join(f"{edge:g}" for edge in edges)
    )
    h_os = rebin_hist(h_os_raw, edges, "h_os_compare")
    h_ss = rebin_hist(h_ss_raw, edges, "h_ss_compare")

    h_light_os = None
    h_light_ss = None
    if args.overlay_lightjet:
        h_light_os = rebin_hist(
            h_light_os_raw, edges, "h_lightjet_os_compare"
        )
        h_light_ss = rebin_hist(
            h_light_ss_raw, edges, "h_lightjet_ss_compare"
        )

    if args.apply_scale:
        h_ss.Scale(scale_factor)
        if args.overlay_lightjet:
            h_light_ss.Scale(light_scale_factor)

    if args.divide_by_width:
        divide_by_bin_width(h_os)
        divide_by_bin_width(h_ss)
        if args.overlay_lightjet:
            divide_by_bin_width(h_light_os)
            divide_by_bin_width(h_light_ss)

    h_os.SetMarkerStyle(20)
    h_os.SetMarkerSize(0.85)
    h_os.SetMarkerColor(ROOT.kBlack)
    h_os.SetLineColor(ROOT.kBlack)
    h_os.SetLineWidth(1)

    h_ss.SetMarkerSize(0.0)
    h_ss.SetLineColor(ROOT.kRed + 1)
    h_ss.SetLineWidth(2)
    h_ss.SetFillStyle(0)

    h_ss_band = h_ss.Clone("h_ss_compare_band")
    h_ss_band.SetDirectory(0)
    h_ss_band.SetMarkerSize(0.0)
    h_ss_band.SetLineWidth(0)
    h_ss_band.SetFillColor(ROOT.kGray + 1)
    h_ss_band.SetFillStyle(3144)

    h_light_ss_band = None
    if args.overlay_lightjet:
        h_light_os.SetMarkerStyle(24)
        h_light_os.SetMarkerSize(0.90)
        h_light_os.SetMarkerColor(ROOT.kBlue + 1)
        h_light_os.SetLineColor(ROOT.kBlue + 1)
        h_light_os.SetLineWidth(1)

        h_light_ss.SetMarkerSize(0.0)
        h_light_ss.SetLineColor(ROOT.kBlue + 1)
        h_light_ss.SetLineStyle(2)
        h_light_ss.SetLineWidth(2)
        h_light_ss.SetFillStyle(0)

        h_light_ss_band = h_light_ss.Clone("h_lightjet_ss_compare_band")
        h_light_ss_band.SetDirectory(0)
        h_light_ss_band.SetMarkerSize(0.0)
        h_light_ss_band.SetLineWidth(0)
        h_light_ss_band.SetFillColor(ROOT.kBlue - 9)
        h_light_ss_band.SetFillStyle(3354)

    canvas = ROOT.TCanvas("c_os_ss", "", 900, 900)
    upper = ROOT.TPad("upper", "", 0.0, 0.30, 1.0, 1.0)
    lower = ROOT.TPad("lower", "", 0.0, 0.00, 1.0, 0.30)

    upper.SetLeftMargin(0.120)
    upper.SetRightMargin(0.050)
    upper.SetTopMargin(0.100)
    upper.SetBottomMargin(0.030)

    lower.SetLeftMargin(0.120)
    lower.SetRightMargin(0.050)
    lower.SetTopMargin(0.040)
    lower.SetBottomMargin(0.350)

    if args.logy:
        upper.SetLogy(True)

    upper.Draw()
    lower.Draw()

    upper.cd()

    plotted_hists = [h_os, h_ss]
    if args.overlay_lightjet:
        plotted_hists.extend([h_light_os, h_light_ss])

    max_y = max(float(hist.GetMaximum()) for hist in plotted_hists)
    ymax = args.ymax if args.ymax is not None else max_y * (50.0 if args.logy else 1.45)

    if args.ymin is not None:
        ymin = args.ymin
    elif args.logy:
        ymin = max(1.0e-3, 0.5 * positive_minimum(*plotted_hists))
    else:
        ymin = 0.0

    h_os.SetMinimum(ymin)
    h_os.SetMaximum(max(ymax, ymin * 10.0 if args.logy else ymax))

    h_os.GetYaxis().SetTitle(
        "Events / GeV" if args.divide_by_width else "Events / bin"
    )
    h_os.GetYaxis().SetTitleFont(42)
    h_os.GetYaxis().SetLabelFont(42)
    h_os.GetYaxis().SetTitleSize(0.055)
    h_os.GetYaxis().SetLabelSize(0.045)
    h_os.GetYaxis().SetTitleOffset(1.05)
    h_os.GetXaxis().SetLabelSize(0.0)
    h_os.GetXaxis().SetTitleSize(0.0)

    h_os.Draw("PE")
    h_ss_band.Draw("E2 SAME")
    if args.overlay_lightjet:
        h_light_ss_band.Draw("E2 SAME")
    h_ss.Draw("HIST SAME")
    if args.overlay_lightjet:
        h_light_ss.Draw("HIST SAME")
        h_light_os.Draw("PE SAME")
    h_os.Draw("PE SAME")

    if args.overlay_lightjet:
        legend = ROOT.TLegend(0.50, 0.56, 0.93, 0.86)
    else:
        legend = ROOT.TLegend(0.55, 0.66, 0.93, 0.86)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.027 if args.overlay_lightjet else 0.030)
    legend.AddEntry(
        h_os,
        f"OS {content_label}, {category_root_tag(os_category)}",
        "lep",
    )

    ss_label = f"SS {content_label}, {category_root_tag(ss_category)}"
    if args.apply_scale:
        ss_label += f" #times {scale_factor:.3g}"
    legend.AddEntry(h_ss, ss_label, "l")
    legend.AddEntry(
        h_ss_band,
        (
            f"{category_root_tag(ss_category)} SS stat. unc."
            if args.overlay_lightjet
            else "SS stat. unc."
        ),
        "f",
    )

    if args.overlay_lightjet:
        legend.AddEntry(
            h_light_os,
            f"OS {content_label}, LightJet",
            "lep",
        )
        light_ss_label = f"SS {content_label}, LightJet"
        if args.apply_scale:
            light_ss_label += f" #times {light_scale_factor:.3g}"
        legend.AddEntry(h_light_ss, light_ss_label, "l")
        legend.AddEntry(
            h_light_ss_band,
            "LightJet SS stat. unc.",
            "f",
        )
    legend.Draw()

    if os_category == ss_category:
        info = [category_label(os_category)]
    else:
        info = [
            f"OS: {category_label(os_category)}",
            f"SS: {category_label(ss_category)}",
        ]
    if args.overlay_lightjet:
        info.append("LightJet OS/SS overlay")

    if args.mode == "subtract-nonqcd":
        info.append("Top, DY, Others subtracted")
    elif args.mode == "qcd-mc":
        info.append("QCD simulation only")
    if args.apply_scale:
        if args.overlay_lightjet:
            info.append(
                f"SS scaled independently using "
                f"{args.norm_min:g} < m < {args.norm_max:g} GeV"
            )
        else:
            info.append(
                f"SS scaled using {args.norm_min:g} < m < {args.norm_max:g} GeV"
            )

    keep = draw_cms_header(ROOT, upper, args, info)
    upper.SetTickx()
    upper.SetTicky()
    upper.RedrawAxis()

    lower.cd()

    h_ratio = h_os.Clone("h_os_over_ss")
    h_ratio.SetDirectory(0)
    h_ratio.Divide(h_ss)

    h_ratio.SetMarkerStyle(20)
    h_ratio.SetMarkerSize(0.75)
    h_ratio.SetMarkerColor(ROOT.kBlack)
    h_ratio.SetLineColor(ROOT.kBlack)

    h_light_ratio = None
    if args.overlay_lightjet:
        h_light_ratio = h_light_os.Clone("h_lightjet_os_over_ss")
        h_light_ratio.SetDirectory(0)
        h_light_ratio.Divide(h_light_ss)
        h_light_ratio.SetMarkerStyle(24)
        h_light_ratio.SetMarkerSize(0.80)
        h_light_ratio.SetMarkerColor(ROOT.kBlue + 1)
        h_light_ratio.SetLineColor(ROOT.kBlue + 1)

    h_ratio.GetYaxis().SetRangeUser(args.ratio_min, args.ratio_max)
    h_ratio.GetYaxis().SetTitle("OS / SS")
    h_ratio.GetYaxis().SetTitleFont(42)
    h_ratio.GetYaxis().SetLabelFont(42)
    h_ratio.GetYaxis().SetTitleSize(0.11)
    h_ratio.GetYaxis().SetTitleOffset(0.50)
    h_ratio.GetYaxis().SetLabelSize(0.09)
    h_ratio.GetYaxis().SetNdivisions(505)

    h_ratio.GetXaxis().SetTitle("m_{#mu#mu} [GeV]")
    h_ratio.GetXaxis().SetTitleFont(42)
    h_ratio.GetXaxis().SetLabelFont(42)
    h_ratio.GetXaxis().SetTitleSize(0.13)
    h_ratio.GetXaxis().SetTitleOffset(1.05)
    h_ratio.GetXaxis().SetLabelSize(0.10)

    h_ratio.Draw("PE")

    unity = ROOT.TLine(args.xmin, 1.0, args.xmax, 1.0)
    unity.SetLineColor(ROOT.kRed + 1)
    unity.SetLineStyle(2)
    unity.SetLineWidth(2)
    unity.Draw("SAME")
    h_ratio.Draw("PE SAME")
    if args.overlay_lightjet:
        h_light_ratio.Draw("PE SAME")
    keep.append(unity)

    lower.SetGridy(True)
    lower.SetTickx()
    lower.SetTicky()
    lower.RedrawAxis()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_tag = {
        "raw-data": "Data",
        "subtract-nonqcd": "DataMinusNonQCD",
        "qcd-mc": "QCDMC",
    }[args.mode]
    scale_tag = (
        f"_SSScaled_{args.norm_min:g}to{args.norm_max:g}"
        if args.apply_scale
        else ""
    )

    os_tag = category_root_tag(os_category)
    ss_tag = category_root_tag(ss_category)
    category_tag = os_tag if os_tag == ss_tag else f"OS{os_tag}_SS{ss_tag}"

    overlay_tag = "_OverlayLightJet" if args.overlay_lightjet else ""

    base_name = (
        f"OS_SS_Comparison_{canonical_era(args.era)}_"
        f"{args.muon_id}_{args.jet_id}_{category_tag}_{mode_tag}"
        f"{scale_tag}{overlay_tag}"
    ).replace(".", "p")

    extensions = [
        x.strip().lstrip(".")
        for x in args.extensions.split(",")
        if x.strip()
    ]

    for ext in extensions:
        output = output_dir / f"{base_name}.{ext}"
        canvas.SaveAs(str(output))
        print(f"[SAVED] {output}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PlotError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
