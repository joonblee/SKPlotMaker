#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot QCD dimuon-jet GEN-flavour components in the BJet selection.

The NIsoMuon.C QCD flavour study stores, for each OS/SS region,

  <SIGN>_POGMedium_tight_BJet_NIsoDimuon_QCDDimuonJet_<FLAVOUR>/
    Dilepton_Mass___<SIGN>_POGMedium_tight_BJet_NIsoDimuon_QCDDimuonJet_<FLAVOUR>

with FLAVOUR = B, C, G, Light, Other.

This script overlays all requested flavour components in one figure:
  * upper panel:
      - OS: filled marker + solid line
      - SS: open marker + dashed line
      - one fixed colour per GEN-flavour category
  * lower panel:
      - OS / SS for each flavour, using the same colour

By default raw weighted QCD MC yields are drawn.
Use --apply-scale to scale SS to OS independently for each flavour in a
normalisation window, or --shape-only to normalise every OS/SS distribution
independently to unit area over the plotted range.

Examples:
  python3 qcd_flavour_os_ss_comparison.py --era Run2
  python3 qcd_flavour_os_ss_comparison.py --era Run2 --logy
  python3 qcd_flavour_os_ss_comparison.py --era Run2 --apply-scale
  python3 qcd_flavour_os_ss_comparison.py --era Run2 --shape-only
  python3 qcd_flavour_os_ss_comparison.py --era 2018 --flavours B,C,G
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from array import array
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


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
YEARS: Tuple[str, ...] = RUN2_ERAS + RUN3_ERAS

ERA_GROUPS: Dict[str, Tuple[str, ...]] = {
    **{era: (era,) for era in YEARS},
    "Run2": RUN2_ERAS,
    "Run3": RUN3_ERAS,
    "Run2+3": YEARS,
    "full": YEARS,
}

DEFAULT_BASE_DIR = (
    "/data6/Users/joonblee/SKOutput/"
    "Run2UL_v3_Run3_v13/NIsoMuon"
)
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
DEFAULT_INPUT_FILE = "NIsoMuon_QCD_Inclusive.root"
DEFAULT_HIST_NAME = "Dilepton_Mass"

ALL_FLAVOURS: Tuple[str, ...] = ("B", "C", "G", "Light", "Other")

DEFAULT_NORM_MIN = 6.0
DEFAULT_NORM_MAX = 9.0

# Same mass binning used by os_ss_comparison.py / dy_bkg_estimation.py.
DEFAULT_MASS_BINS = [
    0.0, 0.5, 1.0, 1.5, 2.0, 2.5,
    3.0, 3.5, 4.0, 4.5,
    5.0, 6.0, 7.0, 8.0, 9.0,
    10.0, 11., 13.0, 15.,
    20.0, 40.0, 80.0,
    120.0, 130.0, 150.0,
]


class PlotError(RuntimeError):
    pass


def import_root():
    try:
        import ROOT  # type: ignore
    except Exception as exc:
        raise PlotError(
            "Could not import PyROOT. Run inside the ROOT/CMSSW environment. "
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


def canonical_flavour(value: str) -> str:
    key = value.strip().lower()
    aliases = {
        "b": "B",
        "bottom": "B",
        "c": "C",
        "charm": "C",
        "g": "G",
        "gluon": "G",
        "light": "Light",
        "lightquark": "Light",
        "light-quark": "Light",
        "other": "Other",
        "others": "Other",
        "unmatched": "Other",
    }
    if key not in aliases:
        raise PlotError(
            f"Unknown flavour {value!r}. "
            "Use B, C, G, Light, or Other."
        )
    return aliases[key]


def parse_flavours(value: str) -> List[str]:
    out: List[str] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        flavour = canonical_flavour(item)
        if flavour not in out:
            out.append(flavour)

    if not out:
        raise PlotError("--flavours produced an empty flavour list.")
    return out


def base_region(sign: str, muon_id: str, jet_id: str) -> str:
    return f"{sign}_{muon_id}_{jet_id}_BJet_NIsoDimuon"


def flavour_region(
    sign: str,
    muon_id: str,
    jet_id: str,
    flavour: str,
) -> str:
    return (
        f"{base_region(sign, muon_id, jet_id)}"
        f"_QCDDimuonJet_{flavour}"
    )


def hist_path(
    sign: str,
    muon_id: str,
    jet_id: str,
    flavour: str,
    hist_name: str,
) -> str:
    reg = flavour_region(sign, muon_id, jet_id, flavour)
    return f"{reg}/{hist_name}___{reg}"


def era_dir(args, year: str) -> str:
    parts = [args.base_dir, year]
    if args.trigger:
        parts.append(args.trigger)
    return os.path.join(*parts)


def open_hist(ROOT, filename: str, path: str, clone_name: str):
    if not os.path.isfile(filename):
        raise PlotError(f"Missing ROOT file: {filename}")

    root_file = ROOT.TFile.Open(filename, "READ")
    if not root_file or root_file.IsZombie():
        if root_file:
            root_file.Close()
        raise PlotError(f"Could not open ROOT file: {filename}")

    hist = root_file.Get(path)
    if not hist:
        root_file.Close()
        raise PlotError(f"Missing histogram: {filename}:{path}")

    out = hist.Clone(clone_name)
    out.SetDirectory(0)
    out.Sumw2()
    root_file.Close()
    return out


def sum_hists(hists: Sequence[object], name: str):
    if not hists:
        raise PlotError("No histograms were loaded.")

    out = hists[0].Clone(name)
    out.SetDirectory(0)
    out.Sumw2()

    for hist in hists[1:]:
        out.Add(hist)
    return out


def load_flavour_hist(
    ROOT,
    args,
    sign: str,
    flavour: str,
):
    path = hist_path(
        sign,
        args.muon_id,
        args.jet_id,
        flavour,
        args.hist_name,
    )

    pieces = []
    for year in years_for_era(args.era):
        filename = os.path.join(
            era_dir(args, year),
            args.input_file,
        )
        pieces.append(
            open_hist(
                ROOT,
                filename,
                path,
                f"h_{sign}_{flavour}_{year}",
            )
        )

    return sum_hists(
        pieces,
        f"h_{sign}_{flavour}_{canonical_era(args.era)}",
    )


def make_variable_edges(
    xmin: float,
    xmax: float,
    requested_edges: Sequence[float] = DEFAULT_MASS_BINS,
) -> List[float]:
    if xmax <= xmin:
        raise PlotError("--xmax must be larger than --xmin")

    edges = [
        float(x)
        for x in requested_edges
        if xmin <= float(x) <= xmax
    ]

    if (
        not edges
        or not math.isclose(
            edges[0],
            xmin,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        edges.insert(0, float(xmin))

    if not math.isclose(
        edges[-1],
        xmax,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        edges.append(float(xmax))

    unique: List[float] = []
    for value in edges:
        if (
            not unique
            or not math.isclose(
                value,
                unique[-1],
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            unique.append(value)

    if len(unique) < 2:
        raise PlotError("Could not construct variable mass binning.")
    return unique


def make_uniform_edges(
    xmin: float,
    xmax: float,
    width: float,
) -> List[float]:
    if width <= 0.0:
        raise PlotError("--bin-width must be positive")
    if xmax <= xmin:
        raise PlotError("--xmax must be larger than --xmin")

    n = int(round((xmax - xmin) / width))
    if (
        n <= 0
        or not math.isclose(
            xmin + n * width,
            xmax,
            rel_tol=0.0,
            abs_tol=1.0e-8,
        )
    ):
        raise PlotError(
            f"[{xmin:g},{xmax:g}] is not an integer number of "
            f"{width:g}-GeV bins"
        )

    return [xmin + i * width for i in range(n + 1)]


def make_edges(args) -> List[float]:
    if args.variable_binning:
        return make_variable_edges(args.xmin, args.xmax)
    return make_uniform_edges(args.xmin, args.xmax, args.bin_width)


def rebin_hist(hist, edges: Sequence[float], name: str):
    arr = array("d", [float(x) for x in edges])
    out = hist.Rebin(len(edges) - 1, name, arr)
    out.SetDirectory(0)
    out.Sumw2()
    return out


def integral_open(hist, xmin: float, xmax: float) -> Tuple[float, float]:
    value = 0.0
    variance = 0.0

    for ibin in range(1, hist.GetNbinsX() + 1):
        x = float(hist.GetXaxis().GetBinCenter(ibin))
        if not (xmin < x < xmax):
            continue

        value += float(hist.GetBinContent(ibin))
        error = float(hist.GetBinError(ibin))
        variance += error * error

    return value, math.sqrt(max(0.0, variance))


def integral_in_plot(hist, xmin: float, xmax: float) -> float:
    value = 0.0
    for ibin in range(1, hist.GetNbinsX() + 1):
        x = float(hist.GetXaxis().GetBinCenter(ibin))
        if xmin <= x <= xmax:
            value += float(hist.GetBinContent(ibin))
    return value


def divide_by_bin_width(hist) -> None:
    for ibin in range(1, hist.GetNbinsX() + 1):
        width = float(hist.GetXaxis().GetBinWidth(ibin))
        if width <= 0.0:
            continue

        hist.SetBinContent(
            ibin,
            float(hist.GetBinContent(ibin)) / width,
        )
        hist.SetBinError(
            ibin,
            float(hist.GetBinError(ibin)) / width,
        )


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


def draw_cms_header(ROOT, pad, args, mode_text: str):
    keep = []
    pad.cd()

    latex = ROOT.TLatex()
    latex.SetNDC(True)
    latex.SetTextFont(42)
    latex.SetTextColor(ROOT.kBlack)

    latex.SetTextAlign(11)
    latex.SetTextSize(0.047)
    latex.DrawLatex(
        0.120,
        0.925,
        "#bf{CMS} #it{Preliminary}",
    )

    latex.SetTextAlign(31)
    latex.SetTextSize(0.038)
    latex.DrawLatex(
        0.950,
        0.925,
        cms_lumi_label(args.era),
    )

    latex.SetTextAlign(11)
    latex.SetTextSize(0.032)
    latex.DrawLatex(
        0.150,
        0.860,
        "QCD MC, b-jet selection",
    )

    latex.SetTextSize(0.028)
    latex.SetTextColor(ROOT.kGray + 2)
    latex.DrawLatex(
        0.150,
        0.820,
        "dimuon-jet GEN flavour",
    )
    latex.DrawLatex(
        0.150,
        0.782,
        mode_text,
    )

    keep.append(latex)
    return keep


def positive_minimum(hists: Sequence[object]) -> float:
    values: List[float] = []

    for hist in hists:
        for ibin in range(1, hist.GetNbinsX() + 1):
            value = float(hist.GetBinContent(ibin))
            if value > 0.0:
                values.append(value)

    return min(values) if values else 1.0


def style_maps(ROOT):
    # One colour per GEN flavour. Sign is encoded independently by
    # marker fill and line style.
    colours = {
        "B": ROOT.kRed + 1,
        "C": ROOT.kBlue + 1,
        "G": ROOT.kGreen + 2,
        "Light": ROOT.kMagenta + 2,
        "Other": ROOT.kGray + 2,
    }

    # Filled/open pairs for OS/SS.
    os_markers = {
        "B": 20,
        "C": 21,
        "G": 22,
        "Light": 23,
        "Other": 33,
    }
    ss_markers = {
        "B": 24,
        "C": 25,
        "G": 26,
        "Light": 32,
        "Other": 27,
    }

    return colours, os_markers, ss_markers


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay B/C/G/Light/Other QCD dimuon-jet GEN-flavour "
            "OS and SS distributions in the BJet selection."
        )
    )

    parser.add_argument(
        "--era",
        required=True,
        choices=list(ERA_GROUPS.keys()),
    )
    parser.add_argument(
        "--base-dir",
        default=DEFAULT_BASE_DIR,
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--trigger",
        default="",
    )
    parser.add_argument(
        "--input-file",
        default=DEFAULT_INPUT_FILE,
    )

    parser.add_argument(
        "--muon-id",
        default=DEFAULT_MUON_ID,
    )
    parser.add_argument(
        "--jet-id",
        default=DEFAULT_JET_ID,
    )
    parser.add_argument(
        "--hist-name",
        default=DEFAULT_HIST_NAME,
    )

    parser.add_argument(
        "--flavours",
        default="B,C,G,Light,Other",
        help=(
            "Comma-separated subset of B,C,G,Light,Other. "
            "Default: all five."
        ),
    )

    parser.add_argument(
        "--xmin",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--xmax",
        type=float,
        default=80.0,
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=1.0,
        help=(
            "Uniform bin width used only with --no-variable-binning."
        ),
    )
    parser.add_argument(
        "--no-variable-binning",
        dest="variable_binning",
        action="store_false",
    )
    parser.set_defaults(variable_binning=True)

    parser.add_argument(
        "--logy",
        action="store_true",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--ratio-min",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--ratio-max",
        type=float,
        default=5.,
    )

    parser.add_argument(
        "--apply-scale",
        action="store_true",
        help=(
            "For each GEN flavour independently, scale SS by OS/SS "
            "in --norm-min/--norm-max."
        ),
    )
    parser.add_argument(
        "--norm-min",
        type=float,
        default=DEFAULT_NORM_MIN,
    )
    parser.add_argument(
        "--norm-max",
        type=float,
        default=DEFAULT_NORM_MAX,
    )

    parser.add_argument(
        "--shape-only",
        action="store_true",
        help=(
            "Normalise every OS and SS histogram independently to unit "
            "area over the plotted range. Useful for pure shape comparison."
        ),
    )

    parser.add_argument(
        "--no-bin-width",
        dest="divide_by_width",
        action="store_false",
        help="Do not divide rebinned histograms by bin width.",
    )
    parser.set_defaults(divide_by_width=True)

    parser.add_argument(
        "--extensions",
        default="pdf,png",
        help="Comma-separated output extensions. Default: pdf,png",
    )

    args = parser.parse_args(argv)

    if args.apply_scale and args.shape_only:
        raise PlotError(
            "Use either --apply-scale or --shape-only, not both."
        )

    if args.norm_max <= args.norm_min:
        raise PlotError(
            "--norm-max must be larger than --norm-min."
        )

    if args.ratio_max <= args.ratio_min:
        raise PlotError(
            "--ratio-max must be larger than --ratio-min."
        )

    flavours = parse_flavours(args.flavours)
    ROOT = import_root()

    edges = make_edges(args)
    print(
        "[BINNING] "
        + ("variable: " if args.variable_binning else "uniform: ")
        + ", ".join(f"{edge:g}" for edge in edges)
    )

    hists: Dict[str, Dict[str, object]] = {}
    scale_factors: Dict[str, float] = {}

    for flavour in flavours:
        h_os_raw = load_flavour_hist(
            ROOT,
            args,
            "OS",
            flavour,
        )
        h_ss_raw = load_flavour_hist(
            ROOT,
            args,
            "SS",
            flavour,
        )

        os_norm, os_norm_err = integral_open(
            h_os_raw,
            args.norm_min,
            args.norm_max,
        )
        ss_norm, ss_norm_err = integral_open(
            h_ss_raw,
            args.norm_min,
            args.norm_max,
        )

        print(
            f"[{flavour}] "
            f"{args.norm_min:g}<m<{args.norm_max:g} GeV: "
            f"OS={os_norm:.10g} +/- {os_norm_err:.10g}, "
            f"SS={ss_norm:.10g} +/- {ss_norm_err:.10g}"
        )

        if ss_norm > 0.0:
            scale_factors[flavour] = os_norm / ss_norm
            print(
                f"[{flavour}] OS/SS scale = "
                f"{scale_factors[flavour]:.10g}"
            )
        else:
            scale_factors[flavour] = 1.0
            if args.apply_scale:
                raise PlotError(
                    f"{flavour} SS integral is non-positive in the "
                    "requested normalisation window."
                )

        h_os = rebin_hist(
            h_os_raw,
            edges,
            f"h_{flavour}_OS_rebinned",
        )
        h_ss = rebin_hist(
            h_ss_raw,
            edges,
            f"h_{flavour}_SS_rebinned",
        )

        if args.apply_scale:
            h_ss.Scale(scale_factors[flavour])

        if args.shape_only:
            os_area = integral_in_plot(
                h_os,
                args.xmin,
                args.xmax,
            )
            ss_area = integral_in_plot(
                h_ss,
                args.xmin,
                args.xmax,
            )

            if os_area <= 0.0 or ss_area <= 0.0:
                raise PlotError(
                    f"Cannot unit-normalise {flavour}: "
                    f"OS area={os_area:g}, SS area={ss_area:g}."
                )

            h_os.Scale(1.0 / os_area)
            h_ss.Scale(1.0 / ss_area)

        if args.divide_by_width:
            divide_by_bin_width(h_os)
            divide_by_bin_width(h_ss)

        hists[flavour] = {
            "OS": h_os,
            "SS": h_ss,
        }

    colours, os_markers, ss_markers = style_maps(ROOT)

    for flavour in flavours:
        h_os = hists[flavour]["OS"]
        h_ss = hists[flavour]["SS"]
        colour = colours[flavour]

        h_os.SetLineColor(colour)
        h_os.SetLineWidth(2)
        h_os.SetLineStyle(1)
        h_os.SetMarkerColor(colour)
        h_os.SetMarkerStyle(os_markers[flavour])
        h_os.SetMarkerSize(0.85)

        h_ss.SetLineColor(colour)
        h_ss.SetLineWidth(2)
        h_ss.SetLineStyle(2)
        h_ss.SetMarkerColor(colour)
        h_ss.SetMarkerStyle(ss_markers[flavour])
        h_ss.SetMarkerSize(0.85)

    canvas = ROOT.TCanvas(
        "c_qcd_flavour_os_ss",
        "",
        1000,
        900,
    )

    upper = ROOT.TPad(
        "upper",
        "",
        0.0,
        0.30,
        1.0,
        1.0,
    )
    lower = ROOT.TPad(
        "lower",
        "",
        0.0,
        0.00,
        1.0,
        0.30,
    )

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

    all_upper_hists = [
        hists[flavour][sign]
        for flavour in flavours
        for sign in ("OS", "SS")
    ]

    max_y = max(
        float(hist.GetMaximum())
        for hist in all_upper_hists
    )

    if args.ymax is not None:
        ymax = args.ymax
    else:
        ymax = max_y * (80.0 if args.logy else 1.55)

    if args.ymin is not None:
        ymin = args.ymin
    elif args.logy:
        ymin = max(
            1.0e-6,
            0.4 * positive_minimum(all_upper_hists),
        )
    else:
        ymin = 0.0

    first = hists[flavours[0]]["OS"]
    first.SetMinimum(ymin)
    first.SetMaximum(
        max(
            ymax,
            ymin * 10.0 if args.logy else ymax,
        )
    )

    if args.shape_only:
        y_title = (
            "Arbitrary units / GeV"
            if args.divide_by_width
            else "Arbitrary units / bin"
        )
    else:
        y_title = (
            "Events / GeV"
            if args.divide_by_width
            else "Events / bin"
        )

    first.GetYaxis().SetTitle(y_title)
    first.GetYaxis().SetTitleFont(42)
    first.GetYaxis().SetLabelFont(42)
    first.GetYaxis().SetTitleSize(0.055)
    first.GetYaxis().SetLabelSize(0.045)
    first.GetYaxis().SetTitleOffset(1.05)

    first.GetXaxis().SetLabelSize(0.0)
    first.GetXaxis().SetTitleSize(0.0)

    first.Draw("E1PL")

    for flavour in flavours:
        h_os = hists[flavour]["OS"]
        h_ss = hists[flavour]["SS"]

        if h_os is not first:
            h_os.Draw("E1PL SAME")
        h_ss.Draw("E1PL SAME")

    # Draw OS again last so filled OS markers remain visible above SS.
    for flavour in flavours:
        hists[flavour]["OS"].Draw("E1P SAME")

    legend = ROOT.TLegend(
        0.49,
        0.52,
        0.94,
        0.87,
    )
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.027)
    legend.SetNColumns(2)

    for flavour in flavours:
        os_label = f"{flavour} OS"
        ss_label = f"{flavour} SS"

        if args.apply_scale:
            ss_label += (
                f" #times {scale_factors[flavour]:.3g}"
            )

        legend.AddEntry(
            hists[flavour]["OS"],
            os_label,
            "lep",
        )
        legend.AddEntry(
            hists[flavour]["SS"],
            ss_label,
            "lep",
        )

    legend.Draw()

    if args.shape_only:
        mode_text = "OS and SS independently unit-normalised"
    elif args.apply_scale:
        mode_text = (
            f"SS scaled to OS in "
            f"{args.norm_min:g}<m_{{#mu#mu}}<{args.norm_max:g} GeV"
        )
    else:
        mode_text = "Raw weighted QCD yields"

    keep = draw_cms_header(
        ROOT,
        upper,
        args,
        mode_text,
    )

    upper.SetTickx()
    upper.SetTicky()
    upper.RedrawAxis()

    lower.cd()

    ratio_hists: Dict[str, object] = {}
    first_ratio = None

    for flavour in flavours:
        ratio = hists[flavour]["OS"].Clone(
            f"h_ratio_{flavour}"
        )
        ratio.SetDirectory(0)
        ratio.Divide(hists[flavour]["SS"])

        colour = colours[flavour]
        ratio.SetLineColor(colour)
        ratio.SetLineWidth(2)
        ratio.SetMarkerColor(colour)
        ratio.SetMarkerStyle(os_markers[flavour])
        ratio.SetMarkerSize(0.75)

        ratio_hists[flavour] = ratio

        if first_ratio is None:
            first_ratio = ratio

    first_ratio.GetYaxis().SetRangeUser(
        args.ratio_min,
        args.ratio_max,
    )
    first_ratio.GetYaxis().SetTitle("OS / SS")
    first_ratio.GetYaxis().SetTitleFont(42)
    first_ratio.GetYaxis().SetLabelFont(42)
    first_ratio.GetYaxis().SetTitleSize(0.11)
    first_ratio.GetYaxis().SetTitleOffset(0.50)
    first_ratio.GetYaxis().SetLabelSize(0.09)
    first_ratio.GetYaxis().SetNdivisions(505)

    first_ratio.GetXaxis().SetTitle(
        "m_{#mu#mu} [GeV]"
    )
    first_ratio.GetXaxis().SetTitleFont(42)
    first_ratio.GetXaxis().SetLabelFont(42)
    first_ratio.GetXaxis().SetTitleSize(0.13)
    first_ratio.GetXaxis().SetTitleOffset(1.05)
    first_ratio.GetXaxis().SetLabelSize(0.10)

    first_ratio.Draw("E1P")

    for flavour in flavours[1:]:
        ratio_hists[flavour].Draw("E1P SAME")

    unity = ROOT.TLine(
        args.xmin,
        1.0,
        args.xmax,
        1.0,
    )
    unity.SetLineColor(ROOT.kBlack)
    unity.SetLineStyle(2)
    unity.SetLineWidth(1)
    unity.Draw("SAME")

    for flavour in flavours:
        ratio_hists[flavour].Draw("E1P SAME")

    keep.append(unity)

    lower.SetGridy(True)
    lower.SetTickx()
    lower.SetTicky()
    lower.RedrawAxis()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    flavour_tag = "-".join(flavours)

    if args.shape_only:
        mode_tag = "ShapeOnly"
    elif args.apply_scale:
        mode_tag = (
            f"SSScaled_{args.norm_min:g}to"
            f"{args.norm_max:g}"
        ).replace(".", "p")
    else:
        mode_tag = "Raw"

    base_name = (
        f"QCDFlavor_OS_SS_BJet_"
        f"{canonical_era(args.era)}_"
        f"{flavour_tag}_{mode_tag}"
    )

    extensions = [
        item.strip().lstrip(".")
        for item in args.extensions.split(",")
        if item.strip()
    ]

    for extension in extensions:
        output = output_dir / (
            f"{base_name}.{extension}"
        )
        canvas.SaveAs(str(output))
        print(f"[SAVED] {output}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PlotError as exc:
        print(
            f"[ERROR] {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2)

