#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the dedicated convener-review diagnostics written by SKFlatAnalyzer
NIsoMuon.C with the ConvenerStudy and DYValidationDRStudy flags.

Expected production layout by default:

  <base>/ConvenerStudy/2018/
  <base>/DYValidationDRStudy/2018/

Run hadd.sh first so the usual merged process files exist in those directories.

Examples:

  source hadd.sh 2018 ConvenerStudy
  source hadd.sh 2018 DYValidationDRStudy

  python3 convener_studies.py --era 2018 --study all
  python3 convener_studies.py --era 2018 --study lepton-veto
  python3 convener_studies.py --era 2018 --study jet-composition
  python3 convener_studies.py --era 2018 --study dijet-mass
  python3 convener_studies.py --era 2018 --study dy-vr

The ConvenerStudy plots are MC-only.  In particular, this script never draws
collision data from the nominal b-jet signal-region-like ConvenerStudy
histograms.  The DY validation-region mode is explicitly orthogonal to the
nominal in-jet signal selection and is intended for an unblinded data closure
test.

Produced diagnostics:

  lepton-veto
    Survival fractions for no veto, N_e=0, N_tau=0, and N_e=N_tau=0.

  jet-composition
    Shape-normalised dimuon-jet constituent multiplicity and charged-hadron
    energy fraction.

  dijet-mass
    Shape-normalised invariant mass of the dimuon jet and independent tag jet.

  dy-vr
    In the DeltaR(mu, reference jet)>0.4 validation region, compare
    background-subtracted b-jet-category data with background-subtracted
    light-jet data multiplied by an aMC@NLO DY transfer factor measured in
    11<m(mumu)<80 GeV.

The script intentionally does not modify the nominal background-estimation
outputs or datacards.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
from array import array
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_BASE_DIR = "/data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13/NIsoMuon"
DEFAULT_OUTPUT_DIR = "convener_studies"

PROCESS_FILES = {
    "data": "data.root",
    "QCD": "NIsoMuon_QCD_Inclusive.root",
    "DY": "NIsoMuon_DYJets_Inclusive.root",
    "tt": "NIsoMuon_tt.root",
    "ST": "NIsoMuon_ST.root",
    "Others": "NIsoMuon_Others.root",
}

PROCESS_LABELS = {
    "QCD": "QCD",
    "DY": "DY",
    "Top": "Top",
    "Others": "Others",
}

LUMI_FB = {
    "2016preVFP": 19.52,
    "2016postVFP": 16.81,
    "2017": 41.48,
    "2018": 59.83,
}

DEFAULT_DY_VR_EDGES = [
    5.0, 6.0, 7.0, 8.0, 9.0,
    11.0, 13.0, 15.0, 20.0, 25.0,
    30.0, 40.0, 50.0, 60.0, 70.0, 80.0,
]


class PlotError(RuntimeError):
    pass


class NameFactory:
    def __init__(self) -> None:
        self._counter = 0

    def get(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"


NAMES = NameFactory()


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
    ROOT.gStyle.SetEndErrorSize(3)
    try:
        ROOT.TH1.AddDirectory(False)
    except Exception:
        pass
    return ROOT


def set_cms_style(ROOT) -> None:
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)
    ROOT.gStyle.SetLegendBorderSize(0)
    ROOT.gStyle.SetFrameLineWidth(2)
    ROOT.gStyle.SetHistLineWidth(2)
    ROOT.gStyle.SetLineWidth(2)


def draw_cms_header(
    ROOT,
    pad,
    era: str,
    subtitle: str,
    *,
    subtitle_x: float = 0.15,
    subtitle_y: float = 0.845,
):
    keep = []
    pad.cd()

    latex = ROOT.TLatex()
    latex.SetNDC(True)
    latex.SetTextFont(42)
    latex.SetTextColor(ROOT.kBlack)

    latex.SetTextAlign(11)
    latex.SetTextSize(0.050)
    latex.DrawLatex(0.12, 0.925, "#bf{CMS} #it{Preliminary}")

    lumi = LUMI_FB.get(era)
    if lumi is not None:
        energy = "13 TeV"
        latex.SetTextAlign(31)
        latex.SetTextSize(0.039)
        latex.DrawLatex(
            0.95,
            0.925,
            f"{lumi:.1f} fb^{{-1}} ({energy})",
        )

    latex.SetTextAlign(11)
    latex.SetTextSize(0.033)
    latex.DrawLatex(subtitle_x, subtitle_y, subtitle)

    keep.append(latex)
    return keep


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_float_list(value: str) -> List[float]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(float(item))
    return out


def mass_label(value: float) -> str:
    if abs(value - round(value)) < 1.0e-9:
        return str(int(round(value)))
    return ("%g" % value).replace(".", "p")


def collection_dir(
    base_dir: str,
    era: str,
    collection: str,
    override: Optional[str],
) -> str:
    if override:
        return os.path.abspath(override)
    return os.path.join(base_dir, collection, era)


def read_hist(ROOT, filename: str, path: str, *, required: bool = True):
    if not os.path.isfile(filename):
        if required:
            raise PlotError(f"Missing ROOT file: {filename}")
        return None

    f = ROOT.TFile.Open(filename, "READ")
    if not f or f.IsZombie():
        if f:
            f.Close()
        if required:
            raise PlotError(f"Could not open ROOT file: {filename}")
        return None

    h = f.Get(path)
    if not h:
        f.Close()
        if required:
            raise PlotError(f"Missing histogram: {filename}:{path}")
        return None

    out = h.Clone(NAMES.get("h"))
    out.SetDirectory(0)
    out.Sumw2()
    f.Close()
    return out


def sum_hists(hists: Iterable[object], prefix: str):
    pieces = [h for h in hists if h]
    if not pieces:
        return None
    out = pieces[0].Clone(NAMES.get(prefix))
    out.SetDirectory(0)
    out.Sumw2()
    for hist in pieces[1:]:
        out.Add(hist)
    return out


def process_file(root_dir: str, process: str) -> str:
    if process not in PROCESS_FILES:
        raise PlotError(f"Unknown process: {process}")
    return os.path.join(root_dir, PROCESS_FILES[process])


def signal_file(root_dir: str, mass: float) -> str:
    label = mass_label(mass)
    direct = os.path.join(root_dir, f"NIsoMuon_Zp_M-{label}.root")
    if os.path.isfile(direct):
        return direct

    for path in sorted(glob.glob(os.path.join(root_dir, "NIsoMuon_Zp_M-*.root"))):
        base = os.path.basename(path)
        token = base[len("NIsoMuon_Zp_M-"):-len(".root")]
        try:
            parsed = float(token.replace("p", "."))
        except ValueError:
            continue
        if abs(parsed - mass) < 1.0e-6:
            return path

    raise PlotError(
        f"Could not find merged signal file for M={mass:g} under {root_dir}"
    )


def convener_region(category: str) -> str:
    return f"OS_POGMedium_tight_{category}_NIsoDimuon_ConvenerStudy"


def dy_vr_region(category: str) -> str:
    return f"OS_POGMedium_tight_{category}_DYValidationDRGt0p4"


def hist_path(region: str, hist_name: str) -> str:
    return f"{region}/{hist_name}___{region}"


def integral_and_error(hist, xmin: float, xmax: float) -> Tuple[float, float]:
    value = 0.0
    variance = 0.0
    for ibin in range(1, hist.GetNbinsX() + 1):
        x = float(hist.GetXaxis().GetBinCenter(ibin))
        if not (xmin < x < xmax):
            continue
        value += float(hist.GetBinContent(ibin))
        err = float(hist.GetBinError(ibin))
        variance += err * err
    return value, math.sqrt(max(0.0, variance))


def normalize_shape(hist, xmin: Optional[float] = None, xmax: Optional[float] = None):
    if xmin is None or xmax is None:
        area = float(hist.Integral(1, hist.GetNbinsX()))
    else:
        area, _ = integral_and_error(hist, xmin, xmax)
    if area <= 0.0:
        raise PlotError(
            f"Cannot shape-normalise histogram with non-positive area {area:g}"
        )
    hist.Scale(1.0 / area)
    return hist


def rebin_variable(hist, edges: Sequence[float], prefix: str):
    arr = array("d", [float(x) for x in edges])
    out = hist.Rebin(len(edges) - 1, NAMES.get(prefix), arr)
    out.SetDirectory(0)
    out.Sumw2()
    return out


def mask_upsilon_bin(hist) -> None:
    for ibin in range(1, hist.GetNbinsX() + 1):
        low = float(hist.GetXaxis().GetBinLowEdge(ibin))
        high = low + float(hist.GetXaxis().GetBinWidth(ibin))
        if low < 11.0 and high > 9.0:
            hist.SetBinContent(ibin, 0.0)
            hist.SetBinError(ibin, 0.0)


def process_hist(
    ROOT,
    root_dir: str,
    process: str,
    region: str,
    hist_name: str,
    *,
    required: bool = True,
):
    if process == "Top":
        h_tt = read_hist(
            ROOT,
            process_file(root_dir, "tt"),
            hist_path(region, hist_name),
            required=required,
        )
        h_st = read_hist(
            ROOT,
            process_file(root_dir, "ST"),
            hist_path(region, hist_name),
            required=required,
        )
        return sum_hists([h_tt, h_st], "top")

    return read_hist(
        ROOT,
        process_file(root_dir, process),
        hist_path(region, hist_name),
        required=required,
    )


def signal_hist(
    ROOT,
    root_dir: str,
    mass: float,
    region: str,
    hist_name: str,
):
    return read_hist(
        ROOT,
        signal_file(root_dir, mass),
        hist_path(region, hist_name),
    )


def style_lines(ROOT):
    # Keep the convener diagnostics visually close to the validation figures
    # used in the AN: simple solid/dashed lines and the usual process colours.
    return {
        "QCD": (ROOT.kAzure + 2, 1),
        "DY": (ROOT.kGray + 2, 2),
        "Top": (ROOT.kOrange + 7, 3),
        "Others": (ROOT.kGreen + 2, 4),
    }


def signal_styles(ROOT):
    # The two benchmark masses shown by default in the review diagnostics.
    return {
        12.0: (ROOT.kRed + 1, 1),
        70.0: (ROOT.kMagenta + 2, 1),
    }


def save_canvas(canvas, base: str, extensions: Sequence[str]) -> None:
    ensure_dir(os.path.dirname(base))
    for ext in extensions:
        canvas.SaveAs(f"{base}.{ext}")


def draw_shape_overlay(
    ROOT,
    hists: Sequence[Tuple[str, object, int, int]],
    *,
    era: str,
    subtitle: str,
    x_title: str,
    output_base: str,
    extensions: Sequence[str],
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    rebin: int = 1,
) -> None:
    if not hists:
        raise PlotError(f"No histograms supplied for {subtitle}")

    styled = []
    max_y = 0.0

    for label, hist, colour, line_style in hists:
        h = hist.Clone(NAMES.get("shape"))
        h.SetDirectory(0)
        if rebin > 1:
            h.Rebin(rebin)
        normalize_shape(h, xmin, xmax)
        h.SetLineColor(colour)
        h.SetLineStyle(line_style)
        h.SetLineWidth(3)
        h.SetMarkerColor(colour)
        max_y = max(max_y, float(h.GetMaximum()))
        styled.append((label, h))

    canvas = ROOT.TCanvas(NAMES.get("c"), "", 900, 800)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.05)
    canvas.SetTopMargin(0.11)
    canvas.SetBottomMargin(0.13)

    first = styled[0][1]
    first.SetMaximum(max_y * 1.45 if max_y > 0.0 else 1.0)
    first.SetMinimum(0.0)
    first.GetXaxis().SetTitle(x_title)
    first.GetYaxis().SetTitle("Arbitrary units")
    first.GetXaxis().SetTitleSize(0.050)
    first.GetYaxis().SetTitleSize(0.050)
    first.GetXaxis().SetLabelSize(0.042)
    first.GetYaxis().SetLabelSize(0.042)
    first.GetYaxis().SetTitleOffset(1.15)
    if xmin is not None and xmax is not None:
        first.GetXaxis().SetRangeUser(xmin, xmax)
    first.Draw("HIST")

    for _, hist in styled[1:]:
        hist.Draw("HIST SAME")

    legend = ROOT.TLegend(0.62, 0.60, 0.92, 0.84)
    legend.SetFillStyle(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.035)
    for label, hist in styled:
        legend.AddEntry(hist, label, "l")
    legend.Draw()

    keep = draw_cms_header(
        ROOT,
        canvas,
        era,
        subtitle,
        subtitle_x=0.15,
        subtitle_y=0.845,
    )
    keep.extend([legend])
    canvas.RedrawAxis()
    save_canvas(canvas, output_base, extensions)


def draw_veto_mass_comparison(
    ROOT,
    args,
    label: str,
    before,
    after,
    colour: int,
    output_base: str,
) -> None:
    """Draw absolute dimuon-mass yields before/after the electron veto.

    The lower panel shows the bin-by-bin electron-veto survival fraction,
    after/before.  This is more informative than a single integrated number:
    it tests both the overall acceptance loss and any mass-dependent shape
    distortion.
    """
    h_before = before.Clone(NAMES.get("veto_before"))
    h_after = after.Clone(NAMES.get("veto_after"))
    h_before.SetDirectory(0)
    h_after.SetDirectory(0)

    # The analyzer stores very fine 20-MeV mass bins. Rebin for a readable
    # review diagnostic while preserving the 11--80 GeV search interval.
    if args.veto_mass_rebin > 1:
        h_before.Rebin(args.veto_mass_rebin)
        h_after.Rebin(args.veto_mass_rebin)

    h_before.GetXaxis().SetRangeUser(args.mass_min, args.mass_max)
    h_after.GetXaxis().SetRangeUser(args.mass_min, args.mass_max)

    canvas = ROOT.TCanvas(NAMES.get("c_veto_mass"), "", 900, 900)
    upper = ROOT.TPad(NAMES.get("veto_upper"), "", 0.0, 0.30, 1.0, 1.0)
    lower = ROOT.TPad(NAMES.get("veto_lower"), "", 0.0, 0.00, 1.0, 0.30)

    upper.SetLeftMargin(0.13)
    upper.SetRightMargin(0.05)
    upper.SetTopMargin(0.11)
    upper.SetBottomMargin(0.03)
    lower.SetLeftMargin(0.13)
    lower.SetRightMargin(0.05)
    lower.SetTopMargin(0.04)
    lower.SetBottomMargin(0.34)
    upper.Draw()
    lower.Draw()

    upper.cd()
    h_before.SetLineColor(ROOT.kBlack)
    h_before.SetMarkerColor(ROOT.kBlack)
    h_before.SetLineWidth(3)
    h_before.SetLineStyle(1)

    h_after.SetLineColor(colour)
    h_after.SetMarkerColor(colour)
    h_after.SetLineWidth(3)
    h_after.SetLineStyle(2)

    ymax = max(float(h_before.GetMaximum()), float(h_after.GetMaximum()))
    h_before.SetMaximum(1.40 * ymax if ymax > 0.0 else 1.0)
    h_before.SetMinimum(0.0)
    h_before.GetYaxis().SetTitle("Events")
    h_before.GetYaxis().SetTitleSize(0.055)
    h_before.GetYaxis().SetLabelSize(0.045)
    h_before.GetYaxis().SetTitleOffset(1.05)
    h_before.GetXaxis().SetLabelSize(0.0)
    h_before.Draw("HIST")
    h_after.Draw("HIST SAME")

    legend = ROOT.TLegend(0.66, 0.66, 0.92, 0.83)
    legend.SetFillStyle(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.036)
    legend.AddEntry(h_before, "No electron veto", "l")
    legend.AddEntry(h_after, "N_{e}=0", "l")
    legend.Draw()

    keep = draw_cms_header(
        ROOT,
        upper,
        args.era,
        f"Electron-veto impact: {label}",
        subtitle_x=0.16,
        subtitle_y=0.78,
    )
    keep.extend([legend, h_before, h_after])

    lower.cd()
    ratio = h_after.Clone(NAMES.get("veto_ratio"))
    ratio.SetDirectory(0)
    ratio.Divide(h_before)
    ratio.SetLineColor(colour)
    ratio.SetMarkerColor(colour)
    ratio.SetMarkerStyle(20)
    ratio.SetMarkerSize(0.8)
    ratio.SetLineWidth(2)
    ratio.SetMinimum(args.veto_ratio_min)
    ratio.SetMaximum(args.veto_ratio_max)
    ratio.GetYaxis().SetTitle("N_{e}=0 / no veto")
    ratio.GetXaxis().SetTitle("m_{#mu#mu} [GeV]")
    ratio.GetYaxis().SetNdivisions(505)
    ratio.GetYaxis().SetTitleSize(0.100)
    ratio.GetYaxis().SetLabelSize(0.085)
    ratio.GetYaxis().SetTitleOffset(0.58)
    ratio.GetXaxis().SetTitleSize(0.120)
    ratio.GetXaxis().SetLabelSize(0.100)
    ratio.GetXaxis().SetTitleOffset(1.05)
    ratio.GetXaxis().SetRangeUser(args.mass_min, args.mass_max)
    ratio.Draw("E1")

    line = ROOT.TLine(args.mass_min, 1.0, args.mass_max, 1.0)
    line.SetLineStyle(2)
    line.SetLineColor(ROOT.kGray + 2)
    line.Draw()

    keep.extend([ratio, line])
    save_canvas(canvas, output_base, args.extensions)


def run_lepton_veto(ROOT, args, root_dir: str) -> List[str]:
    region = convener_region("BJet")

    # The current SKFlat ntuples do not contain reconstructed tau candidates.
    # Therefore this review plot deliberately shows only the electron-veto
    # comparison; no statement about a tau veto is inferred from it.
    hist_names = {
        "No veto": "ConvenerStudy_DileptonMass_NoVeto",
        "N_{e}=0": "ConvenerStudy_DileptonMass_ElectronVeto",
    }

    # DY is intentionally omitted from this diagnostic. The useful comparison
    # for the convener question is QCD, Top, and the representative signals.
    veto_processes = ["QCD", "Top"]
    entries: List[Tuple[str, Dict[str, Tuple[float, float]]]] = []
    mass_hists: List[Tuple[str, object, object, int]] = []

    process_colours = {
        "QCD": ROOT.kAzure + 2,
        "Top": ROOT.kOrange + 7,
    }

    for process in veto_processes:
        values = {}
        loaded = {}
        for label, hist_name in hist_names.items():
            hist = process_hist(
                ROOT,
                root_dir,
                process,
                region,
                hist_name,
            )
            loaded[label] = hist
            values[label] = integral_and_error(
                hist,
                args.mass_min,
                args.mass_max,
            )
        entries.append((PROCESS_LABELS.get(process, process), values))
        mass_hists.append(
            (
                PROCESS_LABELS.get(process, process),
                loaded["No veto"],
                loaded["N_{e}=0"],
                process_colours[process],
            )
        )

    sig_styles = signal_styles(ROOT)
    for idx, mass in enumerate(args.signal_masses):
        values = {}
        loaded = {}
        for label, hist_name in hist_names.items():
            hist = signal_hist(ROOT, root_dir, mass, region, hist_name)
            loaded[label] = hist
            values[label] = integral_and_error(
                hist,
                args.mass_min,
                args.mass_max,
            )
        signal_name = f"Z' {mass:g} GeV"
        entries.append((signal_name, values))
        colour, _ = sig_styles.get(
            float(mass),
            (ROOT.kRed + 1 + idx, 1),
        )
        mass_hists.append(
            (
                signal_name,
                loaded["No veto"],
                loaded["N_{e}=0"],
                colour,
            )
        )

    labels = list(hist_names)
    canvas = ROOT.TCanvas(NAMES.get("c_veto"), "", 1000, 800)
    canvas.SetLeftMargin(0.12)
    canvas.SetRightMargin(0.05)
    canvas.SetTopMargin(0.11)
    canvas.SetBottomMargin(0.20)

    axis = ROOT.TH1D(
        NAMES.get("axis_veto"),
        "",
        len(entries),
        0.5,
        len(entries) + 0.5,
    )
    axis.SetDirectory(0)
    axis.SetMinimum(0.92)
    axis.SetMaximum(1.005)
    axis.GetYaxis().SetTitle("Fraction of nominal selected yield")
    axis.GetXaxis().SetLabelSize(0.040)
    axis.GetYaxis().SetLabelSize(0.042)
    axis.GetYaxis().SetTitleSize(0.050)
    axis.GetYaxis().SetTitleOffset(1.10)

    for idx, (name, _) in enumerate(entries, 1):
        axis.GetXaxis().SetBinLabel(idx, name)
    axis.LabelsOption("v", "X")
    axis.Draw("AXIS")

    colours = [ROOT.kBlack, ROOT.kBlue + 1]
    markers = [20, 21]
    graphs = []

    for ivar, label in enumerate(labels):
        graph = ROOT.TGraph(len(entries))
        graph.SetName(NAMES.get("g_veto"))
        graph.SetLineColor(colours[ivar])
        graph.SetMarkerColor(colours[ivar])
        graph.SetMarkerStyle(markers[ivar])
        graph.SetMarkerSize(1.25)
        graph.SetLineWidth(2)

        for idx, (_, values) in enumerate(entries):
            denom = values["No veto"][0]
            num = values[label][0]
            ratio_value = num / denom if denom > 0.0 else 0.0
            graph.SetPoint(idx, idx + 1.0, ratio_value)

        graph.Draw("LP SAME")
        graphs.append((label, graph))

    line = ROOT.TLine(0.5, 1.0, len(entries) + 0.5, 1.0)
    line.SetLineStyle(2)
    line.SetLineColor(ROOT.kGray + 2)
    line.Draw()

    # Keep both the title and the compact two-entry legend inside the lower
    # half of the frame, but away from the vertical x-axis labels.
    legend = ROOT.TLegend(0.69, 0.25, 0.91, 0.36)
    legend.SetFillStyle(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.034)
    for label, graph in graphs:
        legend.AddEntry(graph, label, "lp")
    legend.Draw()

    keep = draw_cms_header(
        ROOT,
        canvas,
        args.era,
        f"Electron-veto impact, {args.mass_min:g}<m_{{#mu#mu}}<{args.mass_max:g} GeV",
        subtitle_x=0.16,
        subtitle_y=0.27,
    )
    keep.extend([axis, line, legend])
    keep.extend(graph for _, graph in graphs)

    out_base = os.path.join(args.output_dir, "electron_veto_survival")
    save_canvas(canvas, out_base, args.extensions)

    csv_path = os.path.join(args.output_dir, "electron_veto_yields.csv")
    ensure_dir(args.output_dir)
    with open(csv_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(
            [
                "process",
                "selection",
                "yield",
                "stat_error",
                "fraction_of_no_veto",
            ]
        )
        for process_name, values in entries:
            denom = values["No veto"][0]
            for label in labels:
                value, error = values[label]
                fraction = value / denom if denom > 0.0 else float("nan")
                writer.writerow(
                    [
                        process_name,
                        label,
                        f"{value:.12g}",
                        f"{error:.12g}",
                        f"{fraction:.12g}",
                    ]
                )

    outputs = [out_base, csv_path]

    # Produce one before/after mass-distribution plot per process. Separate
    # canvases avoid meaningless comparisons of very different absolute
    # normalisations and make the lower-panel survival fraction directly
    # interpretable.
    for label, before, after, colour in mass_hists:
        safe = (
            label.replace(" ", "_")
            .replace("'", "p")
            .replace(".", "p")
        )
        mass_base = os.path.join(
            args.output_dir,
            f"electron_veto_mass_{safe}",
        )
        draw_veto_mass_comparison(
            ROOT,
            args,
            label,
            before,
            after,
            colour,
            mass_base,
        )
        outputs.append(mass_base)

    print(f"[lepton-veto] wrote {csv_path}")
    for process_name, values in entries:
        denom = values["No veto"][0]
        value = values["N_{e}=0"][0]
        ratio_value = value / denom if denom > 0.0 else float("nan")
        print(
            f"[lepton-veto] {process_name}: "
            f"N_e=0 / no veto = {ratio_value:.5f}"
        )

    return outputs

def build_mc_shape_inputs(
    ROOT,
    args,
    root_dir: str,
    hist_name: str,
) -> List[Tuple[str, object, int, int]]:
    region = convener_region("BJet")
    colours = style_lines(ROOT)
    out: List[Tuple[str, object, int, int]] = []

    for process in args.processes:
        hist = process_hist(
            ROOT,
            root_dir,
            process,
            region,
            hist_name,
            required=(process != "Others"),
        )
        if hist is None:
            continue
        colour, line_style = colours.get(
            process,
            (ROOT.kGray + 2, len(out) + 1),
        )
        out.append(
            (
                PROCESS_LABELS.get(process, process),
                hist,
                colour,
                line_style,
            )
        )

    sig_styles = signal_styles(ROOT)
    fallback_signal_colours = [
        ROOT.kRed + 1,
        ROOT.kMagenta + 2,
        ROOT.kBlue + 1,
        ROOT.kViolet + 1,
        ROOT.kCyan + 2,
    ]
    for idx, mass in enumerate(args.signal_masses):
        hist = signal_hist(ROOT, root_dir, mass, region, hist_name)
        colour, line_style = sig_styles.get(
            float(mass),
            (
                fallback_signal_colours[
                    idx % len(fallback_signal_colours)
                ],
                1,
            ),
        )
        out.append(
            (
                f"Z' {mass:g} GeV",
                hist,
                colour,
                line_style,
            )
        )

    return out


def run_jet_composition(ROOT, args, root_dir: str) -> List[str]:
    outputs = []

    specs = [
        (
            "ConvenerStudy_DimuonJet_ConstituentMultiplicity",
            "dimuon_jet_constituent_multiplicity",
            "Dimuon-jet constituent multiplicity",
            "Constituent multiplicity",
            args.constituent_xmin,
            args.constituent_xmax,
            args.constituent_rebin,
        ),
        (
            "ConvenerStudy_DimuonJet_ChargedHadronFraction",
            "dimuon_jet_charged_hadron_fraction",
            "Dimuon-jet charged-hadron fraction",
            "Charged-hadron energy fraction",
            0.0,
            1.0,
            args.charged_fraction_rebin,
        ),
    ]

    for hist_name, file_tag, subtitle, x_title, xmin, xmax, rebin in specs:
        hists = build_mc_shape_inputs(
            ROOT,
            args,
            root_dir,
            hist_name,
        )
        out_base = os.path.join(args.output_dir, file_tag)
        draw_shape_overlay(
            ROOT,
            hists,
            era=args.era,
            subtitle=subtitle,
            x_title=x_title,
            output_base=out_base,
            extensions=args.extensions,
            xmin=xmin,
            xmax=xmax,
            rebin=rebin,
        )
        outputs.append(out_base)

    return outputs


def run_dijet_mass(ROOT, args, root_dir: str) -> List[str]:
    hists = build_mc_shape_inputs(
        ROOT,
        args,
        root_dir,
        "ConvenerStudy_DijetMass",
    )
    out_base = os.path.join(args.output_dir, "dijet_mass")
    draw_shape_overlay(
        ROOT,
        hists,
        era=args.era,
        subtitle="Dimuon-jet + tag-jet invariant mass",
        x_title="m(j_{#mu#mu}, j_{tag}) [GeV]",
        output_base=out_base,
        extensions=args.extensions,
        xmin=args.dijet_xmin,
        xmax=args.dijet_xmax,
        rebin=args.dijet_rebin,
    )
    return [out_base]


def subtract_non_dy(
    ROOT,
    root_dir: str,
    region: str,
    hist_name: str,
):
    data = process_hist(ROOT, root_dir, "data", region, hist_name)
    out = data.Clone(NAMES.get("data_sub"))
    out.SetDirectory(0)
    out.Sumw2()

    for process in ("QCD", "tt", "ST", "Others"):
        hist = process_hist(
            ROOT,
            root_dir,
            process,
            region,
            hist_name,
            required=(process != "Others"),
        )
        if hist:
            out.Add(hist, -1.0)

    return out


def ratio_graph(ROOT, numerator, denominator):
    graph = ROOT.TGraphErrors()
    graph.SetName(NAMES.get("ratio"))
    point = 0

    for ibin in range(1, numerator.GetNbinsX() + 1):
        x = float(numerator.GetXaxis().GetBinCenter(ibin))
        if 9.0 < x < 11.0:
            continue

        n = float(numerator.GetBinContent(ibin))
        ne = float(numerator.GetBinError(ibin))
        d = float(denominator.GetBinContent(ibin))
        de = float(denominator.GetBinError(ibin))

        if d <= 0.0:
            continue

        ratio = n / d
        variance = (ne / d) ** 2 + ((n * de) / (d * d)) ** 2

        graph.SetPoint(point, x, ratio)
        graph.SetPointError(
            point,
            0.5 * float(numerator.GetXaxis().GetBinWidth(ibin)),
            math.sqrt(max(0.0, variance)),
        )
        point += 1

    return graph


def run_dy_vr(ROOT, args, root_dir: str) -> List[str]:
    hist_name = "Dilepton_Mass"
    b_region = dy_vr_region("BJet")
    l_region = dy_vr_region("LightJet")

    b_data_sub_raw = subtract_non_dy(
        ROOT,
        root_dir,
        b_region,
        hist_name,
    )
    l_data_sub_raw = subtract_non_dy(
        ROOT,
        root_dir,
        l_region,
        hist_name,
    )

    b_dy_mc_raw = process_hist(
        ROOT,
        root_dir,
        "DY",
        b_region,
        hist_name,
    )
    l_dy_mc_raw = process_hist(
        ROOT,
        root_dir,
        "DY",
        l_region,
        hist_name,
    )

    b_nf, b_nf_err = integral_and_error(
        b_dy_mc_raw,
        args.nf_min,
        args.nf_max,
    )
    l_nf, l_nf_err = integral_and_error(
        l_dy_mc_raw,
        args.nf_min,
        args.nf_max,
    )

    if l_nf <= 0.0:
        raise PlotError(
            "DY light-jet MC integral is non-positive in the NF interval."
        )

    factor = b_nf / l_nf
    rel_var = 0.0
    if b_nf > 0.0:
        rel_var += (b_nf_err / b_nf) ** 2
    rel_var += (l_nf_err / l_nf) ** 2
    factor_err = abs(factor) * math.sqrt(max(0.0, rel_var))

    print(
        "[dy-vr] aMC@NLO transfer factor "
        f"({args.nf_min:g}<m<{args.nf_max:g} GeV) = "
        f"{factor:.8g} +/- {factor_err:.8g}"
    )

    b_data_sub = rebin_variable(
        b_data_sub_raw,
        args.dy_vr_edges,
        "b_data_sub_reb",
    )
    l_data_sub = rebin_variable(
        l_data_sub_raw,
        args.dy_vr_edges,
        "l_data_sub_reb",
    )
    prediction = l_data_sub.Clone(NAMES.get("pred"))
    prediction.SetDirectory(0)
    prediction.Scale(factor)

    # Propagate both the source-histogram uncertainty and the finite-MC
    # uncertainty of the transfer factor into the prediction error.
    for ibin in range(1, prediction.GetNbinsX() + 1):
        source_value = float(l_data_sub.GetBinContent(ibin))
        source_error = float(l_data_sub.GetBinError(ibin))
        variance = (
            (factor * source_error) ** 2
            + (source_value * factor_err) ** 2
        )
        prediction.SetBinError(ibin, math.sqrt(max(0.0, variance)))

    mask_upsilon_bin(b_data_sub)
    mask_upsilon_bin(prediction)

    canvas = ROOT.TCanvas(NAMES.get("c_dyvr"), "", 900, 900)

    upper = ROOT.TPad(NAMES.get("upper"), "", 0.0, 0.30, 1.0, 1.0)
    lower = ROOT.TPad(NAMES.get("lower"), "", 0.0, 0.00, 1.0, 0.30)

    upper.SetLeftMargin(0.13)
    upper.SetRightMargin(0.05)
    upper.SetTopMargin(0.11)
    upper.SetBottomMargin(0.03)

    lower.SetLeftMargin(0.13)
    lower.SetRightMargin(0.05)
    lower.SetTopMargin(0.04)
    lower.SetBottomMargin(0.34)

    upper.Draw()
    lower.Draw()

    upper.cd()

    b_data_sub.SetMarkerStyle(20)
    b_data_sub.SetMarkerSize(1.0)
    b_data_sub.SetMarkerColor(ROOT.kBlack)
    b_data_sub.SetLineColor(ROOT.kBlack)
    b_data_sub.SetLineWidth(2)

    prediction.SetMarkerStyle(24)
    prediction.SetMarkerSize(1.0)
    prediction.SetMarkerColor(ROOT.kRed + 1)
    prediction.SetLineColor(ROOT.kRed + 1)
    prediction.SetLineWidth(2)

    values = []
    for hist in (b_data_sub, prediction):
        for ibin in range(1, hist.GetNbinsX() + 1):
            values.append(
                float(hist.GetBinContent(ibin))
                + float(hist.GetBinError(ibin))
            )
    ymax = max(values) if values else 1.0
    ymin = min(
        [
            float(hist.GetBinContent(ibin))
            - float(hist.GetBinError(ibin))
            for hist in (b_data_sub, prediction)
            for ibin in range(1, hist.GetNbinsX() + 1)
        ]
        or [0.0]
    )

    span = max(ymax - min(0.0, ymin), 1.0)
    b_data_sub.SetMaximum(ymax + 0.35 * span)
    b_data_sub.SetMinimum(min(0.0, ymin - 0.10 * span))
    b_data_sub.GetYaxis().SetTitle("Background-subtracted events")
    b_data_sub.GetYaxis().SetTitleSize(0.055)
    b_data_sub.GetYaxis().SetLabelSize(0.045)
    b_data_sub.GetYaxis().SetTitleOffset(1.05)
    b_data_sub.GetXaxis().SetLabelSize(0.0)
    b_data_sub.Draw("E1")
    prediction.Draw("E1 SAME")

    legend = ROOT.TLegend(0.51, 0.66, 0.92, 0.84)
    legend.SetFillStyle(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.035)
    legend.AddEntry(
        b_data_sub,
        "B-jet-category data - non-DY",
        "lep",
    )
    legend.AddEntry(
        prediction,
        "Light-jet data - non-DY, #times F_{DY}^{VR}",
        "lep",
    )
    legend.Draw()

    keep = draw_cms_header(
        ROOT,
        upper,
        args.era,
        "#DeltaR(#mu, reference jet)>0.4 DY validation region",
    )
    keep.append(legend)

    nf_text = ROOT.TLatex()
    nf_text.SetNDC(True)
    nf_text.SetTextFont(42)
    nf_text.SetTextSize(0.032)
    nf_text.DrawLatex(
        0.16,
        0.79,
        f"F_{{DY}}^{{VR}} = {factor:.3f} #pm {factor_err:.3f} (MC stat.)",
    )
    keep.append(nf_text)

    lower.cd()
    ratio = ratio_graph(ROOT, b_data_sub, prediction)
    ratio.SetMarkerStyle(20)
    ratio.SetMarkerSize(0.9)
    ratio.SetLineColor(ROOT.kBlack)
    ratio.SetMarkerColor(ROOT.kBlack)

    axis = ROOT.TH1D(
        NAMES.get("ratio_axis"),
        "",
        len(args.dy_vr_edges) - 1,
        array("d", args.dy_vr_edges),
    )
    axis.SetDirectory(0)
    axis.SetMinimum(args.ratio_min)
    axis.SetMaximum(args.ratio_max)
    axis.GetYaxis().SetTitle("Target / prediction")
    axis.GetXaxis().SetTitle("m_{#mu#mu} [GeV]")
    axis.GetYaxis().SetNdivisions(505)
    axis.GetYaxis().SetTitleSize(0.105)
    axis.GetYaxis().SetLabelSize(0.090)
    axis.GetYaxis().SetTitleOffset(0.55)
    axis.GetXaxis().SetTitleSize(0.120)
    axis.GetXaxis().SetLabelSize(0.100)
    axis.GetXaxis().SetTitleOffset(1.05)
    axis.Draw("AXIS")
    ratio.Draw("P SAME")

    line = ROOT.TLine(
        args.dy_vr_edges[0],
        1.0,
        args.dy_vr_edges[-1],
        1.0,
    )
    line.SetLineStyle(2)
    line.SetLineColor(ROOT.kGray + 2)
    line.Draw()

    out_base = os.path.join(args.output_dir, "dy_validation_drgt0p4")
    save_canvas(canvas, out_base, args.extensions)

    csv_path = os.path.join(args.output_dir, "dy_validation_drgt0p4.csv")
    ensure_dir(args.output_dir)
    with open(csv_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(
            [
                "bin_low",
                "bin_high",
                "target_data_minus_nonDY",
                "target_error",
                "scaled_light_data_minus_nonDY",
                "prediction_error",
                "ratio",
            ]
        )
        for ibin in range(1, b_data_sub.GetNbinsX() + 1):
            low = float(b_data_sub.GetXaxis().GetBinLowEdge(ibin))
            high = low + float(b_data_sub.GetXaxis().GetBinWidth(ibin))
            if low < 11.0 and high > 9.0:
                continue
            target = float(b_data_sub.GetBinContent(ibin))
            target_err = float(b_data_sub.GetBinError(ibin))
            pred = float(prediction.GetBinContent(ibin))
            pred_err = float(prediction.GetBinError(ibin))
            ratio_value = target / pred if pred > 0.0 else float("nan")
            writer.writerow(
                [
                    f"{low:.12g}",
                    f"{high:.12g}",
                    f"{target:.12g}",
                    f"{target_err:.12g}",
                    f"{pred:.12g}",
                    f"{pred_err:.12g}",
                    f"{ratio_value:.12g}",
                ]
            )

    print(f"[dy-vr] wrote {csv_path}")
    return [out_base, csv_path]


def inspect_root(ROOT, filename: str) -> None:
    if not os.path.isfile(filename):
        raise PlotError(f"Missing ROOT file: {filename}")

    f = ROOT.TFile.Open(filename, "READ")
    if not f or f.IsZombie():
        raise PlotError(f"Could not open ROOT file: {filename}")

    def walk(directory, prefix=""):
        keys = directory.GetListOfKeys()
        for key in keys:
            name = key.GetName()
            obj = key.ReadObj()
            path = f"{prefix}{name}"
            if obj.InheritsFrom("TDirectory"):
                walk(obj, path + "/")
            else:
                print(path)

    walk(f)
    f.Close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot NIsoMuon convener-review diagnostic studies."
    )
    parser.add_argument(
        "--era",
        default="2018",
        help="Intended first-pass study era. Default: 2018.",
    )
    parser.add_argument(
        "--study",
        default="all",
        choices=[
            "all",
            "lepton-veto",
            "jet-composition",
            "dijet-mass",
            "dy-vr",
        ],
    )
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument(
        "--convener-collection",
        default="ConvenerStudy",
    )
    parser.add_argument(
        "--dy-vr-collection",
        default="DYValidationDRStudy",
    )
    parser.add_argument(
        "--convener-dir",
        default=None,
        help="Override <base>/<ConvenerStudy>/<era>.",
    )
    parser.add_argument(
        "--dy-vr-dir",
        default=None,
        help="Override <base>/<DYValidationDRStudy>/<era>.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Default: convener_studies/<era>. "
            "Plots are written as PDF and PNG."
        ),
    )
    parser.add_argument(
        "--extensions",
        default="pdf,png",
    )
    parser.add_argument(
        "--signal-masses",
        default="12,70",
        help=(
            "Comma-separated signal masses for MC shape comparisons. "
            "Default: 12,70."
        ),
    )
    parser.add_argument(
        "--processes",
        default="QCD,DY,Top",
        help="Comma-separated background processes: QCD,DY,Top,Others.",
    )
    parser.add_argument("--mass-min", type=float, default=11.0)
    parser.add_argument("--mass-max", type=float, default=80.0)
    parser.add_argument(
        "--veto-mass-rebin",
        type=int,
        default=50,
        help=(
            "Rebin factor for before/after electron-veto dimuon-mass plots. "
            "The native analyzer bin width is 20 MeV; default 50 gives 1 GeV."
        ),
    )
    parser.add_argument("--veto-ratio-min", type=float, default=0.85)
    parser.add_argument("--veto-ratio-max", type=float, default=1.02)

    parser.add_argument("--constituent-xmin", type=float, default=0.0)
    parser.add_argument("--constituent-xmax", type=float, default=60.0)
    parser.add_argument("--constituent-rebin", type=int, default=1)
    parser.add_argument("--charged-fraction-rebin", type=int, default=2)

    parser.add_argument("--dijet-xmin", type=float, default=0.0)
    parser.add_argument("--dijet-xmax", type=float, default=1000.0)
    parser.add_argument("--dijet-rebin", type=int, default=5)

    parser.add_argument("--nf-min", type=float, default=11.0)
    parser.add_argument("--nf-max", type=float, default=80.0)
    parser.add_argument(
        "--dy-vr-bins",
        default=",".join(str(x) for x in DEFAULT_DY_VR_EDGES),
        help="Comma-separated m_mumu bin edges for DY VR closure.",
    )
    parser.add_argument("--ratio-min", type=float, default=0.0)
    parser.add_argument("--ratio-max", type=float, default=2.0)

    parser.add_argument(
        "--inspect-file",
        default=None,
        help="Print all ROOT keys in this file and exit.",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    args.signal_masses = parse_float_list(args.signal_masses)
    args.processes = [
        item.strip()
        for item in args.processes.split(",")
        if item.strip()
    ]
    unknown = set(args.processes) - {"QCD", "DY", "Top", "Others"}
    if unknown:
        raise PlotError(
            "Unknown --processes entries: " + ", ".join(sorted(unknown))
        )

    args.extensions = [
        item.strip().lstrip(".")
        for item in args.extensions.split(",")
        if item.strip()
    ]
    args.dy_vr_edges = parse_float_list(args.dy_vr_bins)
    if len(args.dy_vr_edges) < 2:
        raise PlotError("--dy-vr-bins needs at least two edges.")
    if any(
        right <= left
        for left, right in zip(args.dy_vr_edges, args.dy_vr_edges[1:])
    ):
        raise PlotError("--dy-vr-bins must be strictly increasing.")

    if args.output_dir is None:
        args.output_dir = os.path.join(
            DEFAULT_OUTPUT_DIR,
            args.era,
        )

    ROOT = import_root()
    set_cms_style(ROOT)

    if args.inspect_file:
        inspect_root(ROOT, args.inspect_file)
        return 0

    convener_dir = collection_dir(
        args.base_dir,
        args.era,
        args.convener_collection,
        args.convener_dir,
    )
    dy_vr_dir = collection_dir(
        args.base_dir,
        args.era,
        args.dy_vr_collection,
        args.dy_vr_dir,
    )

    print(f"[INFO] ConvenerStudy input: {convener_dir}")
    print(f"[INFO] DYValidationDRStudy input: {dy_vr_dir}")
    print(f"[INFO] Output directory: {args.output_dir}")

    outputs = []

    if args.study in {"all", "lepton-veto"}:
        outputs.extend(run_lepton_veto(ROOT, args, convener_dir))

    if args.study in {"all", "jet-composition"}:
        outputs.extend(run_jet_composition(ROOT, args, convener_dir))

    if args.study in {"all", "dijet-mass"}:
        outputs.extend(run_dijet_mass(ROOT, args, convener_dir))

    if args.study in {"all", "dy-vr"}:
        outputs.extend(run_dy_vr(ROOT, args, dy_vr_dir))

    print("[DONE] Produced:")
    for output in outputs:
        print(f"  {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PlotError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
