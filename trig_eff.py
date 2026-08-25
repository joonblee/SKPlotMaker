#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Measure and plot the NIsoMuon high-pT single-muon trigger efficiency from the
current 2D TriggerEfficiency histograms.

Current inputs
--------------
  /data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13/NIsoMuon/
    TriggerEfficiency/<era>/DATA/data.root
    TriggerEfficiency/<era>/NIsoMuon_QCD_Inclusive.root
    TriggerEfficiency/<era>/NIsoMuon_tt.root

Current analyzer histograms
---------------------------
  TriggerEfficiency_DENOM/Probe_absEta_Pt___TriggerEfficiency_DENOM
  TriggerEfficiency_NUM/Probe_absEta_Pt___TriggerEfficiency_NUM

The 2D histograms use x=|eta| and y=probe-pT.  This script projects the requested
eta region onto pT, rebins to the exact binning used by the supplied
plot_trigeff.py reference, and draws the efficiency plot with the same ROOT
style, colours, TLatex strings, positions, legend, uncertainty band and pad
geometry as the supplied reference PDF.

Bare invocation prints this help page and does not run ROOT.

Examples
--------
  python3 trig_eff.py
  python3 trig_eff.py --year 2023
  python3 trig_eff.py --year 2016postVFP --object LeptonBarrel
  python3 trig_eff.py --year Run2
  python3 trig_eff.py --year Run3
  python3 trig_eff.py --year all
  python3 trig_eff.py --year 2023 --inspect-hists

With --object omitted, Barrel/Overlap/Endcap/Forward are all produced.
"""
from __future__ import annotations

import argparse
from array import array
import csv
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


VALID_ERAS: Tuple[str, ...] = (
    "2016preVFP",
    "2016postVFP",
    "2017",
    "2018",
    "2022",
    "2022EE",
    "2023",
    "2023BPix",
)

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

INPUT_BASE = Path(
    "/data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13/NIsoMuon"
)
PLOT_BASE = Path("/data6/Users/joonblee/PlotMaker/plots/TriggerEfficiency")

# Exact default binning in the supplied updated plot_trigeff.py.
REFERENCE_PT_EDGES: Tuple[float, ...] = (
    40,42,44,46,48,50,52,55,60,70,80,120,200,500,
)

DEFAULT_OBJECTS: Tuple[str, ...] = (
    "LeptonBarrel",
    "LeptonOverlap",
    "LeptonEndcap",
    "LeptonForward",
)

ETA_RANGE: Dict[str, Tuple[float, float]] = {
    "LeptonBarrel": (0.0, 0.9),
    "LeptonOverlap": (0.9, 1.2),
    "LeptonEndcap": (1.2, 2.1),
    "LeptonForward": (2.1, 2.4),
    "LeptonAll": (0.0, 2.4),
}

REGION_INFO: Dict[str, Tuple[str, str, str]] = {
    "LeptonBarrel": ("barrel", "Barrel", "|#eta| < 0.9"),
    "LeptonOverlap": ("overlap", "Overlap", "0.9 < |#eta| < 1.2"),
    "LeptonEndcap": ("endcap", "Endcap", "1.2 < |#eta| < 2.1"),
    "LeptonForward": ("forward", "Forward", "2.1 < |#eta| < 2.4"),
    "LeptonAll": ("all", "All", "|#eta| < 2.4"),
}

# Run-2 values follow the supplied hadd_trigeff.py/plot_trigeff.py reference.
# Run-3 values are retained from the current PlotMaker setup.
LUMI_LABEL: Dict[str, str] = {
    "2016preVFP": "19.5 fb^{-1} (13 TeV)",
    "2016postVFP": "16.8 fb^{-1} (13 TeV)",
    "2017": "41.5 fb^{-1} (13 TeV)",
    "2018": "59.8 fb^{-1} (13 TeV)",
    "2022": "7.98 fb^{-1} (13.6 TeV)",
    "2022EE": "26.67 fb^{-1} (13.6 TeV)",
    "2023": "17.7 fb^{-1} (13.6 TeV)",
    "2023BPix": "9.5 fb^{-1} (13.6 TeV)",
}

# Style constants copied from the supplied updated plot_trigeff.py so that the
# output geometry/text placement matches trig_eff_barrel_2016postVFP.pdf.
X_MIN = 40.0
X_MAX = 500.0
Y_MIN = 0.0
Y_MAX = 1.05
RATIO_MIN = 0.8
RATIO_MAX = 1.2
MC_BAND_FILL_STYLE = 3004
ETA_EDGE_TOL = 1.0e-5
PT_EDGE_TOL = 1.0e-4
SELECTION_LABEL = "OS, POGMedium, tight jet, DeepJet medium b tag"


class NameFactory:
    def __init__(self) -> None:
        self.counter = 0

    def unique(self, prefix: str) -> str:
        self.counter += 1
        cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in prefix)
        return f"{cleaned}_{self.counter}"


_NAMES = NameFactory()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trig_eff.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Measure the NIsoMuon high-pT muon-trigger efficiency using the current "
            "2D TriggerEfficiency histograms.\n\n"
            "No arguments prints this page and exits.\n"
            "The plot style and default pT binning reproduce the supplied "
            "plot_trigeff.py / reference PDF."
        ),
        epilog=(
            "Year groups:\n"
            "  Run2 = 2016preVFP, 2016postVFP, 2017, 2018\n"
            "  Run3 = 2022, 2022EE, 2023, 2023BPix\n"
            "  all  = all eight eras, plotted separately\n\n"
            "Object behaviour:\n"
            "  default/all = Barrel, Overlap, Endcap, Forward\n"
            "  LeptonAll   = one inclusive |eta| < 2.4 projection\n\n"
            "Reference pT binning:\n"
            "  0,10,20,30,35,40,42,44,46,48,50,52,55,60,70,80,120,200,500 GeV\n"
            "  displayed range: 40--500 GeV\n\n"
            "Main examples:\n"
            "  python3 trig_eff.py\n"
            "  python3 trig_eff.py --year 2023\n"
            "  python3 trig_eff.py --year 2016postVFP --object LeptonBarrel\n"
            "  python3 trig_eff.py --year Run2\n"
            "  python3 trig_eff.py --year Run3\n"
            "  python3 trig_eff.py --year all\n"
            "  python3 trig_eff.py --year 2023 --inspect-hists\n"
            "  python3 trig_eff.py --year 2023 --qcd-rescale 0.58\n\n"
            "Default QCD extra rescale is 1.0, matching the supplied "
            "hadd_trigeff.py + plot_trigeff.py workflow."
        ),
    )
    parser.add_argument(
        "--year",
        "--era",
        dest="year",
        choices=(*VALID_ERAS, "Run2", "Run3", "all"),
        default="2023",
        help="single era, Run2, Run3, or all; default: %(default)s",
    )
    parser.add_argument(
        "--object",
        choices=("all", *DEFAULT_OBJECTS, "LeptonAll"),
        default="all",
        help="eta region; default 'all' draws Barrel/Overlap/Endcap/Forward",
    )
    parser.add_argument(
        "--qcd-rescale",
        type=float,
        default=1.0,
        help=(
            "additional QCD normalisation factor after the merged input; "
            "default: %(default)s"
        ),
    )
    parser.add_argument(
        "--inspect-hists",
        action="store_true",
        help="print/check expected ROOT keys without plotting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print resolved inputs/outputs without importing ROOT",
    )
    parser.add_argument(
        "--quiet-bin-values",
        action="store_true",
        help="do not print final bin-by-bin data/MC efficiencies and SFs",
    )
    return parser


def selected_eras(value: str) -> Sequence[str]:
    if value == "Run2":
        return RUN2_ERAS
    if value == "Run3":
        return RUN3_ERAS
    if value == "all":
        return VALID_ERAS
    return (value,)


def selected_objects(value: str) -> Sequence[str]:
    return DEFAULT_OBJECTS if value == "all" else (value,)


def import_root():
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        import ROOT  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Could not import PyROOT. Load a ROOT/CMSSW environment before running. "
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
    ROOT.gStyle.SetOptTitle(0)
    try:
        ROOT.TGaxis.SetMaxDigits(4)
    except Exception:
        pass
    return ROOT


def histogram_paths() -> Tuple[str, str]:
    return (
        "TriggerEfficiency_DENOM/Probe_absEta_Pt___TriggerEfficiency_DENOM",
        "TriggerEfficiency_NUM/Probe_absEta_Pt___TriggerEfficiency_NUM",
    )


def era_input_dir(era: str) -> Path:
    return INPUT_BASE / "TriggerEfficiency" / era


def era_files(era: str) -> Dict[str, Path]:
    directory = era_input_dir(era)
    return {
        "Data": directory / "DATA" / "data.root",
        "QCD": directory / "NIsoMuon_QCD_Inclusive.root",
        "Top": directory / "NIsoMuon_tt.root",
    }


def output_dir(era: str) -> Path:
    return PLOT_BASE / era


def print_resolved_configuration(args: argparse.Namespace, era: str) -> None:
    denominator, numerator = histogram_paths()
    eta_low, eta_high = ETA_RANGE[args.object]
    print(f"[ERA] {era}")
    print(f"[INPUT DIR] {era_input_dir(era)}")
    for label, path in era_files(era).items():
        print(f"[INPUT] {label:4s}: {path}")
    print(f"[HIST] denominator: {denominator}")
    print(f"[HIST] numerator  : {numerator}")
    print(f"[ETA] {args.object}: {eta_low:g} <= |eta| < {eta_high:g}")
    print("[PT BINS] " + ",".join(f"{x:g}" for x in REFERENCE_PT_EDGES))
    print(f"[DISPLAY] {X_MIN:g} <= pT <= {X_MAX:g} GeV")
    print(f"[QCD RESCALE] {args.qcd_rescale:g}")
    print(f"[OUTPUT DIR] {output_dir(era)}")


def open_root_file(ROOT, path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing ROOT file: {path}")
    root_file = ROOT.TFile.Open(str(path), "READ")
    if not root_file or root_file.IsZombie():
        if root_file:
            root_file.Close()
        raise OSError(f"Could not open ROOT file: {path}")
    return root_file


def clone_hist(hist, name: str):
    if not hist:
        return None
    out = hist.Clone(name)
    out.SetDirectory(0)
    if out.GetSumw2N() == 0:
        out.Sumw2()
    out.SetTitle("")
    return out


def get_histogram(root_file, path: str, label: str):
    print(f"[HIST] {label}:{path} -> ", end="", flush=True)
    hist = root_file.Get(path)
    if not hist:
        print("MISSING")
        raise KeyError(f"Missing histogram {path} in {label}")
    if not hist.InheritsFrom("TH2"):
        print(f"WRONG TYPE ({hist.ClassName()})")
        raise TypeError(
            f"Expected TH2 histogram {path} in {label}, got {hist.ClassName()}"
        )
    print(f"FOUND ({hist.ClassName()})")
    return clone_hist(hist, _NAMES.unique(label + "_hist2d"))


def project_probe_pt(hist2d, object_name: str, name: str):
    eta_low, eta_high = ETA_RANGE[object_name]
    xaxis = hist2d.GetXaxis()
    selected_bins: List[int] = []

    # TH2F stores nominal eta edges with float32 round-off.  Select bins by centre.
    for ibin in range(1, xaxis.GetNbins() + 1):
        centre = float(xaxis.GetBinCenter(ibin))
        if eta_low <= centre < eta_high:
            selected_bins.append(ibin)

    if not selected_bins:
        raise ValueError(
            f"No source eta bins found for {object_name}: [{eta_low},{eta_high})"
        )

    first_bin = min(selected_bins)
    last_bin = max(selected_bins)
    actual_low = float(xaxis.GetBinLowEdge(first_bin))
    actual_high = float(xaxis.GetBinUpEdge(last_bin))
    if not math.isclose(actual_low, eta_low, rel_tol=0.0, abs_tol=ETA_EDGE_TOL) or not math.isclose(
        actual_high, eta_high, rel_tol=0.0, abs_tol=ETA_EDGE_TOL
    ):
        raise ValueError(
            f"Requested eta range [{eta_low},{eta_high}) is incompatible with source "
            f"eta bins [{actual_low},{actual_high})"
        )

    print(
        f"[ETA PROJECTION] {object_name}: source x bins {first_bin}-{last_bin}, "
        f"stored edges [{actual_low:.9g},{actual_high:.9g})"
    )

    projected = hist2d.ProjectionY(name, first_bin, last_bin, "e")
    if not projected:
        raise RuntimeError(f"ProjectionY failed for {object_name}")
    projected.SetDirectory(0)
    if projected.GetSumw2N() == 0:
        projected.Sumw2()
    projected.SetTitle("")
    return projected


def iter_key_paths(ROOT, directory, prefix: str = "") -> Iterable[str]:
    keys = directory.GetListOfKeys()
    if not keys:
        return
    for key in keys:
        name = key.GetName()
        path = f"{prefix}/{name}" if prefix else name
        klass = ROOT.gROOT.GetClass(key.GetClassName())
        if klass and klass.InheritsFrom("TDirectory"):
            subdir = directory.GetDirectory(name)
            if subdir:
                yield from iter_key_paths(ROOT, subdir, path)
        else:
            yield path


def inspect_file(ROOT, root_file, label: str, expected: Sequence[str]) -> None:
    print(f"\n[INSPECT] {label}: {root_file.GetName()}")
    for path in expected:
        state = "FOUND" if root_file.Get(path) else "MISSING"
        print(f"  [{state}] {path}")
    print("  Matching keys:")
    matches = [
        path
        for path in iter_key_paths(ROOT, root_file)
        if "TriggerEfficiency" in path and "Probe_absEta_Pt" in path
    ]
    for path in matches[:300]:
        print(f"    {path}")
    if not matches:
        print("    (none)")


def source_edges(hist) -> List[float]:
    axis = hist.GetXaxis()
    edges = [float(axis.GetBinLowEdge(1))]
    edges.extend(float(axis.GetBinUpEdge(i)) for i in range(1, axis.GetNbins() + 1))
    return edges


def rebin_to_reference(ROOT, hist, name: str):
    """
    Merge source pT bins into the exact reference binning without ever splitting
    a source bin.  Target edges outside the source range (notably 0 GeV when the
    analyzer starts at 10 GeV) are allowed and remain empty.
    """
    src_edges = source_edges(hist)
    src_min = src_edges[0]
    src_max = src_edges[-1]

    missing: List[float] = []
    for edge in REFERENCE_PT_EDGES:
        # An edge outside the populated source axis does not require a source edge.
        if edge < src_min - PT_EDGE_TOL or edge > src_max + PT_EDGE_TOL:
            continue
        if not any(math.isclose(edge, src, rel_tol=0.0, abs_tol=PT_EDGE_TOL) for src in src_edges):
            missing.append(edge)

    if missing:
        raise ValueError(
            "Cannot reproduce the supplied plot_trigeff.py pT binning because the "
            "source histogram would need to be split at: "
            + ", ".join(f"{x:g}" for x in missing)
            + " GeV. Source edges are: "
            + ", ".join(f"{x:g}" for x in src_edges)
            + ". Update RunTriggerEfficiency ptEdges and rerun TriggerEfficiency production."
        )

    target = ROOT.TH1D(
        name,
        "",
        len(REFERENCE_PT_EDGES) - 1,
        array("d", REFERENCE_PT_EDGES),
    )
    target.SetDirectory(0)
    target.Sumw2()

    for ibin in range(1, hist.GetNbinsX() + 1):
        low = float(hist.GetXaxis().GetBinLowEdge(ibin))
        high = float(hist.GetXaxis().GetBinUpEdge(ibin))
        if high <= REFERENCE_PT_EDGES[0] + PT_EDGE_TOL:
            continue
        if low >= REFERENCE_PT_EDGES[-1] - PT_EDGE_TOL:
            continue

        centre = float(hist.GetXaxis().GetBinCenter(ibin))
        out_bin = int(target.GetXaxis().FindFixBin(centre))
        if out_bin < 1 or out_bin > target.GetNbinsX():
            continue

        out_low = float(target.GetXaxis().GetBinLowEdge(out_bin))
        out_high = float(target.GetXaxis().GetBinUpEdge(out_bin))
        if low < out_low - PT_EDGE_TOL or high > out_high + PT_EDGE_TOL:
            raise ValueError(
                f"Source bin [{low:g},{high:g}) straddles reference bin "
                f"[{out_low:g},{out_high:g}); splitting source bins is forbidden."
            )

        content = float(target.GetBinContent(out_bin)) + float(hist.GetBinContent(ibin))
        error2 = float(target.GetBinError(out_bin)) ** 2 + float(hist.GetBinError(ibin)) ** 2
        target.SetBinContent(out_bin, content)
        target.SetBinError(out_bin, math.sqrt(max(0.0, error2)))

    return target


def add_histograms(first, second, name: str):
    out = clone_hist(first, name)
    out.Add(second)
    return out


def efficiency(num, den, name: str):
    eff = clone_hist(num, name)
    eff.Divide(num, den, 1.0, 1.0, "B")
    eff.SetStats(0)
    eff.SetMinimum(0.0)
    eff.SetMaximum(1.1)
    return eff


def data_over_mc_ratio(data_eff, mc_eff, name: str):
    """Data/MC ratio points with data uncertainty only; MC error is a band."""
    out = clone_hist(data_eff, name)
    for ibin in range(1, out.GetNbinsX() + 1):
        d = float(data_eff.GetBinContent(ibin))
        ed = float(data_eff.GetBinError(ibin))
        m = float(mc_eff.GetBinContent(ibin))
        if m > 0.0:
            out.SetBinContent(ibin, d / m)
            out.SetBinError(ibin, ed / m)
        else:
            out.SetBinContent(ibin, 0.0)
            out.SetBinError(ibin, 0.0)
    return out


def weighted_efficiency_sigma(num, den, eff, ibin: int) -> float:
    """
    Weighted-binomial uncertainty used by the supplied plot_trigeff.py.

      Var(e) = [Var(N) + e^2 Var(D) - 2 e Cov(N,D)] / D^2,
      Cov(N,D) = Var(N), because numerator is a subset of denominator.
    """
    d = float(den.GetBinContent(ibin))
    e = float(eff.GetBinContent(ibin))
    if d <= 0.0:
        return 0.0

    v_num = float(num.GetBinError(ibin)) ** 2
    v_den = float(den.GetBinError(ibin)) ** 2
    var = ((1.0 - 2.0 * e) * v_num + e * e * v_den) / (d * d)

    if var < 0.0:
        n_eff = d * d / v_den if v_den > 0.0 else d
        var = e * max(0.0, 1.0 - e) / n_eff if n_eff > 0.0 else 0.0
    return math.sqrt(max(0.0, var))


def make_efficiency_band(ROOT, eff, num, den, name: str, color: int):
    xs = array("d")
    ys = array("d")
    exl = array("d")
    exh = array("d")
    eyl = array("d")
    eyh = array("d")

    for ibin in range(1, eff.GetNbinsX() + 1):
        low = float(eff.GetXaxis().GetBinLowEdge(ibin))
        high = float(eff.GetXaxis().GetBinUpEdge(ibin))
        if high <= X_MIN or low >= X_MAX:
            continue
        draw_low = max(low, X_MIN)
        draw_high = min(high, X_MAX)
        if draw_low <= 0.0 or draw_high <= 0.0:
            continue

        # Reference uses geometric centre for boxes on the log-x axis.
        x = math.sqrt(draw_low * draw_high)
        y = float(eff.GetBinContent(ibin))
        if den.GetBinContent(ibin) <= 0.0:
            continue
        sigma = weighted_efficiency_sigma(num, den, eff, ibin)
        y_low = max(0.0, y - sigma)
        y_high = min(1.0, y + sigma)

        xs.append(x)
        ys.append(y)
        exl.append(x - draw_low)
        exh.append(draw_high - x)
        eyl.append(max(0.0, y - y_low))
        eyh.append(max(0.0, y_high - y))

    if len(xs) == 0:
        return None

    graph = ROOT.TGraphAsymmErrors(len(xs), xs, ys, exl, exh, eyl, eyh)
    graph.SetName(name)
    graph.SetTitle("")
    graph.SetFillColor(color)
    graph.SetFillStyle(MC_BAND_FILL_STYLE)
    graph.SetLineColor(color)
    graph.SetLineWidth(0)
    return graph


def make_ratio_band(ROOT, eff, num, den, name: str, color: int):
    xs = array("d")
    ys = array("d")
    exl = array("d")
    exh = array("d")
    eyl = array("d")
    eyh = array("d")

    for ibin in range(1, eff.GetNbinsX() + 1):
        low = float(eff.GetXaxis().GetBinLowEdge(ibin))
        high = float(eff.GetXaxis().GetBinUpEdge(ibin))
        if high <= X_MIN or low >= X_MAX:
            continue
        draw_low = max(low, X_MIN)
        draw_high = min(high, X_MAX)
        if draw_low <= 0.0 or draw_high <= 0.0:
            continue

        x = math.sqrt(draw_low * draw_high)
        y = float(eff.GetBinContent(ibin))
        if y <= 0.0 or den.GetBinContent(ibin) <= 0.0:
            continue
        sigma = weighted_efficiency_sigma(num, den, eff, ibin)
        rel = sigma / y if y > 0.0 else 0.0

        xs.append(x)
        ys.append(1.0)
        exl.append(x - draw_low)
        exh.append(draw_high - x)
        eyl.append(rel)
        eyh.append(rel)

    if len(xs) == 0:
        return None

    graph = ROOT.TGraphAsymmErrors(len(xs), xs, ys, exl, exh, eyl, eyh)
    graph.SetName(name)
    graph.SetTitle("")
    graph.SetFillColor(color)
    graph.SetFillStyle(MC_BAND_FILL_STYLE)
    graph.SetLineColor(color)
    graph.SetLineWidth(0)
    return graph


def set_eff_style(hist, color: int, marker: int, line_style: int = 1):
    hist.SetStats(0)
    hist.SetTitle("")
    hist.SetLineColor(color)
    hist.SetMarkerColor(color)
    hist.SetMarkerStyle(marker)
    hist.SetMarkerSize(1.0)
    hist.SetLineWidth(2)
    hist.SetLineStyle(line_style)
    hist.SetFillStyle(0)
    return hist


def draw_label(ROOT, text: str, x: float, y: float, size: float, align: int):
    latex = ROOT.TLatex()
    latex.SetNDC(True)
    latex.SetTextFont(42)
    latex.SetTextSize(size)
    latex.SetTextAlign(align)
    latex.DrawLatex(x, y, text)
    return latex


def configure_log_axis(axis) -> None:
    try:
        axis.SetMoreLogLabels(True)
        axis.SetNoExponent(True)
    except Exception:
        pass
    axis.SetNdivisions(510)


def configure_axes(hist) -> None:
    hist.SetTitle("")
    hist.GetXaxis().SetRangeUser(X_MIN, X_MAX)
    hist.GetYaxis().SetRangeUser(Y_MIN, Y_MAX)
    hist.GetYaxis().SetTitle("Trigger efficiency")
    hist.GetYaxis().SetTitleSize(0.055)
    hist.GetYaxis().SetTitleOffset(0.95)
    hist.GetYaxis().SetLabelSize(0.045)
    hist.GetXaxis().SetTitle("Probe muon p_{T} [GeV]")
    hist.GetXaxis().SetTitleSize(0.0)
    hist.GetXaxis().SetLabelSize(0.0)
    configure_log_axis(hist.GetXaxis())


def configure_ratio_axes(hist) -> None:
    hist.SetTitle("")
    hist.GetXaxis().SetRangeUser(X_MIN, X_MAX)
    hist.GetYaxis().SetRangeUser(RATIO_MIN, RATIO_MAX)
    hist.GetYaxis().SetTitle("Data / MC")
    hist.GetYaxis().CenterTitle(True)
    hist.GetYaxis().SetTitleSize(0.105)
    hist.GetYaxis().SetTitleOffset(0.43)
    hist.GetYaxis().SetLabelSize(0.085)
    hist.GetYaxis().SetNdivisions(505)
    hist.GetXaxis().SetTitle("Probe muon p_{T} [GeV]")
    hist.GetXaxis().SetTitleSize(0.105)
    hist.GetXaxis().SetTitleOffset(1.05)
    hist.GetXaxis().SetLabelSize(0.085)
    configure_log_axis(hist.GetXaxis())


def draw_reference_efficiency(
    ROOT,
    *,
    era: str,
    object_name: str,
    data_eff,
    mc_eff,
    mc_num,
    mc_den,
    output_base: Path,
) -> None:
    region_key, region_label, eta_text = REGION_INFO[object_name]
    data_ratio = data_over_mc_ratio(data_eff, mc_eff, _NAMES.unique("ratio_data_over_mc"))

    # Exact reference colours/styles.
    set_eff_style(data_eff, ROOT.kBlack, 20)
    set_eff_style(mc_eff, ROOT.kRed + 1, 1)
    data_ratio.SetStats(0)
    data_ratio.SetMarkerStyle(20)
    data_ratio.SetMarkerSize(0.9)
    data_ratio.SetLineColor(ROOT.kBlack)
    data_ratio.SetMarkerColor(ROOT.kBlack)
    data_ratio.SetTitle("")

    mc_band = make_efficiency_band(
        ROOT,
        mc_eff,
        mc_num,
        mc_den,
        _NAMES.unique("mc_eff_band"),
        ROOT.kRed + 1,
    )
    ratio_band = make_ratio_band(
        ROOT,
        mc_eff,
        mc_num,
        mc_den,
        _NAMES.unique("mc_ratio_band"),
        ROOT.kRed + 1,
    )

    # Exact reference canvas and pad geometry.
    canvas = ROOT.TCanvas(
        _NAMES.unique(f"c_trig_eff_{region_key}_{era}"),
        "",
        950,
        950,
    )
    canvas.cd()

    upper = ROOT.TPad(_NAMES.unique("upper"), "", 0.0, 0.30, 1.0, 1.0)
    lower = ROOT.TPad(_NAMES.unique("lower"), "", 0.0, 0.00, 1.0, 0.30)
    upper.SetLeftMargin(0.13)
    upper.SetRightMargin(0.04)
    upper.SetTopMargin(0.08)
    upper.SetBottomMargin(0.03)
    upper.SetTickx(1)
    upper.SetTicky(1)
    lower.SetLeftMargin(0.13)
    lower.SetRightMargin(0.04)
    lower.SetTopMargin(0.04)
    lower.SetBottomMargin(0.34)
    lower.SetTickx(1)
    lower.SetTicky(1)
    upper.SetLogx()
    lower.SetLogx()
    upper.Draw()
    lower.Draw()

    upper.cd()
    configure_axes(data_eff)
    data_eff.Draw("E1")
    if mc_band is not None:
        mc_band.Draw("2 SAME")
    mc_eff.Draw("HIST SAME")
    data_eff.Draw("E1 SAME")
    upper.SetGridx(True)
    upper.SetGridy(True)
    upper.RedrawAxis()

    # Exact reference legend box/text.
    leg = ROOT.TLegend(0.55, 0.18, 0.92, 0.39)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.034)
    leg.AddEntry(data_eff, "SingleMuon data", "pe")
    if mc_band is not None:
        leg.AddEntry(mc_band, "Top + QCD MC stat.", "f")
    else:
        leg.AddEntry(mc_eff, "Top + QCD MC", "l")
    leg.Draw()

    # Exact reference TLatex strings, font sizes and NDC positions.
    latex_objs = [
        draw_label(ROOT, "#bf{CMS} #it{Preliminary}", 0.14, 0.965, 0.043, 13),
        draw_label(ROOT, LUMI_LABEL.get(era, era), 0.96, 0.965, 0.037, 33),
        draw_label(ROOT, f"{region_label}: {eta_text}", 0.40, 0.56, 0.036, 13),
        draw_label(ROOT, SELECTION_LABEL, 0.40, 0.50, 0.030, 13),
    ]

    lower.cd()
    configure_ratio_axes(data_ratio)
    data_ratio.Draw("E1")
    if ratio_band is not None:
        ratio_band.Draw("2 SAME")
    unity = ROOT.TLine(X_MIN, 1.0, X_MAX, 1.0)
    unity.SetLineColor(ROOT.kRed + 1)
    unity.SetLineStyle(2)
    unity.SetLineWidth(2)
    unity.Draw("SAME")
    data_ratio.Draw("E1 SAME")
    lower.SetGridx(True)
    lower.SetGridy(True)
    lower.RedrawAxis()

    keep = [upper, lower, leg, unity, data_ratio, *latex_objs]
    if mc_band is not None:
        keep.append(mc_band)
    if ratio_band is not None:
        keep.append(ratio_band)
    canvas._keepalive = keep

    output_base.parent.mkdir(parents=True, exist_ok=True)
    canvas.Modified()
    canvas.Update()
    for suffix in (".png", ".pdf"):
        path = output_base.with_suffix(suffix)
        canvas.SaveAs(str(path))
        print(f"[SAVE] {path}")


def write_measurement_outputs(
    ROOT,
    *,
    era: str,
    object_name: str,
    data_den,
    data_num,
    qcd_den,
    qcd_num,
    top_den,
    top_num,
    mc_den,
    mc_num,
    data_eff,
    mc_eff,
) -> Tuple[Path, Path]:
    outdir = output_dir(era)
    outdir.mkdir(parents=True, exist_ok=True)
    region_key = REGION_INFO[object_name][0]
    root_path = outdir / f"trigger_efficiency_{region_key}_{era}.root"
    csv_path = outdir / f"trigger_efficiency_{region_key}_{era}.csv"

    data_ratio = data_over_mc_ratio(data_eff, mc_eff, _NAMES.unique("sf_output"))

    fout = ROOT.TFile.Open(str(root_path), "RECREATE")
    if not fout or fout.IsZombie():
        raise OSError(f"Could not create ROOT output: {root_path}")
    try:
        for name, hist in (
            ("data_denominator", data_den),
            ("data_numerator", data_num),
            ("qcd_denominator", qcd_den),
            ("qcd_numerator", qcd_num),
            ("top_denominator", top_den),
            ("top_numerator", top_num),
            ("mc_denominator", mc_den),
            ("mc_numerator", mc_num),
            ("data_efficiency", data_eff),
            ("mc_efficiency", mc_eff),
            ("data_over_mc", data_ratio),
        ):
            fout.cd()
            hist.Write(name, ROOT.TObject.kOverwrite)
        fout.Write()
    finally:
        fout.Close()

    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "pt_low",
                "pt_high",
                "data_den",
                "data_num",
                "mc_den",
                "mc_num",
                "data_eff",
                "data_eff_err",
                "mc_eff",
                "mc_eff_stat",
                "data_over_mc",
                "data_over_mc_data_err",
            ]
        )
        for ibin in range(1, data_eff.GetNbinsX() + 1):
            low = float(data_eff.GetXaxis().GetBinLowEdge(ibin))
            high = float(data_eff.GetXaxis().GetBinUpEdge(ibin))
            if high <= X_MIN or low >= X_MAX:
                continue
            mc_err = weighted_efficiency_sigma(mc_num, mc_den, mc_eff, ibin)
            writer.writerow(
                [
                    low,
                    high,
                    data_den.GetBinContent(ibin),
                    data_num.GetBinContent(ibin),
                    mc_den.GetBinContent(ibin),
                    mc_num.GetBinContent(ibin),
                    data_eff.GetBinContent(ibin),
                    data_eff.GetBinError(ibin),
                    mc_eff.GetBinContent(ibin),
                    mc_err,
                    data_ratio.GetBinContent(ibin),
                    data_ratio.GetBinError(ibin),
                ]
            )

    print(f"[SAVE] {root_path}")
    print(f"[SAVE] {csv_path}")
    return root_path, csv_path


def print_bin_values(data_eff, mc_eff, mc_num, mc_den) -> None:
    print("\n[BIN-BY-BIN RESULT]")
    print("  {:>12s} {:>18s} {:>18s} {:>18s}".format("pT [GeV]", "eff(data)", "eff(MC)", "Data/MC"))
    ratio = data_over_mc_ratio(data_eff, mc_eff, _NAMES.unique("ratio_print"))
    for ibin in range(1, data_eff.GetNbinsX() + 1):
        low = float(data_eff.GetXaxis().GetBinLowEdge(ibin))
        high = float(data_eff.GetXaxis().GetBinUpEdge(ibin))
        if high <= X_MIN or low >= X_MAX:
            continue
        mc_err = weighted_efficiency_sigma(mc_num, mc_den, mc_eff, ibin)
        print(
            f"  {low:5.0f}-{high:<5.0f} "
            f"{data_eff.GetBinContent(ibin):8.5f} +/- {data_eff.GetBinError(ibin):7.5f} "
            f"{mc_eff.GetBinContent(ibin):8.5f} +/- {mc_err:7.5f} "
            f"{ratio.GetBinContent(ibin):8.5f} +/- {ratio.GetBinError(ibin):7.5f}"
        )


def run_one_era(ROOT, args: argparse.Namespace, era: str) -> int:
    print("\n" + "=" * 78)
    print(f"[TRIGGER EFFICIENCY] {era} / {args.object}")
    print("=" * 78)
    print_resolved_configuration(args, era)

    paths = era_files(era)
    handles = []
    try:
        files = {label: open_root_file(ROOT, path) for label, path in paths.items()}
        handles.extend(files.values())
        denominator_path, numerator_path = histogram_paths()

        if args.inspect_hists:
            for label, root_file in files.items():
                inspect_file(ROOT, root_file, label, (denominator_path, numerator_path))
            return 0

        data_den_2d = get_histogram(files["Data"], denominator_path, "Data denominator")
        data_num_2d = get_histogram(files["Data"], numerator_path, "Data numerator")
        qcd_den_2d = get_histogram(files["QCD"], denominator_path, "QCD denominator")
        qcd_num_2d = get_histogram(files["QCD"], numerator_path, "QCD numerator")
        top_den_2d = get_histogram(files["Top"], denominator_path, "Top denominator")
        top_num_2d = get_histogram(files["Top"], numerator_path, "Top numerator")

        data_den = rebin_to_reference(ROOT, project_probe_pt(data_den_2d, args.object, _NAMES.unique("data_den")), _NAMES.unique("data_den_ref"))
        data_num = rebin_to_reference(ROOT, project_probe_pt(data_num_2d, args.object, _NAMES.unique("data_num")), _NAMES.unique("data_num_ref"))
        qcd_den = rebin_to_reference(ROOT, project_probe_pt(qcd_den_2d, args.object, _NAMES.unique("qcd_den")), _NAMES.unique("qcd_den_ref"))
        qcd_num = rebin_to_reference(ROOT, project_probe_pt(qcd_num_2d, args.object, _NAMES.unique("qcd_num")), _NAMES.unique("qcd_num_ref"))
        top_den = rebin_to_reference(ROOT, project_probe_pt(top_den_2d, args.object, _NAMES.unique("top_den")), _NAMES.unique("top_den_ref"))
        top_num = rebin_to_reference(ROOT, project_probe_pt(top_num_2d, args.object, _NAMES.unique("top_num")), _NAMES.unique("top_num_ref"))

        if args.qcd_rescale != 1.0:
            qcd_den.Scale(args.qcd_rescale)
            qcd_num.Scale(args.qcd_rescale)

        mc_den = add_histograms(top_den, qcd_den, _NAMES.unique("mc_den"))
        mc_num = add_histograms(top_num, qcd_num, _NAMES.unique("mc_num"))
        data_eff = efficiency(data_num, data_den, _NAMES.unique("eff_data"))
        mc_eff = efficiency(mc_num, mc_den, _NAMES.unique("eff_mc"))

        region_key = REGION_INFO[args.object][0]
        outdir = output_dir(era)
        draw_reference_efficiency(
            ROOT,
            era=era,
            object_name=args.object,
            data_eff=data_eff,
            mc_eff=mc_eff,
            mc_num=mc_num,
            mc_den=mc_den,
            output_base=outdir / f"trig_eff_{region_key}_{era}",
        )

        write_measurement_outputs(
            ROOT,
            era=era,
            object_name=args.object,
            data_den=data_den,
            data_num=data_num,
            qcd_den=qcd_den,
            qcd_num=qcd_num,
            top_den=top_den,
            top_num=top_num,
            mc_den=mc_den,
            mc_num=mc_num,
            data_eff=data_eff,
            mc_eff=mc_eff,
        )

        if not args.quiet_bin_values:
            print_bin_values(data_eff, mc_eff, mc_num, mc_den)
        return 0
    finally:
        for handle in handles:
            try:
                handle.Close()
            except Exception:
                pass


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    if not argv:
        parser.print_help(sys.stdout)
        return 0

    args = parser.parse_args(argv)
    if not math.isfinite(args.qcd_rescale) or args.qcd_rescale < 0.0:
        print("[ERROR] --qcd-rescale must be finite and non-negative.", file=sys.stderr)
        return 2

    eras = selected_eras(args.year)
    objects = selected_objects(args.object)

    for era in eras:
        for object_name in objects:
            run_args = argparse.Namespace(**vars(args))
            run_args.object = object_name
            print_resolved_configuration(run_args, era)

    if args.dry_run:
        return 0

    try:
        ROOT = import_root()
        for era in eras:
            for object_name in objects:
                run_args = argparse.Namespace(**vars(args))
                run_args.object = object_name
                status = run_one_era(ROOT, run_args, era)
                if status != 0:
                    return status
    except (FileNotFoundError, OSError, KeyError, RuntimeError, TypeError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

