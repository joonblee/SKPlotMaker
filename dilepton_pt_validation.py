#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Standalone Dilepton_pT validation plotter for NIsoMuon.

Reads the one-dimensional histogram

  Dilepton_pT___<region>

from data and nominal MC files and draws Data/MC validation plots for the
B-jet and light-jet regions.  The plot style follows dy_bkg_estimation.py.

Examples
--------
  python3 dilepton_pt_validation.py --era 2018 --blind
  python3 dilepton_pt_validation.py --era Run2 --blind
  python3 dilepton_pt_validation.py --era Run3
  python3 dilepton_pt_validation.py --era 2018 --blind --with-mg
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from array import array
from typing import List, Tuple

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
DY_FILE = "NIsoMuon_DYJets_Inclusive.root"
MG_DY_FILE = "NIsoMuon_DYJets_MG_Inclusive.root"
DILEPTON_PT_BINS = (50., 100., 120., 150., 200., 250., 300., 400., 600., 1000.)

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
    "Run2": 137.94,
    "Run3": 61.85,
    "Run2+3": 199.79,
}


def selected_eras(period: str) -> Tuple[str, ...]:
    return ERA_GROUPS[period]


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


def import_root():
    import ROOT  # type: ignore

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)
    return ROOT


def era_dir(args, era: str) -> str:
    parts = [args.base_dir, era]
    if args.trigger:
        parts.append(args.trigger)
    return os.path.join(*parts)


def clone_hist(hist, name: str):
    out = hist.Clone(name)
    out.SetDirectory(0)
    out.Sumw2()
    return out


def assert_same_binning(reference, candidate, reference_label: str, candidate_label: str) -> None:
    if reference.GetNbinsX() != candidate.GetNbinsX():
        raise ValueError(
            f"[ERROR] Binning mismatch: {reference_label} has {reference.GetNbinsX()} bins, "
            f"{candidate_label} has {candidate.GetNbinsX()} bins."
        )

    ref_axis = reference.GetXaxis()
    cand_axis = candidate.GetXaxis()
    for ibin in range(1, reference.GetNbinsX() + 1):
        ref_low = float(ref_axis.GetBinLowEdge(ibin))
        cand_low = float(cand_axis.GetBinLowEdge(ibin))
        if not math.isclose(ref_low, cand_low, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(
                f"[ERROR] Binning mismatch at bin {ibin}: "
                f"{reference_label}={ref_low}, {candidate_label}={cand_low}."
            )

    ref_high = float(ref_axis.GetBinUpEdge(reference.GetNbinsX()))
    cand_high = float(cand_axis.GetBinUpEdge(candidate.GetNbinsX()))
    if not math.isclose(ref_high, cand_high, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(
            f"[ERROR] Upper-edge mismatch: {reference_label}={ref_high}, "
            f"{candidate_label}={cand_high}."
        )


def load_histogram_across_eras(
    ROOT,
    args,
    filename: str,
    folder: str,
    hist_name: str,
    clone_name: str,
    *,
    required: bool,
):
    combined = None
    missing: List[str] = []

    for era in selected_eras(args.era):
        path = os.path.join(era_dir(args, era), filename)
        f = ROOT.TFile.Open(path)
        if not f or f.IsZombie():
            if f:
                f.Close()
            missing.append(f"{era}:{path}")
            continue

        full_name = f"{folder}/{hist_name}"
        hist = f.Get(full_name)
        if not hist:
            f.Close()
            missing.append(f"{era}:{path}:{full_name}")
            continue

        hist = clone_hist(hist, f"{clone_name}_{era}")
        f.Close()

        if combined is None:
            combined = clone_hist(hist, clone_name)
        else:
            assert_same_binning(combined, hist, clone_name, f"{era}/{filename}")
            combined.Add(hist)

    if required and missing:
        raise FileNotFoundError(
            "[ERROR] Required input(s) missing:\n  " + "\n  ".join(missing)
        )

    return combined


def rebin_dilepton_pt(hist, name: str):
    """Use exactly the Dilepton_pT binning from dy_bkg_estimation.py."""
    edges = array("d", DILEPTON_PT_BINS)
    out = hist.Rebin(len(edges) - 1, name, edges)
    out.SetDirectory(0)
    out.Sumw2()
    return out


def scale_to_yield_per_gev(hist) -> None:
    axis = hist.GetXaxis()
    for ibin in range(1, hist.GetNbinsX() + 1):
        width = float(axis.GetBinWidth(ibin))
        if width <= 0.0 or not math.isfinite(width):
            continue
        hist.SetBinContent(ibin, float(hist.GetBinContent(ibin)) / width)
        hist.SetBinError(ibin, float(hist.GetBinError(ibin)) / width)
    hist.GetYaxis().SetTitle("Yield / GeV")


def apply_full_blinding(hist) -> None:
    for ibin in range(1, hist.GetNbinsX() + 1):
        hist.SetBinContent(ibin, -9999.0)
        hist.SetBinError(ibin, 0.0)


def visible_max(hist, xmin: float, xmax: float) -> float:
    axis = hist.GetXaxis()
    first = max(1, axis.FindFixBin(xmin))
    last = min(hist.GetNbinsX(), axis.FindFixBin(xmax))
    out = 0.0
    for ibin in range(first, last + 1):
        out = max(out, float(hist.GetBinContent(ibin) + hist.GetBinError(ibin)))
    return out


def apply_style(hist, is_ratio: bool = False) -> None:
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


def draw_cms_text(ROOT, era: str, region_name: str, blinded: bool) -> None:
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
    latex.DrawLatex(0.16, 0.84, "p_{T}(#mu#mu) Validation")

    latex.SetTextSize(0.030)
    latex.SetTextColor(ROOT.kGray + 2)
    latex.DrawLatex(0.16, 0.79, f"Region: {region_name}")
    if blinded:
        latex.DrawLatex(0.16, 0.752, "Data fully blinded")


def draw_validation_plot(
    ROOT,
    args,
    folder: str,
    region_token: str,
    region_name: str,
    *,
    dy_file: str,
    output_suffix: str = "",
    dy_label: str = "DY",
) -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)

    hist_name = f"Dilepton_pT___{folder}"
    object_token = f"{region_token}{output_suffix}"
    h_data_raw = load_histogram_across_eras(
        ROOT,
        args,
        "data.root",
        folder,
        hist_name,
        f"data_{object_token}",
        required=True,
    )
    h_data = rebin_dilepton_pt(h_data_raw, f"data_rebin_{object_token}")
    scale_to_yield_per_gev(h_data)

    is_blinded = args.blind and region_token == "BJet"
    if is_blinded:
        apply_full_blinding(h_data)

    color_dy = ROOT.TColor.GetColor("#FFCC66")
    color_top = ROOT.TColor.GetColor("#669966")
    color_qcd = ROOT.TColor.GetColor("#99CCFF")
    color_others = ROOT.TColor.GetColor("#CCCCCC")

    mc_files = (
        ("NIsoMuon_Others.root", color_others, "Others"),
        ("NIsoMuon_Top.root", color_top, "Top"),
        ("NIsoMuon_QCD_Inclusive.root", color_qcd, "QCD"),
        (dy_file, color_dy, dy_label),
    )

    stack = ROOT.THStack(f"stack_{object_token}", "")
    h_total_mc = h_data.Clone(f"total_mc_{object_token}")
    h_total_mc.Reset()
    mc_hists = []

    for filename, color, label in mc_files:
        h_raw = load_histogram_across_eras(
            ROOT,
            args,
            filename,
            folder,
            hist_name,
            f"{label}_{object_token}",
            required=False,
        )
        if h_raw is None:
            print(f"[WARNING] Missing optional MC input: {filename}")
            continue

        h_mc = rebin_dilepton_pt(h_raw, f"{label}_rebin_{object_token}")
        scale_to_yield_per_gev(h_mc)
        h_mc.SetFillColor(color)
        h_mc.SetLineColor(ROOT.kBlack)
        h_mc.SetLineWidth(1)

        stack.Add(h_mc)
        h_total_mc.Add(h_mc)
        mc_hists.append((h_mc, label))

    xmin = args.xmin
    xmax = args.xmax

    canvas = ROOT.TCanvas(f"c_{object_token}", "", 900, 900)
    upper = ROOT.TPad(f"upper_{object_token}", "", 0.0, 0.30, 1.0, 1.0)
    lower = ROOT.TPad(f"lower_{object_token}", "", 0.0, 0.00, 1.0, 0.30)

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

    max_val = max(
        visible_max(h_data, xmin, xmax),
        visible_max(h_total_mc, xmin, xmax),
    )

    h_data.GetXaxis().SetRangeUser(xmin, xmax)
    h_data.GetYaxis().SetTitle("Yield / GeV")
    apply_style(h_data, is_ratio=False)

    if args.logy:
        h_data.SetMaximum(max(max_val * 100.0, 1.0))
        h_data.SetMinimum(1e-3)
    else:
        h_data.SetMaximum(max(max_val * 1.45, 1.0))
        h_data.SetMinimum(0.0)

    h_data.SetMarkerStyle(20)
    h_data.SetMarkerColor(ROOT.kBlack)
    h_data.SetLineColor(ROOT.kBlack)

    h_data.Draw("PE")
    stack.Draw("HIST SAME")
    h_data.Draw("PE SAME")

    legend = ROOT.TLegend(0.60, 0.60, 0.94, 0.89)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.030)
    legend.SetTextFont(42)
    legend.AddEntry(h_data, "Data", "lep")
    for h_mc, label in reversed(mc_hists):
        legend.AddEntry(h_mc, label, "f")
    legend.Draw()

    draw_cms_text(ROOT, args.era, region_name, is_blinded)
    upper.SetTickx()
    upper.SetTicky()
    upper.RedrawAxis()

    lower.cd()
    h_ratio = h_data.Clone(f"ratio_{object_token}")
    h_ratio.Divide(h_total_mc)
    if is_blinded:
        apply_full_blinding(h_ratio)

    h_ratio.GetXaxis().SetRangeUser(xmin, xmax)
    apply_style(h_ratio, is_ratio=True)
    h_ratio.GetYaxis().SetTitle("Data / MC")
    h_ratio.GetYaxis().SetRangeUser(0.0, 2.0)
    h_ratio.GetXaxis().SetTitle("p_{T}(#mu#mu) [GeV]")
    h_ratio.Draw("PE")

    line = ROOT.TF1(f"line_one_{object_token}", "1.0", xmin, xmax)
    line.SetLineColor(ROOT.kBlack)
    line.SetLineStyle(2)
    line.Draw("SAME")

    lower.SetTickx()
    lower.SetTicky()
    lower.SetGridy()
    lower.RedrawAxis()

    output_pdf = os.path.join(
        PLOT_DIR, f"Validation_{args.era}_Dilepton_pT_{region_token}{output_suffix}.pdf"
    )
    canvas.SaveAs(output_pdf)
    print(f"[SAVED] {output_pdf}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw standalone NIsoMuon Dilepton_pT validation plots."
    )
    parser.add_argument(
        "--era",
        required=True,
        choices=VALID_ERAS,
        help="Individual era, Run2, Run3, or Run2+3",
    )
    parser.add_argument(
        "--blind",
        action="store_true",
        help="Fully blind B-jet data and Data/MC ratio",
    )
    parser.add_argument(
        "--base-dir",
        default=DEFAULT_BASE_DIR,
        help="NIsoMuon SKOutput base directory",
    )
    parser.add_argument(
        "--trigger",
        default="",
        help="Optional legacy trigger subdirectory",
    )
    parser.add_argument(
        "--dy-file",
        default=DY_FILE,
        help="Nominal aMC@NLO DY ROOT filename",
    )
    parser.add_argument(
        "--with-mg",
        action="store_true",
        help="Also draw validation plots using the MG LO DY sample",
    )
    parser.add_argument(
        "--mg-dy-file",
        default=MG_DY_FILE,
        help="MG LO DY ROOT filename used with --with-mg",
    )
    parser.add_argument(
        "--xmin",
        type=float,
        default=DILEPTON_PT_BINS[0],
        help="Minimum dimuon pT shown",
    )
    parser.add_argument(
        "--xmax",
        type=float,
        default=DILEPTON_PT_BINS[-1],
        help="Maximum dimuon pT shown",
    )
    parser.add_argument(
        "--linear-y",
        dest="logy",
        action="store_false",
        help="Use linear y axis",
    )
    parser.set_defaults(logy=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.xmax <= args.xmin:
        raise ValueError("--xmax must be larger than --xmin")

    ROOT = import_root()

    regions = (
        ("OS_POGMedium_tight_BJet_NIsoDimuon", "BJet", "B-Jet"),
        ("OS_POGMedium_tight_LightJet_NIsoDimuon", "LightJet", "Light-Jet"),
    )

    for folder, region_token, region_name in regions:
        draw_validation_plot(
            ROOT, args, folder, region_token, region_name,
            dy_file=args.dy_file,
        )

    if args.with_mg:
        for folder, region_token, region_name in regions:
            draw_validation_plot(
                ROOT, args, folder, region_token, region_name,
                dy_file=args.mg_dy_file,
                output_suffix="_MG",
                dy_label="DY (MG LO)",
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
