#!/usr/bin/env python3
"""Compare BJet and LightJet dimuon-mass shapes using only nominal NLO DY MC.

Default: sum 2016preVFP, 2016postVFP, 2017, 2018, 2022, 2022EE,
2023 and 2023BPix, then match the LightJet integral to BJet in 11--80 GeV.
Stored MC event weights are retained: eras are summed before matching, with
no extra luminosity rescaling and no per-era shape normalisation.

Only NIsoMuon_DYJets_Inclusive.root is opened, in READ mode. No collision data,
data-driven templates or MG LO samples are used, and no ROOT files are written.
The output is PDF, PNG and a ratio-fit JSON report. Errors are finite-MC Sumw2
errors with the matching factor held fixed; they are not a total uncertainty
or a closure-test p-value.
The ratio panel fits y=c (dashed) and y=ax+b (solid), using diagonal statistical
errors with the matching factor fixed. Use --ratio-fits none to hide them or
--ratio-fit-range LOW HIGH to restrict the fit to whole bins in that range.

Examples:
  python3 dy_mc_shape_comparison.py
  python3 dy_mc_shape_comparison.py --era Run2+3 --xmin 11 --xmax 80
  python3 dy_mc_shape_comparison.py --era 2018
  python3 dy_mc_shape_comparison.py --logx
  python3 dy_mc_shape_comparison.py --bin-edges 11 12 15 20 30 50 80 --logx
  python3 dy_mc_shape_comparison.py --era Run3 --rebin 50 --linear-y

Requires PyROOT for plotting. --help does not require ROOT.
"""

from __future__ import annotations

import argparse
import math
import sys
from array import array
from pathlib import Path

import ratio_fits

RUN2_ERAS: tuple[str, ...] = (
    "2016preVFP", "2016postVFP", "2017", "2018",
)
RUN3_ERAS: tuple[str, ...] = (
    "2022", "2022EE", "2023", "2023BPix",
)
ALL_ERAS: tuple[str, ...] = RUN2_ERAS + RUN3_ERAS
ERA_GROUPS = {
    **{era: (era,) for era in ALL_ERAS},
    "Run2": RUN2_ERAS,
    "Run3": RUN3_ERAS,
    "Run2+3": ALL_ERAS,
}
VALID_ERAS = list(ERA_GROUPS.keys())
DEFAULT_BASE_DIR = "/data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13/NIsoMuon"
PLOT_DIR = "/data6/Users/joonblee/PlotMaker/plots"
DY_AMC_DEFAULT_FILE = "NIsoMuon_DYJets_Inclusive.root"

DEFAULT_MASS_BINS = [
    0.0, 0.5, 1.0, 1.5, 2.0, 2.5,
    3.0, 3.5, 4.0, 4.5,
    5.0, 6.0, 7.0, 8.0, 9.0,
    11.0, 13., 15., 20.0, 25.,
    30.0, 40., 60., 80., 85., 90., 95., 100., 110., 150.
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

REG_B = "OS_POGMedium_tight_BJet_NIsoDimuon"
REG_L = "OS_POGMedium_tight_LightJet_NIsoDimuon"


def selected_eras(period: str) -> tuple[str, ...]:
    try:
        return ERA_GROUPS[period]
    except KeyError as exc:
        raise ValueError(
            f"Unknown era/period {period!r}; use one of: {', '.join(VALID_ERAS)}"
        ) from exc


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
    import ROOT # type: ignore
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)
    return ROOT

def clone_hist(hist, name: str):
    out = hist.Clone(name)
    out.SetDirectory(0)
    out.Sumw2()
    return out


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


def scale_to_yield_per_gev(hist) -> None:
    axis = hist.GetXaxis()
    for ibin in range(1, hist.GetNbinsX() + 1):
        width = float(axis.GetBinWidth(ibin))
        if width <= 0.0 or not math.isfinite(width):
            continue
        hist.SetBinContent(ibin, float(hist.GetBinContent(ibin)) / width)
        hist.SetBinError(ibin, float(hist.GetBinError(ibin)) / width)
    hist.GetYaxis().SetTitle("Yield / GeV")


def axis_edges(hist):
    axis = hist.GetXaxis()
    return [float(axis.GetBinLowEdge(i))
            for i in range(1, hist.GetNbinsX() + 2)]


def edge_index(hist, edges, value):
    for i, edge in enumerate(edges):
        if math.isclose(value, edge, rel_tol=0.0, abs_tol=1e-8):
            return i
    raise ValueError(
        f"{hist.GetName()}: {value:g} GeV is not an input bin edge; "
        "choose a compatible mass range and binning"
    )


def window_bins(hist, low, high):
    """Require exact histogram boundaries and exclude underflow and overflow."""
    edges = axis_edges(hist)
    start = edge_index(hist, edges, low)
    stop = edge_index(hist, edges, high)
    if start >= stop:
        raise ValueError("Require a nonempty mass window")
    return range(start + 1, stop + 1)


def assert_same_1d_binning(reference, candidate, reference_label, candidate_label):
    first, second = axis_edges(reference), axis_edges(candidate)
    if len(first) != len(second) or any(
        not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)
        for a, b in zip(first, second)
    ):
        raise ValueError(f"Histogram binning differs: {reference_label} vs {candidate_label}")


def rebin_hist(hist, name, args):
    if args.rebin == 1:
        return clone_hist(hist, name)
    if args.rebin > 1:
        out = hist.Rebin(args.rebin, name)
    else:
        native_edges = axis_edges(hist)
        if args.bin_edges is None:
            edges = [native_edges[0]]
            edges.extend(edge for edge in DEFAULT_MASS_BINS
                         if native_edges[0] < edge < native_edges[-1])
            edges.append(native_edges[-1])
        else:
            edges = args.bin_edges
        # Give ROOT the original boundaries, including their floating-point
        # representation, so no input bin is split between two output bins.
        indices = [edge_index(hist, native_edges, edge) for edge in edges]
        if any(left >= right for left, right in zip(indices, indices[1:])):
            raise ValueError("Bin edges must select distinct input boundaries")
        root_edges = array("d", (native_edges[index] for index in indices))
        out = hist.Rebin(len(root_edges) - 1, name, root_edges)
    if not out:
        raise ValueError(f"Could not rebin {hist.GetName()}")
    out.SetDirectory(0)
    out.Sumw2()
    return out


def integral_and_error(hist, low, high):
    bins = window_bins(hist, low, high)
    value = math.fsum(float(hist.GetBinContent(i)) for i in bins)
    error = math.sqrt(math.fsum(float(hist.GetBinError(i)) ** 2 for i in bins))
    if not math.isfinite(value) or not math.isfinite(error):
        raise ValueError(f"Non-finite integral/error in {hist.GetName()}")
    return value, error


def load_dy_pair(ROOT, args):
    """Sum both categories over every requested era; missing inputs are fatal."""
    b_total = l_total = None
    for era in selected_eras(args.era):
        path = Path(args.base_dir) / era / DY_AMC_DEFAULT_FILE
        handle = ROOT.TFile.Open(str(path), "READ")
        if not handle or handle.IsZombie():
            if handle:
                handle.Close()
            raise OSError(f"Cannot open NLO DY input: {path}")
        try:
            pair = []
            for category, region in (("BJet", REG_B), ("LightJet", REG_L)):
                key = f"{region}/Dilepton_Mass___{region}"
                hist = handle.Get(key)
                if not hist or not hist.InheritsFrom("TH1") or hist.GetDimension() != 1:
                    raise ValueError(f"Missing one-dimensional histogram: {path}:{key}")
                hist = clone_hist(hist, f"dy_shape_{category}_{era}")
                # Validate the requested integration interval before adding eras.
                integral_and_error(hist, args.xmin, args.xmax)
                pair.append(hist)
            b_hist, l_hist = pair
            assert_same_1d_binning(b_hist, l_hist, era + "/BJet", era + "/LightJet")
            if b_total is None:
                b_total, l_total = b_hist, l_hist
            else:
                assert_same_1d_binning(b_total, b_hist, "BJet sum", era + "/BJet")
                assert_same_1d_binning(l_total, l_hist, "LightJet sum", era + "/LightJet")
                b_total.Add(b_hist)
                l_total.Add(l_hist)
            print(f"[INPUT] {era}: {path}")
        finally:
            handle.Close()
    return b_total, l_total


def match_lightjet(ROOT, b_hist, l_hist, low, high):
    """Return a detached, scaled LightJet histogram and the matching factor."""
    b_yield, b_error = integral_and_error(b_hist, low, high)
    l_yield, l_error = integral_and_error(l_hist, low, high)
    if b_yield <= 0 or l_yield <= 0:
        raise ValueError("BJet and LightJet integrals must both be positive")
    factor = b_yield / l_yield
    scaled = clone_hist(l_hist, "dy_shape_LightJet_matched")
    scaled.Scale(factor)
    print(f"[MATCH] {low:g} < m(mumu) < {high:g} GeV")
    print(f"[YIELD] BJet={b_yield:.9g} +/- {b_error:.9g}; "
          f"LightJet={l_yield:.9g} +/- {l_error:.9g} (MC stat.)")
    print(f"[SCALE] LightJet *= {factor:.9g}; sum eras before matching")
    matched, _ = integral_and_error(scaled, low, high)
    if not math.isclose(matched, b_yield, rel_tol=1e-10, abs_tol=1e-9):
        raise ValueError("LightJet integral matching failed")
    return scaled, factor


def ratio_points(b_hist, l_hist, low, high):
    """Independent bin errors, conditional on a fixed shape-matching factor."""
    points = []
    for i in window_bins(b_hist, low, high):
        b, eb = float(b_hist.GetBinContent(i)), float(b_hist.GetBinError(i))
        l, el = float(l_hist.GetBinContent(i)), float(l_hist.GetBinError(i))
        if not all(math.isfinite(x) for x in (b, eb, l, el)):
            raise ValueError(f"Non-finite plotted bin {i}")
        if l <= 0:
            continue
        points.append((
            float(b_hist.GetXaxis().GetBinCenter(i)),
            0.5 * float(b_hist.GetXaxis().GetBinWidth(i)),
            b / l,
            math.hypot(eb / l, b * el / l ** 2),
        ))
    return points


def draw(ROOT, args, b_hist, l_hist, factor):
    canvas = ROOT.TCanvas("dy_mc_shape", "", 900, 900)
    upper = ROOT.TPad("dy_shape_upper", "", 0, 0.30, 1, 1)
    lower = ROOT.TPad("dy_shape_lower", "", 0, 0, 1, 0.30)
    for pad in (upper, lower):
        pad.SetLeftMargin(0.14)
        pad.SetRightMargin(0.05)
        pad.SetLogx(args.logx)
        pad.SetTickx()
        pad.SetTicky()
    upper.SetTopMargin(0.10)
    upper.SetBottomMargin(0.03)
    lower.SetTopMargin(0.04)
    lower.SetBottomMargin(0.35)
    lower.SetGridy()
    upper.Draw()
    lower.Draw()
    upper.cd()
    upper.SetLogy(not args.linear_y)

    visible = [
        (float(h.GetBinContent(i)), float(h.GetBinError(i)))
        for h in (b_hist, l_hist)
        for i in window_bins(h, args.xmin, args.xmax)
    ]
    ymax = max(y + error for y, error in visible)
    if args.linear_y:
        ymin = min(0.0, min(y - error for y, error in visible))
        span = max(ymax - ymin, 1.0)
        ymin, ymax = ymin - 0.05 * span, ymax + 0.5 * span
    else:
        positive = [y for y, _ in visible if y > 0]
        if not positive:
            raise ValueError("No positive plotted bins for logarithmic y axis")
        ymin, ymax = min(positive) * 0.3, max(ymax * 20.0, min(positive) * 10.0)

    apply_style(b_hist)
    b_hist.SetTitle("")
    b_hist.GetXaxis().SetRangeUser(args.xmin, args.xmax)
    b_hist.GetYaxis().SetTitle("Yield / GeV")
    b_hist.SetMinimum(ymin)
    b_hist.SetMaximum(ymax)
    b_hist.SetLineColor(ROOT.kBlack)
    b_hist.SetMarkerColor(ROOT.kBlack)
    b_hist.SetMarkerStyle(20)
    b_hist.SetMarkerSize(0.8)
    b_hist.Draw("E1")

    l_hist.SetLineColor(ROOT.kBlue + 1)
    l_hist.SetMarkerColor(ROOT.kBlue + 1)
    l_hist.SetFillColor(ROOT.kBlue + 1)
    l_hist.SetFillStyle(3004)
    l_hist.SetMarkerSize(0)
    l_hist.Draw("E2 SAME")
    line = clone_hist(l_hist, "dy_shape_LightJet_line")
    line.SetFillStyle(0)
    line.SetLineWidth(2)
    line.Draw("HIST SAME")
    b_hist.Draw("E1 SAME")

    legend = ROOT.TLegend(0.49, 0.66, 0.94, 0.83)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.032)
    legend.AddEntry(b_hist, "b-jet category", "pe")
    legend.AddEntry(line, f"light-jet category (scale = {factor:.5g})", "l")
    legend.Draw()

    text = ROOT.TLatex()
    text.SetNDC()
    text.SetTextFont(42)
    text.SetTextSize(0.043)
    text.DrawLatex(0.14, 0.925, "#bf{CMS} #it{Simulation}")
    text.SetTextSize(0.033)
    text.SetTextAlign(31)
    text.DrawLatex(0.95, 0.925, lumi_label(args.era))
    text.SetTextAlign(11)
    text.SetTextSize(0.032)
    text.DrawLatex(0.18, 0.85, "DY aMC@NLO, POG Medium, nonisolated dimuon")
    text.SetTextSize(0.029)
    text.DrawLatex(0.18, 0.16, f"Light-jet scale = {factor:.5g}")
    text.DrawLatex(0.18, 0.115,
                   f"Integral matched in {args.xmin:g} < m(#mu#mu) < {args.xmax:g} GeV")
    text.DrawLatex(0.18, 0.07, "MC statistical errors; matching factor fixed")

    lower.cd()
    frame = ROOT.TH1D("dy_shape_ratio_frame", "", 1, args.xmin, args.xmax)
    frame.SetDirectory(0)
    apply_style(frame, is_ratio=True)
    frame.GetXaxis().SetTitle("m(#mu#mu) [GeV]")
    if args.logx:
        frame.GetXaxis().SetMoreLogLabels()
        frame.GetXaxis().SetNoExponent()
    frame.GetYaxis().SetTitle("b / scaled light")
    frame.GetYaxis().SetTitleSize(0.10)
    frame.GetYaxis().SetTitleOffset(0.65)
    frame.SetMinimum(args.ratio_min)
    frame.SetMaximum(args.ratio_max)
    frame.Draw("AXIS")
    unity = ROOT.TLine(args.xmin, 1.0, args.xmax, 1.0)
    unity.SetLineStyle(2)
    unity.Draw()
    ratio = ROOT.TGraphErrors()
    for i, (x, ex, y, ey) in enumerate(ratio_points(
            b_hist, l_hist, args.xmin, args.xmax)):
        ratio.SetPoint(i, x, y)
        ratio.SetPointError(i, ex, ey)
    ratio.SetMarkerStyle(20)
    ratio.SetMarkerSize(0.8)
    ratio.SetLineColor(ROOT.kBlack)
    ratio.Draw("PE SAME")
    fit_series = [("DY MC", ROOT.kGreen + 2, ratio_fits.analyse(
        ratio_points(b_hist, l_hist, args.xmin, args.xmax), args))]
    fit_objects = ratio_fits.draw(ROOT, fit_series)
    lower.RedrawAxis()
    upper.cd()
    upper.RedrawAxis()
    canvas.cd()
    canvas.Update()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.bin_edges is not None:
        binning_tag = "customBins"
    elif args.rebin:
        binning_tag = f"rebin{args.rebin}"
    else:
        binning_tag = "varBin"
    stem = (
        f"DY_BJet_vs_LightJet_{args.era}_POGMedium_NIsoDimuon_"
        f"yieldPerGeV_LightJetScaled_m{args.xmin:g}to{args.xmax:g}"
        f"_{binning_tag}"
        f"{'_logx' if args.logx else ''}"
    )
    for suffix in ("pdf", "png"):
        path = output / f"{stem}.{suffix}"
        canvas.SaveAs(str(path))
        print(f"[SAVED] {path}")
    if args.ratio_fits != "none":
        ratio_fits.save(output / f"{stem}.json", fit_series, era=args.era,
                        matching_factor=factor)


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--era", choices=VALID_ERAS, default="Run2+3")
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--output-dir", default=PLOT_DIR)
    parser.add_argument("--xmin", type=float, help="Lower mass bound (default: 11 GeV or first custom edge)")
    parser.add_argument("--xmax", type=float, help="Upper mass bound (default: 80 GeV or last custom edge)")
    binning = parser.add_mutually_exclusive_group()
    binning.add_argument(
        "--rebin", type=int, default=0,
        help="0: default variable mass bins; positive: native-bin merge factor")
    binning.add_argument(
        "--bin-edges", type=float, nargs="+", metavar="EDGE",
        help="Custom mass bin edges in GeV, in increasing order")
    parser.add_argument("--logx", action="store_true", help="Use a logarithmic mass axis")
    parser.add_argument("--linear-y", action="store_true", help="Use a linear y axis")
    parser.add_argument("--ratio-min", type=float, default=0.0)
    parser.add_argument("--ratio-max", type=float, default=2.0)
    ratio_fits.add_arguments(parser)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    edges = args.bin_edges
    if edges is not None and (
        len(edges) < 2 or not all(map(math.isfinite, edges))
        or any(left >= right for left, right in zip(edges, edges[1:]))
    ):
        parser.error("--bin-edges requires at least two finite, increasing edges")
    if args.xmin is None:
        args.xmin = edges[0] if edges is not None else 11.0
    if args.xmax is None:
        args.xmax = edges[-1] if edges is not None else 80.0
    if edges is not None and (args.xmin != edges[0] or args.xmax != edges[-1]):
        parser.error("--xmin and --xmax must match the first and last custom bin edges")
    if not (math.isfinite(args.xmin) and math.isfinite(args.xmax)
            and args.xmin < args.xmax):
        parser.error("Require finite xmin < xmax")
    if args.logx and args.xmin <= 0:
        parser.error("--logx requires --xmin > 0")
    if args.rebin < 0:
        parser.error("--rebin must be nonnegative")
    if not (math.isfinite(args.ratio_min) and math.isfinite(args.ratio_max)
            and args.ratio_min < args.ratio_max):
        parser.error("Require finite ratio-min < ratio-max")
    try:
        ratio_fits.fit_range(args)
    except ValueError as exc:
        parser.error(str(exc))
    ROOT = import_root()
    print("[ERAS] " + ", ".join(selected_eras(args.era)))
    b_raw, l_raw = load_dy_pair(ROOT, args)
    l_matched, factor = match_lightjet(ROOT, b_raw, l_raw, args.xmin, args.xmax)
    b_plot = rebin_hist(b_raw, "dy_shape_BJet_plot", args)
    l_plot = rebin_hist(l_matched, "dy_shape_LightJet_plot", args)
    # Reject display bins straddling either normalisation boundary.
    integral_and_error(b_plot, args.xmin, args.xmax)
    integral_and_error(l_plot, args.xmin, args.xmax)
    for hist in (b_plot, l_plot):
        scale_to_yield_per_gev(hist)
    print("[ERRORS] Finite-MC Sumw2 only; integral-matching factor held fixed")
    draw(ROOT, args, b_plot, l_plot, factor)
    return 0


if __name__ == "__main__":
    sys.exit(main())
