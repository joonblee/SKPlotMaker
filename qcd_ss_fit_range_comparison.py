#!/usr/bin/env python3
"""Compare SS Data - Top - Others fits in 5--20, 5--30 and 5--50 GeV.

Reuse the production six function families, nominal Power x Exp x Logistic,
bin-integrated chi-square and fit-validity rules. Signed bins are never clipped.
All ranges use one common grid: adaptive below 30 GeV with a protected
30 GeV boundary, and one fixed 30--50 GeV bin. No 20 GeV edge is forced.
First complete statistical adaptive binning and the width floor
max(0.1 GeV, 0.02*final bin centre). Extra forward merges use whole adaptive
bins only: next yield <=0, combined yield >0, within the base width cap
(default 5 GeV). No native-bin early stop or width-floor exemption is used.
After fixing all edges, merge the last bin ending at 30 GeV backward once
only if its signed yield is >0 and the preceding bin's signed yield is <=0.
Never merge across 5 or 30 GeV. This final conditional merge and the fixed
tail may exceed the adaptive width cap; all signed yields/errors are retained.
The common-grid edges and all fit diagnostics are exported.
Refit to apply the new grid; --plot-json keeps saved bin edges.

Bands are pointwise nominal +/- max_alternative |alternative - nominal|,
without zero-clipping. They are functional envelopes, not confidence intervals
or the total statistical + systematic uncertainty. Default display: log x,
5--50 GeV and y >= 0.001 events/GeV, with decimal chi-square/ndf. These are
viewport settings only: signed bins, all three fit ranges and full 5--50 GeV
curves/bands remain unchanged in the fit and JSON.
The lower panel divides every curve/envelope by the fixed 5--30 GeV nominal
curve. Data ratios use the corresponding bin-averaged reference prediction;
statistical data errors are divided by the same fixed denominator. Reference
uncertainty is shown as a band around one, not added in quadrature to ratios.
The reference/envelope note is inside the upper panel. Its transparent legend
is compact and aligned near the right frame edge; no canvas footer is drawn.
Canvas proportions match the DY/OS-SS comparison plots, with no extra bottom
strip. Axis titles use one pixel font size across both panels; legend text is
enlarged. Ratio default: linear 0--2. An inset at the lower left of the upper
panel shows 20--50 GeV on a logarithmic ratio y axis from 1 to an automatic
upper bound covering all functional bands in that mass interval. A box at
20--50 GeV, ratio 0.9--1.9 and connectors identify its source mass region;
the inset expands the y range beyond the box. Reference/Bands notes sit above
the inset on separate lines. Its title identifies the fixed reference and
20--50 GeV mass interval. --no-ratio-inset hides the inset and callout.
--ratio-scale hybrid keeps ratios <=1.5
linear and compresses larger values logarithmically, with physical-value ticks.
Its upper bound is automatic unless --ratio-max is specified. Display options
never change the saved fits, envelopes or JSON.

For combined periods, sum the SS data and each subtraction MC histogram over
the requested eras first, then fit that summed histogram once per range.
Only SS histograms are read. No production anchors/templates are written.

Examples (base-dir includes NIsoMuon, like the comparison plotters):
  python3 qcd_ss_fit_range_comparison.py --era 2018
  python3 qcd_ss_fit_range_comparison.py --era full
  python3 qcd_ss_fit_range_comparison.py --all-periods
  python3 qcd_ss_fit_range_comparison.py --plot-json plots/SSFitRangeComparison/*.json
--all-periods runs eight eras, Run2, Run3 and Run2+3 sequentially in separate
processes, forwarding the same options and retaining per-period output names.
Requires PyROOT only; plotting uses Python scalar maths and ROOT graphics,
without importing NumPy/Matplotlib or running BLAS. --plot-json redraws saved
results without opening input ROOT files or repeating fits. --help needs no ROOT.
"""

import argparse
from array import array
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import subprocess
import sys

import qcd_bkg_estimation as qcd

FIT_MAX = 50.
RANGES = (30., 20., FIT_MAX)
FIXED_HIGH_MASS_EDGES = (30., FIT_MAX)
FORCED_EDGES = FIXED_HIGH_MASS_EDGES
ALL_PERIODS = (*qcd.RUN2_ERAS, *qcd.RUN3_ERAS, "Run2", "Run3", "Run2+3")
DEFAULT_BASE = str(Path(qcd.DEFAULT_BASE_DIR) / "NIsoMuon")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    periods = parser.add_mutually_exclusive_group()
    periods.add_argument("--era", "--year", choices=(*qcd.VALID_YEARS, "full"), default="full")
    periods.add_argument("--all-periods", action="store_true",
                         help="Fit/plot eight eras plus Run2, Run3 and Run2+3 sequentially")
    parser.add_argument("--base-dir", default=DEFAULT_BASE,
                        help="Input directory containing era directories (includes NIsoMuon)")
    parser.add_argument("--trigger", default="")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: plots/SSFitRangeComparison; for --plot-json, beside each input JSON")
    parser.add_argument("--plot-json", type=Path, nargs="+", metavar="JSON",
                        help="Redraw saved JSON results only: no input histograms, fits or JSON writes")
    parser.add_argument("--ss-min-effective-count", type=qcd.positive_float, default=10.)
    parser.add_argument("--ss-max-bin-width", type=qcd.positive_float, default=5.,
                        help="Base adaptive width cap (default: 5); width-floor/final conditional merging and the fixed 30--50 bin can exceed it")
    parser.add_argument("--fit-max-attempts", type=qcd.positive_integer, default=30)
    parser.add_argument("--fit-attempt-details", action="store_true")
    parser.add_argument("--linthresh", type=qcd.positive_float, default=.1,
                        help="Symmetric-log scale near zero when --ymin 0 is used, in events/GeV")
    parser.add_argument("--ymin", type=float, default=.001,
                        help="Displayed lower y bound in events/GeV (default: 0.001; 0 is allowed)")
    parser.add_argument("--xmax", type=qcd.positive_float, default=FIT_MAX,
                        help="Displayed upper mass bound only, 5 < xmax <= 50 (default: 50 GeV)")
    parser.add_argument("--ratio-min", type=float, default=0., help="Lower ratio-panel bound (default: 0)")
    parser.add_argument("--ratio-max", type=float,
                        help="Upper physical ratio bound (default: 2 for linear; automatic for hybrid)")
    parser.add_argument("--ratio-scale", choices=("linear", "hybrid"), default="linear",
                        help="Linear ratio axis (default), or linear through 1.5 then logarithmic")
    parser.add_argument("--no-ratio-inset", dest="ratio_inset", action="store_false", default=True,
                        help="Hide the upper-panel 20--50 GeV log-ratio inset and its box/connectors")
    xscale = parser.add_mutually_exclusive_group()
    xscale.add_argument("--logx", dest="logx", action="store_true", help="Logarithmic x axis (default)")
    xscale.add_argument("--linear-x", dest="logx", action="store_false", help="Linear x axis")
    parser.set_defaults(logx=True)
    return parser


def common_histograms(ROOT, source, args):
    """Fix 30--50, then conditionally merge the positive terminal bin backward."""
    mode = replace(qcd.SS_MODE, fit_max=FIT_MAX)
    edges = qcd.make_adaptive_ss_binning(source, mode, args.ss_min_effective_count,
                                        args.ss_max_bin_width, protected_edges=(30.,),
                                        merge_nonpositive_next=True)
    edges = [x for x in edges if x <= 30. or x >= FIT_MAX]
    terminal = edges.index(30.)
    if terminal >= 2 and edges[terminal-2] >= mode.fit_min:
        previous_low, last_low, last_high = edges[terminal-2:terminal+1]
        previous_yield = float(source.Integral(source.FindBin(previous_low+1e-7),
                                               source.FindBin(last_low-1e-7)))
        last_yield = float(source.Integral(source.FindBin(last_low+1e-7),
                                           source.FindBin(last_high-1e-7)))
        if last_yield > 0. and previous_yield <= 0.:
            del edges[terminal-1]
            print(f"[SS COMPARISON BINNING] final conditional backward merge: "
                  f"[{previous_low:g},{last_low:g}] yield={previous_yield:g} + "
                  f"[{last_low:g},{last_high:g}] yield={last_yield:g}")
    print("[SS COMPARISON BINNING] final edges [GeV] = " + ", ".join(f"{x:g}" for x in edges))
    counts_raw = source.Rebin(len(edges)-1, "ss_comparison_counts_raw", array("d", edges))
    counts_raw.SetDirectory(0)
    counts = qcd.clone_to_th1d(ROOT, counts_raw, "ss_comparison_counts", edges)
    density = qcd.clone_to_th1d(ROOT, counts, "ss_comparison_density", edges)
    for i in range(1, density.GetNbinsX()+1):
        width = density.GetBinWidth(i)
        density.SetBinContent(i, density.GetBinContent(i) / width)
        density.SetBinError(i, density.GetBinError(i) / width)
    return qcd.PreparedHistograms(counts=counts, density=density, edges=edges)


def envelope(central, alternatives):
    if not alternatives:
        raise ValueError("All five alternative functions are required")
    if not all(math.isfinite(x) for x in (central, *alternatives)):
        raise ValueError("Nonfinite extrapolated function/envelope")
    delta = max(abs(x-central) for x in alternatives)
    return central-delta, central+delta


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value


def fit_ranges(ROOT, prepared, args):
    fit_args = qcd.build_parser().parse_args(["--mode", "ss-data", "--year", args.era])
    fit_args.fit_max_attempts = args.fit_max_attempts
    fit_args.fit_attempt_details = args.fit_attempt_details
    output = []
    # Evaluate each range before the next fit can reuse ROOT function names.
    x = [5. + i * .05 for i in range(round((FIT_MAX-5.)/.05)+1)]
    for upper in RANGES:
        mode = replace(qcd.SS_MODE, fit_max=upper)
        data = qcd.build_fit_data(ROOT, prepared, mode, "chi2", fit_args.log_relative_error_floor)
        selected, _, _ = qcd.fit_all_models(ROOT, prepared.density, data, mode, "chi2", fit_args)
        by_key = {fit.model.key: fit for fit in selected}
        invalid = [key for key in qcd.SS_REQUIRED_ROOT_MODELS if not by_key[key].accepted]
        report = dict(range=[5., upper], valid=not invalid, invalid_models=invalid,
                      models={fit.model.key: dict(accepted=fit.accepted,
                              diagnostics=asdict(fit.diagnostics), metrics=asdict(fit.metrics),
                              parameters=[float(fit.function.GetParameter(i))
                                          for i in range(fit.model.npar)]) for fit in selected})
        if not invalid:
            main = by_key[qcd.SS_NOMINAL_MODEL].function
            alternatives = [by_key[key].function for key in qcd.SS_SHAPE_ALTERNATIVES]
            central = [float(main.Eval(mass)) for mass in x]
            bands = [envelope(y, [float(f.Eval(mass)) for f in alternatives])
                     for mass, y in zip(x, central)]
            report.update(x=x, central=central, low=[b[0] for b in bands], high=[b[1] for b in bands])
            # Integrated envelopes for exactly the plotted data bins as well.
            bins = []
            for left, right in zip(prepared.edges, prepared.edges[1:]):
                if left < 5. or right > FIT_MAX:
                    continue
                nominal = float(main.Integral(left, right)) / (right-left)
                lower, higher = envelope(nominal, [float(f.Integral(left, right)) / (right-left)
                                                  for f in alternatives])
                bins.append(dict(low=left, high=right, central=nominal,
                                 envelope_low=lower, envelope_high=higher))
            report["bin_averages"] = bins
        output.append(report)
    return output


def signed_log(value, threshold):
    """Smooth symmetric log mapping; linear near zero, valid for signed values."""
    magnitude = abs(value)
    # Log-domain fallback avoids overflow for very large envelopes.
    if magnitude <= threshold:
        mapped = math.log1p(magnitude / threshold) / math.log(10.)
    else:
        mapped = (math.log(magnitude) - math.log(threshold)
                  + math.log1p(threshold / magnitude)) / math.log(10.)
    return math.copysign(mapped, value)


def validate_plot_data(points, reports):
    """Reject incomplete/failed JSON results rather than drawing partial bands."""
    if not points or len(reports) != len(RANGES):
        raise ValueError("Need SS data points and all three fit-range reports")
    if sorted(r["range"] for r in reports) != sorted([[5., high] for high in RANGES]):
        raise ValueError("Expected 5--20, 5--30 and 5--50 GeV reports; refit older 5--80 GeV results")
    for point in points:
        if not all(math.isfinite(point[key]) for key in ("x", "half_width", "value", "error")):
            raise ValueError("Nonfinite SS point in plot input")
        if point["error"] < 0 or point["half_width"] < 0:
            raise ValueError("Negative SS error/bin width in plot input")
    for report in reports:
        if not report.get("valid") or report.get("invalid_models"):
            raise ValueError(f"Cannot plot invalid fit range {report['range']}: {report.get('invalid_models')}")
        if any(not report["models"][key]["accepted"] for key in qcd.SS_REQUIRED_ROOT_MODELS):
            raise ValueError(f"Unaccepted required model in fit range {report['range']}")
        n = len(report["x"])
        if n < 2 or any(len(report[key]) != n for key in ("central", "low", "high")):
            raise ValueError("Inconsistent curve/envelope arrays in plot input")
        if any(not math.isfinite(value) for key in ("x", "central", "low", "high")
               for value in report[key]):
            raise ValueError("Nonfinite curve/envelope in plot input")
        if any(a >= b for a, b in zip(report["x"], report["x"][1:])):
            raise ValueError("Curve mass grid must be strictly increasing")
        if any(not low <= central <= high for low, central, high in
               zip(report["low"], report["central"], report["high"])):
            raise ValueError("Envelope does not bracket nominal curve")


def ratio_to_reference(points, reports):
    """Derive display ratios without altering the saved physical predictions.

    Curve bands share a fixed nominal reference, so no independent-denominator
    error propagation is applied to these correlated fits of the same data.
    Data denominators are bin averages, never centre-point evaluations.
    """
    reference = next(r for r in reports if r["range"] == [5., 30.])
    ratios = []
    for report in reports:
        if report["x"] != reference["x"]:
            raise ValueError("Fit curves must share the reference mass grid")
        indices = [i for i, denominator in enumerate(reference["central"])
                   if math.isfinite(denominator) and denominator > 0.]
        result = dict(range=report["range"], x=[report["x"][i] for i in indices])
        for key in ("central", "low", "high"):
            result[key] = [report[key][i]/reference["central"][i] for i in indices]
            if not all(math.isfinite(value) for value in result[key]):
                raise ValueError("Nonfinite ratio to 5--30 GeV nominal reference")
        ratios.append(result)
    bin_key = lambda low, high: (round(low, 8), round(high, 8))
    bin_averages = {bin_key(b["low"], b["high"]): b["central"]
                    for b in reference["bin_averages"]}
    data = []
    for point in points:
        key = bin_key(point["x"]-point["half_width"], point["x"]+point["half_width"])
        if key not in bin_averages:
            raise ValueError(f"Missing bin-averaged reference for data bin {key}")
        denominator = bin_averages[key]
        if denominator <= 0. or not math.isfinite(denominator):
            continue
        data.append(dict(point, value=point["value"]/denominator, error=point["error"]/denominator))
    return ratios, data


def ratio_display_max(args, reports, points):
    """Physical ratio bound; automatic hybrid range includes visible high bands."""
    if args.ratio_max is not None:
        return args.ratio_max
    if args.ratio_scale == "linear":
        return 2.
    values = [5., args.ratio_min+1.]
    values.extend(value for report in reports for x, value in zip(report["x"], report["high"])
                  if 5. <= x <= args.xmax and math.isfinite(value))
    values.extend(p["value"]+p["error"] for p in points
                  if 5. <= p["x"] <= args.xmax and math.isfinite(p["value"]+p["error"]))
    peak = max(values)
    decade = 10.**math.floor(math.log10(peak))
    return next((factor*decade for factor in (1., 2., 5., 10.)
                 if math.isfinite(factor*decade) and factor*decade >= peak), peak)


def ratio_display_value(value, scale, upper):
    """Continuous hybrid display: 0--1.5 linear; 1.5--upper log into 1.5--2.5."""
    if scale == "linear" or value <= 1.5:
        return value
    return 1.5 + (math.log(value)-math.log(1.5))/(math.log(upper)-math.log(1.5))


def ratio_inset_max(reports):
    """Cover every high-band value in 20--50 GeV with a convenient log bound."""
    peak = max([2.] + [value for report in reports
                       for x, value in zip(report["x"], report["high"])
                       if 20. <= x <= FIT_MAX and math.isfinite(value)])
    decade = 10.**math.floor(math.log10(peak))
    return next((factor*decade for factor in (1., 2., 5., 10.)
                 if math.isfinite(factor*decade) and factor*decade >= peak), peak)


def pad_point_to_canvas(pad, x, y, xmin, xmax, ymin, ymax, logx):
    """Map a frame point to canvas NDC, respecting margins and the log x axis."""
    fraction_x = ((math.log(x)-math.log(xmin))/(math.log(xmax)-math.log(xmin))
                  if logx else (x-xmin)/(xmax-xmin))
    fraction_y = (y-ymin)/(ymax-ymin)
    return (pad.GetAbsXlowNDC() + pad.GetAbsWNDC()*(pad.GetLeftMargin()
            + fraction_x*(1.-pad.GetLeftMargin()-pad.GetRightMargin())),
            pad.GetAbsYlowNDC() + pad.GetAbsHNDC()*(pad.GetBottomMargin()
            + fraction_y*(1.-pad.GetBottomMargin()-pad.GetTopMargin())))


def draw_ratio_inset(ROOT, parent, reports, points, logx):
    """Log-y view of the same physical ratios; never modify their stored values."""
    parent.cd()
    ratio_max = ratio_inset_max(reports)
    inset = ROOT.TPad("ss_ratio_high_mass", "", .18, .055, .58, .34)
    inset.SetFillColor(ROOT.kWhite)
    inset.SetFillStyle(1001)
    inset.SetFrameFillColor(ROOT.kWhite)
    inset.SetBorderMode(0)
    inset.SetLeftMargin(.14)
    inset.SetRightMargin(.035)
    inset.SetBottomMargin(.20)
    inset.SetTopMargin(.20)
    inset.SetLogx(logx)
    inset.SetLogy()
    inset.SetTicks(1, 1)
    inset.Draw()
    inset.cd()
    frame = inset.DrawFrame(20., 1., FIT_MAX, ratio_max)
    frame.SetTitle("")
    for axis in (frame.GetXaxis(), frame.GetYaxis()):
        axis.SetTitle("")
        axis.SetLabelFont(43)
        axis.SetLabelSize(16)
        axis.SetNdivisions(505)
    frame.GetXaxis().SetNoExponent()
    frame.GetXaxis().SetMoreLogLabels(logx)
    frame.GetYaxis().SetNoExponent(False)
    frame.GetYaxis().SetMoreLogLabels(ratio_max < 10.)
    keepalive, curves = [inset, frame], []
    for report, hex_colour in zip(reports, ("#2166ac", "#d95f02", "#1b9e77")):
        selected = [i for i, x in enumerate(report["x"]) if 20. <= x <= FIT_MAX]
        if not selected:
            continue
        colour = ROOT.TColor.GetColor(hex_colour)
        n = len(selected)
        band, curve = ROOT.TGraph(2*n), ROOT.TGraph(n)
        for graph in (band, curve):
            graph.SetBit(ROOT.TGraph.kClipFrame)
            graph.SetLineColor(colour)
            graph.SetLineWidth(1)
        for i, j in enumerate(selected):
            x = report["x"][j]
            # Clip signed band endpoints for this log viewport only.
            band.SetPoint(i, x, min(ratio_max, max(1., report["low"][j])))
            band.SetPoint(2*n-1-i, x, min(ratio_max, max(1., report["high"][j])))
            curve.SetPoint(i, x, max(1e-300, report["central"][j]))
        band.SetFillColorAlpha(colour, .17)
        band.Draw("F SAME")
        curves.append(curve)
        keepalive.extend((band, curve))
    for curve in curves:
        curve.Draw("L SAME")
    data = ROOT.TGraphAsymmErrors()
    data.SetBit(ROOT.TGraph.kClipFrame)
    for point in points:
        if not 20. <= point["x"] <= FIT_MAX or point["value"] <= 0.:
            continue
        i, value, error = data.GetN(), point["value"], point["error"]
        data.SetPoint(i, point["x"], value)
        data.SetPointError(i, point["half_width"], point["half_width"],
                           min(error, max(0., value-1e-300)), error)
    data.SetMarkerStyle(20)
    data.SetMarkerSize(.25)
    data.SetMarkerColor(ROOT.kBlack)
    data.SetLineColor(ROOT.kBlack)
    data.Draw("PZ SAME")
    keepalive.append(data)
    title = ROOT.TLatex()
    title.SetNDC()
    title.SetTextFont(43)
    title.SetTextSize(15)
    keepalive.append(title.DrawLatex(.14, .88, f"Ratio to 5-30, 20 < m_{{#mu#mu}} < {FIT_MAX:g} GeV"))
    keepalive.append(title)
    inset.RedrawAxis()
    parent.cd()
    return keepalive


def plot(args, points, reports, stem, ROOT=None):
    """Draw the physical prediction above and fixed-reference ratios below."""
    validate_plot_data(points, reports)
    ratio_reports, ratio_points = ratio_to_reference(points, reports)
    ratio_max = ratio_display_max(args, ratio_reports, ratio_points)
    ratio_transform = lambda value: ratio_display_value(value, args.ratio_scale, ratio_max)
    ROOT = ROOT if ROOT is not None else qcd.import_root()
    log_y = args.ymin > 0.
    transform = (lambda y: max(y, args.ymin)) if log_y else (lambda y: signed_log(y, args.linthresh))
    visible_points = [p for p in points if 5. <= p["x"] <= args.xmax and p["value"] >= args.ymin]
    visible_highs = [high for r in reports for x, high in zip(r["x"], r["high"])
                     if 5. <= x <= args.xmax]
    ymin = args.ymin
    ymax = max(args.linthresh, ymin*10.,
               20.*max([p["value"]+p["error"] for p in visible_points] + visible_highs))
    bottom, top = transform(ymin), transform(ymax)
    margin = .025*(top-bottom)
    canvas = ROOT.TCanvas("ss_range_comparison", "", 900, 900)
    upper = ROOT.TPad("ss_range_upper", "", 0., .32, 1., 1.)
    lower = ROOT.TPad("ss_range_ratio", "", 0., 0., 1., .32)
    for pad in (upper, lower):
        pad.SetLeftMargin(.14)
        pad.SetRightMargin(.05)
        pad.SetLogx(args.logx)
        pad.SetTicks(1, 1)
        pad.Draw()
    upper.SetTopMargin(.10)
    upper.SetBottomMargin(.03)
    lower.SetTopMargin(.04)
    lower.SetBottomMargin(.32)
    upper.cd()
    upper.SetLogy(log_y)
    frame = upper.DrawFrame(5., bottom, args.xmax, top+margin)
    frame.SetTitle("")
    frame.GetXaxis().SetTitle("")
    frame.GetYaxis().SetTitle("Events / GeV" if log_y else "Events / GeV (symmetric log scale)")
    frame.GetXaxis().SetTitleSize(.04)
    frame.GetXaxis().SetLabelSize(0.)
    if args.logx:
        frame.GetXaxis().SetMoreLogLabels()
        frame.GetXaxis().SetNoExponent()
    for axis in (frame.GetXaxis(), frame.GetYaxis()):
        axis.SetTitleFont(43)
        axis.SetLabelFont(43)
        axis.SetTitleSize(32)
    frame.GetYaxis().SetTitleOffset(1.45)
    frame.GetYaxis().SetLabelSize(25 if log_y else 0.)
    frame.GetYaxis().SetTickLength(.02 if log_y else 0.)
    keepalive = [frame, upper, lower]
    labels = ROOT.TLatex()
    labels.SetTextFont(43)
    labels.SetTextSize(25)
    labels.SetTextAlign(32)
    max_abs = max(abs(ymin), abs(ymax))
    ticks = [0.] if not log_y else []
    first_tick = args.linthresh
    for exponent in range(math.floor(math.log10(first_tick)),
                          math.ceil(math.log10(max_abs))+1):
        value = 10.**exponent
        if not log_y:
            ticks.extend(sign*value for sign in (-1., 1.) if ymin <= sign*value <= ymax)
    for value in sorted(ticks):
        y = transform(value)
        grid = ROOT.TLine(5., y, args.xmax, y)
        grid.SetLineColor(ROOT.kGray if value else ROOT.kGray+2)
        grid.SetLineStyle(3 if value else 1)
        grid.Draw()
        keepalive.append(grid)
        if value == 0.:
            label = "0"
        elif abs(value) == 1.:
            label = "-1" if value < 0 else "1"
        else:
            label = ("-" if value < 0 else "") + f"10^{{{round(math.log10(abs(value)))}}}"
        keepalive.append(labels.DrawLatex(4.3, y, label))
    for boundary in (20., 30.):
        if boundary >= args.xmax:
            continue
        line = ROOT.TLine(boundary, bottom, boundary, top+margin)
        line.SetLineColor(ROOT.kGray+1)
        line.SetLineStyle(3)
        line.Draw()
        keepalive.append(line)
    if log_y:
        upper.SetGridy()
    legend_right = 1.-upper.GetRightMargin()-.015
    legend = ROOT.TLegend(legend_right-.54, .68, legend_right, .89)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(43)
    legend.SetTextSize(25)
    legend.SetMargin(.13)
    curves = []
    for report, hex_colour in zip(reports, ("#2166ac", "#d95f02", "#1b9e77")):
        colour = ROOT.TColor.GetColor(hex_colour)
        n = len(report["x"])
        band = ROOT.TGraph(2*n)
        curve = ROOT.TGraph(n)
        band.SetBit(ROOT.TGraph.kClipFrame)
        curve.SetBit(ROOT.TGraph.kClipFrame)
        for i, (x, central, low, high) in enumerate(zip(report["x"], report["central"],
                                                      report["low"], report["high"])):
            band.SetPoint(i, x, transform(low))
            band.SetPoint(2*n-1-i, x, transform(high))
            curve.SetPoint(i, x, max(central, 1e-300) if log_y else transform(central))
        band.SetFillColorAlpha(colour, .17)
        band.SetLineColor(colour)
        band.SetLineWidth(2)
        band.Draw("F SAME")
        curve.SetLineColor(colour)
        curve.SetLineWidth(2)
        curves.append(curve)
        keepalive.extend((band, curve))
        metric = report["models"][qcd.SS_NOMINAL_MODEL]["metrics"]
        label = (f"5-{report['range'][1]:g} GeV fit + envelope"
                 f" (#chi^{{2}}/ndf={metric['stat_chi2']/metric['stat_ndf']:.3f})")
        legend.AddEntry(band, label, "lf")
    for curve in curves:
        curve.Draw("L SAME")
    data = ROOT.TGraphAsymmErrors(len(visible_points))
    data.SetBit(ROOT.TGraph.kClipFrame)
    for i, point in enumerate(visible_points):
        value, error = point["value"], point["error"]
        y = transform(value)
        data.SetPoint(i, point["x"], y)
        data.SetPointError(i, point["half_width"], point["half_width"],
                           y-transform(value-error), transform(value+error)-y)
    data.SetMarkerStyle(20)
    data.SetMarkerSize(.5)
    data.SetLineColor(ROOT.kBlack)
    data.Draw("PZ SAME")
    legend.AddEntry(data, "SS data - Top - Others", "pe")
    legend.Draw()
    text = ROOT.TLatex()
    text.SetNDC()
    text.SetTextFont(43)
    text.SetTextSize(25)
    text.DrawLatex(.14, .94, f"SS fit-range comparison: {args.era}")
    text.SetTextSize(19)
    text.DrawLatex(.17, .415, "Reference: fixed 5-30 GeV nominal fit.")
    text.DrawLatex(.17, .375, "Bands: functional envelope only.")
    upper.RedrawAxis()

    lower.cd()
    lower.SetGridy(args.ratio_scale == "linear")
    ratio_frame = lower.DrawFrame(5., ratio_transform(args.ratio_min), args.xmax, ratio_transform(ratio_max))
    ratio_frame.SetTitle("")
    ratio_frame.GetXaxis().SetTitle("m_{#mu#mu} [GeV]")
    for axis in (ratio_frame.GetXaxis(), ratio_frame.GetYaxis()):
        axis.SetTitleFont(43)
        axis.SetLabelFont(43)
        axis.SetTitleSize(32)
        axis.SetLabelSize(25)
    ratio_frame.GetXaxis().SetTitleOffset(1.05)
    if args.logx:
        ratio_frame.GetXaxis().SetMoreLogLabels()
        ratio_frame.GetXaxis().SetNoExponent()
    ratio_frame.GetYaxis().SetTitle("Ratio to 5-30")
    ratio_frame.GetYaxis().SetTitleOffset(1.45)
    ratio_frame.GetYaxis().SetNdivisions(505)
    keepalive.append(ratio_frame)
    if args.ratio_scale == "hybrid":
        ratio_frame.GetYaxis().SetLabelSize(0.)
        ratio_frame.GetYaxis().SetTickLength(0.)
        candidates = [0., .5, 1., 1.5]
        for exponent in range(0, min(308, math.ceil(math.log10(ratio_max)))+1):
            candidates.extend(factor*10.**exponent for factor in (1., 2., 5.))
        candidates = sorted(set(v for v in candidates if args.ratio_min <= v <= ratio_max))
        span = ratio_transform(ratio_max)-ratio_transform(args.ratio_min)
        height = canvas.GetWh()*.32*(1.-lower.GetTopMargin()-lower.GetBottomMargin())
        gap = 25.*span/height
        ticks = []
        for value in candidates:
            if not ticks or ratio_transform(value)-ratio_transform(ticks[-1]) >= gap:
                ticks.append(value)
        if not ticks or ticks[-1] != ratio_max:
            if ticks and ratio_transform(ratio_max)-ratio_transform(ticks[-1]) < gap:
                ticks.pop()
            ticks.append(ratio_max)
        tick_text = ROOT.TLatex()
        tick_text.SetTextFont(43)
        tick_text.SetTextSize(25)
        tick_text.SetTextAlign(32)
        label_x = 5./1.04 if args.logx else 5.-.015*(args.xmax-5.)
        for value in ticks:
            y = ratio_transform(value)
            grid = ROOT.TLine(5., y, args.xmax, y)
            grid.SetLineColor(ROOT.kGray+1)
            grid.SetLineStyle(3)
            grid.Draw()
            keepalive.extend((grid, tick_text.DrawLatex(label_x, y, f"{value:g}")))
        keepalive.append(tick_text)
    ratio_curves = []
    for report, hex_colour in zip(ratio_reports, ("#2166ac", "#d95f02", "#1b9e77")):
        colour = ROOT.TColor.GetColor(hex_colour)
        n = len(report["x"])
        if n == 0:
            continue
        band, curve = ROOT.TGraph(2*n), ROOT.TGraph(n)
        for graph in (band, curve):
            graph.SetBit(ROOT.TGraph.kClipFrame)
            graph.SetLineColor(colour)
            graph.SetLineWidth(2)
        for i, (x, value, low, high) in enumerate(zip(report["x"], report["central"],
                                                      report["low"], report["high"])):
            band.SetPoint(i, x, ratio_transform(low))
            band.SetPoint(2*n-1-i, x, ratio_transform(high))
            curve.SetPoint(i, x, ratio_transform(value))
        band.SetFillColorAlpha(colour, .17)
        band.Draw("F SAME")
        ratio_curves.append(curve)
        keepalive.extend((band, curve))
    for curve in ratio_curves:
        curve.Draw("L SAME")
    unity = ROOT.TLine(5., 1., args.xmax, 1.)
    unity.SetLineColor(ROOT.kGray+2)
    unity.SetLineStyle(2)
    unity.Draw()
    ratio_data = ROOT.TGraphAsymmErrors()
    ratio_data.SetBit(ROOT.TGraph.kClipFrame)
    for point in ratio_points:
        if not 5. <= point["x"] <= args.xmax:
            continue
        i = ratio_data.GetN()
        value, error = point["value"], point["error"]
        y = ratio_transform(value)
        ratio_data.SetPoint(i, point["x"], y)
        ratio_data.SetPointError(i, point["half_width"], point["half_width"],
                                y-ratio_transform(value-error), ratio_transform(value+error)-y)
    ratio_data.SetMarkerStyle(20)
    ratio_data.SetMarkerSize(.5)
    ratio_data.SetLineColor(ROOT.kBlack)
    ratio_data.Draw("PZ SAME")
    keepalive.extend((unity, ratio_data))
    if args.ratio_scale == "hybrid":
        note = ROOT.TLatex()
        note.SetNDC()
        note.SetTextFont(43)
        note.SetTextSize(17)
        keepalive.append(note.DrawLatex(.17, .87, "Linear up to 1.5; log above"))
        keepalive.append(note)
    lower.RedrawAxis()
    if args.ratio_inset:
        inset_objects = draw_ratio_inset(ROOT, upper, ratio_reports, ratio_points, args.logx)
        keepalive.extend(inset_objects)
        inset = inset_objects[0]
        # The box identifies the mass interval; the inset expands its y range.
        xleft, xright = 20., min(FIT_MAX, args.xmax)
        ybottom, ytop = max(.9, args.ratio_min), min(1.9, ratio_max)
        if xleft < xright and ybottom < ytop:
            lower.cd()
            box = ROOT.TBox(xleft, ratio_transform(ybottom), xright, ratio_transform(ytop))
            box.SetFillStyle(0)
            box.SetLineColor(ROOT.kGray+2)
            box.SetLineStyle(2)
            box.SetLineWidth(2)
            box.Draw("SAME")
            keepalive.append(box)
            canvas.cd()
            canvas.Update()
            target_left = inset.GetAbsXlowNDC()+inset.GetAbsWNDC()*inset.GetLeftMargin()
            target_right = inset.GetAbsXlowNDC()+inset.GetAbsWNDC()*(1.-inset.GetRightMargin())
            target_y = inset.GetAbsYlowNDC()+inset.GetAbsHNDC()*inset.GetBottomMargin()
            for x, target_x in ((xleft, target_left), (xright, target_right)):
                origin = pad_point_to_canvas(lower, x, ratio_transform(ytop), 5., args.xmax,
                                             ratio_transform(args.ratio_min), ratio_transform(ratio_max), args.logx)
                connector = ROOT.TLine(*origin, target_x, target_y)
                connector.SetNDC(True)
                connector.SetLineColor(ROOT.kGray+2)
                connector.SetLineStyle(2)
                connector.SetLineWidth(1)
                connector.Draw()
                keepalive.append(connector)
    canvas.cd()
    canvas.Update()
    try:
        for extension in ("pdf", "png"):
            path = stem.with_suffix("." + extension)
            canvas.SaveAs(str(path))
            print(f"[SAVED] {path}")
    finally:
        canvas.Close()


def replot_saved(args):
    """Preserve saved fit results and JSON bytes; only regenerate PDF/PNG."""
    ROOT = qcd.import_root()
    for path in args.plot_json:
        payload = json.loads(path.read_text())
        plot_args = argparse.Namespace(**vars(args))
        plot_args.era = payload["era"]
        output = args.output_dir if args.output_dir is not None else path.parent
        output.mkdir(parents=True, exist_ok=True)
        plot(plot_args, payload["points"], payload["fits"], output / path.stem, ROOT=ROOT)
    return 0


def run_all_periods(argv):
    """Use the normal single-period command for each period, isolating ROOT state."""
    # Also remove argparse's accepted abbreviations of this switch.
    forwarded = [arg for arg in argv
                 if not (arg.startswith("--") and "--all-periods".startswith(arg))]
    failed = []
    for period in ALL_PERIODS:
        print(f"[PERIOD] {period}", flush=True)
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                    *forwarded, "--era", period], check=False)
        if completed.returncode != 0:
            failed.append(period)
            print(f"[FAILED] {period}: exit code {completed.returncode}", file=sys.stderr)
    if failed:
        print("[FAILED PERIODS] " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


def main(argv=None):
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    if not math.isfinite(args.ymin) or args.ymin < 0.:
        parser.error("--ymin must be finite and nonnegative")
    if not 5. < args.xmax <= FIT_MAX:
        parser.error("--xmax must satisfy 5 < xmax <= 50 GeV (display only)")
    if not math.isfinite(args.ratio_min) or (args.ratio_max is not None and
            (not math.isfinite(args.ratio_max) or args.ratio_min >= args.ratio_max)):
        parser.error("Require finite --ratio-min < --ratio-max")
    if args.ratio_max is None and args.ratio_scale == "linear" and args.ratio_min >= 2.:
        parser.error("--ratio-min must be below the default linear maximum of 2")
    if args.ratio_scale == "hybrid" and (args.ratio_min >= 1.5 or
            (args.ratio_max is not None and args.ratio_max <= 1.5)):
        parser.error("Hybrid ratio bounds must straddle 1.5")
    if args.all_periods:
        if args.plot_json:
            parser.error("--all-periods runs fits; use --plot-json separately to redraw saved results")
        return run_all_periods(argv)
    if args.plot_json:
        return replot_saved(args)
    args.era = "Run2+3" if args.era == "full" else args.era
    if args.output_dir is None:
        args.output_dir = Path("plots/SSFitRangeComparison")
    ROOT = qcd.import_root()
    qcd.declare_fit_functions(ROOT)
    qcd.configure_minimizer(ROOT)
    base = Path(args.base_dir).expanduser()
    directories = [(era, base / era / args.trigger) for era in qcd.selected_eras(args.era)]
    source, _, handles = qcd.build_input_histogram(ROOT, qcd.SS_MODE, directories)
    try:
        prepared = common_histograms(ROOT, source, args)
        reports = fit_ranges(ROOT, prepared, args)
        points = [dict(x=float(prepared.density.GetBinCenter(i)),
                       half_width=float(prepared.density.GetBinWidth(i))/2.,
                       value=float(prepared.density.GetBinContent(i)),
                       error=float(prepared.density.GetBinError(i)))
                  for i in range(1, prepared.density.GetNbinsX()+1)
                  if 5. <= prepared.density.GetBinCenter(i) <= FIT_MAX]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = args.output_dir / f"QCD_SS_FitRangeComparison_{args.era}"
        payload = dict(era=args.era, eras=list(qcd.selected_eras(args.era)),
                       combination="sum data and each subtraction MC before fitting",
                       objective="bin-integrated diagonal-error chi2; signed contents retained",
                       nominal=qcd.SS_NOMINAL_MODEL, alternatives=list(qcd.SS_SHAPE_ALTERNATIVES),
                       envelope="symmetric functional only; no clipping; not a confidence interval",
                       binning=dict(edges=prepared.edges, min_effective_count=args.ss_min_effective_count,
                                    max_width=args.ss_max_bin_width,
                                    adaptive_range=[5., 30.],
                                    merge_nonpositive_next_native_bin=False,
                                    merge_nonpositive_next_adaptive_bin=True,
                                    merge_terminal_adaptive_bin_backward=True,
                                    terminal_backward_merge_condition="last yield >0 and preceding yield <=0; once",
                                    nonpositive_next_requires_positive_sum=True,
                                    minimum_width=dict(absolute_gev=.1, relative_to_bin_centre=.02),
                                    tail_merge_start=30.,
                                    fixed_high_mass_edges=list(FIXED_HIGH_MASS_EDGES),
                                    forced_edges=list(FORCED_EDGES)),
                       points=points, fits=reports)
        stem.with_suffix(".json").write_text(json.dumps(finite_json(payload), indent=2, allow_nan=False) + "\n")
        print(f"[SAVED] {stem.with_suffix('.json')}")
        failed = [f"5--{r['range'][1]:g}: {', '.join(r['invalid_models'])}"
                  for r in reports if not r["valid"]]
        if failed:
            raise RuntimeError("No comparison band drawn: required fits failed validity checks. "
                               "See JSON; retry with more --fit-max-attempts. " + "; ".join(failed))
        plot(args, points, reports, stem, ROOT=ROOT)
    finally:
        for handle in handles:
            handle.Close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
