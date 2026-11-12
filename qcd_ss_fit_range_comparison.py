#!/usr/bin/env python3
"""Compare SS Data - Top - Others fits in 5--20, 5--30 and 5--80 GeV.

Reuse the production six function families, nominal Power x Exp x Logistic,
bin-integrated chi-square and fit-validity rules. Signed bins are never clipped.
All ranges use one adaptive 5--80 GeV grid, split at 20 and 30 GeV; thus only
the fit range changes. This diagnostic grid can differ from production 5--30
adaptive binning. The common-grid edges and all fit diagnostics are exported.

Bands are pointwise nominal +/- max_alternative |alternative - nominal|,
without zero-clipping. They are functional envelopes, not confidence intervals
or the total statistical + systematic uncertainty. The symmetric-logarithmic
y axis retains negative observations and negative lower band edges.

For combined periods, sum the SS data and each subtraction MC histogram over
the requested eras first, then fit that summed histogram once per range.
Only SS histograms are read. No production anchors/templates are written.

Examples (base-dir includes NIsoMuon, like the comparison plotters):
  python3 qcd_ss_fit_range_comparison.py --era 2018
  python3 qcd_ss_fit_range_comparison.py --era full
  python3 qcd_ss_fit_range_comparison.py --plot-json plots/SSFitRangeComparison/*.json
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

import qcd_bkg_estimation as qcd

RANGES = (30., 20., 80.)
DEFAULT_BASE = str(Path(qcd.DEFAULT_BASE_DIR) / "NIsoMuon")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--era", "--year", choices=(*qcd.VALID_YEARS, "full"), default="full")
    parser.add_argument("--base-dir", default=DEFAULT_BASE,
                        help="Input directory containing era directories (includes NIsoMuon)")
    parser.add_argument("--trigger", default="")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: plots/SSFitRangeComparison; for --plot-json, beside each input JSON")
    parser.add_argument("--plot-json", type=Path, nargs="+", metavar="JSON",
                        help="Redraw saved JSON results only: no input histograms, fits or JSON writes")
    parser.add_argument("--ss-min-effective-count", type=qcd.positive_float, default=10.)
    parser.add_argument("--ss-max-bin-width", type=qcd.positive_float, default=5.)
    parser.add_argument("--fit-max-attempts", type=qcd.positive_integer, default=30)
    parser.add_argument("--fit-attempt-details", action="store_true")
    parser.add_argument("--linthresh", type=qcd.positive_float, default=.1,
                        help="Linear y-axis neighbourhood around zero, in events/GeV")
    return parser


def common_histograms(ROOT, source, args):
    """Use production adaptive merging; force shared boundaries at 20 and 30."""
    mode = replace(qcd.SS_MODE, fit_max=80.)
    edges = qcd.make_adaptive_ss_binning(source, mode, args.ss_min_effective_count,
                                        args.ss_max_bin_width)
    native = [float(source.GetXaxis().GetBinLowEdge(i))
              for i in range(1, source.GetNbinsX()+2)]
    for boundary in (20., 30.):
        if not any(math.isclose(x, boundary, abs_tol=1e-8, rel_tol=0.) for x in native):
            raise ValueError(f"{boundary:g} GeV is not a native bin edge; cannot split an input bin")
    edges = sorted(set([round(x, 9) for x in edges] + [20., 30.]))
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
    x = [5. + i * .05 for i in range(1501)]
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
                if left < 5. or right > 80.:
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
        raise ValueError("Expected 5--20, 5--30 and 5--80 GeV reports")
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


def plot(args, points, reports, stem, ROOT=None):
    """Draw ROOT graphs in transformed y coordinates, with physical y labels.

    Transform each endpoint of an error/band separately: neither negative
    values nor statistical intervals crossing zero are lost or symmetrised
    in the displayed coordinates. No array libraries or BLAS calls are used.
    """
    validate_plot_data(points, reports)
    ROOT = ROOT if ROOT is not None else qcd.import_root()
    transform = lambda y: signed_log(y, args.linthresh)
    ymin = min([p["value"]-p["error"] for p in points] + [min(r["low"]) for r in reports])
    ymax = max([p["value"]+p["error"] for p in points] + [max(r["high"]) for r in reports])
    ymin, ymax = min(-args.linthresh, ymin*1.6), max(args.linthresh, ymax*2.)
    bottom, top = transform(ymin), transform(ymax)
    margin = .025*(top-bottom)
    canvas = ROOT.TCanvas("ss_range_comparison", "", 1100, 850)
    canvas.SetLeftMargin(.14)
    canvas.SetRightMargin(.035)
    canvas.SetTopMargin(.08)
    canvas.SetBottomMargin(.16)
    frame = canvas.DrawFrame(5., bottom-margin, 80., top+margin)
    frame.SetTitle("")
    frame.GetXaxis().SetTitle("m_{#mu#mu} [GeV]")
    frame.GetYaxis().SetTitle("Events / GeV (symmetric log scale)")
    frame.GetXaxis().SetTitleSize(.04)
    frame.GetXaxis().SetLabelSize(.035)
    frame.GetYaxis().SetTitleSize(.04)
    frame.GetYaxis().SetTitleOffset(1.55)
    frame.GetYaxis().SetLabelSize(0.)
    frame.GetYaxis().SetTickLength(0.)
    keepalive = [frame]
    labels = ROOT.TLatex()
    labels.SetTextFont(42)
    labels.SetTextSize(.033)
    labels.SetTextAlign(32)
    max_abs = max(abs(ymin), abs(ymax))
    ticks = [0.]
    for exponent in range(math.floor(math.log10(args.linthresh)),
                          math.ceil(math.log10(max_abs))+1):
        value = 10.**exponent
        ticks.extend(sign*value for sign in (-1., 1.) if ymin <= sign*value <= ymax)
    for value in sorted(ticks):
        y = transform(value)
        grid = ROOT.TLine(5., y, 80., y)
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
        line = ROOT.TLine(boundary, bottom-margin, boundary, top+margin)
        line.SetLineColor(ROOT.kGray+1)
        line.SetLineStyle(3)
        line.Draw()
        keepalive.append(line)
    legend = ROOT.TLegend(.40, .735, .955, .905)
    legend.SetBorderSize(0)
    legend.SetFillColor(ROOT.kWhite)
    legend.SetTextFont(42)
    legend.SetTextSize(.028)
    curves = []
    for report, hex_colour in zip(reports, ("#2166ac", "#d95f02", "#1b9e77")):
        colour = ROOT.TColor.GetColor(hex_colour)
        n = len(report["x"])
        band = ROOT.TGraph(2*n)
        curve = ROOT.TGraph(n)
        for i, (x, central, low, high) in enumerate(zip(report["x"], report["central"],
                                                      report["low"], report["high"])):
            band.SetPoint(i, x, transform(low))
            band.SetPoint(2*n-1-i, x, transform(high))
            curve.SetPoint(i, x, transform(central))
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
                 f" (#chi^{{2}}/ndf={metric['stat_chi2']:.1f}/{metric['stat_ndf']})")
        legend.AddEntry(band, label, "lf")
    for curve in curves:
        curve.Draw("L SAME")
    data = ROOT.TGraphAsymmErrors(len(points))
    for i, point in enumerate(points):
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
    text.SetTextFont(42)
    text.SetTextSize(.034)
    text.DrawLatex(.14, .95, f"SS fit-range comparison: {args.era}")
    text.SetTextSize(.026)
    text.DrawLatex(.14, .065, "Bands: symmetric functional envelope only; signed bins and lower edges retained.")
    text.DrawLatex(.14, .03, f"Symmetric log axis, linear near zero (scale {args.linthresh:g} events/GeV).")
    canvas.RedrawAxis()
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


def main(argv=None):
    args = build_parser().parse_args(argv)
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
                  if 5. <= prepared.density.GetBinCenter(i) <= 80.]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = args.output_dir / f"QCD_SS_FitRangeComparison_{args.era}"
        payload = dict(era=args.era, eras=list(qcd.selected_eras(args.era)),
                       combination="sum data and each subtraction MC before fitting",
                       objective="bin-integrated diagonal-error chi2; signed contents retained",
                       nominal=qcd.SS_NOMINAL_MODEL, alternatives=list(qcd.SS_SHAPE_ALTERNATIVES),
                       envelope="symmetric functional only; no clipping; not a confidence interval",
                       binning=dict(edges=prepared.edges, min_effective_count=args.ss_min_effective_count,
                                    max_width=args.ss_max_bin_width, forced_edges=[20., 30.]),
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
