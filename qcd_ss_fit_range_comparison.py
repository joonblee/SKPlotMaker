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
Requires PyROOT, NumPy and Matplotlib. --help does not require ROOT.
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
    parser.add_argument("--output-dir", type=Path, default=Path("plots/SSFitRangeComparison"))
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


def plot(args, points, reports, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7.5))
    for report, colour in zip(reports, ("#2166ac", "#d95f02", "#1b9e77")):
        upper = report["range"][1]
        metric = report["models"][qcd.SS_NOMINAL_MODEL]["metrics"]
        label = (f"5–{upper:g} GeV fit: nominal + functional envelope"
                 f" ($\\chi^2$/ndf={metric['stat_chi2']:.1f}/{metric['stat_ndf']})")
        ax.fill_between(report["x"], report["low"], report["high"], color=colour, alpha=.15)
        ax.plot(report["x"], report["central"], color=colour, linewidth=1.8, label=label)
    ax.errorbar([p["x"] for p in points], [p["value"] for p in points],
                yerr=[p["error"] for p in points], xerr=[p["half_width"] for p in points],
                fmt="o", markersize=3, color="black", linewidth=.8, label="SS data − Top − Others", zorder=5)
    ax.axhline(0., color="grey", linewidth=.8)
    for boundary in (20., 30.):
        ax.axvline(boundary, color="grey", linestyle=":", linewidth=.8)
    ax.set(xlim=(5., 80.), xlabel=r"$m_{\mu\mu}$ [GeV]", ylabel="Events / GeV")
    ax.set_yscale("symlog", linthresh=args.linthresh)
    ymin = min([p["value"]-p["error"] for p in points] + [min(r["low"]) for r in reports])
    ymax = max([p["value"]+p["error"] for p in points] + [max(r["high"]) for r in reports])
    ax.set_ylim(min(-args.linthresh, ymin*1.6), max(args.linthresh, ymax*2.))
    ax.set_title(f"SS fit-range comparison — {args.era}", loc="left")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=.2)
    fig.text(.12, .025, "Bands: functional envelope only; symmetric and unclipped. Signed bins retained.", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    for extension in ("pdf", "png"):
        path = stem.with_suffix("." + extension)
        fig.savefig(path, dpi=160)
        print(f"[SAVED] {path}")
    plt.close(fig)


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.era = "Run2+3" if args.era == "full" else args.era
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
        plot(args, points, reports, stem)
    finally:
        for handle in handles:
            handle.Close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
