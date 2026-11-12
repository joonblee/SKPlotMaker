"""Conditional, diagonal-error constant/linear fits for ratio diagnostics.

Points are (x, horizontal_half_width, ratio, statistical_error). The horizontal
width defines bin boundaries, not a measurement error. Normalisation factors
are held fixed: no normalisation covariance or denominator likelihood is fitted.
Chi-square/ndf is a descriptive diagnostic, not a calibrated closure p-value.
"""

import json
import math
from pathlib import Path


def add_arguments(parser, default_range=None):
    parser.add_argument("--ratio-fits", choices=("none", "const", "linear", "both"),
                        default="both", help="Ratio-panel fits (default: both)")
    parser.add_argument("--ratio-fit-range", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        help=("Fit only whole bins within LOW HIGH GeV; default: "
                              + (f"{default_range[0]:g}--{default_range[1]:g} GeV, clipped to plot"
                                 if default_range else "plotted range")))
    parser.set_defaults(ratio_fit_default_range=default_range)


def fit_range(args):
    if args.ratio_fit_range:
        low, high = args.ratio_fit_range
    elif getattr(args, "ratio_fit_default_range", None) and args.ratio_fits != "none":
        low = max(args.xmin, args.ratio_fit_default_range[0])
        high = min(args.xmax, args.ratio_fit_default_range[1])
    else:
        low, high = args.xmin, args.xmax
    if not all(map(math.isfinite, (low, high))) or not args.xmin <= low < high <= args.xmax:
        raise ValueError("Require xmin <= ratio-fit LOW < HIGH <= xmax")
    return low, high


def histogram_points(hist):
    return [(float(hist.GetBinCenter(i)), float(hist.GetBinWidth(i)) / 2.,
             float(hist.GetBinContent(i)), float(hist.GetBinError(i)))
            for i in range(1, hist.GetNbinsX() + 1)]


def selected_points(points, low, high, excluded=()):
    selected = []
    for x, ex, y, ey in points:
        if not all(map(math.isfinite, (x, ex, y, ey))) or ey <= 0 or ex < 0:
            continue
        left, right = x - ex, x + ex
        if left < low - 1e-8 or right > high + 1e-8:
            continue
        if any(left < stop - 1e-8 and right > start + 1e-8 for start, stop in excluded):
            continue
        selected.append((x, ex, y, ey))
    return selected


def fit_points(points, kind):
    """Weighted least squares with known errors; retain negative ratio values."""
    if kind not in ("const", "linear"):
        raise ValueError(f"Unknown ratio fit: {kind}")
    npar = 1 if kind == "const" else 2
    result = dict(kind=kind, n_points=len(points), ndf=len(points) - npar)
    if len(points) <= npar:
        return dict(result, status="insufficient bins")
    weights = [1. / p[3]**2 for p in points]
    sw = math.fsum(weights)
    xm = math.fsum(w * p[0] for w, p in zip(weights, points)) / sw
    ym = math.fsum(w * p[2] for w, p in zip(weights, points)) / sw
    if kind == "const":
        parameters, errors = dict(c=ym), dict(c=math.sqrt(1. / sw))
        predict = lambda x: ym
    else:
        sxx = math.fsum(w * (p[0] - xm)**2 for w, p in zip(weights, points))
        if sxx <= 0:
            return dict(result, status="singular mass coordinates")
        a = math.fsum(w * (p[0] - xm) * (p[2] - ym)
                      for w, p in zip(weights, points)) / sxx
        b = ym - a * xm
        parameters, errors = dict(a=a, b=b), dict(a=math.sqrt(1. / sxx),
                                                b=math.sqrt(1. / sw + xm*xm / sxx))
        result["cov_a_b"] = -xm / sxx
        predict = lambda x: a * x + b
    chi2 = math.fsum(((p[2] - predict(p[0])) / p[3])**2 for p in points)
    return dict(result, status="ok", parameters=parameters, errors=errors,
                chi2=chi2, chi2_ndf=chi2 / result["ndf"])


def analyse(points, args, excluded=()):
    low, high = fit_range(args)
    chosen = selected_points(points, low, high, excluded)
    kinds = ("const", "linear") if args.ratio_fits == "both" else (args.ratio_fits,)
    return dict(range=[low, high], excluded=[list(x) for x in excluded],
                bins=[[x-ex, x+ex] for x, ex, _, _ in chosen],
                fits=[] if args.ratio_fits == "none" else [fit_points(chosen, k) for k in kinds])


def equation(result):
    """ROOT TLatex equation with symmetric errors as superscript +/- values."""
    p, e = result["parameters"], result["errors"]
    def number(value, error):
        return f"{value:.4g}^{{#pm {error:.2g}}}"
    if result["kind"] == "const":
        return "y = " + number(p["c"], e["c"])
    sign = " - " if p["b"] < 0 else " + "
    return "y = (" + number(p["a"], e["a"]) + ") x" + sign + number(abs(p["b"]), e["b"])


def draw(ROOT, series):
    """Draw fits and an upper-right results box; return ROOT keepalive objects."""
    objects, lines = [], []
    for label, colour, report in series:
        if not report["fits"]:
            continue
        low, high = report["range"]
        if report["bins"]:
            low, high = report["bins"][0][0], report["bins"][-1][1]
        lines.append((colour, f"{label}: {low:g}--{high:g} GeV (stat., fixed scale)"))
        for result in report["fits"]:
            kind = result["kind"]
            if result["status"] != "ok":
                lines.append((colour, f"{kind}: {result['status']}"))
                continue
            p = result["parameters"]
            quality = f"#chi^{{2}}/ndf = {result['chi2_ndf']:.3f}"
            lines.append((colour, equation(result) + ";  " + quality))
            # Draw within the support of eligible bins, leaving veto/blind gaps.
            segments = []
            for left, right in report["bins"]:
                if segments and abs(left - segments[-1][1]) < 1e-8:
                    segments[-1][1] = right
                else:
                    segments.append([left, right])
            for left, right in segments:
                fn = ROOT.TF1(f"ratio_fit_{len(objects)}_{label.replace(' ', '_')}",
                              "[0]" if kind == "const" else "[0]*x+[1]", left, right)
                if kind == "const":
                    fn.SetParameter(0, p["c"])
                else:
                    fn.SetParameters(p["a"], p["b"])
                fn.SetLineColor(colour)
                fn.SetLineStyle(2 if kind == "const" else 1)
                fn.SetLineWidth(2)
                fn.Draw("SAME")
                objects.append(fn)
    if lines:
        box = ROOT.TPaveText(.54, max(.36, .95 - .075*len(lines)), .945, .95, "NDC")
        box.SetBorderSize(0)
        box.SetFillStyle(0)
        box.SetTextFont(42)
        box.SetTextAlign(12)
        box.SetTextSize(.05)
        for colour, line in lines:
            box.AddText(line).SetTextColor(colour)
        box.Draw()
        objects.append(box)
    return objects


def save(path, series, **metadata):
    payload = dict(error_model="conditional diagonal statistical errors; scale fixed; no closure p-value",
                   **metadata, series={label: report for label, _, report in series})
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"[SAVED] {path}")
