#!/usr/bin/env python3
"""Compare selected prefit background uncertainties in a single mass-ratio panel.

Example (same inputs, binning, data mode and bands as plotter.py)::

    python3 dimuon_mass_systematics.py --era Run2+3 --variable dimuon_mass \
        --uncertainty syst+stat --xmax 80 --strict

Add --blind to hide the search-region data, exactly as in plotter.py. The
inherited default is unblind. --ratio-min/--ratio-max control the vertical range.
--systematics accepts a comma-separated subset of the ten default nuisance
names, with or without CMS_NPS26009_. --list-systematics needs no ROOT or inputs.

The data points and stat/stat+syst bands come directly from plotter.py. Coloured
step boundaries show each source's envelope, 1 - delta_low/B and
1 + delta_high/B, for TOTAL nominal background B. These are prefit yield
responses, not Combine impacts. Correlated era/process shifts are combined
before forming each envelope, using plotter.py's theory and metadata helpers.
All sources, including those not individually drawn, remain in the total band.

Keep plotter.py's statistical bookkeeping: QCD_MCTransferStat_Run2 is already
inside stat; DY NFStat is in syst. An overlay is never added again to the total.
Missing selected templates/invalid metadata are errors; absent eras or an
explicit MC background mode make the corresponding DD source inapplicable.

PDF/PNG and a JSON sidecar are written under --output-dir (default: plots).
The JSON contains per-bin uncertainties, background composition, signed ttbar
scale responses and audit warnings; it contains no observed data. --save-root
also writes the nominal background and ratio uncertainty graphs. Signal/yield
panel options are inapplicable. No input files or production code are modified.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field, replace

import plotter as p


PREFIX = "CMS_NPS26009_"
DEFAULT_SYSTEMATICS = (
    "QCDscale_ren_ttbar",
    "QCDscale_fac_ttbar",
    PREFIX + "NFModel_DY_BJetOS_13TeV",
    PREFIX + "NFModel_DY_BJetOS_13p6TeV",
    PREFIX + "topmass_ttbar_BJetOS",
    PREFIX + "pdf_NNPDF31",
    "pdf_alphas",
    PREFIX + "QCD_MCTransferStat_Run2",
    PREFIX + "NFStat_DY_BJetOS_2018",
    PREFIX + "NFStat_DY_BJetOS_2016preVFP",
)


@dataclass
class Component:
    uncertainty: p.Uncertainty
    included_in: str = "syst"
    signed_shifts: dict = field(default_factory=dict)


def short_name(name):
    return name[len(PREFIX):] if name.startswith(PREFIX) else name


def selected_names(values):
    known = {short_name(name): name for name in DEFAULT_SYSTEMATICS}
    selected = []
    for value in values:
        key = short_name(value)
        if key not in known:
            raise ValueError(f"Unknown systematic {value!r}; use --list-systematics.")
        if known[key] not in selected:
            selected.append(known[key])
    return selected


def pair_component(down, up, nbins, labels=None):
    low2, high2 = [0.0] * nbins, [0.0] * nbins
    p.add_delta_pair_shift(down, up, low2, high2)
    signed = {}
    if labels is not None:
        for label, hist in zip(labels, (down, up)):
            signed[label] = [float(hist.GetBinContent(i + 1)) for i in range(nbins)]
    return Component(p.Uncertainty([math.sqrt(x) for x in low2],
                                   [math.sqrt(x) for x in high2]), signed_shifts=signed)


def symmetric_component(hist, nbins, included_in="syst"):
    low2, high2 = [0.0] * nbins, [0.0] * nbins
    p.add_positive_sigma_hist_shift(hist, low2, high2)
    sigma = [math.sqrt(x) for x in low2]
    return Component(p.Uncertainty(sigma[:], sigma[:]), included_in)


def dy_components(ROOT, cfg, inputs, edges, scale, requested):
    """Read the same NF scalars and check the same factorisation as plotter.py."""
    result, model_parts = {}, {"Run2": [], "Run3": []}
    n = inputs.bkg_total.GetNbinsX()
    for year in inputs.years:
        group = p.run_group(year)
        model_name = PREFIX + "NFModel_DY_BJetOS_" + ("13TeV" if group == "Run2" else "13p6TeV")
        stat_name = PREFIX + "NFStat_DY_BJetOS_" + year
        if not ({model_name, stat_name} & set(requested)):
            continue
        filename = os.path.join(p.root_dir_for_year(cfg, year), cfg.dy_data_driven_file)
        objects = []
        for path in (p.DY_AUX_NF_AMC_PATH, p.DY_AUX_NF_MG_PATH, "DYAux/LightJetSource"):
            hist, error = p.read_hist(ROOT, filename, path)
            if error or not hist:
                raise RuntimeError(f"{year}: {error or 'missing ' + path}")
            objects.append(hist)
        amc, mg, source = objects
        nf, nf_error, nf_mg = float(amc.GetBinContent(1)), abs(float(amc.GetBinError(1))), float(mg.GetBinContent(1))
        if (amc.GetNbinsX() != 1 or mg.GetNbinsX() != 1 or nf <= 0 or nf_mg <= 0
                or not all(math.isfinite(x) for x in (nf, nf_error, nf_mg))):
            raise RuntimeError(f"{year}: invalid DY NF metadata.")
        nominal = inputs.bkg_by_year[year]["DY"]
        source = p.rebin_hist(source, edges)
        p.apply_scale(source, scale * nf * p.period_prediction_scale(cfg, year))
        p.apply_bin_width_normalization(cfg, source)
        for getter in (lambda h, i: h.GetBinContent(i), lambda h, i: h.GetBinError(i)):
            if any(not math.isclose(getter(source, i), getter(nominal, i), rel_tol=1e-5, abs_tol=1e-280)
                   for i in range(1, n + 1)):
                raise RuntimeError(f"{year}: DY nominal/LightJetStat disagree with LightJetSource * NF.")
        if stat_name in requested:
            delta = nominal.Clone(p._NAMES.unique("nf_stat"))
            delta.SetDirectory(0)
            delta.Scale(nf_error / nf)
            result[stat_name] = symmetric_component(delta, n)
        if model_name in requested:
            delta = nominal.Clone(p._NAMES.unique("nf_model"))
            delta.SetDirectory(0)
            delta.Scale(abs(nf_mg / nf - 1.0))
            model_parts[group].append(delta)
    for group, parts in model_parts.items():
        if parts:
            name = PREFIX + "NFModel_DY_BJetOS_" + ("13TeV" if group == "Run2" else "13p6TeV")
            result[name] = symmetric_component(p.sum_hists(parts, "nf_model_shared"), n)
    return result


def qcd_transfer_component(ROOT, cfg, inputs, scale):
    """Extract the existing Run-2 shared statistical term from its derivative basis."""
    sigma = [0.0] * inputs.bkg_total.GetNbinsX()
    per_width = cfg.divide_by_bin_width and p.variable_spec(cfg).divide_by_bin_width
    identity = None
    for year in inputs.years:
        if year not in p.RUN2_ERAS:
            continue
        filename = p.process_file(cfg, year, "QCD", -1.0, "", [], report_missing=False)
        f = ROOT.TFile.Open(filename, "READ")
        try:
            if not f or f.IsZombie():
                raise RuntimeError(f"Could not open {filename}")
            basis = p.read_qcd_stat_inputs(f, year, p.hist_path(cfg, p.base_region(cfg)))
            transfer = basis[0]["transfer_statistics"]
            if transfer.get("method") != "run-common":
                raise RuntimeError(f"{year}: QCD_MCTransferStat_Run2 requires run-common QCD metadata.")
            common = transfer["common_transfer"]
            if identity is not None and identity != common["id"]:
                raise RuntimeError("Mixed Run-2 QCD common-transfer identities.")
            identity = common["id"]
            nominal, native = inputs.bkg_by_year[year]["QCD"], basis[1]
            axis = nominal.GetXaxis()
            for i in range(len(sigma)):
                low, high = axis.GetBinLowEdge(i + 1), axis.GetBinUpEdge(i + 1)
                if high <= native.GetXaxis().GetXmin() or low >= native.GetXaxis().GetXmax():
                    continue
                actual_low, actual_high = p.whole_bin_window(native, low, high)
                if not (math.isclose(low, actual_low, abs_tol=1e-8)
                        and math.isclose(high, actual_high, abs_tol=1e-8)):
                    raise ValueError("QCD plotting edges must coincide with native template bin edges.")
                factor = scale * p.period_prediction_scale(cfg, year)
                if per_width:
                    factor /= axis.GetBinWidth(i + 1)
                statistics = p.qcd_root_window_statistics(
                    basis, low, high, float(nominal.GetBinContent(i + 1)) / factor)
                sigma[i] += abs(factor) * statistics["sigma_mc_transfer_stat"]
        finally:
            if f:
                f.Close()
    return Component(p.Uncertainty(sigma[:], sigma[:]), "stat")


def build_components(ROOT, cfg, inputs, edges, norm, requested):
    n = inputs.bkg_total.GetNbinsX()
    scale = p.plot_scale(cfg, norm, inputs.lumi_pb)
    result, summary, errors = {}, [], []

    def indexed(proc, prefix, count, label):
        return p._load_indexed_theory_variations(
            ROOT, cfg, inputs.years, proc, inputs.bkg_by_year, edges, scale,
            prefix, count, label, summary, errors)

    scale_names = {"muR": "QCDscale_ren_ttbar", "muF": "QCDscale_fac_ttbar"}
    if set(scale_names.values()) & set(requested):
        nominal = inputs.bkg["tt"]
        variations = indexed("tt", p.SCALE_PREFIX, p.SCALE_COUNT, "PDF_scale")
        for direction, name in scale_names.items():
            if name in requested:
                down_idx, up_idx = p.SCALE_PAIRS[direction]
                result[name] = pair_component(
                    p.make_delta_hist(variations[down_idx], nominal, name + "_down"),
                    p.make_delta_hist(variations[up_idx], nominal, name + "_up"), n,
                    (direction + "_0p5", direction + "_2"))

    name = PREFIX + "pdf_NNPDF31"
    if name in requested:
        pieces = []
        for proc in p.THEORY_PROCESSES:
            variations = indexed(proc, p.PDF_ERROR_PREFIX, p.PDF_ERROR_COUNT, "PDF_error")
            pieces.append(p.hessian_sigma_hist(inputs.bkg[proc], variations, "pdf_" + proc))
        result[name] = symmetric_component(p.sum_hists(pieces, "pdf_shared"), n)

    if "pdf_alphas" in requested:
        down, up = [], []
        for proc in p.THEORY_PROCESSES:
            variations = indexed(proc, p.ALPHAS_PREFIX, 2, "PDF_alphas")
            for parts, idx in zip((down, up), p.ALPHAS_PAIR):
                parts.append(p.make_delta_hist(variations[idx], inputs.bkg[proc], "alphas_" + proc))
        result["pdf_alphas"] = pair_component(p.sum_hists(down, "alphas_down"),
                                               p.sum_hists(up, "alphas_up"), n)

    name = PREFIX + "topmass_ttbar_BJetOS"
    if name in requested:
        down, up = [], []
        for year in inputs.years:
            nominal = inputs.bkg_by_year[year]["tt"]
            for parts, factor in zip((down, up), p.TT_MASS_FACTORS[p.run_group(year)]):
                parts.append(p._scaled_delta(nominal, factor, "topmass_" + year))
        result[name] = pair_component(p.sum_hists(down, "mass_down"), p.sum_hists(up, "mass_up"), n)

    if not p.use_dy_mc(cfg):
        result.update(dy_components(ROOT, cfg, inputs, edges, scale, requested))
    name = PREFIX + "QCD_MCTransferStat_Run2"
    if name in requested and not p.use_qcd_mc(cfg) and any(y in p.RUN2_ERAS for y in inputs.years):
        result[name] = qcd_transfer_component(ROOT, cfg, inputs, scale)

    p._finish_syst_audit(replace(cfg, strict=True), inputs.warnings,
                         "[component-check] selected sources", summary, errors)
    # A component is a subset of its parent band, including shared QCD statistics.
    for name, component in result.items():
        parent = inputs.stat if component.included_in == "stat" else inputs.syst
        for side in ("low", "high"):
            if any(a > b + 1e-8 * max(1.0, abs(b)) for a, b in
                   zip(getattr(component.uncertainty, side), getattr(parent, side))):
                raise RuntimeError(f"{name}: component exceeds plotter.py's {component.included_in} band.")
    skipped = [name for name in requested if name not in result]
    for name in skipped:
        print(f"[component-check] {name}: not applicable to these eras/background methods")
    return {name: result[name] for name in requested if name in result}, skipped


def output_base(cfg, norm):
    mode = "unblind" if not cfg.blind else {"data": "blind", "asimov": "asimov", "toy": "toy"}[cfg.blind_point_mode]
    qcd = "QCDMC_" + ("norm" if cfg.qcd_normalise else "raw") if p.use_qcd_mc(cfg) else "QCDDD"
    dy = "DYMC" if p.use_dy_mc(cfg) else "DYDD"
    edges = p.make_bin_edges(cfg)
    suffix = f"_rebin{cfg.rebin_factor}" if cfg.rebin_factor > 1 else ""
    return os.path.join(cfg.output_dir, f"{p.plot_era_tag(cfg)}_dimuon_mass_systematics_{norm}_{mode}_BJet_{qcd}_{dy}"
                        f"{suffix}_x{p.mass_label(edges[0])}to{p.mass_label(edges[-1])}")


def write_diagnostics(path, cfg, norm, inputs, edges, components, skipped):
    nominal = [float(inputs.bkg_total.GetBinContent(i + 1)) for i in range(len(edges) - 1)]

    def relative(values, denominators=nominal):
        return [v / b if b > 0 else None for v, b in zip(values, denominators)]

    def describe(unc):
        return {"low": unc.low, "high": unc.high,
                "relative_low": relative(unc.low), "relative_high": relative(unc.high)}

    process_yields = {proc: [float(h.GetBinContent(i + 1)) for i in range(len(nominal))]
                      for proc, h in inputs.bkg.items()}
    payload = {
        "era": cfg.era, "normalisation": norm, "prefit": True, "bin_edges_GeV": list(edges),
        "units": p.y_axis_title(cfg, norm), "background": nominal, "process_yields": process_yields,
        "process_fractions": {proc: relative(values) for proc, values in process_yields.items()},
        "stat": describe(inputs.stat), "syst": describe(inputs.syst), "total": describe(inputs.total),
        "components": {}, "not_applicable": skipped, "warnings": list(inputs.warnings),
        "note": "Total includes every plotter.py source exactly once. QCD_MCTransferStat is in stat; DY NFStat is in syst. These are yield responses, not Combine impacts.",
    }
    for name, component in components.items():
        item = dict(describe(component.uncertainty), label=short_name(name), included_in=component.included_in)
        if component.signed_shifts:
            item["signed_shifts"] = component.signed_shifts
            item["signed_relative_to_background"] = {k: relative(v) for k, v in component.signed_shifts.items()}
            item["signed_relative_to_ttbar"] = {k: relative(v, process_yields["tt"]) for k, v in component.signed_shifts.items()}
        payload["components"][name] = item
    with open(path, "w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
        output.write("\n")


def draw_ratio(ROOT, cfg, norm, inputs, edges, components, path):
    canvas = ROOT.TCanvas(p._NAMES.unique("mass_systematics"), "", 1400, 1050)
    canvas.SetLeftMargin(cfg.left_margin)
    canvas.SetRightMargin(cfg.right_margin)
    canvas.SetTopMargin(0.32)
    canvas.SetBottomMargin(0.13)
    canvas.SetTicks(1, 1)
    frame = inputs.bkg_total.Clone(p._NAMES.unique("systematics_frame"))
    frame.SetDirectory(0)
    frame.Reset("ICESM")
    frame.SetMinimum(cfg.ratio_min)
    frame.SetMaximum(cfg.ratio_max)
    frame.GetXaxis().SetRangeUser(edges[0], edges[-1])
    frame.GetXaxis().SetTitle(p.variable_spec(cfg).x_title)
    frame.GetYaxis().SetTitle("Data / Bkg")
    for axis in (frame.GetXaxis(), frame.GetYaxis()):
        axis.SetLabelSize(0.035)
        axis.SetTitleSize(0.040)
    frame.GetYaxis().SetTitleOffset(1.25)
    frame.GetYaxis().SetNdivisions(505)
    frame.Draw("AXIS")
    total = p.make_ratio_unc_graph(ROOT, inputs.bkg_total, inputs.total, "total")
    stat = p.make_ratio_unc_graph(ROOT, inputs.bkg_total, inputs.stat, "stat")
    p.style_unc_graph(ROOT, total, "total")
    p.style_unc_graph(ROOT, stat, "stat")
    stat.SetFillStyle(1001)
    stat.SetFillColorAlpha(ROOT.kGray + 3, 0.15)
    total.Draw("E2 SAME")
    stat.Draw("E2 SAME")
    outline = p.make_band_outline_graph(ROOT, inputs.bkg_total, inputs.stat, "stat_outline", ratio=True)
    p.style_band_outline(ROOT, outline)
    outline.Draw("L SAME")
    line = ROOT.TLine(edges[0], 1.0, edges[-1], 1.0)
    line.SetLineStyle(2)
    line.Draw("SAME")

    colours = ("#D62728", "#1F77B4", "#009E73", "#007D7E", "#B8860B",
               "#9467BD", "#CC79A7", "#E66101", "#6B6B00", "#333333")
    styles = (1, 1, 1, 7, 1, 1, 7, 2, 2, 7)
    graphs = {}
    # No filled component bands: ten overlapping fills would obscure the sizes.
    for name, component in components.items():
        idx = DEFAULT_SYSTEMATICS.index(name)
        colour = ROOT.TColor.GetColor(colours[idx])
        pair = []
        for side, sign in (("high", 1), ("low", -1)):
            graph = ROOT.TGraph(2 * (len(edges) - 1))
            graph.SetName(p._NAMES.unique(short_name(name) + "_" + side))
            graph.SetLineColor(colour)
            #graph.SetLineStyle(styles[idx])
            #graph.SetLineWidth(2 if idx < 2 else 1)
            graph.SetLineStyle(1)
            graph.SetLineWidth(1)
            for i, delta in enumerate(getattr(component.uncertainty, side)):
                nominal = inputs.bkg_total.GetBinContent(i + 1)
                value = 1.0 + sign * delta / nominal if nominal > 0 else 1.0
                graph.SetPoint(2 * i, edges[i], value)
                graph.SetPoint(2 * i + 1, edges[i + 1], value)
            graph.Draw("L SAME")
            pair.append(graph)
        graphs[name] = pair

    data = p.make_data_graph(ROOT, cfg, norm, inputs.lumi_pb, inputs.data, inputs.bkg_total)
    ratio = p.make_ratio_graph(ROOT, data, inputs.bkg_total)
    p.style_data_graph(ROOT, ratio, cfg)
    if ratio:
        ratio.Draw("PZ SAME")
    frame.Draw("AXIS SAME")

    # Header/legend occupy dedicated whitespace, leaving the ratio panel clear.
    legend = ROOT.TLegend(0.12, 0.705, 0.95, 0.875)
    legend.SetNColumns(2)
    legend.SetColumnSeparation(0.035)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.022)
    legend.AddEntry(total, "Bkg stat #oplus syst unc. (all sources)", "f")
    legend.AddEntry(stat, "Bkg stat unc.", "f")
    for name in components:
        label = short_name(name).replace("_", " ")
        if components[name].included_in == "stat":
            label += " *"
        legend.AddEntry(graphs[name][0], label, "l")
    if ratio:
        label = "Bkg Asimov" if cfg.blind and cfg.blind_point_mode == "asimov" else (
            "Bkg toy" if cfg.blind and cfg.blind_point_mode == "toy" else "Data")
        legend.AddEntry(ratio, label, "pe")
    legend.Draw()

    text = ROOT.TLatex()
    text.SetNDC(True)
    text.SetTextFont(61)
    text.SetTextSize(0.050)
    text.DrawLatex(0.12, 0.947, "CMS")
    text.SetTextFont(52)
    text.SetTextSize(0.034)
    text.DrawLatex(0.23, 0.947, cfg.cms_label)
    text.SetTextFont(42)
    text.SetTextAlign(31)
    text.SetTextSize(0.032)
    lumi = (f"{inputs.lumi_pb / 1000.:.3f} fb^{{-1}} (13 TeV)" if cfg.data_period
            else p.lumi_label(cfg.era, inputs.lumi_pb))
    text.DrawLatex(0.95, 0.947, lumi)
    text.SetTextAlign(11)
    text.SetTextSize(0.028)
    period = f", SingleMuon {cfg.data_period}" if cfg.data_period else ""
    text.DrawLatex(0.12, 0.895, f"{cfg.era}{period}, OS, b-jet; prefit background uncertainties")
    text.SetTextSize(0.020)
    text.DrawLatex(0.12, 0.055, "Coloured lines: 1 - #DeltaB_{low}/B and 1 + #DeltaB_{high}/B; B = total background.")
    text.DrawLatex(0.12, 0.027, "* QCD MCTransferStat is included in stat; DY NFStat is included in syst. Total includes all sources once.")
    if cfg.blind:
        text.SetTextSize(0.022)
        label = (f"Data blinded: {cfg.blind_low:g} < m_{{#mu#mu}} < {cfg.blind_high:g} GeV"
                 if cfg.blind_point_mode == "data" else "Blind: background-only " + cfg.blind_point_mode + " points")
        text.DrawLatex(0.15, 0.65, label)
    if inputs.warnings:
        text.SetTextColor(ROOT.kRed + 1)
        text.SetTextSize(0.022)
        text.DrawLatex(0.15, 0.615, "Input warnings: see log/JSON; use --strict to require complete inputs.")
    if cfg.data_period and (not p.use_qcd_mc(cfg) or not p.use_dy_mc(cfg)):
        text.SetTextColor(ROOT.kBlack)
        text.SetTextSize(0.020)
        text.DrawLatex(0.15, 0.585, "DD: full-era estimate scaled by luminosity")
    canvas.RedrawAxis()
    for ext in cfg.extensions:
        canvas.SaveAs(path + "." + ext)
    if cfg.save_root:
        output = ROOT.TFile.Open(path + ".root", "RECREATE")
        if not output or output.IsZombie():
            raise RuntimeError("Could not create " + path + ".root")
        try:
            inputs.bkg_total.Write("bkg_total")
            stat.Write("ratio_stat_unc")
            total.Write("ratio_stat_plus_syst_unc")
            for name, component in components.items():
                graph = p.make_ratio_unc_graph(ROOT, inputs.bkg_total, component.uncertainty, name)
                graph.Write(name)
        finally:
            output.Close()
    canvas.Close()


def main(argv=None):
    argv = p.normalise_cli_hyphen_typos(list(sys.argv[1:] if argv is None else argv))
    parser = p.make_parser()
    parser.prog = "dimuon_mass_systematics.py"
    parser.description = "Single-panel prefit dimuon-mass uncertainty comparison, using plotter.py inputs."
    parser.epilog = __doc__
    parser.set_defaults(uncertainty="syst+stat")
    parser.add_argument("--systematics", type=p.parse_str_list, default=list(DEFAULT_SYSTEMATICS),
                        help="comma-separated subset of the ten supported nuisance names")
    parser.add_argument("--list-systematics", action="store_true", help="list names without loading ROOT or data")
    if "--list-systematics" in argv:
        print("\n".join(DEFAULT_SYSTEMATICS))
        return 0
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    cfg = p.config_from_args(args)
    if cfg.variable != "dimuon_mass" or cfg.jet_mode != "bjet" or cfg.dilepton_sign != "OS":
        parser.error("This diagnostic supports dimuon_mass in the OS BJet region only.")
    if not cfg.draw_systematics:
        parser.error("Use --uncertainty syst+stat; the comparison requires systematic inputs.")
    if not cfg.draw_ratio:
        parser.error("--no-ratio is incompatible with this ratio-only plot.")
    if cfg.draw_signal or cfg.qcd_normalisation_diagnostics:
        parser.error("Signal overlays and QCD normalisation diagnostics belong to plotter.py.")
    requested = selected_names(args.systematics)
    cfg = p.apply_variable_blinding_mode(replace(cfg, era=p.canonical_era(cfg.era)))
    ROOT = p.import_root()
    p.set_readable_cms_style(ROOT)
    edges = p.make_bin_edges(cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)
    for norm in cfg.normalisations:
        if norm not in {"events", "xsec"}:
            parser.error("Normalisation must be events or xsec.")
        inputs = p.build_plot_inputs(ROOT, cfg, norm, edges)
        components, skipped = build_components(ROOT, cfg, inputs, edges, norm, requested)
        path = output_base(cfg, norm)
        draw_ratio(ROOT, cfg, norm, inputs, edges, components, path)
        write_diagnostics(path + ".json", cfg, norm, inputs, edges, components, skipped)
        p.print_warnings(cfg, norm, inputs.warnings)
        for ext in list(cfg.extensions) + ["json"] + (["root"] if cfg.save_root else []):
            print("[DONE] " + path + "." + ext)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
