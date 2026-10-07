#!/usr/bin/env python3
"""Print native event yields and QCD closure without modifying ROOT inputs.

Run after DY and QCD estimation, with the same --base-dir used by plotter.py:
  python3 qcd_yield_diagnostics.py --era 2016preVFP 2016postVFP --unblind

OS subtraction: Data - Top - Others - DY DD.
SS subtraction: Data - Top - Others (DY MC is printed, not subtracted).
Calibration windows come from QCDStat metadata/derivatives when available.
Also print 11--15 GeV and its four 1-GeV bins. The separate validation-plot
scale factors use --plot-range (default 5--120 GeV), never the transfer windows.
No fitting, histogram scaling, template regeneration or output files.
High-mass OS observations and data-derived plot factors require --unblind.
Ratios are central-value diagnostics, not significance estimates.
"""

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import plotter as p
from qcd_bkg_estimation import QCD_TRANSFER_LOW_WINDOW, QCD_TRANSFER_HIGH_WINDOW


def ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def fmt(value):
    return "n/a" if value is None else f"{value:.8g}"


def count(hist, low, high):
    axis = hist.GetXaxis()
    if low < axis.GetBinLowEdge(1) - 1e-8 or high > axis.GetBinUpEdge(hist.GetNbinsX()) + 1e-8:
        raise ValueError(f"{hist.GetName()}: [{low:g}, {high:g}] outside histogram range")
    result = p.qcd_diagnostic_count(hist, low, high)
    if not math.isfinite(result):
        raise ValueError(f"{hist.GetName()}: non-finite integral")
    return result


def component_yields(histograms, low, high, unblind):
    rows = {}
    for sign in ("OS", "SS"):
        hidden = sign == "OS" and not unblind and low < 80 and high > 11
        row = {proc: (None if hidden and proc == "Data" else count(hist, low, high))
               for (charge, proc), hist in histograms.items() if charge == sign}
        row["NonQCD_estimator"] = row["Top"] + row["Others"] + (row["DYDD"] if sign == "OS" else 0)
        row["NonQCD_plot"] = row["tt"] + row["ST"] + row["Others"] + row["DYMC"]
        row["MC_total"] = row["NonQCD_plot"] + row["QCDMC"]
        if sign == "OS":
            row["NonQCD_plot"] += row["DYDD"] - row["DYMC"]
        row["Residual"] = None if row["Data"] is None else row["Data"] - row["NonQCD_estimator"]
        row["Top_minus_tt_ST"] = row["Top"] - row["tt"] - row["ST"]
        rows[sign] = row
    return rows


def plot_factors(histograms, plot_range, unblind):
    """Reproduce unblinded global QCDMC_norm factors without altering histograms."""
    rows = component_yields(histograms, *plot_range, unblind)
    result = {}
    for sign, row in rows.items():
        residual = None if row["Data"] is None else row["Data"] - row["NonQCD_plot"]
        value = ratio(residual, row["QCDMC"])
        result[sign] = None if value is None else max(0., value)
    os_row = rows["OS"]
    nonqcd_mc = os_row["tt"] + os_row["ST"] + os_row["Others"] + os_row["DYMC"]
    value = ratio(None if os_row["Data"] is None else os_row["Data"] - nonqcd_mc, os_row["QCDMC"])
    result["OS_DYMC"] = None if value is None else max(0., value)
    return result


def print_window(histograms, low, high, factors, inputs, cfg):
    rows = component_yields(histograms, low, high, not cfg.blind)
    print(f"\n[window] {low:g}--{high:g} GeV; native event counts")
    print(f"{'Yield':<24} {'OS':>17} {'SS':>17}")
    for key in ("Data", "tt", "ST", "Top", "Others", "DYMC", "DYDD", "QCDMC", "MC_total", "QCDDD",
                "Top_minus_tt_ST", "NonQCD_estimator", "Residual"):
        print(f"{key:<24} {fmt(rows['OS'].get(key)):>17} {fmt(rows['SS'].get(key)):>17}")
    for sign, row in rows.items():
        scaled = None if factors[sign] is None else factors[sign] * row["QCDMC"]
        total = None if scaled is None else row["NonQCD_plot"] + scaled
        plot_residual = None if row["Data"] is None else row["Data"] - row["NonQCD_plot"]
        print(f"[closure/{sign}] residual/raw-QCDMC={fmt(ratio(row['Residual'], row['QCDMC']))}, "
              f"scaled-QCDMC={fmt(scaled)}, plot-residual/scaled-QCDMC={fmt(ratio(plot_residual, scaled))}, "
              f"Data/plot-total={fmt(ratio(row['Data'], total))}")
    r_mc = ratio(rows["OS"]["QCDMC"], rows["SS"]["QCDMC"])
    r_data = ratio(rows["OS"]["Residual"], rows["SS"]["Residual"])
    r_scaled = None if factors["OS"] is None or factors["SS"] is None else ratio(
        factors["OS"] * rows["OS"]["QCDMC"], factors["SS"] * rows["SS"]["QCDMC"])
    print(f"[ratio] R_MC(raw)={fmt(r_mc)}, R_data(residual)={fmt(r_data)}, "
          f"R_data/R_MC={fmt(ratio(r_data, r_mc))}, R_MC(plot-scaled)={fmt(r_scaled)}")
    print(f"[DD] QCDDD/OS-residual={fmt(ratio(rows['OS']['QCDDD'], rows['OS']['Residual']))}")
    if inputs is not None:
        h = {sign + "_" + proc: hist for (sign, proc), hist in histograms.items()}
        for sign in ("OS", "SS"):
            h[sign + "_data"] = h[sign + "_Data"]
            h[sign + "_QCD"] = h[sign + "_QCDMC"]
        h["OS_DY"] = h["OS_DYDD"]
        diag = p.qcd_normalisation_diagnostic_row(inputs, h, low, high, cfg)
        print(f"[SS-fit] yield={fmt(diag['ss_fit'])}, fit/SS-residual={fmt(diag['ss_fit_over_residual'])}, "
              f"fit/SS-QCDMC={fmt(diag['ss_fit_over_mc'])}, effective-transfer={fmt(diag['effective_transfer'])}")
        rebuilt = None if r_data is None else diag["ss_fit"] * r_data
        print(f"[closure] SS-fit * local residual OS/SS={fmt(rebuilt)} (diagnostic only)")
    return rows, r_data, r_mc


def inspect_era(ROOT, cfg, plot_range):
    directory = Path(p.root_dir_for_year(cfg, cfg.era))
    files = {"Data": "data.root", "tt": cfg.tt_file, "ST": cfg.st_file,
             "Top": "NIsoMuon_Top.root", "Others": cfg.others_file,
             "QCDMC": cfg.qcd_mc_file, "DYMC": cfg.dy_mc_file,
             "DYDD": cfg.dy_data_driven_file, "QCDDD": cfg.qcd_data_driven_file}
    histograms = {}
    print(f"\n{'=' * 72}\n[era] {cfg.era}; directory={directory}")
    for proc, filename in files.items():
        signs = ("OS",) if proc in ("DYDD", "QCDDD") else ("OS", "SS")
        source = directory / filename
        print(f"[input] {proc}: {source}")
        for sign in signs:
            region_cfg = replace(cfg, dilepton_sign=sign)
            path = p.hist_path(region_cfg, p.base_region(region_cfg))
            hist, error = p.read_hist(ROOT, str(source), path)
            if error:
                raise ValueError(error)
            histograms[sign, proc] = hist

    root_file = ROOT.TFile.Open(str(directory / cfg.qcd_data_driven_file), "READ")
    try:
        inputs = None
        windows = [QCD_TRANSFER_LOW_WINDOW, QCD_TRANSFER_HIGH_WINDOW]
        window_source = "current producer constants (stored metadata unavailable)"
        metadata_obj = root_file.Get(p.QCD_STAT_PATH)
        metadata = json.loads(str(metadata_obj.GetString())) if metadata_obj else None
        if metadata and metadata.get("transfer_statistics", {}).get("windows"):
            recorded = metadata["transfer_statistics"]["windows"]
            windows = [tuple(recorded[key]) for key in ("low", "high")]
            window_source = "production metadata"
        try:
            inputs = p.read_qcd_stat_inputs(root_file, cfg.era, p.hist_path(cfg, p.base_region(cfg)))
            windows, window_source = p.qcd_transfer_windows(inputs)
        except (ValueError, KeyError, TypeError) as exc:
            print(f"[WARNING] Stored SS-fit/stat-basis audit unavailable: {exc}; component yields still printed.")
        print(f"[windows] low={windows[0]}, high={windows[1]}; {window_source}")
        print("[subtraction] OS: Data-Top-Others-DYDD; SS: Data-Top-Others; SS DYDD=n/a")
        factors = plot_factors(histograms, plot_range, not cfg.blind)
        print(f"[plot-normalisation] range={tuple(plot_range)}; OS(DYDD)={fmt(factors['OS'])}, "
              f"OS(DYMC)={fmt(factors['OS_DYMC'])}, SS(DYMC)={fmt(factors['SS'])}")
        if cfg.blind:
            print("[blind] OS observations intersecting 11--80 GeV and OS global plot factors are hidden.")
        selected = list(dict.fromkeys([*windows, (11., 15.), *((float(m), float(m+1)) for m in range(11, 15))]))
        records = [print_window(histograms, *window, factors, inputs, cfg) for window in selected]
        _, r_low, mc_low = records[0]
        _, r_high, mc_high = records[1]
        double_mc = ratio(mc_high, mc_low)
        transported = None if r_low is None or double_mc is None else r_low * double_mc
        print(f"\n[transfer/current] R_low_data={fmt(r_low)}, R_low_MC={fmt(mc_low)}, "
              f"R_high_data={fmt(r_high)}, R_high_MC={fmt(mc_high)}, "
              f"MC-double-ratio={fmt(double_mc)}, T_high={fmt(transported)}, "
              f"T_high/R_high_data={fmt(ratio(transported, r_high))}")
        if metadata is not None:
            transfer = metadata["transfer_statistics"]
            print(f"[transfer/stored] R_low_data={fmt(transfer['low_transfer'])}, "
                  f"MC-double-ratio={fmt(transfer['mc_double_ratio'])}, T_high={fmt(transfer['high_transfer'])}")
            for label, key, sign, proc, index in (
                ("data_os_low", "data_os_low", "OS", "Residual", 0),
                ("data_ss_low", "data_ss_low", "SS", "Residual", 0),
                ("mc_os_low", "mc_os_low", "OS", "QCDMC", 0),
                ("mc_ss_low", "mc_ss_low", "SS", "QCDMC", 0),
                ("mc_os_high", "mc_os_high", "OS", "QCDMC", 1),
                ("mc_ss_high", "mc_ss_high", "SS", "QCDMC", 1)):
                current = records[index][0][sign][proc]
                stored = transfer["primitive"][key]["value"]
                diff = None if current is None else current - stored
                print(f"[stored/current] {label}: stored={fmt(stored)}, current={fmt(current)}, difference={fmt(diff)}")
    finally:
        root_file.Close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--era", nargs="+", default=["2016preVFP", "2016postVFP", "2017", "2018"],
                        help="individual eras or Run2/Run3 groups; default: four Run-2 eras")
    parser.add_argument("--base-dir", default=p.Config().base_dir)
    parser.add_argument("--trigger", default="")
    parser.add_argument("--unblind", action="store_true", help="print observed OS yields in 11--80 GeV")
    parser.add_argument("--plot-range", type=float, nargs=2, default=(5., 120.), metavar=("LOW", "HIGH"),
                        help="global QCDMC_norm validation range; default: 5 120")
    args = parser.parse_args(argv)
    if not all(math.isfinite(v) for v in args.plot_range) or args.plot_range[0] >= args.plot_range[1]:
        parser.error("--plot-range requires finite LOW < HIGH")
    try:
        import ROOT
        ROOT.gROOT.SetBatch(True)
        years = list(dict.fromkeys(year for era in args.era for year in p.years_for_era(era)))
        for year in years:
            cfg = p.Config(base_dir=args.base_dir, era=year, trigger=args.trigger, blind=not args.unblind)
            inspect_era(ROOT, cfg, args.plot_range)
    except (ImportError, ValueError, OSError, KeyError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
