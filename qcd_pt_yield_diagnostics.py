#!/usr/bin/env python3
"""Read-only OS/SS yield and weighted-statistics audit of QCD MuEnriched samples.

  python3 qcd_pt_yield_diagnostics.py --era 2016preVFP 2016postVFP 2017 2018
  python3 qcd_pt_yield_diagnostics.py --era 2016postVFP --sample 80To120

Discover Skim_NIsoMuon_QCD_Pt-*_MuEnriched.root exactly as hadd.sh does, including
any low-pT samples. Read native event-count histograms without fitting, rebinning,
scaling, or rewriting inputs. Stored QCDStat metadata supplies the production
low/high windows; current producer constants are the fallback. Also print
11--15 GeV and its four 1-GeV bins.

Errors are sqrt(stored Sumw2); N_eff = sumw**2 / sumw2. Missing Sumw2 is reported
as unknown, never replaced with Poisson errors. Ratio errors use first-order
propagation for independent OS/SS selections and disjoint low/high windows;
they contain MC statistics only. Leave-one-sample-out double ratios diagnose
composition sensitivity, not statistical significance or a replacement estimate.
Check the sample sum against NIsoMuon_QCD_Inclusive.root in yield and Sumw2.
--sample restricts displayed samples; all files still enter the sum and audit.
Missing histograms/files are listed explicitly; their yields remain unknown.
An incomplete available-histogram sum is labelled KNOWN_SUM rather than ALL.
Only MC histograms and optional QCD window metadata are read; no observed data
histograms are read.
"""

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import plotter as p
from qcd_bkg_estimation import QCD_TRANSFER_LOW_WINDOW, QCD_TRANSFER_HIGH_WINDOW


@dataclass(frozen=True)
class Count:
    value: float
    variance: Optional[float] = None

    @property
    def error(self):
        return None if self.variance is None else math.sqrt(self.variance)

    @property
    def effective(self):
        return None if self.variance is None or self.variance <= 0 else self.value**2 / self.variance


def fmt(value):
    return "n/a" if value is None else f"{value:.7g}"


def with_error(count):
    return "n/a" if count is None else f"{fmt(count.value)} +/- {fmt(count.error)}"


def add_counts(counts):
    counts = [c for c in counts if c is not None]
    if not counts:
        return Count(0., None)
    variance = None if any(c.variance is None for c in counts) else math.fsum(c.variance for c in counts)
    return Count(math.fsum(c.value for c in counts), variance)


def ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator.value <= 0:
        return None
    value = numerator.value / denominator.value
    variance = None
    if numerator.variance is not None and denominator.variance is not None:
        variance = (numerator.variance / denominator.value**2
                    + value**2 * denominator.variance / denominator.value**2)
    return Count(value, variance)


def native_count(hist, low, high):
    if hist.GetDimension() != 1:
        raise ValueError(f"{hist.GetName()}: expected a one-dimensional histogram")
    axis = hist.GetXaxis()
    if low < axis.GetBinLowEdge(1) - 1e-8 or high > axis.GetBinUpEdge(hist.GetNbinsX()) + 1e-8:
        raise ValueError(f"{hist.GetName()}: [{low:g}, {high:g}] outside histogram range")
    value = p.qcd_diagnostic_count(hist, low, high)  # Reject non-aligned edges.
    variance = None
    if hist.GetSumw2N() > 0:
        first, last = axis.FindFixBin(low + 1e-8), axis.FindFixBin(high - 1e-8)
        weights = [float(hist.GetSumw2().At(i)) for i in range(first, last + 1)]
        if any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError(f"{hist.GetName()}: invalid stored Sumw2")
        variance = math.fsum(weights)
    if not math.isfinite(value):
        raise ValueError(f"{hist.GetName()}: non-finite integral")
    return Count(value, variance)


def open_file(ROOT, filename):
    root_file = ROOT.TFile.Open(str(filename), "READ")
    if not root_file or root_file.IsZombie():
        if root_file:
            root_file.Close()
        raise OSError(f"Cannot open ROOT file: {filename}")
    return root_file


def production_windows(ROOT, directory, cfg):
    filename = directory / cfg.qcd_data_driven_file
    if filename.is_file():
        root_file = open_file(ROOT, filename)
        try:
            obj = root_file.Get(p.QCD_STAT_PATH)
            if obj:
                metadata = json.loads(str(obj.GetString()))
                recorded = metadata.get("transfer_statistics", {}).get("windows")
                if recorded:
                    return [tuple(map(float, recorded[key])) for key in ("low", "high")], "production metadata"
        finally:
            root_file.Close()
    return [QCD_TRANSFER_LOW_WINDOW, QCD_TRANSFER_HIGH_WINDOW], "current producer constants (no recorded windows)"


def read_counts(ROOT, filename, cfg, windows):
    root_file = open_file(ROOT, filename)
    counts = {}
    try:
        for sign in ("OS", "SS"):
            sign_cfg = replace(cfg, dilepton_sign=sign)
            path = p.hist_path(sign_cfg, p.base_region(sign_cfg))
            hist = root_file.Get(path)
            if not hist:
                print(f"[WARNING] Missing histogram: {filename}:{path}; yield unknown, excluded from KNOWN_SUM")
                counts.update({(sign, window): None for window in windows})
                continue
            if not hist.InheritsFrom("TH1"):
                raise ValueError(f"Expected TH1: {filename}:{path}")
            if not hist.GetSumw2N():
                print(f"[WARNING] Missing stored Sumw2: {filename}:{path}; statistical errors unknown")
            for window in windows:
                counts[sign, window] = native_count(hist, *window)
    finally:
        root_file.Close()
    return counts


def sample_name(filename):
    return filename.name[len("Skim_NIsoMuon_QCD_Pt-"):-len("_MuEnriched.root")]


def canonical_sample(name):
    prefix, suffix = "Skim_NIsoMuon_QCD_Pt-", "_MuEnriched.root"
    if name.startswith(prefix):
        name = name[len(prefix):]
    if name.endswith(suffix):
        name = name[:-len(suffix)]
    name = name.lower()
    return "1000toinf" if name == "1000" else name


def sample_sort(filename):
    name = sample_name(filename)
    match = re.match(r"\d+", name)
    return (int(match.group()) if match else math.inf, name)


def transfer_ratios(counts, low, high):
    r_low = ratio(counts["OS", low], counts["SS", low])
    r_high = ratio(counts["OS", high], counts["SS", high])
    return r_low, r_high, ratio(r_high, r_low)


def print_window(samples, selected, totals, window, total_label):
    print(f"\n[window] {window[0]:g}--{window[1]:g} GeV; weighted event yields, no additional normalisation")
    print(f"{'pT sample':<15} {'OS sumw +/- stat':>24} {'SS sumw +/- stat':>24} "
          f"{'N_eff OS':>12} {'N_eff SS':>12} {'OS share':>12} {'SS share':>12} {'OS/SS +/- stat':>24}")
    for name in [*selected, total_label]:
        counts = totals if name == total_label else samples[name]
        os_count, ss_count = counts["OS", window], counts["SS", window]
        os_share = ratio(os_count, totals["OS", window])
        ss_share = ratio(ss_count, totals["SS", window])
        print(f"{name:<15} {with_error(os_count):>24} {with_error(ss_count):>24} "
              f"{fmt(None if os_count is None else os_count.effective):>12} "
              f"{fmt(None if ss_count is None else ss_count.effective):>12} "
              f"{fmt(None if os_share is None else os_share.value):>12} "
              f"{fmt(None if ss_share is None else ss_share.value):>12} "
              f"{with_error(ratio(os_count, ss_count)):>24}")


def inspect_era(ROOT, cfg, requested):
    directory = Path(p.root_dir_for_year(cfg, cfg.era))
    files = sorted(directory.glob("Skim_NIsoMuon_QCD_Pt-*_MuEnriched.root"), key=sample_sort)
    if not files:
        raise OSError(f"No Skim_NIsoMuon_QCD_Pt-*_MuEnriched.root files in {directory}")
    print(f"\n{'=' * 100}\n[era] {cfg.era}; directory={directory}")
    print(f"[discovery] {len(files)} files; same pattern as hadd.sh")
    low_high, source = production_windows(ROOT, directory, cfg)
    if any(len(w) != 2 or not all(math.isfinite(v) for v in w) or w[0] >= w[1] for w in low_high):
        raise ValueError("Invalid low/high mass windows")
    low, high = low_high
    if max(low[0], high[0]) < min(low[1], high[1]):
        raise ValueError("Low/high windows overlap; independent ratio-error propagation is inapplicable")
    windows = list(dict.fromkeys([low, high, (11., 15.), *((float(m), float(m + 1)) for m in range(11, 15))]))
    print(f"[windows] low={low}; high={high}; {source}")
    samples, aliases = {}, {}
    for filename in files:
        name = sample_name(filename)
        key = canonical_sample(name)
        if key in aliases:
            print(f"[WARNING] Multiple spellings of pT sample {key}: {aliases[key]}, {filename.name}; "
                  "both enter hadd.sh and this sum; check for duplicate production")
        aliases[key] = filename.name
        print(f"[input] {filename}")
        try:
            samples[name] = read_counts(ROOT, filename, cfg, windows)
        except (ValueError, OSError) as exc:
            print(f"[WARNING] Sample read failed: {exc}; yields unknown, excluded from KNOWN_SUM")
            samples[name] = {(sign, window): None for sign in ("OS", "SS") for window in windows}
    selected = [name for name in samples if not requested or canonical_sample(name) in requested]
    missing = requested - {canonical_sample(name) for name in samples}
    if missing:
        raise ValueError(f"Requested pT samples not found: {', '.join(sorted(missing))}")
    totals = {key: add_counts(counts[key] for counts in samples.values()) for key in next(iter(samples.values()))}
    complete = all(c is not None for counts in samples.values() for c in counts.values())
    total_label = "ALL" if complete else "KNOWN_SUM"
    for sign in ("OS", "SS"):
        missing_hist = [name for name, counts in samples.items() if counts[sign, low] is None]
        print(f"[coverage] {sign}: {len(samples) - len(missing_hist)}/{len(samples)} histograms available; "
              f"unknown samples={','.join(missing_hist) or 'none'}")
    if not complete:
        print("[WARNING] KNOWN_SUM, shares and leave-out ratios use only available histograms; total coverage is incomplete")
    print("[statistics] stat=sqrt(stored Sumw2); N_eff=sumw^2/Sumw2; ratio errors are first-order MC stat only")
    print(f"[shares] Fractions of {total_label}; --sample only filters displayed sample rows")
    for window in windows:
        print_window(samples, selected, totals, window, total_label)

    inclusive = directory / cfg.qcd_mc_file
    if inclusive.is_file():
        merged = read_counts(ROOT, inclusive, cfg, windows)
        for (sign, window), count in totals.items():
            expected = merged[sign, window]
            if expected is None:
                print(f"[inclusive-check] {sign} {window[0]:g}--{window[1]:g}: inclusive histogram unavailable")
                continue
            agree = math.isclose(count.value, expected.value, rel_tol=1e-8, abs_tol=1e-8)
            stat_agree = None if count.variance is None or expected.variance is None else math.isclose(
                count.variance, expected.variance, rel_tol=1e-8, abs_tol=1e-8)
            print(f"[inclusive-check] {sign} {window[0]:g}--{window[1]:g}: "
                  f"{total_label}={fmt(count.value)}, inclusive={fmt(expected.value)}, "
                  f"delta={fmt(count.value - expected.value)}, yield={'OK' if agree else 'MISMATCH'}; "
                  f"sample-Sumw2={fmt(count.variance)}, inclusive-Sumw2={fmt(expected.variance)}, "
                  f"Sumw2={'unknown' if stat_agree is None else ('OK' if stat_agree else 'MISMATCH')}")
    else:
        print(f"[WARNING] Inclusive comparison unavailable: {inclusive}")

    _, _, full_double = transfer_ratios(totals, low, high)
    print("\n[composition] H/L uses the production high/low windows; D=(OS/SS)_high/(OS/SS)_low")
    print(f"[leave-out] D_without removes one file from {total_label}; "
          f"delta_D=D_without-D_{total_label} (correlated, no significance assigned)")
    print(f"{'pT sample':<15} {'R_low +/- stat':>24} {'R_high +/- stat':>24} "
          f"{'OS H/L +/- stat':>24} {'SS H/L +/- stat':>24} {'D +/- stat':>24} {'D_without':>12} {'delta_D':>12}")
    for name in [*selected, total_label]:
        counts = totals if name == total_label else samples[name]
        r_low, r_high, double = transfer_ratios(counts, low, high)
        removed = None
        if name != total_label:
            remainder = {key: add_counts(c[key] for other, c in samples.items() if other != name) for key in totals}
            removed = transfer_ratios(remainder, low, high)[2]
        delta = None if removed is None or full_double is None else removed.value - full_double.value
        print(f"{name:<15} {with_error(r_low):>24} {with_error(r_high):>24} "
              f"{with_error(ratio(counts['OS', high], counts['OS', low])):>24} "
              f"{with_error(ratio(counts['SS', high], counts['SS', low])):>24} "
              f"{with_error(double):>24} {fmt(None if removed is None else removed.value):>12} {fmt(delta):>12}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--era", nargs="+", default=["2016preVFP", "2016postVFP", "2017", "2018"],
                        help="individual eras or Run2/Run3 groups; default: four Run-2 eras")
    parser.add_argument("--base-dir", default=p.Config().base_dir)
    parser.add_argument("--trigger", default="")
    parser.add_argument("--sample", nargs="+", default=[], metavar="PT_BIN",
                        help="restrict displayed samples, e.g. 80To120; ALL/inclusive audit still uses every file")
    args = parser.parse_args(argv)
    requested = {canonical_sample(name) for name in args.sample}
    try:
        years = list(dict.fromkeys(year for era in args.era for year in p.years_for_era(era)))
        import ROOT
        ROOT.gROOT.SetBatch(True)
    except (ImportError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    status = 0
    for year in years:
        cfg = p.Config(base_dir=args.base_dir, era=year, trigger=args.trigger)
        try:
            inspect_era(ROOT, cfg, requested)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            print(f"[ERROR] {year}: {exc}", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
