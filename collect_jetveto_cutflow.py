#!/usr/bin/env python3
"""Summarise NIsoMuon DY jet-veto studies and signal cutflows using PyROOT.

Select Run2 or Run3 (or one era) with --era. Read native, merged SKFlatAnalyzer
or SKNanoAnalyzer outputs. Only the two DY MC slices and Zprime signal MC are
opened. Signal slices are read from the analyser's submitted signal list.
Use --mass to collect every slice of only the requested mass points.
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

# MC normalisation uses GetTriggerLumi("Full"). These are the fb^-1 values
# from SKFlatAnalyzer/Run2UL_v3 Event.C and SKNanoAnalyzer/NPS-26-009
# data/Run3_v13_Run2_v9/<era>/Trigger/HLT_Path.json, respectively.
LUMI = {
    "2016preVFP": 19.517523849863,
    "2016postVFP": 16.812151722482,
    "2017": 41.477877400009,
    "2018": 59.827879502925,
    "2022": 7.9804,
    "2022EE": 26.6717,
    "2023": 18.064,
    "2023BPix": 9.693,
}
ERAS = {
    "Run2": ("2016preVFP", "2016postVFP", "2017", "2018"),
    "Run3": ("2022", "2022EE", "2023", "2023BPix"),
}
DY = ("DYJets", "DYJets10to50")
REGIONS = ("BJet", "LightJet")
STAGES = (
    "Input", "Trigger and event quality", "Two separated tight AK4 jets",
    "DeepJet Medium tag", "OS muon kinematics and Medium ID",
    "Both muons in the dimuon jet", "Reco dimuon mass in 11-80 GeV",
)
TRUTH_STAGES = (
    "Input after CMS GEN filter", "OS truth dimuon with a common Zprime ancestor",
    "Two separated AK4 GenJets", "Hadron-flavour b tag", "Truth muon kinematics",
    "Both muons in the dimuon GenJet",
)


@contextmanager
def open_root(root, path):
    if not path.is_file():
        raise ValueError(f"Missing merged ROOT file: {path}")
    handle = root.TFile.Open(str(path), "READ")
    if not handle or handle.IsZombie():
        raise ValueError(f"Cannot open ROOT file: {path}")
    try:
        if handle.TestBit(root.TFile.kRecovered):
            raise ValueError(f"Recovered ROOT file: {path}")
        yield handle
    finally:
        handle.Close()


def hist(handle, key, optional=False):
    obj = handle.Get(key)
    if obj and obj.InheritsFrom("TH1") and obj.GetDimension() == 1:
        return obj
    if not obj and optional:
        return None
    raise ValueError(f"Missing or invalid histogram {key} in {handle.GetName()}")


def ratio(num, den):
    return num / den if den > 0 else None


def ratio_fields(on, off):
    r = ratio(on, off)
    return {"on": on, "off": off, "on_over_off": r,
            "change_pct": 100 * (r - 1) if r is not None else None}


def file_path(base, mode, era, sample):
    prefix = "NIsoMuon_" if era in ERAS["Run2"] else ""
    return base / mode / era / f"{prefix}{sample}.root"


def collect_dy(root, base, era_names, overrides):
    spectra, sources = {}, []
    for era in era_names:
        for sample in DY:
            path = overrides.get((era, sample), file_path(base, "JetVetoMapStudy", era, sample))
            sources.append(("JetVetoMapStudy", era, sample, path))
            with open_root(root, path) as handle:
                for region in REGIONS:
                    for state in ("on", "off"):
                        obj = hist(handle, f"JetVetoMap/{region}_{state}")
                        axis = obj.GetXaxis()
                        for low in range(11, 80):
                            first = axis.FindFixBin(low + 1e-6)
                            last = axis.FindFixBin(low + 1 - 1e-6)
                            if abs(axis.GetBinLowEdge(first) - low) > 1e-5 or abs(
                                axis.GetBinUpEdge(last) - low - 1
                            ) > 1e-5:
                                raise ValueError(f"Unexpected mass binning in {path}")
                            key = era, region, state, low
                            spectra[key] = spectra.get(key, 0.) + obj.Integral(first, last)

    summary, shapes = [], []
    groups = [(era, [era]) for era in era_names]
    for run, run_eras in ERAS.items():
        if set(era_names) == set(run_eras):
            groups.append((run, run_eras))
    for label, members in groups:
        lumi = sum(LUMI[era] for era in members)
        totals = {}
        for region in REGIONS:
            on, off = 0., 0.
            for low in range(11, 80):
                bin_on = sum(spectra[era, region, "on", low] for era in members)
                bin_off = sum(spectra[era, region, "off", low] for era in members)
                shapes.append({"era": label, "region": region, "mass_low_GeV": low,
                               "mass_high_GeV": low + 1, **ratio_fields(bin_on, bin_off)})
                on += bin_on
                off += bin_off
            totals[region] = (on, off)
            summary.append({"era": label, "quantity": region, "lumi_fb_inv": lumi,
                            **ratio_fields(on, off), "on_per_fb_inv": on / lumi,
                            "off_per_fb_inv": off / lumi})
        tf_on = ratio(totals["BJet"][0], totals["LightJet"][0])
        tf_off = ratio(totals["BJet"][1], totals["LightJet"][1])
        if tf_on is not None and tf_off is not None:
            summary.append({"era": label, "quantity": "SR_over_CR", "lumi_fb_inv": lumi,
                            **ratio_fields(tf_on, tf_off)})
    return summary, shapes, sources


def collect_cutflow(root, base, era_names, signal_list, masses=None):
    samples = {}
    for line in signal_list.read_text().splitlines():
        sample = line.split("#", 1)[0].strip()
        if not sample:
            continue
        match = re.fullmatch(r"Zp_M-(\d+)_Pt-\d+to\d+_hw7", sample)
        if not match or sample in samples:
            raise ValueError(f"Unexpected or repeated sample in {signal_list}: {sample}")
        samples[sample] = int(match[1])
    if not samples:
        raise ValueError(f"Empty signal list: {signal_list}")
    if masses is not None:
        requested = set(masses)
        missing = requested.difference(samples.values())
        if missing:
            raise ValueError(f"Signal masses not listed in {signal_list}: {sorted(missing)} GeV")
        samples = {sample: mass for sample, mass in samples.items() if mass in requested}

    sums = defaultdict(lambda: [0.] * (len(STAGES) + 1))
    truth_sums = defaultdict(lambda: [0.] * len(TRUTH_STAGES))
    fiducial_sums = defaultdict(lambda: [0.] * len(STAGES))
    sources = []
    has_truth_by_mass = {}
    for era in era_names:
        for sample, mass in samples.items():
            path = file_path(base, "Cutflow", era, sample)
            sources.append(("Cutflow", era, sample, path))
            with open_root(root, path) as handle:
                reco = hist(handle, "Cutflow/Reco")
                add_cutflow(reco, sums[era, mass], STAGES, path)
                nominal = hist(handle, "Cutflow/RecoNominalYield", optional=True)
                if nominal:
                    if nominal.GetNbinsX() != len(STAGES):
                        raise ValueError(f"Unexpected nominal-yield binning in {path}")
                    sums[era, mass][-1] += nominal.GetBinContent(len(STAGES))

                truth = hist(handle, "Cutflow/Truth", optional=era in ERAS["Run2"])
                has_truth = bool(truth)
                if has_truth != has_truth_by_mass.setdefault((era, mass), has_truth):
                    raise ValueError(f"Mixed reco-only and fiducial cutflow inputs in {era}, "
                                     f"m(Zprime)={mass} GeV: {path}")
                if truth:
                    add_cutflow(truth, truth_sums[era, mass], TRUTH_STAGES, path)
                    # The analyser only books this histogram when a fiducial event exists.
                    fiducial = hist(handle, "Cutflow/RecoFiducial", optional=True)
                    values = fiducial_sums[era, mass]
                    if fiducial:
                        add_cutflow(fiducial, values, STAGES, path)
                    elif truth.GetBinContent(len(TRUTH_STAGES)) != 0.:
                        raise ValueError(f"Missing fiducial reco cutflow in {path}")

    reco_rows = cutflow_rows(sums, STAGES, corrected=True)
    fiducial_rows = cutflow_rows(truth_sums, TRUTH_STAGES, level="truth")
    fiducial_rows += cutflow_rows(fiducial_sums, STAGES, level="reco_fiducial")
    return reco_rows, fiducial_rows, sources


def add_cutflow(obj, totals, stages, path):
    if obj.GetNbinsX() != len(stages):
        raise ValueError(f"Unexpected cutflow binning for {obj.GetName()} in {path}")
    for step in range(len(stages)):
        totals[step] += obj.GetBinContent(step + 1)


def cutflow_rows(sums, stages, *, corrected=False, level=None):
    rows = []
    for (era, mass), values in sorted(sums.items()):
        for step, name in enumerate(stages):
            row = {"era": era, "mass_GeV": mass, "step": step, "selection": name,
                   "sum_weight_pb": values[step],
                   "step_fraction": ratio(values[step], values[max(0, step - 1)])}
            if corrected:
                row["cumulative_reco_fraction"] = ratio(values[step], values[0])
                row["corrected_expected_yield"] = values[-1] if step == len(stages) - 1 else None
            else:
                row["level"] = level
                row["cumulative_fraction"] = ratio(values[step], values[0])
            rows.append(row)
    return rows


def write_csv(path, columns, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="Directory containing Cutflow and JetVetoMapStudy")
    parser.add_argument("--out", default=Path("run2_summary"), type=Path)
    parser.add_argument("--study", choices=("all", "dy", "cutflow"), default="all")
    parser.add_argument("--era", choices=(*ERAS, *LUMI), default="Run2",
                        help="Run or individual era (default: Run2)")
    parser.add_argument("--signal-list", type=Path,
                        help="Analyser SampleLists/Run2signal.txt or Run3signal.txt for cutflow")
    parser.add_argument("--mass", type=int, nargs="+",
                        help="Collect all slices of these signal masses in GeV (default: all masses)")
    parser.add_argument("--dy-file", action="append", default=[], metavar="ERA:SAMPLE=FILE",
                        help="Override one merged DY ROOT file (repeatable)")
    args = parser.parse_args()
    era_names = ERAS.get(args.era, (args.era,))
    if args.study in ("all", "cutflow") and args.signal_list is None:
        parser.error("--signal-list is required for signal cutflow")
    if args.study == "dy" and args.mass is not None:
        parser.error("--mass applies only to signal cutflow")
    overrides = {}
    for override in args.dy_file:
        label, sep, path = override.partition("=")
        era, _, sample = label.partition(":")
        if not sep or not path or era not in era_names or sample not in DY or (era, sample) in overrides:
            parser.error(f"Invalid or repeated --dy-file: {override}")
        overrides[era, sample] = Path(path)

    import ROOT

    ROOT.gROOT.SetBatch(True)
    files, sources = [], []
    if args.study in ("all", "dy"):
        summary, shapes, inputs = collect_dy(ROOT, args.base, era_names, overrides)
        files.extend([
            ("dy_veto_summary.csv",
             ["era", "quantity", "lumi_fb_inv", "on", "off", "on_over_off",
              "change_pct", "on_per_fb_inv", "off_per_fb_inv"], summary),
            ("dy_veto_mass_bins.csv", list(shapes[0]), shapes),
        ])
        sources.extend(inputs)
    if args.study in ("all", "cutflow"):
        rows, fiducial_rows, inputs = collect_cutflow(
            ROOT, args.base, era_names, args.signal_list, masses=args.mass,
        )
        files.append(("signal_reco_cutflow.csv", list(rows[0]), rows))
        if fiducial_rows:
            files.append(("signal_fiducial_cutflow.csv", list(fiducial_rows[0]), fiducial_rows))
        sources.extend(inputs)

    args.out.mkdir(parents=True, exist_ok=True)
    files.append(("inputs.csv", ["study", "era", "sample", "path"], [
        {"study": mode, "era": era, "sample": sample, "path": os.path.abspath(path)}
        for mode, era, sample, path in sources
    ]))
    for name, columns, rows in files:
        path = args.out / name
        write_csv(path, columns, rows)
        print(path)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, ImportError) as error:
        sys.exit(str(error))
