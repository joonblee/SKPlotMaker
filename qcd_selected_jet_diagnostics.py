#!/usr/bin/env python3
"""Read AK4 jet kinematics for already traced, selected QCD MC events.

Uses events in qcd_event_sources_2016postVFP.json from the previous source trace.
Reads only their file-local entries and verifies exact run/lumi/event IDs.
No analyser replay, event selection, normalisation or production files changed.

jet_pt is the stored reconstructed pT before nominal JER smearing (not an
uncorrected raw pT). MC GetAllJets() uses jet_pt * jet_smearedRes. All AK4 jets
are printed, sorted by that nominal pT, retaining their original vector index.
The old audit did not record selected dimuon-jet/tag-jet indices: do not identify
those jets solely from rank, DeepJet or jet ID in this output.
"""

import argparse
import json
import math
from pathlib import Path
import sys


def source_entry(row, source):
    if source == "skim":
        return row["input_file"], row["local_entry"]
    matches = row.get("original_matches", [])
    if len(matches) != 1 or matches[0].get("raw_muons_match") is not True:
        raise ValueError("Original source must have one verified raw-muon match.")
    return matches[0]["input_file"], matches[0]["local_entry"]


def read_jets(tree, entry, row):
    required = ("run", "lumi", "event", "jet_pt", "jet_eta", "jet_phi", "jet_smearedRes")
    missing = [name for name in required if not tree.GetBranch(name)]
    if missing:
        raise ValueError(f"Missing branches: {missing}")
    optional = ("jet_DeepFlavour", "jet_tightJetID")
    available = [name for name in optional if tree.GetBranch(name)]
    tree.SetBranchStatus("*", 0)
    for name in (*required, *available):
        tree.SetBranchStatus(name, 1)
    if type(entry) is not int or not 0 <= entry < int(tree.GetEntries()):
        raise ValueError(f"Invalid file-local entry {entry}")
    if tree.GetEntry(entry) <= 0:
        raise ValueError(f"Cannot read entry {entry}")
    # Branch access retains ULong64_t IDs; GetLeaf().GetValue() uses doubles.
    for name in ("run", "lumi", "event"):
        observed, expected = getattr(tree, name), row[name]
        if not isinstance(observed, int) or type(expected) is not int:
            raise ValueError(f"{name}: integer event IDs required; no float conversion")
        if observed != expected:
            raise ValueError(f"Entry/ID mismatch: {name}={observed}, expected={expected}")
    vectors = {name: list(getattr(tree, name)) for name in (*required[3:], *available)}
    if len({len(v) for v in vectors.values()}) != 1:
        raise ValueError("Jet vector lengths disagree")
    jets = []
    for index, pt in enumerate(vectors["jet_pt"]):
        pt, eta, phi, jer = (float(vectors[name][index])
                             for name in ("jet_pt", "jet_eta", "jet_phi", "jet_smearedRes"))
        if not all(math.isfinite(v) for v in (pt, eta, phi, jer)) or pt < 0 or jer < 0:
            raise ValueError(f"Invalid jet kinematics at index {index}")
        deepjet = float(vectors["jet_DeepFlavour"][index]) if "jet_DeepFlavour" in available else None
        if deepjet is not None and not math.isfinite(deepjet):
            raise ValueError(f"Invalid DeepJet score at index {index}")
        jets.append(dict(index=index, stored_pt=pt, jer=jer, nominal_pt=pt*jer,
                         eta=eta, phi=phi, deepjet=deepjet,
                         tight=int(vectors["jet_tightJetID"][index]) if "jet_tightJetID" in available else None))
    return sorted(jets, key=lambda jet: jet["nominal_pt"], reverse=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-report", default="plots/qcd_event_sources_2016postVFP.json")
    parser.add_argument("--input", choices=("skim", "original"), default="skim")
    parser.add_argument("--sign", choices=("ss", "os", "all"), default="ss")
    parser.add_argument("--mass-range", nargs=2, type=float, default=(11., 80.), metavar=("LOW", "HIGH"))
    args = parser.parse_args(argv)
    low, high = args.mass_range
    if not all(math.isfinite(v) for v in (low, high)) or low >= high:
        parser.error("Mass range must be finite and LOW < HIGH.")
    report = json.loads(Path(args.source_report).read_text())
    rows = [row for row in report["events"] if low <= float(row["mass"]) < high
            and (args.sign == "all" or row["sign"].lower() == args.sign)]
    if not rows:
        raise ValueError("No selected events in the supplied report/window/sign.")
    for row in rows:
        if not all(math.isfinite(float(row[name])) for name in ("mass", "weight")):
            raise ValueError("Non-finite mass or weight in source report")
    import ROOT
    ROOT.gROOT.SetBatch(True)
    tree_name = report.get("tree", "recoTree/SKFlat")
    print(f"[selection] era={report.get('era')}; input={args.input}; "
          f"sign={args.sign}; mass=[{low:g},{high:g}); selected fills={len(rows)}", flush=True)
    print("[definition] all stored AK4 jets; nominal pT = jet_pt * jet_smearedRes; "
          "no jet cuts applied; selected dimuon/tag jet indices unavailable", flush=True)
    sources = set()
    identities = set()
    for row in rows:
        filename, entry = source_entry(row, args.input)
        identity = (row["run"], row["lumi"], row["event"])
        identities.add(identity)
        sources.add(filename)
        print(f"\n[event] source={filename}; entry={entry}; "
              f"run/lumi/event={identity}; mass={row['mass']:.6f} GeV; "
              f"applied weight={row['weight']:.8g}", flush=True)
        handle = ROOT.TFile.Open(filename, "READ")
        try:
            if not handle or handle.IsZombie():
                raise ValueError(f"Cannot open {filename}")
            tree = handle.Get(tree_name)
            if not tree or not tree.InheritsFrom("TTree"):
                raise ValueError(f"Missing TTree {tree_name} in {filename}")
            jets = read_jets(tree, entry, row)
            print("  index   stored_pT[GeV]   JER_factor   nominal_pT[GeV]       eta       phi    DeepJet  tightID")
            for jet in jets:
                deepjet = "n/a" if jet["deepjet"] is None else f"{jet['deepjet']:.5f}"
                tight = "n/a" if jet["tight"] is None else str(jet["tight"])
                print(f"  {jet['index']:5d}   {jet['stored_pt']:14.4f}   {jet['jer']:10.5f}   "
                      f"{jet['nominal_pt']:15.4f}   {jet['eta']:8.4f}  {jet['phi']:8.4f}  {deepjet:>9}  {tight:>7}")
            if not jets:
                print("  (no stored AK4 jets)")
        finally:
            if handle:
                handle.Close()
    print(f"\n[summary] selected fills={len(rows)}; unique event IDs={len(identities)}; "
          f"source files={len(sources)}; sum of applied weights={math.fsum(row['weight'] for row in rows):.9g}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
