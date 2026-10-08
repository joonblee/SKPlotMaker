#!/usr/bin/env python3
"""Read AK4 jet kinematics for already traced, selected QCD MC events.

Uses events in qcd_event_sources_2016postVFP.json from the previous source trace.
Alternatively --audit-dir reads selected_weights_*.root for any audited era,
without generating a source-trace JSON. Reads only their file-local entries.
JSON event IDs are verified exactly; audit-only inputs acquire IDs from the skim.
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


def read_audit(ROOT, directory):
    """Read completed nominal selected-event audit outputs, never replay jobs."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    files = sorted(directory.glob("selected_weights_*.root"))
    jobs = manifest.get("jobs")
    expected = len(jobs) if isinstance(jobs, list) else jobs if isinstance(jobs, int) else None
    if not files or (expected is not None and len(files) != expected):
        raise ValueError(f"Incomplete audit: found {len(files)} selected_weights files; expected={expected}")
    if expected is None:
        print("[audit] job count not recorded; full production coverage cannot be verified", flush=True)
    rows, seen = [], set()
    fields = ("mass", "weight", "sign", "input_file", "local_entry")
    for filename in files:
        handle = ROOT.TFile.Open(str(filename), "READ")
        try:
            if not handle or handle.IsZombie():
                raise ValueError(f"Cannot open audit output {filename}")
            tree = handle.Get("QCDWeightAudit")
            if not tree or not tree.InheritsFrom("TTree"):
                raise ValueError(f"Missing QCDWeightAudit in {filename}")
            if any(not tree.GetBranch(name) for name in fields):
                raise ValueError(f"Incomplete QCDWeightAudit branches in {filename}")
            tree.SetBranchStatus("*", 0)
            for name in fields:
                tree.SetBranchStatus(name, 1)
            for entry in range(int(tree.GetEntries())):
                if tree.GetEntry(entry) <= 0:
                    raise ValueError(f"Unreadable audit entry {entry} in {filename}")
                sign = int(tree.sign)
                if sign not in (-1, 1) or not isinstance(tree.local_entry, int):
                    raise ValueError(f"Invalid sign/file-local entry in {filename}")
                row = dict(mass=float(tree.mass), weight=float(tree.weight),
                           sign="SS" if sign == -1 else "OS", input_file=str(tree.input_file),
                           local_entry=int(tree.local_entry))
                if not all(math.isfinite(row[k]) for k in ("mass", "weight")):
                    raise ValueError(f"Non-finite audit mass/weight in {filename}")
                identity = (row["input_file"], row["local_entry"], row["sign"])
                if identity in seen:
                    raise ValueError(f"Duplicate selected fill across audit jobs: {identity}")
                seen.add(identity)
                rows.append(row)
        finally:
            if handle:
                handle.Close()
    return dict(era=manifest.get("era"), tree="recoTree/SKFlat", events=rows)


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
        observed = getattr(tree, name)
        expected = row.get(name, observed)
        if not isinstance(observed, int) or type(expected) is not int:
            raise ValueError(f"{name}: integer event IDs required; no float conversion")
        if observed != expected:
            raise ValueError(f"Entry/ID mismatch: {name}={observed}, expected={expected}")
        row[name] = observed
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
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--source-report", help="source-trace JSON; default: plots/qcd_event_sources_2016postVFP.json")
    inputs.add_argument("--audit-dir", help="completed era audit directory containing manifest.json and selected_weights_*.root")
    parser.add_argument("--input", choices=("skim", "original"), default="skim")
    parser.add_argument("--sign", choices=("ss", "os", "all"), default="ss")
    parser.add_argument("--mass-range", nargs=2, type=float, default=(11., 80.), metavar=("LOW", "HIGH"))
    args = parser.parse_args(argv)
    if args.audit_dir and args.input == "original":
        parser.error("--input original requires --source-report with verified original matches.")
    low, high = args.mass_range
    if not all(math.isfinite(v) for v in (low, high)) or low >= high:
        parser.error("Mass range must be finite and LOW < HIGH.")
    ROOT = None
    if args.audit_dir:
        import ROOT
        report = read_audit(ROOT, args.audit_dir)
    else:
        report = json.loads(Path(args.source_report or "plots/qcd_event_sources_2016postVFP.json").read_text())
    rows = [row for row in report["events"] if low <= float(row["mass"]) < high
            and (args.sign == "all" or row["sign"].lower() == args.sign)]
    for row in rows:
        if not all(math.isfinite(float(row[name])) for name in ("mass", "weight")):
            raise ValueError("Non-finite mass or weight in source report")
    tree_name = report.get("tree", "recoTree/SKFlat")
    print(f"[selection] era={report.get('era')}; input={args.input}; "
          f"sign={args.sign}; mass=[{low:g},{high:g}); selected fills={len(rows)}", flush=True)
    if not rows:
        print("[summary] selected fills=0; sum of applied weights=0; no selected jets to inspect", flush=True)
        return 0
    if ROOT is None:
        import ROOT
    ROOT.gROOT.SetBatch(True)
    if args.audit_dir:
        print("[audit] file-local entries from audit; event IDs read from skim (no independent ID trace)", flush=True)
    print("[definition] all stored AK4 jets; nominal pT = jet_pt * jet_smearedRes; "
          "no jet cuts applied; selected dimuon/tag jet indices unavailable", flush=True)
    sources = set()
    identities = set()
    for row in rows:
        filename, entry = source_entry(row, args.input)
        sources.add(filename)
        print(f"\n[read] source={filename}; entry={entry}", flush=True)
        handle = ROOT.TFile.Open(filename, "READ")
        try:
            if not handle or handle.IsZombie():
                raise ValueError(f"Cannot open {filename}")
            tree = handle.Get(tree_name)
            if not tree or not tree.InheritsFrom("TTree"):
                raise ValueError(f"Missing TTree {tree_name} in {filename}")
            jets = read_jets(tree, entry, row)
            identity = (row["run"], row["lumi"], row["event"])
            identities.add(identity)
            print(f"[event] run/lumi/event={identity}; mass={row['mass']:.6f} GeV; "
                  f"applied weight={row['weight']:.8g}", flush=True)
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
