#!/usr/bin/env python3
r"""Trace selected QCD audit fills to skim entries and original SKFlat ntuples.

  python3 -u qcd_event_source_trace.py --audit-dir plots/qcd_weight_audit/<audit>
  python3 -u qcd_event_source_trace.py --audit-dir plots/qcd_weight_audit/<audit> \
    --original-filelist /path/to/ForSNU/QCD_Pt-120To170_MuEnriched.txt

Reads an existing qcd_sample_weight_audit.py directory; no analyser replay is
needed. Default selection: SS, 11 <= dimuon mass < 80 GeV. The stored input_file
and zero-based file-local entry locate the skim event. Reads integer run/lumi/
event IDs and ALL raw muon pt/eta/phi/charge values from that entry. Raw leading
pt is max(muon_pt), not necessarily the selected/corrected analyser muon's pt.

With --original-filelist, scans ALL listed original ntuples for exact integer
run:lumi:event matches and compares the complete raw muon vectors. Scanning
reads only ID branches in C++; muon branches are read for matching entries.
Every ID collision is retained. Zero/multiple verified matches or any unreadable
original file returns nonzero. Uniqueness applies only to the supplied list.
Skim filenames are job numbers, not an assumed mapping to original filenames.

All ROOT files are opened READ-only. Optional --output writes a NEW JSON file
without overwriting an existing file. Original-source matches are numerical
ID/kinematic associations, not recovered historical production provenance.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys


ID_TYPES = {'run': 'Int_t', 'lumi': 'Int_t', 'event': 'ULong64_t'}
MUON_BRANCHES = ('muon_pt', 'muon_eta', 'muon_phi', 'muon_charge')

SCAN_CPP = r'''
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
namespace QCDSourceTrace {
std::vector<Long64_t> findEntries(TTree *tree, const std::vector<std::string>& ids) {
  using Key = std::tuple<Int_t, Int_t, ULong64_t>;
  std::set<Key> wanted;
  for(const auto& text : ids) {
    std::istringstream stream(text);
    Int_t run, lumi; ULong64_t event;
    if(!(stream >> run >> lumi >> event)) throw std::runtime_error("Invalid event ID");
    wanted.emplace(run, lumi, event);
  }
  TTreeReader reader(tree);
  TTreeReaderValue<Int_t> run(reader, "run"), lumi(reader, "lumi");
  TTreeReaderValue<ULong64_t> event(reader, "event");
  std::vector<Long64_t> entries;
  Long64_t scanned = 0;
  while(reader.Next()) {
    ++scanned;
    if(wanted.count(Key(*run, *lumi, *event))) entries.push_back(reader.GetCurrentEntry());
  }
  if(scanned != tree->GetEntries()) throw std::runtime_error("Incomplete event ID scan");
  return entries;
}
}
'''


def emit(label, value):
    print(f'[{label}] ' + json.dumps(value, allow_nan=False), flush=True)


def open_tree(ROOT, filename, name):
    handle = ROOT.TFile.Open(str(filename), 'READ')
    if not handle or handle.IsZombie():
        if handle:
            handle.Close()
        raise OSError(f'Cannot open ROOT file: {filename}')
    tree = handle.Get(name)
    if not tree or not tree.InheritsFrom('TTree'):
        handle.Close()
        raise ValueError(f'Missing TTree {filename}:{name}')
    return handle, tree


def require_branches(tree, names):
    missing = [name for name in names if not tree.GetBranch(name)]
    if missing:
        raise ValueError('Missing branches: ' + ', '.join(missing))


def enable_branches(tree, names):
    require_branches(tree, names)
    tree.SetBranchStatus('*', 0)
    for name in names:
        tree.SetBranchStatus(name, 1)


def validate_id_types(tree):
    require_branches(tree, ID_TYPES)
    for name, expected in ID_TYPES.items():
        leaf = tree.GetLeaf(name)
        actual = str(leaf.GetTypeName()) if leaf else 'no scalar leaf'
        if actual != expected:
            raise ValueError(f'{name} type={actual}; expected {expected}; no lossy ID conversion')


def event_key(row):
    return tuple(row[name] for name in ID_TYPES)


def read_event(tree, entry):
    if entry < 0 or entry >= int(tree.GetEntries()) or tree.GetEntry(entry) <= 0:
        raise ValueError(f'Cannot read file-local entry {entry}')
    ids = {name: int(getattr(tree, name)) for name in ID_TYPES}
    vectors = {name: [int(v) if name == 'muon_charge' else float(v)
                      for v in getattr(tree, name)] for name in MUON_BRANCHES}
    if len({len(values) for values in vectors.values()}) != 1:
        raise ValueError(f'Muon vector lengths disagree at entry {entry}')
    if not all(math.isfinite(value) for name, values in vectors.items()
               if name != 'muon_charge' for value in values):
        raise ValueError(f'Non-finite muon kinematics at entry {entry}')
    return dict(**ids, raw_muons=vectors,
                raw_leading_muon_pt=max(vectors['muon_pt'], default=None))


def read_audit(ROOT, directory, sign, window):
    manifest = json.loads((directory / 'manifest.json').read_text())
    jobs = manifest.get('jobs', [])
    if not jobs:
        raise ValueError('Audit manifest contains no jobs')
    files = [directory / f'selected_weights_{index}.root' for index in range(len(jobs))]
    if any(not filename.is_file() for filename in files):
        raise ValueError('Incomplete audit directory: a selected_weights_<job>.root file is missing')
    selected, seen = [], set()
    wanted = {'OS', 'SS'} if sign == 'both' else {sign.upper()}
    fields = ('mass', 'weight', 'correction', 'sign', 'input_file', 'local_entry')
    for filename in files:
        handle, tree = open_tree(ROOT, filename, 'QCDWeightAudit')
        try:
            enable_branches(tree, fields)
            for index in range(int(tree.GetEntries())):
                if tree.GetEntry(index) <= 0:
                    raise ValueError(f'Cannot read {filename}:entry {index}')
                mass, weight, correction = (float(getattr(tree, name)) for name in fields[:3])
                if not all(math.isfinite(value) for value in (mass, weight, correction)):
                    raise ValueError(f'Non-finite audit values: {filename}:entry {index}')
                code = int(tree.sign)
                if code not in (-1, 1):
                    raise ValueError(f'Invalid audit sign: {code}')
                label = 'OS' if code == 1 else 'SS'
                if label not in wanted or not window[0] <= mass < window[1]:
                    continue
                source, entry = str(tree.input_file), int(tree.local_entry)
                identity = (source, entry, label)
                if not source or entry < 0 or identity in seen:
                    raise ValueError(f'Invalid/repeated selected fill: {identity}')
                seen.add(identity)
                selected.append(dict(audit_file=str(filename), audit_entry=index,
                                     input_file=source, local_entry=entry, sign=label,
                                     mass=mass, weight=weight, correction=correction))
        finally:
            handle.Close()
    return manifest, selected


def enrich_skim_entries(ROOT, records, tree_name):
    groups = defaultdict(list)
    for row in records:
        groups[row['input_file']].append(row)
    for filename, rows in groups.items():
        handle, tree = open_tree(ROOT, filename, tree_name)
        try:
            validate_id_types(tree)
            enable_branches(tree, (*ID_TYPES, *MUON_BRANCHES))
            for row in rows:
                row.update(read_event(tree, row['local_entry']))
                row['original_matches'] = []
                emit('skim-event', row)
        finally:
            handle.Close()


def read_filelist(filename):
    names = [line.split('#', 1)[0].strip() for line in filename.read_text().splitlines()]
    names = [name for name in names if name]
    if not names or len(names) != len(set(names)):
        raise ValueError('Original file list is empty or has duplicate paths')
    if any(not name.endswith('.root') for name in names):
        raise ValueError('Original file list contains a non-ROOT path')
    return names


def scan_originals(ROOT, records, names, tree_name):
    if not records:
        return [], 0
    if not ROOT.gInterpreter.Declare(SCAN_CPP):
        raise RuntimeError('Cannot compile the event ID scanner')
    targets = defaultdict(list)
    for row in records:
        targets[event_key(row)].append(row)
    ids = ROOT.std.vector('string')()
    for key in targets:
        ids.push_back(' '.join(str(value) for value in key))
    errors, scanned_files = [], 0
    for index, filename in enumerate(names):
        emit('original-scan', dict(file=index + 1, total=len(names), path=filename))
        handle = None
        try:
            handle, tree = open_tree(ROOT, filename, tree_name)
            validate_id_types(tree)
            enable_branches(tree, ID_TYPES)
            entries = [int(value) for value in ROOT.QCDSourceTrace.findEntries(tree, ids)]
            enable_branches(tree, (*ID_TYPES, *MUON_BRANCHES))
            for entry in entries:
                event = read_event(tree, entry)
                for row in targets[event_key(event)]:
                    match = dict(input_file=filename, local_entry=entry,
                                 raw_muons_match=event['raw_muons'] == row['raw_muons'])
                    row['original_matches'].append(match)
                    emit('original-match', dict(run=row['run'], lumi=row['lumi'],
                                                event=row['event'], **match))
            scanned_files += 1
        except Exception as exc:
            error = dict(input_file=filename, error=str(exc))
            errors.append(error)
            emit('original-error', error)
        finally:
            if handle:
                handle.Close()
    return errors, scanned_files


def match_status(row, searched, complete):
    if not searched:
        return 'not_searched'
    matches = row['original_matches']
    verified = [match for match in matches if match['raw_muons_match']]
    if not complete:
        return 'incomplete_search'
    if len(verified) > 1:
        return 'ambiguous'
    if len(verified) == 1:
        return 'unique_in_supplied_list'
    return 'id_only_kinematics_mismatch' if matches else 'not_found'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--audit-dir', required=True, help='existing complete replay directory')
    parser.add_argument('--sign', choices=('ss', 'os', 'both'), default='ss')
    parser.add_argument('--mass-range', nargs=2, type=float, default=(11., 80.), metavar=('LOW', 'HIGH'))
    parser.add_argument('--tree', default='recoTree/SKFlat')
    parser.add_argument('--original-filelist', help='explicit original, unskimmed ROOT file list to search')
    parser.add_argument('--output', help='NEW JSON path; existing files are not overwritten')
    args = parser.parse_args(argv)
    try:
        if not all(math.isfinite(value) for value in args.mass_range) or args.mass_range[0] >= args.mass_range[1]:
            raise ValueError('Invalid mass range')
        output = Path(args.output).expanduser() if args.output else None
        if output and (output.exists() or not output.parent.is_dir()):
            raise ValueError('--output must be a new path in an existing directory')
        original_list = Path(args.original_filelist).expanduser() if args.original_filelist else None
        names = read_filelist(original_list) if original_list else []
        import ROOT
        ROOT.gROOT.SetBatch(True)
        directory = Path(args.audit_dir).expanduser().resolve()
        manifest, records = read_audit(ROOT, directory, args.sign, args.mass_range)
        if set(names) & {row['input_file'] for row in records}:
            raise ValueError('Original list includes a recorded skim input; supply the unskimmed list')
        emit('selected', dict(era=manifest.get('era'), fills=len(records), sign=args.sign,
                              mass_range=args.mass_range, audit_dir=str(directory)))
        enrich_skim_entries(ROOT, records, args.tree)
        errors, scanned = scan_originals(ROOT, records, names, args.tree) if original_list else ([], 0)
        complete = not errors and scanned == len(names)
        for row in records:
            row['original_status'] = match_status(row, bool(original_list), complete)
            emit('event-source', row)
        report = dict(era=manifest.get('era'), audit_dir=str(directory), tree=args.tree,
                      sign=args.sign, mass_range=args.mass_range, events=records,
                      original_filelist=str(original_list.resolve()) if original_list else None,
                      original_filelist_sha256=hashlib.sha256(original_list.read_bytes()).hexdigest() if original_list else None,
                      listed_original_files=len(names), scanned_original_files=scanned,
                      search_complete=bool(original_list) and complete, errors=errors)
        if output:
            with output.open('x') as handle:
                json.dump(report, handle, indent=2, allow_nan=False)
                handle.write('\n')
            emit('output', str(output.resolve()))
        unresolved = sum(row['original_status'] != 'unique_in_supplied_list' for row in records) if original_list else 0
        emit('trace-summary', dict(selected_fills=len(records), skim_entries_read=len(records),
                                   scanned_original_files=scanned, original_errors=len(errors),
                                   unresolved_original_fills=unresolved))
        return int(bool(errors or unresolved))
    except KeyboardInterrupt:
        print('[INTERRUPTED] source trace stopped', file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        print(f'[ERROR] {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
