#!/usr/bin/env python3
"""Audit QCD sample normalisation and, optionally, replay exact selected weights.

  python3 qcd_sample_weight_audit.py --era 2016preVFP 2016postVFP
  python3 qcd_sample_weight_audit.py --era 2016postVFP --replay

Default: read sample histograms, current CommonSampleInfo/ForSNU lists and
production run.C/run_*.C configurations. Default sample: 120To170. No files
are changed. Current metadata is NOT assumed to be the metadata used in an
existing ROOT output. All matching nominal productions are listed separately.
Macro discovery matches file contents, not sample-directory names. The search
root, macro counts and rejection reasons are printed. --find-production-only
performs this discovery without ROOT or reading any histogram.

--replay creates a separate audit directory, copies/renames the CURRENT
NIsoMuon source, and records weights at its existing Dilepton_Mass FillHist
call. It runs ALL recorded jobs of one production using their input files,
configuration and saved libraries. Original sources, macros and ROOT outputs
are never rewritten. Multiple candidate productions require --production-tag.
Run in the compatible ROOT/CMSSW environment with access to the recorded inputs.
Replay can take as long as processing the sample; it submits no batch jobs.

The audit tree stores exact selected histogram fills, raw gen_weight,
MCweight(), trigger luminosity, final weight, correction factor and file/entry.
Only MC macros explicitly marked IsDATA=false are accepted. Replay yield and
Sumw2 must match the existing histogram in every reported window before the
event audit is labelled validated. This is a numerical closure check, not proof
that the production source/version is identical. N_eff is never an event count.
The MCweight default uses sumSign when usesign=true; both denominators are shown.
Generated/skim total entries must not be equated to selected fills or N_eff.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import qcd_pt_yield_diagnostics as d


CLASS = "NIsoMuonQCDWeightAudit"
NOMINAL_EXCLUSIONS = {"RunSyst", "RunXSecSyst", "MuonIDEfficiency", "TriggerEfficiency",
                      "ConvenerStudy", "DYValidationDRStudy"}
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def cpp_string(text):
    return json.dumps(str(text), ensure_ascii=True)


def strip_comments(text):
    # Preserve quoted root:// paths and other string literals.
    pattern = r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/'
    return re.sub(pattern, lambda m: m.group() if m.group().startswith('"') else ' ', text, flags=re.S)


def inspect_job(path, era, alias):
    text = strip_comments(path.read_text())
    def string_field(pattern):
        match = re.search(pattern + r'\s*\(?("(?:\\.|[^"\\])*")', text)
        return json.loads(match.group(1)) if match else None
    declaration = re.search(r"\b(\w+)\s+m\s*;", text)
    info = dict(sample=string_field(r"m\.MCSample\s*="),
                era=string_field(r"m\.SetEra"),
                analyser=declaration.group(1) if declaration else None,
                sample_hint=alias in text or alias in path.parts)
    if info['sample'] != alias:
        return None, 'sample mismatch or unsupported assignment', info
    if info['analyser'] != 'NIsoMuon':
        return None, 'analyser mismatch or unsupported declaration', info
    if info['era'] != era:
        return None, 'era mismatch or unsupported SetEra', info
    if not re.search(r"m\.IsDATA\s*=\s*false\s*;", text):
        return None, 'IsDATA is not explicitly false', info
    flag_match = re.search(r"m\.Userflags\s*=\s*\{(.*?)\}\s*;", text, re.S)
    flags = re.findall(r'"([^"\\]*)"', flag_match.group(1)) if flag_match else []
    info['flags'] = flags
    if NOMINAL_EXCLUSIONS.intersection(flags):
        return None, 'non-nominal Userflags', info
    values = {}
    for key in ("xsec", "sumSign", "sumW"):
        match = re.search(r"m\." + key + r"\s*=\s*(" + NUMBER + r")\s*;", text)
        values[key] = float(match.group(1)) if match else None
    inputs = [json.loads(s) for s in re.findall(r'm\.AddFile\s*\(\s*("(?:\\.|[^"\\])*")', text)]
    group = path.parent.parent if re.fullmatch(r"job_\d+", path.parent.name) else path.parent
    job = dict(path=str(path), group=str(group), values=values, flags=flags, inputs=inputs,
               limits=re.findall(r"m\.(?:MaxEvent|SkipEvent)\s*=[^;]+;", text), sha256=digest(path))
    return job, 'matched', info


def read_job(path, era, alias):
    return inspect_job(path, era, alias)[0]


def find_jobs(root, era, alias):
    jobs = []
    root = Path(root).expanduser().resolve() if root else None
    print(f"[production-search] root={root}; era={era}; sample={alias}", flush=True)
    if root is None or not root.is_dir():
        print('[production-search] Directory does not exist; set --runlog-dir to the saved runlog directory', flush=True)
        return jobs
    counts, rejected, symlinks, errors = Counter(macros=0), [], [], []
    for directory, subdirs, filenames in os.walk(root, onerror=lambda exc: errors.append(str(exc))):
        subdirs[:] = [x for x in subdirs if x not in {"lib", "output", "www", ".git"}]
        folder = Path(directory)
        # Do not follow links recursively (archives may contain cycles); a linked
        # target can be selected directly with --runlog-dir.
        symlinks.extend(str(folder / x) for x in subdirs if (folder / x).is_symlink())
        for name in filenames:
            if not re.fullmatch(r"run(?:_[A-Za-z0-9_]+|\d+)?\.C", name):
                continue
            counts['macros'] += 1
            path = folder / name
            try:
                job, reason, info = inspect_job(path, era, alias)
            except (OSError, UnicodeError, ValueError) as exc:
                counts['unreadable macro'] += 1
                errors.append(f'{path}: {exc}')
                continue
            counts[reason] += 1
            if job:
                jobs.append(job)
            elif info['sample_hint'] and len(rejected) < 5:
                rejected.append(dict(path=str(path), reason=reason,
                                     **{k:v for k,v in info.items() if k!='sample_hint'}))
            if counts['macros'] % 5000 == 0:
                print(f"[production-search-progress] macros={counts['macros']}; matched={len(jobs)}", flush=True)
    print('[production-search-summary] ' + json.dumps(dict(counts)), flush=True)
    for item in rejected:
        print('[production-rejected-example] ' + json.dumps(item), flush=True)
    for error in errors[:5]:
        print(f'[production-search-error] {error}', flush=True)
    if symlinks:
        print(f'[production-search] skipped linked directories={len(symlinks)}; examples={symlinks[:3]}; '
              'select their target with --runlog-dir if needed', flush=True)
    return sorted(jobs, key=lambda j: j["path"])


def read_current_metadata(skflat, era, sample, version):
    directory = skflat / "data" / version / era / "Sample"
    matches = [f for f in (directory / "CommonSampleInfo").glob("QCD_Pt-*_MuEnriched.txt")
               if d.canonical_sample(f.stem[len("QCD_Pt-"):-len("_MuEnriched")]) == sample]
    if len(matches) != 1:
        print(f"[metadata] Expected one CommonSampleInfo file; found {len(matches)} in {directory}")
        return None
    filename = matches[0]
    rows = [line.split('#', 1)[0].split() for line in filename.read_text().splitlines()]
    rows = [row for row in rows if row]
    if len(rows) != 1 or len(rows[0]) != 6:
        raise ValueError(f"Unexpected metadata format: {filename}")
    alias, pd, xsec, nmc, sumsign, sumw = rows[0]
    result = dict(alias=alias, PD=pd, xsec=float(xsec), nmc=float(nmc),
                  sumSign=float(sumsign), sumW=float(sumw), path=str(filename), sha256=digest(filename))
    print(f"[current-metadata] {filename}; sha256={result['sha256']}")
    print("[current-values] " + json.dumps({k:result[k] for k in ("alias", "PD", "xsec", "nmc", "sumSign", "sumW")}))
    for key in ("sumSign", "sumW"):
        factor = None if result[key] == 0 else result["xsec"] / result[key]
        print(f"[current-normalisation] xsec/{key}={d.fmt(factor)} pb per generator-weight unit; before lumi/SFs")
    for name in (alias, "SkimTree_NIsoMuon_" + alias):
        filelist = directory / "ForSNU" / (name + ".txt")
        if filelist.is_file():
            inputs = [line.split('#', 1)[0].strip() for line in filelist.read_text().splitlines()]
            inputs = [line for line in inputs if line]
            print(f"[current-filelist] {filelist}; files={len(inputs)}, duplicates={len(inputs)-len(set(inputs))}")
    return result


def show_productions(jobs, current):
    groups = defaultdict(list)
    for job in jobs:
        groups[job["group"]].append(job)
    for group, records in sorted(groups.items()):
        print(f"[candidate-production] tag={Path(group).parent.name}; sample-dir={group}; jobs={len(records)}")
        variants = Counter(tuple(j["values"][k] for k in ("xsec", "sumSign", "sumW")) for j in records)
        for values, number in variants.items():
            params = dict(zip(("xsec", "sumSign", "sumW"), values))
            print(f"[recorded-job-values] jobs={number}: {json.dumps(params)}")
            if current:
                changes = {k: (params[k], current[k]) for k in params
                           if params[k] is None or not math.isclose(params[k], current[k], rel_tol=1e-12, abs_tol=1e-12)}
                print(f"[recorded/current] {'DIFFERENT ' + json.dumps(changes) if changes else 'all three numeric values match'}")
        inputs = [name for job in records for name in job["inputs"]]
        print(f"[recorded-inputs] files={len(inputs)}; duplicates across jobs={len(inputs)-len(set(inputs))}")
        limited = [j["path"] for j in records if j["limits"]]
        if limited:
            print(f"[WARNING] Event limits in {len(limited)} job macros: {limited[:3]}")
        print("[provenance] Candidate job configuration; association with the existing output is not yet verified")
    if not groups:
        print("[production] No matching nominal MC macros found; metadata actually used in production remains unknown")
    return groups


HOOK_DECLARATIONS = r'''
  TTree *_qcdAuditTree = nullptr;
  double _qcdMass=0, _qcdWeight=0, _qcdGen=0, _qcdMC=0, _qcdLumi=0, _qcdCorrection=0;
  double _qcdXsec=0, _qcdSumSign=0, _qcdSumW=0;
  int _qcdSign=0;
  Long64_t _qcdEntry=-1;
  std::string _qcdFile;
  void QCDWeightAuditInit();
  void QCDWeightAuditRecord(TString region, double mass, double weight, double mc, double lumi);
  void QCDWeightAuditWrite(const char *path);
'''

HOOK_IMPLEMENTATION = r'''
void @CLASS@::QCDWeightAuditInit() {
  if(_qcdAuditTree) return;
  _qcdAuditTree = new TTree("QCDWeightAudit", "Selected nominal histogram fills");
  _qcdAuditTree->SetDirectory(nullptr);
  _qcdAuditTree->Branch("mass", &_qcdMass, "mass/D");
  _qcdAuditTree->Branch("weight", &_qcdWeight, "weight/D");
  _qcdAuditTree->Branch("gen_weight", &_qcdGen, "gen_weight/D");
  _qcdAuditTree->Branch("mcweight", &_qcdMC, "mcweight/D");
  _qcdAuditTree->Branch("trigger_lumi", &_qcdLumi, "trigger_lumi/D");
  _qcdAuditTree->Branch("correction", &_qcdCorrection, "correction/D");
  _qcdAuditTree->Branch("xsec", &_qcdXsec, "xsec/D");
  _qcdAuditTree->Branch("sumSign", &_qcdSumSign, "sumSign/D");
  _qcdAuditTree->Branch("sumW", &_qcdSumW, "sumW/D");
  _qcdAuditTree->Branch("sign", &_qcdSign, "sign/I");
  _qcdAuditTree->Branch("local_entry", &_qcdEntry, "local_entry/L");
  _qcdAuditTree->Branch("input_file", &_qcdFile);
}
void @CLASS@::QCDWeightAuditRecord(TString region, double mass, double weight, double mc, double lumi) {
  if(region != @OS@ && region != @SS@) return;
  if(IsDATA) throw std::runtime_error("QCD audit only accepts MC");
  QCDWeightAuditInit();
  _qcdSign = region == @OS@ ? 1 : -1;
  _qcdMass=mass; _qcdWeight=weight; _qcdGen=gen_weight;
  _qcdMC=mc; _qcdLumi=lumi;
  _qcdCorrection=(mc*lumi != 0.) ? weight/(mc*lumi) : 0.;
  _qcdXsec=xsec; _qcdSumSign=sumSign; _qcdSumW=sumW;
  _qcdEntry=fChain && fChain->GetTree() ? fChain->GetTree()->GetReadEntry() : -1;
  _qcdFile=fChain && fChain->GetCurrentFile() ? fChain->GetCurrentFile()->GetName() : "";
  _qcdAuditTree->Fill();
}
void @CLASS@::QCDWeightAuditWrite(const char *path) {
  QCDWeightAuditInit();
  TFile result(path, "CREATE");
  if(result.IsZombie()) throw std::runtime_error("Cannot create new audit ROOT output");
  result.cd(); _qcdAuditTree->Write(); result.Close();
}
'''


def make_source(skflat, work, cfg):
    cpp = skflat / "Analyzers/src/NIsoMuon.C"
    header = skflat / "Analyzers/include/NIsoMuon.h"
    source, declaration = cpp.read_text(), header.read_text()
    pattern = (r'\bFillHist\s*\(\s*this_region\s*\+\s*"/Dilepton_Mass___"\s*\+\s*this_region'
               r'\s*,\s*dimuonMass\s*,\s*weight\s*,[^;]+;')
    matches = list(re.finditer(pattern, source))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one nominal mass FillHist call; found {len(matches)}; no replay generated")
    match = matches[0]
    record = 'QCDWeightAuditRecord(this_region, dimuonMass, weight, MCweight(), ev.GetTriggerLumi("Full"));\n    '
    source = source[:match.start()] + record + source[match.start():]
    source = re.sub(r"\bNIsoMuon\b", CLASS, source)
    declaration = declaration.replace("NIsoMuon_h", CLASS + "_h")
    declaration = re.sub(r"\bNIsoMuon\b", CLASS, declaration)
    end = declaration.rfind("\n};")
    if end < 0:
        raise ValueError("Cannot locate analyser class closing declaration")
    declaration = declaration[:end] + HOOK_DECLARATIONS + declaration[end:]
    declaration = '#include <TTree.h>\n#include <TFile.h>\n#include <string>\n#include <stdexcept>\n' + declaration
    regions = {sign:d.p.base_region(replace(cfg, dilepton_sign=sign)) for sign in ("OS", "SS")}
    hook = HOOK_IMPLEMENTATION.replace('@CLASS@', CLASS).replace('@OS@', cpp_string(regions['OS'])).replace('@SS@', cpp_string(regions['SS']))
    (work / (CLASS + ".h")).write_text(declaration)
    (work / (CLASS + ".C")).write_text(source + '\n' + hook)
    return dict(source=str(cpp), source_sha256=digest(cpp), header=str(header), header_sha256=digest(header))


def make_replay_job(job, work, index):
    original = strip_comments(Path(job['path']).read_text())
    if not re.search(r"m\.IsDATA\s*=\s*false\s*;", original):
        raise ValueError("Replay requires explicit MC IsDATA=false")
    if any(token in original for token in ('gSystem->Exec', 'system(', 'TFile(')):
        raise ValueError("Unexpected file/system operation in production macro; manual review required")
    name = f"audit_job_{index}"
    function = re.search(r"\bvoid\s+(run[A-Za-z0-9_]*)\s*\(\s*\)", original)
    if not function:
        raise ValueError(f"Cannot locate run function: {job['path']}")
    text = original[:function.start()] + original[function.start():].replace(function.group(1), name, 1)
    text, declarations = re.subn(r"\bNIsoMuon\s+m\s*;", CLASS + " m;", text)
    stub = work / f"unused_histogram_output_{index}.root"
    text, setters = re.subn(r'm\.SetOutfilePath\s*\(\s*"(?:\\.|[^"\\])*"\s*\)\s*;',
                           lambda _: 'm.SetOutfilePath(' + cpp_string(stub) + ');', text)
    output = work / f"selected_weights_{index}.root"
    text, writers = re.subn(r"m\.WriteHist\s*\(\s*\)\s*;",
                           lambda _: 'm.QCDWeightAuditWrite(' + cpp_string(output) + ');', text)
    if (declarations, setters, writers) != (1,1,1) or len(re.findall(r"m\.Loop\s*\(\s*\)", text)) != 1:
        raise ValueError("Unexpected job structure; cannot safely redirect replay output")
    text = '#include "' + CLASS + '.h"\n' + text
    path = work / (name + ".C")
    path.write_text(text)
    return path, output


def replay(ROOT, cfg, jobs, skflat, output_dir):
    root_binary = shutil.which("root")
    if not root_binary:
        raise OSError("ROOT executable not found; use the compatible ROOT/CMSSW environment")
    inputs = [name for job in jobs for name in job['inputs']]
    if len(inputs) != len(set(inputs)):
        raise ValueError("Duplicate input files across production jobs; resolve provenance before replay")
    if not inputs or any(not re.search(r'(?:^|[/_])MC(?:[/_.]|$)', name) for name in inputs):
        raise ValueError("Recorded inputs are empty or lack explicit MC path/filename identification")
    if any(any(job['values'][k] is None or not math.isfinite(job['values'][k]) or job['values'][k]<=0
               for k in ('xsec','sumSign','sumW')) for job in jobs):
        raise ValueError("Recorded QCD normalisation contains non-literal, non-finite or non-positive values")
    group = Path(jobs[0]['group'])
    libraries = next((ancestor / 'lib' for ancestor in group.parents if (ancestor / 'lib').is_dir()), None)
    if libraries is None or not list(libraries.glob('*.so')):
        raise OSError(f"Saved production libraries not found above {group}; no silent substitution")
    work = Path(tempfile.mkdtemp(prefix=f"{cfg.era}_", dir=output_dir)).resolve()
    provenance = make_source(skflat, work, cfg)
    manifest = dict(era=cfg.era, source=provenance, jobs=jobs, libraries=str(libraries),
                    library_sha256={f.name:digest(f) for f in libraries.glob('*.so')})
    (work / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f"[replay] NEW audit directory={work}; jobs={len(jobs)}; current source SHA256={provenance['source_sha256']}")
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(libraries) + ':' + env.get('LD_LIBRARY_PATH', '')
    include_paths = [str(work), *(str(skflat / directory / 'include') for directory in ('DataFormats','AnalyzerTools','Analyzers'))]
    env['ROOT_INCLUDE_PATH'] = ':'.join(include_paths + [env.get('ROOT_INCLUDE_PATH','')])
    records = []
    for index, job in enumerate(jobs):
        macro, output = make_replay_job(job, work, index)
        driver_name = f"audit_driver_{index}"
        loads = '\n'.join('  if(gSystem->Load(' + cpp_string(f) + ') < 0) { gSystem->Exit(71); return; }'
                          for f in sorted(libraries.glob('*.so')))
        # Preserve the original external-library directives (e.g. LHAPDF).
        external = '\n'.join(re.findall(r'^\s*R__LOAD_LIBRARY\([^\n]+', Path(job['path']).read_text(), re.M))
        driver = (external + '\n#include <TSystem.h>\n#include <TROOT.h>\nvoid ' + driver_name + '() {\n' + loads
                  + '\n  if(!gSystem->CompileMacro(' + cpp_string(work / (CLASS+'.C'))
                  + ', "kO")) { gSystem->Exit(72); return; }\n  Int_t error=0;\n  gROOT->ProcessLine('
                  + cpp_string('.x "' + str(macro) + '"') + ', &error);\n'
                  + '  if(error) gSystem->Exit(73);\n}\n')
        driver_path = work / (driver_name + '.C')
        driver_path.write_text(driver)
        log = work / f'job_{index}.log'
        print(f"[replay-job] {index+1}/{len(jobs)}: {job['path']}; log={log}", flush=True)
        with log.open('w') as handle:
            status = subprocess.run([root_binary,'-l','-b','-q',str(driver_path)], cwd=work, env=env,
                                    stdout=handle, stderr=subprocess.STDOUT).returncode
        if status != 0 or not output.is_file():
            raise RuntimeError(f"Replay failed (exit={status}); inspect {log}")
        root_file = d.open_file(ROOT, output)
        try:
            tree = root_file.Get('QCDWeightAudit')
            if not tree:
                raise ValueError(f"Missing audit tree in {output}")
            for row in tree:
                record = {key:float(getattr(row,key)) for key in
                          ('mass','weight','gen_weight','mcweight','trigger_lumi','correction','xsec','sumSign','sumW')}
                if not all(math.isfinite(v) for v in record.values()):
                    raise ValueError(f"Non-finite event weight/metadata in {output}")
                record.update(sign='OS' if int(row.sign)==1 else 'SS', local_entry=int(row.local_entry), input_file=str(row.input_file))
                records.append(record)
        finally:
            root_file.Close()
    identities = [(r['input_file'],r['local_entry'],r['sign']) for r in records]
    if len(identities) != len(set(identities)):
        raise ValueError("Repeated input-file/local-entry/sign in replay; audit counts are not validated")
    return records, work


def event_summary(records):
    weights = [r['weight'] for r in records]
    count = len(weights)
    sumw, sumw2 = math.fsum(weights), math.fsum(w*w for w in weights)
    return dict(n_fills=count, sumw=sumw, sumw2=sumw2, n_negative=sum(w<0 for w in weights),
                n_zero=sum(w==0 for w in weights), n_eff=None if sumw2==0 else sumw**2/sumw2,
                mean=None if not count else sumw/count,
                rms=None if not count else math.sqrt(sumw2/count),
                minimum=min(weights) if weights else None, maximum=max(weights) if weights else None)


def report_events(records, histograms, windows, top):
    for label, usesign in (('sign*xsec/sumSign',True),('gen_weight*xsec/sumW',False)):
        checked = [r for r in records if r['sumSign' if usesign else 'sumW'] != 0]
        mismatches = 0
        for row in checked:
            gen = (1. if row['gen_weight']>0 else (-1. if row['gen_weight']<0 else 0.)) if usesign else row['gen_weight']
            expected = gen * row['xsec'] / row['sumSign' if usesign else 'sumW']
            mismatches += not math.isclose(row['mcweight'],expected,rel_tol=1e-8,abs_tol=1e-12)
        print(f'[replay-MCweight] {label}: checked={len(checked)}, mismatches={mismatches}')
    validated = True
    for window in windows:
        for sign in ('OS','SS'):
            selected = [r for r in records if r['sign']==sign and window[0] <= r['mass'] < window[1]]
            result = event_summary(selected)
            stored = histograms[sign,window]
            agree = stored is not None and math.isclose(result['sumw'],stored.value,rel_tol=1e-8,abs_tol=1e-8)
            stat_agree = stored is not None and stored.variance is not None and math.isclose(
                result['sumw2'],stored.variance,rel_tol=1e-8,abs_tol=1e-8)
            validated = validated and agree and stat_agree
            print(f"[event-audit] {sign} {window[0]:g}--{window[1]:g}: " + json.dumps(result))
            print(f"[replay-closure] yield={'OK' if agree else 'MISMATCH/unknown'}; "
                  f"Sumw2={'OK' if stat_agree else 'MISMATCH/unknown'}; "
                  f"existing-yield={d.fmt(None if stored is None else stored.value)}, "
                  f"existing-Sumw2={d.fmt(None if stored is None else stored.variance)}")
            for row in sorted(selected,key=lambda r:abs(r['weight']),reverse=True)[:top]:
                print('[largest-weight] ' + json.dumps(row))
    print(f"[validation] {'PASS: exact fill counts and weights numerically reproduce reported windows' if validated else 'FAIL: replay is not validated against existing inputs; do not interpret it as the original selected-event audit'}")
    return validated


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--era',nargs='+',default=['2016preVFP','2016postVFP'])
    parser.add_argument('--sample',default='120To170',help='one pT bin, default: 120To170')
    parser.add_argument('--base-dir',default=d.p.Config().base_dir)
    parser.add_argument('--trigger',default='')
    parser.add_argument('--skflat-dir',default=os.getenv('SKFlat_WD',str(Path(__file__).resolve().parent.parent/'SKFlatAnalyzer')))
    parser.add_argument('--data-version',default='Run2UltraLegacy_v3')
    parser.add_argument('--runlog-dir',default=os.getenv('SKFlatRunlogDir') or f'/data6/Users/{getpass.getuser()}/SKRunlog',
                        help='saved runlog root or a specific production/sample directory; supports ~')
    parser.add_argument('--find-production-only',action='store_true',
                        help='print metadata and production discovery only; does not require ROOT')
    parser.add_argument('--production-tag',help='exact timestamp/analyser directory name or exact sample-directory path')
    parser.add_argument('--replay',action='store_true',help='run recorded MC jobs on input ntuples into a NEW audit directory')
    parser.add_argument('--output-dir',help='parent for NEW replay directories; default: system temporary directory')
    parser.add_argument('--top',type=int,default=3,help='largest absolute weights printed per sign/window; default: 3')
    args = parser.parse_args(argv)
    if args.top < 0:
        parser.error('--top must be non-negative')
    if args.find_production_only and args.replay:
        parser.error('--find-production-only cannot be combined with --replay')
    try:
        years = list(dict.fromkeys(year for era in args.era for year in d.p.years_for_era(era)))
        if not args.find_production_only:
            import ROOT
            ROOT.gROOT.SetBatch(True)
    except (ImportError,ValueError) as exc:
        print(f'[ERROR] {exc}',file=sys.stderr); return 1
    skflat = Path(args.skflat_dir).expanduser().resolve()
    sample = d.canonical_sample(args.sample)
    if args.output_dir and args.replay:
        Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    status = 0
    for era in years:
        try:
            cfg = d.p.Config(era=era,base_dir=args.base_dir,trigger=args.trigger)
            directory = Path(d.p.root_dir_for_year(cfg,era))
            if args.find_production_only:
                print(f'\n[era] {era}; production discovery only; metadata repo={skflat}',flush=True)
            else:
                files = [f for f in directory.glob('Skim_NIsoMuon_QCD_Pt-*_MuEnriched.root')
                         if d.canonical_sample(d.sample_name(f))==sample]
                if len(files)!=1:
                    raise ValueError(f'Expected one sample ROOT file in {directory}; found {len(files)}')
                print(f'\n[era] {era}; ROOT input={files[0]}; metadata repo={skflat}',flush=True)
            git = subprocess.run(['git','-C',str(skflat),'rev-parse','--abbrev-ref','HEAD'],capture_output=True,text=True)
            sha = subprocess.run(['git','-C',str(skflat),'rev-parse','HEAD'],capture_output=True,text=True)
            print(f'[current-repository] branch={git.stdout.strip() or "unknown"}; commit={sha.stdout.strip() or "unknown"}')
            current = read_current_metadata(skflat,era,sample,args.data_version)
            sample_label = sample.replace('to','To').replace('inf','Inf') if args.find_production_only else d.sample_name(files[0])
            alias = 'QCD_Pt-' + sample_label + '_MuEnriched'
            if current:
                alias = current['alias']
            groups = show_productions(find_jobs(args.runlog_dir,era,alias),current)
            if args.find_production_only:
                if not groups:
                    status = 1
                continue
            low_high, origin = d.production_windows(ROOT,directory,cfg)
            if any(len(w)!=2 or w[0]>=w[1] or not all(math.isfinite(v) for v in w) for w in low_high):
                raise ValueError('Invalid recorded windows')
            windows = list(dict.fromkeys([*low_high,(11.,15.),*((float(m),float(m+1)) for m in range(11,15))]))
            print(f'[windows] {low_high}; {origin}')
            histograms = d.read_counts(ROOT,files[0],cfg,windows)
            root_file = d.open_file(ROOT,files[0])
            try:
                for sign in ('OS','SS'):
                    sign_cfg = replace(cfg,dilepton_sign=sign)
                    hist = root_file.Get(d.p.hist_path(sign_cfg,d.p.base_region(sign_cfg)))
                    if hist:
                        print(f'[stored-entries] {sign}: GetEntries={hist.GetEntries():.12g}; '
                              'whole histogram including flows, not a mass-window count')
            finally:
                root_file.Close()
            for (sign,window), count in histograms.items():
                effective_weight = None if count is None or count.variance is None or count.value==0 else count.variance/count.value
                print(f'[histogram] {sign} {window[0]:g}--{window[1]:g}: sumw={d.fmt(None if count is None else count.value)}, '
                      f'Sumw2={d.fmt(None if count is None else count.variance)}, '
                      f'N_eff={d.fmt(None if count is None else count.effective)}, Sumw2/sumw={d.fmt(effective_weight)}')
            print('[unweighted-count] Per-window selected fills and individual weights are unavailable from weighted histograms alone')
            if not args.replay:
                continue
            candidates = {group:jobs for group,jobs in groups.items() if not args.production_tag
                          or args.production_tag in (group,Path(group).parent.name)}
            if len(candidates)!=1:
                if not groups:
                    raise ValueError('No nominal production found; inspect production-search output and '
                                     'set --runlog-dir to the saved archive. --production-tag only selects '
                                     'among discovered productions; it cannot locate missing macros.')
                raise ValueError(f'Replay needs exactly one production; found {len(candidates)}. '
                                 'Use --production-tag with a candidate-production tag printed above.')
            jobs = next(iter(candidates.values()))
            records, work = replay(ROOT,cfg,jobs,skflat,args.output_dir)
            if not report_events(records,histograms,windows,args.top):
                status = 1
            print(f'[audit-files] {work}')
        except (OSError,ValueError,KeyError,TypeError,RuntimeError) as exc:
            sys.stdout.flush()
            print(f'[ERROR] {era}: {exc}',file=sys.stderr,flush=True); status = 1
    return status


if __name__=='__main__':
    raise SystemExit(main())
