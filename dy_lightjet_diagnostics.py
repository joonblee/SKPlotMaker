#!/usr/bin/env python3
"""Read-only DY NF/source audit. Requires PyROOT; never fits or writes ROOT files.

Read merged LightJet/BJet histograms and saved DYAux metadata for each era.
Compare aMC@NLO and MG LO NF measurements and alternative MG predictions.
With multiple eras, print pooled and log-fit common NF diagnostics; these
assume a common ratio and independent MC statistical errors across eras.
No event trees, source replay, production reweighting or template changes.

Default luminosities reproduce the previous comparison's denominators. Override
with --lumi ERA=VALUE if the luminosity for the actual input dataset differs.
Histogram errors are finite-data/MC errors only, not total systematic errors.
"""
import argparse
import math
from pathlib import Path

LUMI = {'2016preVFP': 19.52, '2016postVFP': 16.81, '2017': 41.48, '2018': 59.83}
REG_B = 'OS_POGMedium_tight_BJet_NIsoDimuon'
REG_L = 'OS_POGMedium_tight_LightJet_NIsoDimuon'


def histogram(handle, path):
    h = handle.Get(path)
    if not h or not h.InheritsFrom('TH1') or h.GetDimension() != 1:
        raise ValueError(f'Missing one-dimensional histogram: {handle.GetName()}:{path}')
    return h


def integral(h, low, high):
    axis = h.GetXaxis()
    if low < axis.GetBinLowEdge(1) or high > axis.GetBinUpEdge(h.GetNbinsX()):
        raise ValueError(f'{h.GetName()}: requested window outside axis')
    bins = [i for i in range(1, h.GetNbinsX() + 1)
            if low < axis.GetBinCenter(i) < high]
    value = math.fsum(float(h.GetBinContent(i)) for i in bins)
    error = math.sqrt(math.fsum(float(h.GetBinError(i))**2 for i in bins))
    if not math.isfinite(value) or not math.isfinite(error):
        raise ValueError(f'Non-finite yield/error in {h.GetName()}')
    return value, error


def audit(ROOT, args, era, luminosity):
    files = []

    def open_file(name):
        path = str(Path(args.base_dir) / era / name)
        f = ROOT.TFile.Open(path, 'READ')
        if not f or f.IsZombie():
            raise OSError(f'Cannot open {path}')
        files.append(f)
        return f

    def get(name, region):
        return histogram(open_file(name), f'{region}/Dilepton_Mass___{region}')

    try:
        data = get('data.root', REG_L)
        qcd = get('NIsoMuon_QCD_Inclusive.root', REG_L)
        top = get('NIsoMuon_Top.root', REG_L)
        others = get('NIsoMuon_Others.root', REG_L)
        source = data.Clone(f'diagnostic_source_{era}')
        source.SetDirectory(0)
        source.Sumw2()
        for h in (qcd, top, others):
            a, b = source.GetXaxis(), h.GetXaxis()
            if source.GetNbinsX() != h.GetNbinsX() or any(
                not math.isclose(a.GetBinLowEdge(i), b.GetBinLowEdge(i), abs_tol=1e-10, rel_tol=0)
                for i in range(1, source.GetNbinsX() + 2)
            ):
                raise ValueError('LightJet component binning differs')
            source.Add(h, -1.)

        dy_file = open_file('NIsoMuon_DYJets_est.root')
        prediction = histogram(dy_file, f'{REG_B}/Dilepton_Mass___{REG_B}')
        low, high = args.mass_range
        print(f'\n[era] {era}; lumi={luminosity:g} fb^-1; window=({low:g},{high:g}) GeV')
        print('[errors] finite-data/MC histogram statistics only; no systematic errors')
        print(f'{"Quantity":<31} {"Yield":>13} {"Hist error":>13} {"Yield/fb^-1":>14}')

        def show(label, h):
            value, error = integral(h, low, high)
            print(f'{label:<31} {value:13.7g} {error:13.7g} {value/luminosity:14.7g}')
            return value

        show('LightJet Data', data)
        show('LightJet QCDMC [subtract]', qcd)
        show('LightJet TopMC [subtract]', top)
        show('LightJet OthersMC [subtract]', others)
        current_source = show('LightJet source [current]', source)
        final_yield = show('BJet DYDD [stored]', prediction)

        names = ('LightJetSource', 'NF_aMC', 'NFInputs_aMC')
        missing = [n for n in names if not dy_file.Get(f'DYAux/{n}')]
        if missing:
            print(f'[WARNING] Missing DYAux objects: {missing}. Cannot audit stored NF/source.')
            print('[WARNING] Keep this log; do not regenerate templates solely for this diagnostic.')
            return None
        stored_source = show('LightJet source [stored]', histogram(dy_file, 'DYAux/LightJetSource'))
        h_nf = histogram(dy_file, 'DYAux/NF_aMC')
        nf, nf_error = h_nf.GetBinContent(1), h_nf.GetBinError(1)
        h_inputs = histogram(dy_file, 'DYAux/NFInputs_aMC')
        inputs = {h_inputs.GetXaxis().GetBinLabel(i): h_inputs.GetBinContent(i)
                  for i in range(1, h_inputs.GetNbinsX() + 1)}
        required = ('BJetYield', 'BJetError', 'LightJetYield', 'LightJetError', 'WindowLow', 'WindowHigh')
        if any(k not in inputs for k in required):
            raise ValueError('Incomplete NFInputs_aMC labels')
        print(f'[NF stored] {nf:.9g} +/- {nf_error:.9g}; extraction window=' 
              f'({inputs["WindowLow"]:g},{inputs["WindowHigh"]:g}) GeV')
        print(f'[NF inputs stored] BJet={inputs["BJetYield"]:.9g} +/- {inputs["BJetError"]:.9g}; '
              f'LightJet={inputs["LightJetYield"]:.9g} +/- {inputs["LightJetError"]:.9g}')

        mc_b = get(args.dy_amc_file, REG_B)
        mc_l = get(args.dy_amc_file, REG_L)
        b, eb = integral(mc_b, inputs['WindowLow'], inputs['WindowHigh'])
        l, el = integral(mc_l, inputs['WindowLow'], inputs['WindowHigh'])
        if b <= 0 or l <= 0:
            raise ValueError('Non-positive current DY MC NF numerator/denominator')
        nf_current = b/l
        error_current = math.hypot(eb/l, b*el/l**2)
        print(f'[NF current] {nf_current:.9g} +/- {error_current:.9g}; BJet={b:.9g}; LightJet={l:.9g}')

        def check(label, observed, expected):
            same = math.isclose(observed, expected, rel_tol=1e-7, abs_tol=1e-6)
            print(f'[check {"OK" if same else "MISMATCH"}] {label}: '
                  f'observed={observed:.9g}, expected={expected:.9g}, delta={observed-expected:.9g}')

        check('source current vs stored', current_source, stored_source)
        check('DYDD vs stored NF * stored source', final_yield, nf*stored_source)
        check('NF stored vs stored B/L', nf, inputs['BJetYield']/inputs['LightJetYield'])
        check('NF current vs stored', nf_current, nf)
        check('DY MC BJet current vs stored', b, inputs['BJetYield'])
        check('DY MC LightJet current vs stored', l, inputs['LightJetYield'])
        measurements = {'aMC': (b, eb, l, el, nf_current, error_current)}
        # The MG prediction is a diagnostic only; never change the saved central.
        mg_b = get(args.dy_mg_file, REG_B)
        mg_l = get(args.dy_mg_file, REG_L)
        mb, meb = integral(mg_b, inputs['WindowLow'], inputs['WindowHigh'])
        ml, mel = integral(mg_l, inputs['WindowLow'], inputs['WindowHigh'])
        if mb <= 0 or ml <= 0:
            raise ValueError('Non-positive current MG DY NF numerator/denominator')
        mg_nf = mb/ml
        mg_error = math.hypot(meb/ml, mb*mel/ml**2)
        measurements['MG'] = (mb, meb, ml, mel, mg_nf, mg_error)
        print(f'[NF current MG] {mg_nf:.9g} +/- {mg_error:.9g}; '
              f'BJet={mb:.9g} +/- {meb:.9g}; LightJet={ml:.9g} +/- {mel:.9g}')
        print(f'[MG MC/fb^-1] BJet={mb/luminosity:.9g}; LightJet={ml/luminosity:.9g}')
        mg_stored = dy_file.Get('DYAux/NF_MG')
        if mg_stored:
            print(f'[NF stored MG] {mg_stored.GetBinContent(1):.9g} '
                  f'+/- {mg_stored.GetBinError(1):.9g}')
            check('MG NF current vs stored', mg_nf, mg_stored.GetBinContent(1))
        else:
            print('[WARNING] No stored NF_MG; current MG measurement is still printed.')
        mg_inputs = dy_file.Get('DYAux/NFInputs_MG')
        if mg_inputs:
            mi = {mg_inputs.GetXaxis().GetBinLabel(i): mg_inputs.GetBinContent(i)
                  for i in range(1, mg_inputs.GetNbinsX() + 1)}
            for key, current in [('BJetYield', mb), ('LightJetYield', ml),
                                 ('WindowLow', inputs['WindowLow']), ('WindowHigh', inputs['WindowHigh'])]:
                if key not in mi:
                    raise ValueError(f'Missing MG NF input label: {key}')
                check(f'MG {key} current vs stored', current, mi[key])
        else:
            print('[WARNING] No stored NFInputs_MG; cannot verify MG input provenance.')
        print(f'[MG/aMC] NF ratio={mg_nf/nf_current:.9g}; '
              f'|ratio-1|={abs(mg_nf/nf_current-1):.9g}')
        print(f'[DYDD MG alternative] yield={mg_nf*stored_source:.9g}; '
              f'yield/fb^-1={mg_nf*stored_source/luminosity:.9g}; same stored LightJet source')
        return {'source_rate': stored_source/luminosity, 'nf': nf,
                'nf_MG': mg_nf, 'dy_rate': final_yield/luminosity,
                'dy_rate_MG': mg_nf*stored_source/luminosity,
                'measurements': measurements}
    finally:
        for f in files:
            f.Close()


def common_nf_summary(results):
    usable = {era: result for era, result in results.items() if result}
    if len(usable) < 2:
        return
    print('\n[common NF diagnostics] eras=' + ','.join(usable))
    print('[assumption] one common NF across selected eras; independent MC statistical errors only')
    for generator in ('aMC', 'MG'):
        rows = [r['measurements'][generator] for r in usable.values()]
        b = math.fsum(x[0] for x in rows)
        l = math.fsum(x[2] for x in rows)
        eb = math.sqrt(math.fsum(x[1]**2 for x in rows))
        el = math.sqrt(math.fsum(x[3]**2 for x in rows))
        pooled = b/l
        pooled_error = math.hypot(eb/l, b*el/l**2)
        print(f'[pooled {generator}] sum(B)/sum(L)={pooled:.9g} +/- {pooled_error:.9g}')
        if any(x[5] <= 0 for x in rows):
            print(f'[WARNING] Cannot fit common log NF for {generator}: zero error')
            continue
        weights = [(x[4]/x[5])**2 for x in rows]
        mean = math.fsum(w*math.log(x[4]) for w, x in zip(weights, rows))/math.fsum(weights)
        error = math.sqrt(1/math.fsum(weights))
        fitted = math.exp(mean)
        chi2 = math.fsum(w*(math.log(x[4])-mean)**2 for w, x in zip(weights, rows))
        print(f'[log-fit {generator}] NF={fitted:.9g} '
              f'-{fitted*(1-math.exp(-error)):.9g}/+{fitted*(math.exp(error)-1):.9g}; '
              f'chi2/ndf={chi2:.9g}/{len(rows)-1}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', default='/data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13/NIsoMuon')
    parser.add_argument('--era', nargs='+', choices=list(LUMI), default=['2016preVFP', '2016postVFP'])
    parser.add_argument('--mass-range', nargs=2, type=float, default=[11., 80.])
    parser.add_argument('--dy-amc-file', default='NIsoMuon_DYJets_Inclusive.root')
    parser.add_argument('--dy-mg-file', default='NIsoMuon_DYJets_MG_Inclusive.root')
    parser.add_argument('--lumi', nargs='*', default=[], metavar='ERA=FB_INV')
    args = parser.parse_args()
    if not all(math.isfinite(x) for x in args.mass_range) or not args.mass_range[0] < args.mass_range[1]:
        parser.error('Require finite LOW < HIGH')
    if args.mass_range[0] < 11 and args.mass_range[1] > 9:
        parser.error('Choose a window outside the excluded 9--11 GeV output interval')
    luminosities = dict(LUMI)
    for item in args.lumi:
        era, value = item.split('=', 1)
        value = float(value)
        if era not in LUMI or not math.isfinite(value) or value <= 0:
            parser.error(f'Invalid luminosity override: {item}')
        luminosities[era] = value
    import ROOT
    ROOT.gROOT.SetBatch(True)
    results = {era: audit(ROOT, args, era, luminosities[era]) for era in dict.fromkeys(args.era)}
    pre, post = results.get('2016preVFP'), results.get('2016postVFP')
    if pre and post:
        print('\n[post/pre ratios; central values only]')
        for key in ('source_rate', 'nf', 'dy_rate', 'nf_MG', 'dy_rate_MG'):
            ratio = post[key]/pre[key] if pre[key] else float('nan')
            print(f'{key}: {ratio:.9g}')
        print('[identity] dy_rate ratio = nf ratio * source_rate ratio when saved closure passes')
    common_nf_summary(results)


if __name__ == '__main__':
    main()
