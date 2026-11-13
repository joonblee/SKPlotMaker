"""Signal-window stability diagnostics, independent of production templates.

Resolution coefficients are copied from higgs_combine/NIsoMuon/sigfit_inputs
at commit 18711487868b6d290e01f49b3a265438c875dbdb. Override the map explicitly
when updating that analysis input. Integrals are SS yields before OS transfer.
"""
import csv
import json
import math
from pathlib import Path

DEFAULT_MASSES = (12., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60., 65., 70.)
DEFAULT_MAP = Path(__file__).with_name('data')/'signal_resolution_coefficients.json'


def parse_masses(text):
    values = [float(x.strip()) for x in text.split(',')]
    if not values or any(not math.isfinite(x) or not 12 <= x <= 70 for x in values) or len(set(values)) != len(values):
        raise ValueError('Use distinct signal masses in 12--70 GeV')
    return sorted(values)


def window_edges(era, mass, coefficients, native_edges, nsigma=5.):
    c = coefficients[era]
    sigma = mass*(c['a']*mass+c['b'])
    low, high = max(11., mass-nsigma*sigma), min(80., mass+nsigma*sigma)
    if not sigma > 0 or not low < high:
        raise ValueError('Invalid signal resolution/window')
    # Match full_spectrum_fit.window_bins: whole native bins, snapped outwards.
    left = max(x for x in native_edges if x <= low+1e-8 and x >= 11.-1e-9)
    right = min(x for x in native_edges if x >= high-1e-8 and x <= 80.+1e-9)
    return low, high, left, right


def integrate_functions(functions, low, high, step=.02):
    """Trapezoidal pointwise envelope; not maximum of integrated functions.

    Pure scalar maths avoids BLAS. Refine the step with --window-integration-step
    for a numerical convergence check; no zero clipping is applied.
    """
    count = max(1, math.ceil((high-low)/step))
    width = (high-low)/count
    totals = [0.]*len(functions)
    delta = 0.
    for i in range(count+1):
        x = low+i*width
        y = [float(f.Eval(x)) for f in functions]
        if not all(math.isfinite(v) for v in y):
            raise ValueError('Nonfinite extrapolated signal-window prediction')
        weight = .5 if i in (0, count) else 1.
        for j, value in enumerate(y):
            totals[j] += weight*value*width
        delta += weight*max(abs(v-y[0]) for v in y[1:])*width
    return totals[0], delta, totals


def window_report(functions, eras, native_edges, args):
    coefficients = json.loads(Path(args.resolution_map).read_text())
    result = []
    for era in eras:
        for mass in args.window_masses:
            low, high, left, right = window_edges(era, mass, coefficients, native_edges, args.window_nsigma)
            nominal, uncertainty, models = integrate_functions(functions, left, right, args.window_integration_step)
            result.append(dict(window_era=era, mass=mass, low=left, high=right,
                               analytic_low=low, analytic_high=high, nominal=nominal,
                               envelope=uncertainty, envelope_low=nominal-uncertainty,
                               envelope_high=nominal+uncertainty, model_integrals=models,
                               relative_envelope=uncertainty/nominal if nominal > 0 else None))
    return result


def write_products(ROOT, args, reports, stem):
    """CSV exact yields plus one stability plot per era's window definition."""
    if not all('signal_windows' in r for r in reports if r.get('valid')):
        return
    reference = next(r for r in reports if r['range'] == [5., 30.])
    lookup = {(w['window_era'], w['mass']): w for w in reference.get('signal_windows', [])}
    rows = []
    for report in reports:
        for window in report.get('signal_windows', []):
            row = {k:v for k,v in window.items() if k != 'model_integrals'}
            nominal0 = lookup[(row['window_era'], row['mass'])]['nominal']
            envelope0 = lookup[(row['window_era'], row['mass'])]['envelope']
            row.update(fit_upper=report['range'][1], spectrum_period=args.era,
                       nominal_over_5to30=row['nominal']/nominal0 if nominal0 > 0 else None,
                       envelope_over_5to30=row['envelope']/envelope0 if envelope0 > 0 else None,
                       yield_definition='SS before OS transfer; combined spectrum integrated in the named era window')
            rows.append(row)
    if not rows:
        return
    csv_path = Path(str(stem)+'_signal_windows.csv')
    with csv_path.open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print('[SAVED]', csv_path)
    colours = {30: '#2166ac', 20: '#d95f02', 40: '#762a83', 50: '#1b9e77'}
    for era in sorted({r['window_era'] for r in rows}):
        canvas = ROOT.TCanvas('window_stability_'+era, '', 1000, 900)
        keep = []
        legend = ROOT.TLegend(.57, .72, .93, .92)
        legend.SetFillStyle(0); legend.SetBorderSize(0); legend.SetTextSize(.035)
        for panel in (1, 2):
            pad = ROOT.TPad(f'window_{era}_{panel}', '', 0., .50 if panel == 1 else 0., 1., 1. if panel == 1 else .50)
            pad.SetLeftMargin(.13); pad.SetRightMargin(.04)
            pad.SetBottomMargin(.16); pad.SetTopMargin(.08)
            pad.Draw(); pad.cd(); keep.append(pad)
            graphs = []
            for report in reports:
                subset = sorted([r for r in rows if r['window_era'] == era and r['fit_upper'] == report['range'][1]], key=lambda r:r['mass'])
                pairs = [(r['mass'], r['nominal_over_5to30'] if panel == 1 else r['relative_envelope']) for r in subset]
                pairs = [(x,y) for x,y in pairs if y is not None and math.isfinite(y) and (panel == 1 or y > 0)]
                graph = ROOT.TGraph(len(pairs))
                for i,(x,y) in enumerate(pairs): graph.SetPoint(i,x,y)
                graph.SetLineColor(ROOT.TColor.GetColor(colours[report['range'][1]]))
                graph.SetMarkerColor(graph.GetLineColor()); graph.SetMarkerStyle(20)
                graph.SetLineWidth(2); graphs.append(graph)
                if panel == 1: legend.AddEntry(graph, f"5--{report['range'][1]:g} GeV fit", 'lp')
            values = [g.GetPointY(i) for g in graphs for i in range(g.GetN())]
            if not values: canvas.cd(); continue
            if panel == 2:
                pad.SetLogy(); ymin, ymax = max(1e-6, min(values)*.5), max(values)*3.
            else:
                ymin, ymax = min(0., min(values)*.9), max(1.5,max(values)*1.2)
            frame = pad.DrawFrame(11., ymin, 71., ymax)
            frame.GetXaxis().SetTitle('Signal mass [GeV]')
            frame.GetYaxis().SetTitle('Nominal / 5--30 nominal' if panel == 1 else 'Functional envelope / nominal')
            for axis in (frame.GetXaxis(),frame.GetYaxis()):
                axis.SetTitleFont(43); axis.SetTitleSize(24); axis.SetLabelFont(43); axis.SetLabelSize(21)
            frame.GetYaxis().SetTitleOffset(2.1)
            for graph in graphs: graph.Draw('LP SAME')
            if panel == 1:
                legend.Draw(); label=ROOT.TLatex(); label.SetNDC(); label.SetTextSize(.033)
                label.DrawLatex(.15,.93,f'{args.era} SS fit; {era} window'); keep.append(label)
            keep.extend([frame,*graphs]); canvas.cd()
        for ext in ('pdf','png'):
            path=Path(str(stem)+f'_signal_windows_{era}.{ext}'); canvas.SaveAs(str(path)); print('[SAVED]',path)
        canvas.Close()
