#!/usr/bin/env python3
"""Plot blinded sideband-fit exports from higgs_combine/NIsoMuon.

This reads JSON only: it never opens OS data or performs another fit. Each era
has its own resolution-based SR; a combined SR-yield comparison is also drawn
when several masses are supplied. The original plotter.py remains unchanged.
"""
import argparse
import json
from pathlib import Path

import numpy as np

# Match plotter.py bottom-to-top order and ROOT palette. Exact colours are read
# from plotter.py when PyROOT is available; these RGB fallbacks use the same ROOT
# palette indices for plotting exports outside CMSSW.
ORDER = ('Others', 'ST', 'tt', 'QCD', 'DY')
LABELS = {'DY': 'DY (light-jet data)', 'QCD': 'QCD (SS data)',
          'tt': r'$t\bar{t}$', 'ST': 'single top', 'Others': 'Others'}
COLOURS = {'DY': '#999999', 'QCD': '#99ccff', 'tt': '#ffcc66',
           'ST': '#cc99ff', 'Others': '#ccff99'}


def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 16, 'axes.labelsize': 21,
                         'xtick.labelsize': 16, 'ytick.labelsize': 16,
                         'legend.fontsize': 12, 'axes.linewidth': 1.2,
                         'xtick.direction': 'in', 'ytick.direction': 'in'})
    colours, order = dict(COLOURS), ORDER
    try:
        import ROOT
        import plotter
        ROOT.gROOT.SetBatch(True)
        for name, index in plotter.process_fill_color(ROOT).items():
            colour = ROOT.gROOT.GetColor(index)
            if colour:
                colours[name] = (colour.GetRed(), colour.GetGreen(), colour.GetBlue())
        order = plotter.B_TAG_STACK_DRAW_ORDER
    except ImportError:
        pass
    return plt, colours, order


def read_result(path):
    model = json.loads(Path(path).read_text())
    if model.get('schema_version') != 1 or model.get('blind') is not True or model.get('observation') != 'nominal_background_Asimov':
        raise ValueError(f'{path}: not a supported blinded sideband export')
    if model.get('fit', {}).get('status') != 0 or model.get('fit', {}).get('covQual') != 3:
        raise ValueError(f'{path}: no validated sideband fit')
    if not model.get('rows') or any('postfit' not in r for r in model['rows']):
        raise ValueError(f'{path}: missing postfit yields')
    return model


def plot_era(model, era, destination, plt, colours, order):
    rows = sorted((r for r in model['rows'] if r['era'] == era), key=lambda r: r['low'])
    fig, (ax, ratio) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [2.1, 1], 'hspace': .04})
    seen = set()
    for row in rows:
        lo, hi = row['low'], row['high']
        width, centre = hi-lo, (hi+lo)/2
        # Unequal bins (especially the wide SR) must be displayed as a density.
        bottom = 0.
        for process in order:
            value = row['postfit'][process]['value']/width
            ax.bar(centre, value, width=width, bottom=bottom, color=colours[process],
                   edgecolor='none', label=LABELS[process] if process not in seen else None)
            bottom += value
        nominal = row['asimov']/width
        post = row['postfit']['total_background']
        total, lower, upper = (post[k]/width for k in ('value', 'low', 'high'))
        ax.fill_between([lo, hi], lower, upper, color='#c62828', alpha=.25,
                        label='Sideband postfit 16–84%' if not seen else None)
        ax.plot([lo, hi], [total]*2, color='#c62828', lw=1.8,
                label='Sideband postfit' if not seen else None)
        ax.plot([lo, hi], [nominal]*2, color='black', lw=1.4, ls='--',
                label='Nominal background' if not seen else None)
        ax.errorbar(centre, nominal, yerr=np.sqrt(row['asimov'])/width,
                    fmt='o', ms=3.5, color='black',
                    label='Bkg Asimov (SR excluded)' if not seen else None)
        if nominal > 0:
            ratio.fill_between([lo, hi], lower/nominal, upper/nominal, color='#c62828', alpha=.25)
            ratio.plot([lo, hi], [total/nominal]*2, color='#c62828', lw=1.8)
            ratio.errorbar(centre, 1., yerr=1/np.sqrt(row['asimov']), fmt='o', ms=3.5, color='black')
        if row['region'] == 'SR':
            for panel in (ax, ratio):
                panel.axvspan(lo, hi, facecolor='0.9', alpha=.25, zorder=-1)
                panel.axvline(lo, color='0.45', ls=':', lw=1)
                panel.axvline(hi, color='0.45', ls=':', lw=1)
            ax.text(centre, .03, 'SR: excluded from fit', transform=ax.get_xaxis_transform(),
                    ha='center', rotation=90, va='bottom', fontsize=10)
        seen.update(order)
    ratio.axhline(1, color='black', lw=1, ls='--')
    ax.set_yscale('log')
    positive = [r['asimov']/(r['high']-r['low']) for r in rows if r['asimov'] > 0]
    ax.set_ylim(max(min(positive)*.15, 1e-5), max(positive)*25)
    ax.set_ylabel('Events / GeV')
    ratio.set_ylabel('Pred. / Asimov')
    ratio.set_xlabel(r'Dimuon mass [GeV]')
    # Preserve excursions; do not silently crop a poorly constrained band.
    bounds = [r['postfit']['total_background'][k]/r['asimov'] for r in rows for k in ('low', 'high')]
    ratio.set_ylim(min(0.5, min(bounds)*.9), max(1.5, max(bounds)*1.1))
    ax.text(0, 1.03, 'CMS', weight='bold', transform=ax.transAxes, fontsize=23)
    ax.text(.13, 1.03, 'Simulation', style='italic', transform=ax.transAxes, fontsize=17)
    ax.text(1, 1.03, f'{era}, '+r'$m(Z\prime)$'+f' = {model["mass"]:g} GeV',
            transform=ax.transAxes, ha='right', fontsize=15)
    if model['fit'].get('boundary_sensitive'):
        ax.text(.99, .02, 'Boundary-sensitive covariance band', transform=ax.transAxes,
                ha='right', color='#b71c1c', fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    entries = dict(zip(labels, handles))
    wanted = ['Bkg Asimov (SR excluded)', 'Nominal background', 'Sideband postfit',
              'Sideband postfit 16–84%', *[LABELS[p] for p in reversed(order)]]
    ax.legend([entries[k] for k in wanted], wanted, ncol=2, loc='upper right',
              frameon=True, facecolor='white', edgecolor='none', framealpha=.95)
    fig.subplots_adjust(left=.13, right=.97, bottom=.11, top=.91)
    stem = destination/f'sideband_fit_{model["target"]}_{era}_M{model["mass"]:g}'
    fig.savefig(str(stem)+'.pdf')
    fig.savefig(str(stem)+'.png', dpi=160)
    plt.close(fig)


def plot_combined_sr(models, destination, plt):
    models = sorted(models, key=lambda m: m['mass'])
    if len({m['mass'] for m in models}) != len(models):
        raise ValueError('Repeated masses: plot each sideband configuration separately.')
    keys = ['exclude_nsigma', 'sr_nsigma', 'outer_nsigma', 'bin_width', 'norm_scope']
    if any(m['target'] != models[0]['target'] or any(m['settings'][k] != models[0]['settings'][k] for k in keys) for m in models):
        raise ValueError('Do not mix sideband configurations in one mass scan.')
    x = np.array([m['mass'] for m in models])
    data = np.array([m['combined_sr']['asimov'] for m in models])
    central, low, high = [np.array([m['combined_sr']['total_background'][k] for m in models]) for k in ('value', 'low', 'high')]
    fig, (ax, ratio) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [2.1, 1], 'hspace': .04})
    ax.errorbar(x, data, yerr=np.sqrt(data), fmt='o', color='black', ms=4, label='Bkg Asimov (excluded from fit)')
    ax.plot(x, data, 'k--', label='Nominal background')
    ax.plot(x, central, color='#c62828', label='Sideband postfit prediction')
    ax.fill_between(x, low, high, color='#c62828', alpha=.25, label='Postfit 16–84%')
    ax.vlines(x, low, high, color='#c62828', alpha=.4, lw=3)
    ratio.errorbar(x, np.ones(len(x)), yerr=1/np.sqrt(data), fmt='o', color='black', ms=4)
    ratio.plot(x, central/data, color='#c62828')
    ratio.fill_between(x, low/data, high/data, color='#c62828', alpha=.25)
    ratio.vlines(x, low/data, high/data, color='#c62828', alpha=.4, lw=3)
    ratio.axhline(1, color='black', ls='--', lw=1)
    ax.set_yscale('log')
    ax.set_ylabel('Events in signal window')
    ratio.set_xlabel(r'$m(Z\prime)$ [GeV]')
    ratio.set_ylabel('Pred. / Asimov')
    ax.text(0, 1.03, 'CMS Simulation', transform=ax.transAxes, fontsize=21)
    ax.text(1, 1.03, models[0]['target'], transform=ax.transAxes, ha='right')
    ax.legend(frameon=False, loc='best')
    ax.text(.02, .03, 'Separate fits at each mass; overlapping windows', transform=ax.transAxes, fontsize=11)
    if any(m['fit'].get('boundary_sensitive') for m in models):
        ax.text(.02, .08, 'Includes boundary-sensitive covariance bands', transform=ax.transAxes, fontsize=11, color='#b71c1c')
    fig.subplots_adjust(left=.13, right=.97, bottom=.11, top=.91)
    stem = destination/f'sideband_fit_{models[0]["target"]}_mass_scan'
    fig.savefig(str(stem)+'.pdf')
    fig.savefig(str(stem)+'.png', dpi=160)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', nargs='+', help='sideband_fit_TARGET_M*.json exports from one configuration')
    p.add_argument('--output-dir', default='plots/sideband_fit')
    args = p.parse_args(argv)
    models = [read_result(path) for path in args.results]
    # Keep 5/7/10-sigma scans separate even with the default plot directory.
    settings = models[0]['settings']
    outer = 'full' if settings['outer_nsigma'] is None else f'outer{settings["outer_nsigma"]:g}'
    destination = Path(args.output_dir)/f'exclude{settings["exclude_nsigma"]:g}sigma_{outer}_bin{settings["bin_width"]:g}_{settings["norm_scope"]}'
    destination.mkdir(parents=True, exist_ok=True)
    plt, colours, order = style()
    plot_combined_sr(models, destination, plt)
    for model in models:
        for era in model['eras']:
            plot_era(model, era, destination, plt, colours, order)
    print('Saved blinded sideband plots under', destination)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
