"""Low-range MC fits must never use held-out tail events."""
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qcd_bkg_estimation as q

try:
    import ROOT
except ImportError:
    ROOT = None


class MCRangeTest(unittest.TestCase):
    def resolve(self, mode='qcd-mc', *options):
        args = q.build_parser().parse_args(['--mode', mode, *options])
        return q.mc_range_modes(args, q.canonical_mode(mode))

    def test_defaults_and_explicit_full_range_keep_filenames(self):
        for options in ((), ('--mc-fit-range', '5', '80')):
            fit, plot = self.resolve('qcd-mc', *options)
            self.assertEqual((fit, plot), (q.QCD_MODE, q.QCD_MODE))
            self.assertEqual(q.mc_range_suffix(fit, plot), '')
        self.assertEqual(self.resolve('ss-data'), (q.SS_MODE, q.SS_MODE))

    def test_low_range_and_invalid_options(self):
        fit, plot = self.resolve('qcd-mc', '--mc-fit-range', '5', '30')
        self.assertEqual((fit.fit_min, fit.fit_max, plot.fit_max), (5., 30., 80.))
        self.assertEqual(q.mc_range_suffix(fit, plot), '-Fit5to30-Plot5to80')
        for mode, options in (
            ('ss-data', ['--mc-fit-range', '5', '30']),
            ('qcd-mc', ['--mc-fit-range', '5', '81']),
            ('qcd-mc', ['--mc-fit-range', '5', '10']),
            ('qcd-mc', ['--mc-fit-range', '9', '11']),
            ('qcd-mc', ['--mc-fit-range', 'nan', '30']),
            ('qcd-mc', ['--mc-fit-range', '5', '30', '--mc-plot-range', '11', '80']),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.resolve(mode, *options)


@unittest.skipIf(ROOT is None, 'PyROOT is not installed')
class MCExtrapolationTest(unittest.TestCase):
    def source(self, tail_scale):
        ROOT.TH1.AddDirectory(False)
        hist = ROOT.TH1D(q._NAMES.unique('extrapolation_input'), '', 5000, 0., 100.)
        hist.Sumw2()
        for i in range(1, hist.GetNbinsX()+1):
            x = hist.GetBinCenter(i)
            y = 100.*math.exp(-x/15.)
            if x > 30.:
                y *= tail_scale
            hist.SetBinContent(i, y)
            hist.SetBinError(i, math.sqrt(y))
        return hist

    def test_tail_changes_cannot_change_fitted_bins_or_log_graph(self):
        mode = q.replace(q.QCD_MODE, fit_max=30.)
        for binning in ('adaptive', 'legacy'):
            for factor in (1, 4):
                snapshots = []
                for scale in (1., 1000.):
                    with redirect_stdout(io.StringIO()):
                        prepared = q.prepare_histograms(ROOT, self.source(scale), q.QCD_MODE, factor,
                                                        mc_binning=binning, mc_protected_edges=(30.,))
                    self.assertIn(30., prepared.edges)
                    self.assertTrue(any(30. < x < 80. for x in prepared.edges))
                    self.assertFalse(any(a < 30. < b for a, b in zip(prepared.edges, prepared.edges[1:])))
                    graph = q.make_log_graph(ROOT, prepared.density, mode, .1)
                    snapshots.append([(graph.GetPointX(i), graph.GetPointY(i), graph.GetErrorY(i))
                                      for i in range(graph.GetN())])
                self.assertEqual(snapshots[0], snapshots[1])

    def test_non_native_boundary_rejected(self):
        with self.assertRaisesRegex(ValueError, 'not an input ROOT bin edge'):
            q.prepare_histograms(ROOT, self.source(1.), q.QCD_MODE, 1, mc_protected_edges=(30.001,))

    def test_held_out_report_uses_bin_widths_and_mc_variances(self):
        hist = ROOT.TH1D(q._NAMES.unique('tail_density'), '', 100, 0., 100.)
        for i in range(1, 101):
            hist.SetBinContent(i, 2.)
            hist.SetBinError(i, .5)
        fn = ROOT.TF1(q._NAMES.unique('constant_prediction'), '[0]', 5., 80.)
        fn.SetParameter(0, 3.)
        selected = SimpleNamespace(model=SimpleNamespace(key='constant', npar=1),
                                   accepted=True, diagnostics=SimpleNamespace(status=0, covariance_status=3),
                                   function=fn, average_function=fn)
        args = SimpleNamespace(year='Run2+3', fit_objective='log-chi2')
        with redirect_stdout(io.StringIO()):
            report = q.mc_extrapolation_report(args, q.replace(q.QCD_MODE, fit_max=30.), q.QCD_MODE, hist, [selected])
        result = report['models']['constant']
        self.assertEqual(len(result['held_out_bins']), 50)
        self.assertEqual(result['held_out_mc_yield'], 100.)
        self.assertEqual(result['held_out_prediction'], 150.)
        self.assertEqual(result['held_out_prediction_over_mc'], 1.5)
        self.assertEqual(result['held_out_chi2_per_bin'], 4.)
        self.assertAlmostEqual(result['held_out_mc_error'], math.sqrt(50)*.5)
        json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
