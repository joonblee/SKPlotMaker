"""Check signed-bin fits, blind/veto boundaries and symmetric envelopes."""
import math
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ratio_fits as r
import dy_mc_shape_comparison as dy
import qcd_ss_fit_range_comparison as ss

try:
    import ROOT
except ImportError:
    ROOT = None


class RatioFitsTest(unittest.TestCase):
    def test_dy_reference_defaults_reach_plot_and_matching(self):
        expected = [11., 13., 15., 17., 19., 22., 26., 40., 70.]
        self.assertEqual(dy.DEFAULT_MASS_BINS, expected)
        with patch.object(dy, 'import_root'), \
             patch.object(dy, 'load_dy_pair', return_value=('b', 'l')), \
             patch.object(dy, 'match_lightjet', return_value=('scaled', 1.)) as match, \
             patch.object(dy, 'rebin_hist'), patch.object(dy, 'integral_and_error'), \
             patch.object(dy, 'scale_to_yield_per_gev'), patch.object(dy, 'draw') as draw:
            self.assertEqual(dy.main([]), 0)
            args = draw.call_args.args[1]
            self.assertEqual((args.xmin, args.xmax), (11., 70.))
            self.assertEqual(r.fit_range(args), (11., 70.))
            self.assertEqual(args.normalisation_range, [11., 70.])
            self.assertEqual(match.call_args.args[-2:], (11., 70.))

    def test_linear_solution_and_known_error_covariance(self):
        points = [(x, .5, 2*x-5, 1.) for x in (0., 1., 2., 3.)]
        fit = r.fit_points(points, "linear")
        self.assertEqual(fit["status"], "ok")
        self.assertEqual(fit["ndf"], 2)
        self.assertAlmostEqual(fit["parameters"]["a"], 2.)
        self.assertAlmostEqual(fit["parameters"]["b"], -5.)
        self.assertAlmostEqual(fit["errors"]["a"], math.sqrt(.2))
        self.assertAlmostEqual(fit["errors"]["b"], math.sqrt(.7))
        self.assertAlmostEqual(fit["cov_a_b"], -.3)
        self.assertAlmostEqual(fit["chi2"], 0.)

    def test_weighted_constant_retains_negative_ratio(self):
        fit = r.fit_points([(1., .5, -2., 1.), (2., .5, 2., 2.)], "const")
        self.assertAlmostEqual(fit["parameters"]["c"], -1.2)
        self.assertAlmostEqual(fit["errors"]["c"], math.sqrt(.8))
        self.assertAlmostEqual(fit["chi2"], 3.2)

    def test_whole_bins_blinding_and_veto(self):
        points = [(7., 2., -1., 1.), (10., 1., 8., 1.),
                  (11., 1., 9., 1.), (20., 9., 100., 1.), (81., 1., 2., 1.)]
        chosen = r.selected_points(points, 5., 82., ((9., 11.), (11., 80.)))
        self.assertEqual([p[0] for p in chosen], [7., 81.])
        self.assertEqual(r.selected_points(points, 6., 8.), [])

    def test_unusable_ratio_points_not_fitted(self):
        points = [(1., .5, -999., 0.), (2., .5, math.nan, 1.),
                  (3., .5, 1., math.inf), (4., .5, -1., 2.)]
        self.assertEqual(r.selected_points(points, 0., 5.), [points[-1]])
        self.assertEqual(r.fit_points([points[-1]], "const")["status"], "insufficient bins")

    def test_fit_range_validation_and_disabled_fits(self):
        args = SimpleNamespace(xmin=5., xmax=80., ratio_fit_range=(11., 80.), ratio_fits="none")
        self.assertEqual(r.analyse([], args)["fits"], [])
        args.ratio_fit_range = (0., 80.)
        with self.assertRaises(ValueError):
            r.fit_range(args)

    def test_default_fit_domains_exclude_low_or_high_mass_outliers(self):
        points = [(8., .5, 100., 1.), (12., .5, 1., 1.),
                  (20., .5, 1., 1.), (69., .5, 1., 1.), (79., 1., 100., 1.)]
        args = SimpleNamespace(xmin=5., xmax=80., ratio_fit_range=None,
                               ratio_fit_default_range=(11., 70.), ratio_fits="both")
        report = r.analyse(points, args)
        self.assertEqual(report["fits"][0]["parameters"]["c"], 1.)
        self.assertEqual(report["fits"][0]["n_points"], 3)
        self.assertEqual(r.fit_range(args), (11., 70.))
        args.xmin = 15.
        self.assertEqual(r.fit_range(args), (15., 70.))
        args.ratio_fit_range = (20., 80.)
        self.assertEqual(r.fit_range(args), (20., 80.))


class SSComparisonTest(unittest.TestCase):
    def test_all_periods_includes_run2_and_run3_and_forwards_options(self):
        with patch.object(ss.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run, \
             patch('builtins.print'):
            self.assertEqual(ss.main(['--all-periods', '--base-dir', '/example',
                                      '--fit-max-attempts', '4']), 0)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual([command[-1] for command in commands], list(ss.ALL_PERIODS))
        self.assertEqual(len(commands), 11)
        self.assertEqual([command[-1] for command in commands][-3:], ['Run2', 'Run3', 'Run2+3'])
        for command in commands:
            self.assertNotIn('--all-periods', command)
            self.assertIn('/example', command)
            self.assertIn('4', command)

    def test_all_periods_continues_after_failure_and_returns_nonzero(self):
        results = [SimpleNamespace(returncode=1)] + [SimpleNamespace(returncode=0)]*10
        with patch.object(ss.subprocess, 'run', side_effect=results) as run, patch('builtins.print'):
            self.assertEqual(ss.main(['--all-periods']), 1)
        self.assertEqual(run.call_count, 11)

    @staticmethod
    def reports():
        return [dict(range=[5., high], valid=True, invalid_models=[],
                     models={key: dict(accepted=True, metrics=dict(stat_chi2=2., stat_ndf=3))
                             for key in ss.qcd.SS_REQUIRED_ROOT_MODELS},
                     x=[5., 80.], central=[1., .1], low=[-1., -.2], high=[3., .4])
                for high in ss.RANGES]

    def test_envelope_symmetric_even_when_negative(self):
        low, high = ss.envelope(1., [.1, 4., 2., .7, 1.])
        self.assertEqual((low, high), (-2., 4.))
        with self.assertRaises(ValueError):
            ss.envelope(1., [math.inf])

    def test_ratios_use_fixed_reference_and_bin_averages_without_mutation(self):
        reports = self.reports()
        reports[0]["bin_averages"] = [dict(low=11., high=13., central=.5)]
        reports[1].update(central=[2., .2], low=[1.5, .15], high=[2.5, .25])
        points = [dict(x=12., half_width=1., value=-1., error=.25)]
        original = copy.deepcopy((reports, points))
        curves, data = ss.ratio_to_reference(points, reports)
        self.assertEqual(curves[0]["central"], [1., 1.])
        self.assertEqual(curves[1]["central"], [2., 2.])
        for value in curves[1]["low"]:
            self.assertAlmostEqual(value, 1.5)
        for value in curves[1]["high"]:
            self.assertAlmostEqual(value, 2.5)
        self.assertEqual(data[0]["value"], -2.)
        self.assertEqual(data[0]["error"], .5)
        self.assertEqual((reports, points), original)

    def test_ratio_reference_zero_is_omitted_and_grids_must_match(self):
        reports = self.reports()
        reports[0]["central"][1] = 0.
        reports[0]["bin_averages"] = [dict(low=11., high=13., central=0.)]
        points = [dict(x=12., half_width=1., value=1., error=.25)]
        curves, data = ss.ratio_to_reference(points, reports)
        self.assertEqual(curves[0]["x"], [5.])
        self.assertEqual(data, [])
        reports[1]["x"] = [5., 70.]
        with self.assertRaises(ValueError):
            ss.ratio_to_reference(points, reports)

    def test_full_alias_parser_and_production_nominal(self):
        args = ss.build_parser().parse_args(["--era", "full"])
        self.assertEqual(args.era, "full")
        self.assertEqual(args.ss_min_effective_count, 10.)
        self.assertEqual((args.xmax, args.ymin, args.logx), (80., .001, True))
        self.assertEqual(ss.qcd.SS_NOMINAL_MODEL, "power_exp_logistic")
        self.assertEqual(len(ss.qcd.SS_SHAPE_ALTERNATIVES), 5)
        self.assertEqual(args.ratio_scale, 'linear')
        self.assertEqual(ss.ratio_display_max(args, [], []), 2.)
        self.assertTrue(args.ratio_inset)

    def test_hybrid_axis_is_continuous_monotonic_and_keeps_linear_core(self):
        values = [-3., 0., .5, 1., 1.5, 2., 5., 10., 1000.]
        transformed = [ss.ratio_display_value(x, 'hybrid', 1000.) for x in values]
        self.assertEqual(transformed, sorted(transformed))
        self.assertEqual(transformed[:5], values[:5])
        self.assertEqual(transformed[-1], 2.5)
        self.assertAlmostEqual(ss.ratio_display_value(1.5+1e-9, 'hybrid', 1000.), 1.5)
        self.assertEqual(ss.ratio_display_value(1000., 'linear', 1000.), 1000.)
        y = ss.ratio_display_value(2., 'hybrid', 1000.)
        low = y-ss.ratio_display_value(1., 'hybrid', 1000.)
        high = ss.ratio_display_value(3., 'hybrid', 1000.)-y
        self.assertGreater(low, high)  # Physical symmetric errors display asymmetrically.

    def test_hybrid_auto_range_covers_visible_high_band_and_data_errors(self):
        args = ss.build_parser().parse_args(['--ratio-scale', 'hybrid', '--xmax', '50'])
        reports = [dict(x=[5., 40., 80.], high=[1., 18., 1e6])]
        self.assertEqual(ss.ratio_display_max(args, reports, []), 20.)
        points = [dict(x=45., value=20., error=7.)]
        self.assertEqual(ss.ratio_display_max(args, reports, points), 50.)
        args.ratio_max = 10.
        self.assertEqual(ss.ratio_display_max(args, reports, points), 10.)
        args.ratio_max = None
        reports[0]['high'][1] = 1.7e308
        self.assertEqual(ss.ratio_display_max(args, reports, points), 1.7e308)

    def test_signed_log_preserves_sign_and_cross_zero_error_endpoints(self):
        values = [-100., -1., -.01, 0., .01, 1., 100.]
        transformed = [ss.signed_log(x, .1) for x in values]
        self.assertEqual(sorted(transformed), transformed)
        self.assertEqual(transformed[3], 0.)
        self.assertAlmostEqual(ss.signed_log(-1., .1), -ss.signed_log(1., .1))
        self.assertTrue(math.isfinite(ss.signed_log(1.e308, 1.e-308)))
        value, error = -.02, .1
        y = ss.signed_log(value, .1)
        self.assertLess(ss.signed_log(value-error, .1), y)
        self.assertGreater(ss.signed_log(value+error, .1), 0.)

    def test_invalid_saved_bands_are_rejected(self):
        points = [dict(x=20., half_width=1., value=-1., error=2.)]
        reports = self.reports()
        ss.validate_plot_data(points, reports)
        bad = copy.deepcopy(reports)
        bad[0]["valid"] = False
        with self.assertRaises(ValueError):
            ss.validate_plot_data(points, bad)
        bad = copy.deepcopy(reports)
        bad[0]["high"][1] = float("nan")
        with self.assertRaises(ValueError):
            ss.validate_plot_data(points, bad)

    def test_plot_json_never_refits_or_changes_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'saved.json'
            payload = dict(era='2022EE', points=[dict(x=20., half_width=1., value=-1., error=2.)],
                           fits=self.reports())
            original = json.dumps(payload).encode()
            path.write_bytes(original)
            with patch.object(ss.qcd, 'import_root'), patch.object(ss, 'plot') as plot, \
                 patch.object(ss.qcd, 'build_input_histogram') as load, \
                 patch.object(ss.qcd, 'declare_fit_functions') as declare, \
                 patch.object(ss, 'fit_ranges') as fit:
                self.assertEqual(ss.main(['--plot-json', str(path)]), 0)
                self.assertEqual(plot.call_args.args[0].era, '2022EE')
                self.assertEqual(plot.call_args.args[3], path.with_suffix(''))
                load.assert_not_called()
                declare.assert_not_called()
                fit.assert_not_called()
            self.assertEqual(path.read_bytes(), original)


@unittest.skipIf(ROOT is None, 'PyROOT is not installed')
class SSRatioInsetTest(unittest.TestCase):
    def test_log_inset_preserves_signed_inputs_and_clips_only_display_band(self):
        ROOT.gROOT.SetBatch(True)
        reports = [dict(x=[5., 20., 80.], central=[1., 1., 2.],
                        low=[-10., -5., -1.], high=[10., 5., 2000.])]
        points = [dict(x=25., half_width=1., value=-2., error=3.),
                  dict(x=50., half_width=2., value=.5, error=2.)]
        original = copy.deepcopy((reports, points))
        canvas = ROOT.TCanvas('ss_inset_test', '', 600, 600)
        try:
            objects = ss.draw_ratio_inset(ROOT, canvas, reports, points, True)
            inset, frame = objects[:2]
            self.assertEqual(inset.GetLogy(), 1)
            self.assertAlmostEqual(frame.GetXaxis().GetXmin(), 20.)
            self.assertAlmostEqual(frame.GetXaxis().GetXmax(), 80.)
            self.assertAlmostEqual(frame.GetMinimum(), 1.)
            self.assertAlmostEqual(frame.GetMaximum(), 1000.)
            bands = [obj for obj in objects if obj.InheritsFrom('TGraph') and obj.GetN() == 4]
            self.assertEqual(len(bands), 1)
            self.assertEqual([bands[0].GetPointY(i) for i in range(4)], [1., 1., 1000., 5.])
            data = [obj for obj in objects if obj.InheritsFrom('TGraphAsymmErrors')][0]
            self.assertEqual(data.GetN(), 1)
            self.assertEqual(data.GetPointY(0), .5)
            self.assertGreaterEqual(data.GetErrorYlow(0), 0.)
            self.assertEqual((reports, points), original)
        finally:
            canvas.Close()


@unittest.skipIf(ROOT is None, 'PyROOT is not installed')
class SSFixedTailBinningTest(unittest.TestCase):
    def test_fixed_tail_preserves_signed_yields_variances_and_low_mass_grid(self):
        ROOT.TH1.AddDirectory(False)
        source = ROOT.TH1D('ss_fixed_tail_input', '', 1000, 0., 100.)
        source.Sumw2()
        for i in range(1, 1001):
            source.SetBinContent(i, 100. if source.GetBinCenter(i) < 40. else -1.)
            source.SetBinError(i, 2.)
        args = ss.build_parser().parse_args([])
        original = ss.qcd.make_adaptive_ss_binning(
            source, ss.replace(ss.qcd.SS_MODE, fit_max=80.),
            args.ss_min_effective_count, args.ss_max_bin_width, protected_edges=(20., 30., 40.))
        expected_low = sorted(set([round(x, 9) for x in original if x < 40.] + [20., 30.]))
        prepared = ss.common_histograms(ROOT, source, args)
        self.assertEqual([x for x in prepared.edges if x < 40.], expected_low)
        self.assertEqual([x for x in prepared.edges if 40. <= x <= 80.],
                         [40., 45., 50., 60., 70., 80.])
        for left, right in zip(prepared.edges, prepared.edges[1:]):
            if left >= 5. and right <= 80.:
                self.assertGreaterEqual(right-left+1e-9, max(.1, .02*(left+right)/2.))
        for boundary in (20., 30., 40.):
            self.assertIn(boundary, prepared.edges)
        for left, right in zip(ss.FIXED_HIGH_MASS_EDGES, ss.FIXED_HIGH_MASS_EDGES[1:]):
            i = prepared.counts.FindBin((left+right)/2.)
            n_native = round((right-left)/.1)
            self.assertEqual(prepared.counts.GetBinContent(i), -n_native)
            self.assertAlmostEqual(prepared.counts.GetBinError(i)**2, 4.*n_native)
            self.assertAlmostEqual(prepared.density.GetBinContent(i), -10.)
            self.assertAlmostEqual(prepared.density.GetBinError(i),
                                   2.*math.sqrt(n_native)/(right-left))
        original_variance = sum(source.GetBinError(i)**2 for i in range(1, 1001))
        self.assertAlmostEqual(prepared.counts.Integral(), source.Integral())
        self.assertAlmostEqual(sum(prepared.counts.GetBinError(i)**2
                                   for i in range(1, prepared.counts.GetNbinsX()+1)),
                               original_variance)

    def test_fixed_edge_inside_native_bin_is_rejected(self):
        # All requested boundaries except 45 GeV coincide with this native grid.
        source = ROOT.TH1D('ss_fixed_tail_unsplittable', '', 50, 0., 100.)
        with patch.object(ss.qcd, 'make_adaptive_ss_binning', return_value=[0., 5., 80., 100.]), \
             self.assertRaisesRegex(ValueError, '45 GeV is not a native bin edge'):
            ss.common_histograms(ROOT, source, ss.build_parser().parse_args([]))


if __name__ == "__main__":
    unittest.main()
