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
import qcd_ss_fit_range_comparison as ss


class RatioFitsTest(unittest.TestCase):
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

    def test_full_alias_parser_and_production_nominal(self):
        args = ss.build_parser().parse_args(["--era", "full"])
        self.assertEqual(args.era, "full")
        self.assertEqual(args.ss_min_effective_count, 10.)
        self.assertEqual(ss.qcd.SS_NOMINAL_MODEL, "power_exp_logistic")
        self.assertEqual(len(ss.qcd.SS_SHAPE_ALTERNATIVES), 5)

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


if __name__ == "__main__":
    unittest.main()
