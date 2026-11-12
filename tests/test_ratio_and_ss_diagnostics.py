"""Check signed-bin fits, blind/veto boundaries and symmetric envelopes."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

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


class SSComparisonTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
