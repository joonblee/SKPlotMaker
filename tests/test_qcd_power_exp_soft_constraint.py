"""Check the fitted soft objective, its strength, and separation from data chi2."""
from array import array
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qcd_bkg_estimation as q

try:
    import ROOT
except ImportError:
    ROOT = None


class SoftConstraintOptionsTest(unittest.TestCase):
    def resolve(self, mode='qcd-mc', objective='auto', width=None):
        args = q.build_parser().parse_args(['--mode', mode, '--fit-objective', objective])
        args.power_exp_n_soft_width = width
        config = q.canonical_mode(mode)
        return q.resolve_power_exp_n_soft_width(args, config, q.resolve_objective(objective, config))

    def test_scope_and_opt_out_preserve_other_fit_modes(self):
        self.assertEqual(self.resolve(), .2)
        self.assertEqual(self.resolve(width=0.), 0.)
        self.assertEqual(self.resolve(mode='ss-data'), 0.)
        for objective in ('log-chi2', 'weighted-likelihood'):
            self.assertEqual(self.resolve(objective=objective), 0.)
            with self.assertRaises(ValueError):
                self.resolve(objective=objective, width=.2)
        with self.assertRaises(ValueError):
            self.resolve(mode='ss-data', width=.2)
        for width in (-.1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                self.resolve(width=width)


@unittest.skipIf(ROOT is None, 'PyROOT is not installed')
class SoftObjectiveFitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not hasattr(ROOT, 'BkgFitFnVariationPy'):
            q.declare_fit_functions(ROOT)
        q.configure_minimizer(ROOT)

    def fit_toy(self, width):
        model = next(m for m in q.FIT_MODELS if m.key == 'power_exp_erf')
        function = ROOT.BkgFitFnVariationPy.MakeObjectiveFunction(
            q._NAMES.unique('soft_objective_test'), model.cpp_id, 0, 5., 30., 5)
        truth = [10000., .5, .1, 0., 1.]
        low, high, value, error = [ROOT.std.vector('double')() for _ in range(4)]
        for left in (5., 12., 18., 25.):
            right = left+1.
            y = ROOT.BkgFitFnVariationPy.IntegrateDensity(model.cpp_id, left, right, array('d', truth))
            for vec, x in zip((low, high, value, error), (left, right, y, 1000.)):
                vec.push_back(x)
        for i, p in enumerate(truth):
            function.SetParameter(i, p)
        function.SetParLimits(1, 0., 20.)  # Explicitly allow n < 1.
        function.SetParError(1, .05)
        for i in (0, 2, 3, 4):
            function.FixParameter(i, truth[i])  # Includes fixed zero m0.
        result = ROOT.BkgFitFnVariationPy.FitPowerExpSoftChi2(
            function, model.cpp_id, low, high, value, error, width)
        params = array('d', [function.GetParameter(i) for i in range(5)])
        data_chi2 = sum(((ROOT.BkgFitFnVariationPy.IntegrateDensity(model.cpp_id, a, b, params)/(b-a)-y)/e)**2
                        for a, b, y, e in zip(low, high, value, error))
        return function, result, data_chi2

    def test_soft_preference_not_hard_bound_and_penalty_added_once(self):
        function, result, data_chi2 = self.fit_toy(.2)
        n = function.GetParameter(1)
        self.assertGreater(n, .5)
        self.assertLess(n, 1.)  # Data can still pull below the preferred threshold.
        penalty = ((1.-n)/.2)**2
        self.assertAlmostEqual(result.MinFcnValue(), data_chi2+penalty, places=8)
        self.assertEqual(result.Ndf(), 3)  # Four data bins, one free parameter.
        self.assertGreater(result.CovMatrix(1, 1), 0.)
        self.assertEqual(int(result), 0)
        self.assertEqual(function.GetParameter(3), 0.)
        model = next(m for m in q.FIT_MODELS if m.key == 'power_exp_erf')
        fit_data = q.FitData('chi2', None, '', 4, power_exp_n_soft_width=.2)
        diag = q.extract_diagnostics(result, function, model, fit_data, None)
        self.assertAlmostEqual(diag.penalty_value, penalty)
        self.assertAlmostEqual(diag.data_objective_value, data_chi2, places=8)
        self.assertAlmostEqual(q.FitMetrics(stat_chi2=data_chi2, stat_ndf=3).stat_chi2_ndf, data_chi2/3.)

    def test_strength_and_disabled_limit(self):
        weak, _, _ = self.fit_toy(.5)
        strong, _, _ = self.fit_toy(.1)
        off, result, chi2 = self.fit_toy(0.)
        self.assertGreater(strong.GetParameter(1), weak.GetParameter(1))
        self.assertAlmostEqual(off.GetParameter(1), .5, places=4)
        self.assertAlmostEqual(result.MinFcnValue(), chi2, places=8)

    def test_fit_wrapper_ignores_poisoned_veto_and_held_out_bins(self):
        model = next(m for m in q.FIT_MODELS if m.key == 'power_exp_erf')
        edges = [5., 6., 9., 11., 12., 13., 18., 19., 25., 26., 30., 35., 80.]
        hist = ROOT.TH1D(q._NAMES.unique('soft_veto_tail'), '', len(edges)-1, array('d', edges))
        hist.SetDirectory(0)
        truth = [10000., .5, .1, 0., 1.]
        for left in (5., 12., 18., 25.):
            i = hist.FindBin(left+.1)
            y = ROOT.BkgFitFnVariationPy.IntegrateDensity(model.cpp_id, left, left+1., array('d', truth))
            hist.SetBinContent(i, y)
            hist.SetBinError(i, 1000.)
        for x in (10., 32., 60.):
            hist.SetBinContent(hist.FindBin(x), 1e12)
            hist.SetBinError(hist.FindBin(x), 1.)
        function = ROOT.BkgFitFnVariationPy.MakeObjectiveFunction(q._NAMES.unique('soft_wrapper'), model.cpp_id, 0, 5., 30., 5)
        for i, p in enumerate(truth):
            function.SetParameter(i, p)
        function.SetParLimits(1, 0., 20.)
        function.SetParError(1, .05)
        for i in (0, 2, 3, 4):
            function.FixParameter(i, truth[i])
        fit_data = q.FitData('chi2', hist, '', 4, power_exp_n_soft_width=.2)
        result = q.run_objective_fit(ROOT, fit_data, function, model, q.replace(q.QCD_MODE, fit_max=30.), 'chi2')
        reference, reference_result, _ = self.fit_toy(.2)
        self.assertAlmostEqual(function.GetParameter(1), reference.GetParameter(1), places=10)
        self.assertAlmostEqual(result.MinFcnValue(), reference_result.MinFcnValue(), places=10)
        self.assertEqual(tuple(fit_data._soft_chi2_bins[0]), (5., 12., 18., 25.))


if __name__ == '__main__':
    unittest.main()
