"""Check the fitted soft objective, its strength, and separation from data chi2."""
from array import array
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
            self.assertEqual(self.resolve(objective=objective), .2)
            self.assertEqual(self.resolve(objective=objective, width=.1), .1)
            self.assertEqual(self.resolve(objective=objective, width=0.), 0.)
        with self.assertRaises(ValueError):
            self.resolve(mode='ss-data', width=.2)
        for width in (-.1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                self.resolve(width=width)

    def test_plot_quality_is_ordinary_data_chi2_for_every_objective(self):
        selected = SimpleNamespace(metrics=q.FitMetrics(stat_chi2=20., stat_ndf=10, log_chi2=90., log_ndf=10))
        for objective in ('chi2', 'log-chi2', 'weighted-likelihood'):
            self.assertEqual(q.objective_metric_for_plot(selected, objective), '#chi^{2}/ndf = 2.000')

    def test_log_recovery_ranks_total_objective_including_penalty(self):
        def fit(data, penalty):
            diag = q.FitDiagnostics(status=0, covariance_status=3, edm=1e-8,
                                    objective_value=data+penalty, data_objective_value=data,
                                    ndf=10, result_valid=True, finite_parameters=True,
                                    power_exp_n_soft_width=.2, penalty_value=penalty)
            return SimpleNamespace(accepted=True, diagnostics=diag,
                                   metrics=q.FitMetrics(log_chi2=data, log_ndf=10))
        self.assertLess(q.selected_fit_rank(fit(2., 1.), 'log-chi2'),
                        q.selected_fit_rank(fit(1., 10.), 'log-chi2'))


@unittest.skipIf(ROOT is None, 'PyROOT is not installed')
class SoftObjectiveFitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ROOT.TH1.AddDirectory(False)
        if not hasattr(ROOT, 'BkgFitFnVariationPy'):
            q.declare_fit_functions(ROOT)
        ROOT.gInterpreter.ProcessLine('SetErrorHandler(DefaultErrorHandler);')
        q.configure_minimizer(ROOT)

    def fit_toy(self, width, kind=0):
        model = next(m for m in q.FIT_MODELS if m.key == 'power_exp_erf')
        function = ROOT.BkgFitFnVariationPy.MakeObjectiveFunction(
            q._NAMES.unique('soft_objective_test'), model.cpp_id, 0, 5., 30., 5)
        truth = [10. if kind == 1 else 10000., .5, .1, 0., 1.]
        low, high, value, error = [ROOT.std.vector('double')() for _ in range(4)]
        for left in (5., 12., 18., 25.):
            right = left+1.
            y = ROOT.BkgFitFnVariationPy.IntegrateDensity(model.cpp_id, left, right, array('d', truth))
            observed = math.log(y) if kind == 2 else y
            sigma = math.sqrt(3.*y) if kind == 1 else (.5 if kind == 2 else 1000.)
            for vec, x in zip((low, high, value, error), (left, right, observed, sigma)):
                vec.push_back(x)
        for i, p in enumerate(truth):
            function.SetParameter(i, p)
        function.SetParLimits(1, 0., 20.)  # Explicitly allow n < 1.
        function.SetParError(1, .05)
        for i in (0, 2, 3, 4):
            function.FixParameter(i, truth[i])  # Includes fixed zero m0.
        result = ROOT.BkgFitFnVariationPy.FitPowerExpSoftObjective(
            function, model.cpp_id, low, high, value, error, width, kind)
        params = array('d', [function.GetParameter(i) for i in range(5)])
        data_objective = 0.
        for a, b, y, e in zip(low, high, value, error):
            mu = ROOT.BkgFitFnVariationPy.IntegrateDensity(model.cpp_id, a, b, params)
            if kind == 1:
                data_objective += mu-y+y*math.log(y/mu)
            else:
                prediction = math.log(mu/(b-a)) if kind == 2 else mu/(b-a)
                data_objective += ((prediction-y)/e)**2
        return function, result, data_objective

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
        self.assertEqual(tuple(fit_data._soft_objective_bins[0]), (5., 12., 18., 25.))

    def test_log_and_likelihood_objective_scales_and_data_separation(self):
        model = next(m for m in q.FIT_MODELS if m.key == 'power_exp_erf')
        for kind, objective, multiplier in ((2, 'log-chi2', 1.), (1, 'weighted-likelihood', .5)):
            with self.subTest(objective=objective):
                function, result, data = self.fit_toy(.2, kind)
                n = function.GetParameter(1)
                self.assertTrue(.5 < n < 1.)
                penalty = ((1.-n)/.2)**2
                self.assertAlmostEqual(result.MinFcnValue(), data+multiplier*penalty, places=8)
                fit_data = q.FitData(objective, None, '', 4, power_exp_n_soft_width=.2)
                diag = q.extract_diagnostics(result, function, model, fit_data, None)
                self.assertAlmostEqual(diag.penalty_chi2, penalty)
                self.assertAlmostEqual(diag.penalty_value, multiplier*penalty)
                self.assertAlmostEqual(diag.data_objective_value, data, places=8)
                self.assertEqual(result.Ndf(), 3)

    def test_weighted_covariance_includes_unweighted_constraint_once(self):
        function, result, _ = self.fit_toy(.2, 1)
        n = function.GetParameter(1)
        model = next(m for m in q.FIT_MODELS if m.key == 'power_exp_erf')
        truth = [10., .5, .1, 0., 1.]
        def nll(exponent):
            parameters = array('d', [10., exponent, .1, 0., 1.])
            total = 0.
            for a in (5., 12., 18., 25.):
                y = ROOT.BkgFitFnVariationPy.IntegrateDensity(model.cpp_id, a, a+1., array('d', truth))
                mu = ROOT.BkgFitFnVariationPy.IntegrateDensity(model.cpp_id, a, a+1., parameters)
                total += mu-y+y*math.log(y/mu)
            return total
        step = 1e-4
        h_data = (nll(n+step)-2.*nll(n)+nll(n-step))/step**2
        h_constraint = 1./.2**2
        expected = (3.*h_data+h_constraint)/(h_data+h_constraint)**2
        self.assertGreaterEqual(result.CovMatrixStatus(), 2)
        self.assertAlmostEqual(result.CovMatrix(1, 1)/expected, 1., delta=.01)

    def test_weighted_native_parity_including_empty_bins(self):
        model = next(m for m in q.FIT_MODELS if m.key == 'power_exp_erf')
        edges = [5., 6., 7., 8., 9.]
        h = ROOT.TH1D(q._NAMES.unique('weighted_native'), '', 4, array('d', edges))
        h.SetDirectory(0)
        truth = [10., .5, .1, 0., 1.]
        vectors = [ROOT.std.vector('double')() for _ in range(4)]
        for i, (a, b) in enumerate(zip(edges, edges[1:]), 1):
            y = ROOT.BkgFitFnVariationPy.IntegrateDensity(model.cpp_id, a, b, array('d', truth)) if i < 4 else 0.
            error = math.sqrt(3.*y)
            h.SetBinContent(i, y)
            h.SetBinError(i, error)
            for vec, x in zip(vectors, (a, b, y, error if error > 0. else 1.)):
                vec.push_back(x)
        mode = q.replace(q.QCD_MODE, fit_max=9.)
        q.configure_objective_binning(ROOT, q.PreparedHistograms(h, h, edges), mode)
        functions = []
        for _ in range(2):
            f = ROOT.BkgFitFnVariationPy.MakeObjectiveFunction(q._NAMES.unique('weighted_parity'), model.cpp_id, 1, 5., 9., 5)
            for i, p in enumerate(truth): f.SetParameter(i, p)
            f.SetParLimits(1, 0., 20.)
            f.SetParError(1, .05)
            for i in (0, 2, 3, 4): f.FixParameter(i, truth[i])
            functions.append(f)
        native = h.Fit(functions[0], 'S R WL Q 0 N', '', 5., 9.)
        custom = ROOT.BkgFitFnVariationPy.FitPowerExpSoftObjective(functions[1], model.cpp_id, *vectors, 0., 1)
        self.assertAlmostEqual(functions[0].GetParameter(1), functions[1].GetParameter(1), delta=1e-4)
        self.assertAlmostEqual(native.MinFcnValue(), custom.MinFcnValue(), places=6)
        self.assertAlmostEqual(native.CovMatrix(1, 1)/custom.CovMatrix(1, 1), 1., delta=.01)


if __name__ == '__main__':
    unittest.main()
