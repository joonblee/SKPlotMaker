import copy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qcd_stat_uncertainty as q
import qcd_bkg_estimation as producer
import plotter


def metadata():
    covariance = [[0.0] * 5 for _ in range(5)]
    covariance[0][0], covariance[2][2] = 0.04, 0.001
    covariance[0][2] = covariance[2][0] = -0.005
    return dict(schema=q.QCD_STAT_SCHEMA, treatment=q.QCD_STAT_TREATMENT,
                era="2018", template_path="OS/template",
                fit=dict(model="power_exp_logistic", coordinates="log(A),shape",
                         reliable=True, covariance_status=3, boundary_parameters=[],
                         parameters=[math.log(10), 0, 0, -1000, 1], covariance=covariance),
                transfer_statistics=dict(complete=True, low_transfer=2.0,
                    mc_double_ratio=1.5, high_transfer=3.0,
                    low_variance=0.04, double_ratio_variance=0.0225))


class QCDStatistics(unittest.TestCase):
    def test_integral_covariance_includes_off_diagonal(self):
        m = metadata()
        result = q.qcd_window_statistics(m, 12, 13, 30)
        g0, gk = 30.0, -375.0
        expected = math.sqrt(g0 ** 2 * .04 + gk ** 2 * .001 + 2 * g0 * gk * -.005)
        self.assertAlmostEqual(result["sigma_fit_stat"], expected)
        self.assertAlmostEqual(result["sigma_nf_stat"], math.sqrt(1.5 ** 2 * 10 ** 2 * .04 + (2 * 10) ** 2 * .0225))
        self.assertAlmostEqual(result["sigma_stat_bound"], result["sigma_nf_stat"] + expected)
        self.assertGreater(result["sigma_stat_bound"], result["stat_quadrature_assuming_independent"])

    def test_integrate_before_propagating_bin_errors(self):
        m = metadata()
        for row in m["fit"]["covariance"]:
            for i in range(5):
                row[i] = 0
        m["fit"]["covariance"][0][0] = .04
        full = q.qcd_window_statistics(m, 12, 14)
        halves = [q.qcd_window_statistics(m, a, b) for a, b in ((12, 13), (13, 14))]
        self.assertAlmostEqual(full["sigma_fit_stat"], sum(h["sigma_fit_stat"] for h in halves))
        self.assertGreater(full["sigma_fit_stat"], math.hypot(*(h["sigma_fit_stat"] for h in halves)))

    def test_mixed_regions_share_transfer_calibration(self):
        m = metadata()
        r = q.qcd_window_statistics(m, 5, 12)
        self.assertAlmostEqual(r["central"], 2 * 40 + 3 * 10)
        self.assertAlmostEqual(r["sigma_nf_stat"], math.sqrt((40 + 1.5 * 10) ** 2 * .04 + (2 * 10) ** 2 * .0225))

    def test_analytic_gradient_matches_parameter_derivatives(self):
        p = [math.log(2649826.918009), 2.4993597, .24571796, 9.713789, 1.6273213]
        x = 12.0
        analytic = q._density_gradient(x, p)
        for i in range(5):
            up, down = p[:], p[:]
            step = 1e-5
            up[i] += step
            down[i] -= step
            numeric = (q._density_gradient(x, up)[0] - q._density_gradient(x, down)[0]) / (2 * step)
            self.assertTrue(math.isclose(analytic[i + 1], numeric, rel_tol=1e-6))

    def test_logged_2018_central_windows_and_nonzero_tail(self):
        m = metadata()
        m["fit"]["parameters"] = [math.log(2649826.918009), 2.4993597, .24571796, 9.713789, 1.6273213]
        t = m["transfer_statistics"]
        t.update(low_transfer=2.80911, high_transfer=1.92385, mc_double_ratio=1.92385 / 2.80911)
        for mass, central in ((12, 103.403700772), (30, .3917296133), (70, 5.953888487e-6)):
            r = q.qcd_window_statistics(m, mass * .99, mass * 1.01)
            self.assertTrue(math.isclose(r["central"], central, rel_tol=1e-5))
            self.assertGreater(r["sigma_fit_stat"], 0)

    def test_missing_stale_or_unreliable_metadata_rejected(self):
        with self.assertRaises(ValueError):
            q.validate_qcd_stat_metadata({})
        m = metadata()
        for mutate in (lambda x: x["fit"].update(covariance_status=2),
                       lambda x: x["fit"].update(boundary_parameters=["k"]),
                       lambda x: x["transfer_statistics"].update(complete=False)):
            bad = copy.deepcopy(m)
            mutate(bad)
            with self.assertRaises(ValueError):
                q.validate_qcd_stat_metadata(bad)
        with self.assertRaises(ValueError):
            q.validate_qcd_stat_metadata(m, "2022")
        with self.assertRaises(ValueError):
            q.qcd_window_statistics(m, 12, 13, nominal=999)

    def test_producer_metadata_matches_retained_covariance(self):
        p = [math.log(1000), 2.5, .24, 9.7, 1.6]
        covariance = [[.01 if i == j else 0 for j in range(5)] for i in range(5)]

        class Function:
            def __init__(self):
                self.values = [math.exp(p[0]), *p[1:]]
            def GetParameter(self, i): return self.values[i]
            def SetParameter(self, i, value): self.values[i] = value
            def Integral(self, a, b, tolerance=None):
                values = [math.log(self.values[0]), *self.values[1:]]
                return q._integral_gradient(values, a, b, .25)[0]

        result = NS(NPar=lambda: 5, Parameter=lambda i: p[i],
                    CovMatrix=lambda i, j: covariance[i][j])
        selected = NS(model=NS(npar=5, key="power_exp_logistic", cpp_id=3),
                      result=result, function=Function(), accepted=True,
                      diagnostics=NS(covariance_status=3, status=0, boundary_parameters=[]))
        root = NS(BkgFitFnVariationPy=NS(MakeFitFunction=lambda *args: Function()))
        args = NS(year="2018", ss_binning="adaptive", ss_min_effective_count=10, ss_max_bin_width=5)
        transfer = metadata()["transfer_statistics"]
        original_parameters = selected.function.values[:]
        built = producer.build_qcd_stat_metadata(root, args, selected, transfer)
        q.validate_qcd_stat_metadata(built, "2018", producer.hist_path(producer.OS_REGION))
        self.assertEqual(built["fit"]["covariance"], covariance)
        self.assertEqual(selected.function.values, original_parameters)
        bad = copy.deepcopy(transfer)
        bad["complete"] = False
        with self.assertRaises(ValueError):
            producer.build_qcd_stat_metadata(root, args, selected, bad)
        selected.diagnostics.covariance_status = 2
        with self.assertRaisesRegex(RuntimeError, "existing ROOT files were not overwritten"):
            producer.build_qcd_stat_metadata(root, args, selected, transfer)

    def test_plot_stat_scaling_and_independent_era_combination(self):
        class Axis:
            def GetXmin(self): return 12.0
            def GetXmax(self): return 15.0
            def FindFixBin(self, x): return 1 if x < 13 else 2
            def GetBinLowEdge(self, i): return [12., 13.][i - 1]
            def GetBinUpEdge(self, i): return [13., 15.][i - 1]
            def GetBinWidth(self, i): return [1., 2.][i - 1]

        hist = NS(GetNbinsX=lambda: 2, GetXaxis=Axis, GetBinContent=lambda i: 60.0)
        files = {}
        for era in ("2017", "2018"):
            m = metadata()
            m["era"] = era
            obj = NS(GetString=lambda m=m: json.dumps(m))
            files[era] = NS(Get=lambda path, obj=obj: obj if path == q.QCD_STAT_PATH else hist,
                            IsZombie=lambda: False, Close=lambda: None)
        root = NS(TFile=NS(Open=lambda filename, mode: files[filename]))
        cfg = NS(variable="dimuon_mass", qcd_method="data-driven", divide_by_bin_width=True)
        by_year = {era: {"QCD": hist} for era in files}
        stat = plotter.Uncertainty(low=[1, 2], high=[1, 2])
        with patch.object(plotter, "process_file", side_effect=lambda cfg, year, *a, **k: year), \
             patch.object(plotter, "variable_spec", return_value=NS(divide_by_bin_width=True)), \
             patch.object(plotter, "base_region", return_value="OS"), \
             patch.object(plotter, "hist_path", return_value="OS/template"):
            result = plotter.add_qcd_stat_uncertainty(root, cfg, list(files), by_year, stat, 2.0)
        for i, (low, high, scale) in enumerate(((12, 13, 2), (13, 15, 1))):
            sigma = q.qcd_window_statistics(metadata(), low, high)["sigma_stat_bound"] * scale
            self.assertAlmostEqual(result.low[i], math.sqrt(stat.low[i] ** 2 + 2 * sigma ** 2))
            self.assertEqual(result.low[i], result.high[i])


if __name__ == "__main__":
    unittest.main()
