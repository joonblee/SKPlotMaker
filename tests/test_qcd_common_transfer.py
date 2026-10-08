"""Physics/statistics contract checks without ROOT; run with unittest discovery."""
import bisect
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qcd_common_transfer as c
import qcd_bkg_estimation as q
import plotter as p

WINDOWS = dict(low=[5., 9.], high=[11., 80.])


def records(group="Run2", ratios=(.8, .8, .8, .8), log_variances=(.04, .04, .04, .04)):
    result = {}
    for era, d, v in zip(c.GROUP_ERAS[group], ratios, log_variances):
        yields = dict(mc_ss_low=100., mc_os_low=300., mc_ss_high=50., mc_os_high=150. * d)
        result[era] = dict(primitive={key: dict(value=y, variance=y*y*v/4) for key, y in yields.items()},
                           source_file=era+".root")
    return result


class Axis:
    edges = [0., 5., 9., 11., 15., 20., 30., 80., 100.]
    def GetXmin(self): return self.edges[0]
    def GetXmax(self): return self.edges[-1]
    def GetBinLowEdge(self, i): return self.edges[i-1]
    def GetBinUpEdge(self, i): return self.edges[i]
    def FindFixBin(self, x): return bisect.bisect_right(self.edges, x)


class Hist:
    def __init__(self, values=None, variances=None):
        self.values = values or [0.] * 8
        self.variances = variances or [abs(x) for x in self.values]
    def GetNbinsX(self): return 8
    def GetXaxis(self): return Axis()
    def GetBinContent(self, i): return self.values[i-1] if 1 <= i <= 8 else 0.
    def GetBinError(self, i): return math.sqrt(self.variances[i-1])


def hist(low, high, low_var=None, high_var=None):
    fractions = [0., 0., 0., .6, .3, .05, .05, 0.]
    values = [high*f for f in fractions]
    values[1] = low
    variances = [(high if high_var is None else high_var)*f for f in fractions]
    variances[1] = low if low_var is None else low_var
    return Hist(values, variances)


FIT = dict(model="power_exp_logistic", coordinates="log(A),shape", usable=True,
           covariance_status=3, parameters=[math.log(1000.), 2., .05, 5., 1.2],
           covariance=[[1e-4 if i == j else 0. for j in range(5)] for i in range(5)])


class Function:
    def Integral(self, low, high):
        # Independent Simpson integration checks the producer's Gauss-Legendre rule.
        n = 2000
        step = (high-low)/n
        a, power, slope, centre, width = FIT["parameters"]
        def f(x):
            return math.exp(a)*x**(-power)*math.exp(-slope*x)/(1+math.exp(-(x-centre)/width))
        return step/3 * math.fsum((1 if i in (0, n) else 4 if i%2 else 2)*f(low+i*step) for i in range(n+1))


def transfer_and_inputs(method="run-common", era="2016postVFP"):
    rec = records()
    def mc_hist(row, sign):
        lo, hi = row[f"mc_{sign}_low"], row[f"mc_{sign}_high"]
        return hist(lo["value"], hi["value"], lo["variance"], hi["variance"])
    # Use the exact sums read by production, including floating-point roundoff.
    histograms = {}
    for year, row in rec.items():
        for sign in ("ss", "os"):
            h = histograms[year, sign] = mc_hist(row["primitive"], sign)
            for name, window in WINDOWS.items():
                y, v = q.diagnostic_count_variance(h, *window)
                row["primitive"][f"mc_{sign}_{name}"] = dict(value=y, variance=v)
    ms, mo = histograms[era, "ss"], histograms[era, "os"]
    common = c.fit_common_double_ratio(rec, "Run2", WINDOWS)
    nf = SimpleNamespace(GetNbinsX=lambda: 1, GetBinContent=lambda i: .2, GetBinError=lambda i: .01)
    transfer = q.diagnostic_transfer_statistics(hist(100., 40.), hist(250., 90.), ms, mo, hist(20., 10.),
                    SimpleNamespace(Get=lambda name: nf), method=method,
                    common_transfer=common if method == "run-common" else None)
    metadata = dict(schema=c.COMMON_STAT_SCHEMA if method == "run-common" else q.QCD_STAT_SCHEMA,
                    treatment=c.COMMON_STAT_TREATMENT if method == "run-common" else q.QCD_STAT_TREATMENT,
                    era=era, template_path="OS/mass", basis=q.QCD_STAT_BASIS,
                    fit=FIT, transfer_statistics=transfer)
    rows = q.build_qcd_stat_basis(metadata, Hist(), Function())
    native = Hist([row[0] for row in rows])
    basis = [Hist([row[i] for row in rows]) for i in range(8)]
    objects = {q.QCD_STAT_PATH: SimpleNamespace(GetString=lambda: json.dumps(metadata)),
               metadata["template_path"]: native,
               **{"QCDStat/"+name: h for name, h in zip(q.QCD_STAT_BASIS, basis)}}
    inputs = p.read_qcd_stat_inputs(SimpleNamespace(Get=lambda name: objects.get(name)), era, "OS/mass")
    return transfer, metadata, inputs


class CommonTransferTests(unittest.TestCase):
    def test_known_common_fit_and_chi2(self):
        report = c.fit_common_double_ratio(records(), "Run2", WINDOWS)
        self.assertAlmostEqual(report["double_ratio"], .8)
        self.assertAlmostEqual(report["log_variance"], .01)
        self.assertEqual(report["pvalue"], 1.)
        self.assertTrue(report["compatible"])
        self.assertAlmostEqual(c.chi2_survival(3., 3), .39162517627108895)
        self.assertAlmostEqual(c.chi2_survival(2., 2), math.exp(-1))
        incompatible = c.fit_common_double_ratio(records(ratios=(.4, .8, 1.6, 3.2),
                                    log_variances=(.001,)*4), "Run2", WINDOWS)
        self.assertFalse(incompatible["compatible"])

    def test_correlated_covariance_and_leave_one_out(self):
        values = records(ratios=(.7, .8, .9, 1.0))
        cov = [[.04 if i == j else .02 for j in range(4)] for i in range(4)]
        report = c.fit_common_double_ratio(values, "Run2", WINDOWS, cov)
        self.assertAlmostEqual(report["log_variance"], .025)
        loo = report["rows"]["2016preVFP"]
        expected = (math.log(.7) - sum(map(math.log, (.8, .9, 1.0)))/3) / math.sqrt(.04+.04*(.5+.5/3)-.04)
        self.assertAlmostEqual(loo["leave_one_out_pull"], expected)
        with self.assertRaises(ValueError):
            c.fit_common_double_ratio(values, "Run2", WINDOWS, [[.04]*4 for _ in range(4)])
        with self.assertRaises(ValueError):
            c.fit_common_double_ratio(values, "Run2", WINDOWS, [[.05 if i == j else 0 for j in range(4)] for i in range(4)])

    def test_missing_era_and_group_separation(self):
        rec = records()
        del rec["2018"]
        with self.assertRaises(ValueError): c.fit_common_double_ratio(rec, "Run2", WINDOWS)
        run3 = c.fit_common_double_ratio(records("Run3", ratios=(1.1,)*4), "Run3", WINDOWS)
        self.assertAlmostEqual(run3["double_ratio"], 1.1)
        self.assertEqual(c.transfer_group("2022EE"), "Run3")

    def test_common_fit_is_not_ratio_of_merged_era_yields(self):
        rec = records()
        for i, row in enumerate(rec.values()):
            ss_low, ss_high, rlow = (1000., 10., 2.) if i < 2 else (10., 1000., 4.)
            yields = dict(mc_ss_low=ss_low, mc_os_low=ss_low*rlow,
                          mc_ss_high=ss_high, mc_os_high=ss_high*rlow*.8)
            row["primitive"] = {key: dict(value=y, variance=y) for key, y in yields.items()}
        report = c.fit_common_double_ratio(rec, "Run2", WINDOWS)
        merged = {key: {field: sum(row["primitive"][key][field] for row in rec.values())
                  for field in ("value", "variance")} for key in c.MC_KEYS}
        self.assertAlmostEqual(report["double_ratio"], .8)
        self.assertGreater(abs(c.mc_double_ratio(merged)[0]-.8), .3)

    def test_producer_plotter_and_legacy_statistics(self):
        for method in ("run-common", "mc-double-ratio", "data-low"):
            t, m, inputs = transfer_and_inputs(method)
            q.validate_qcd_stat_metadata(m)
            self.assertAlmostEqual(p.qcd_transfer_factors(t)[0], 2.5)
            for low, high in ((5., 9.), (11., 15.), (11., 80.), (5., 80.)):
                producer = q.qcd_window_statistics(m, low, high)
                consumer = p.qcd_root_window_statistics(inputs, low, high)
                for key in ("central", "sigma_nf_stat", "sigma_fit_stat", "sigma_stat_bound",
                            "sigma_local_stat_bound", "sigma_mc_transfer_stat"):
                    self.assertTrue(math.isclose(producer[key], consumer[key], rel_tol=1e-10), (method, key))
                if method == "run-common":
                    self.assertAlmostEqual(consumer["sigma_stat_bound"], math.hypot(
                        consumer["sigma_nf_low_stat"]+consumer["sigma_fit_stat"], consumer["sigma_mc_transfer_stat"]))
                else:
                    self.assertAlmostEqual(consumer["sigma_stat_bound"], consumer["sigma_nf_stat"]+consumer["sigma_fit_stat"])
            if method == "data-low": self.assertTrue(all(h.GetBinContent(4) == 0 for h in inputs[2][-1:]))
            hists = {name: hist(20., 30.) for name in ("SS_data", "SS_Top", "SS_Others", "SS_QCD", "OS_QCD",
                     "SS_tt", "SS_ST", "OS_Top", "OS_tt", "OS_ST", "OS_data", "OS_Others", "OS_DY")}
            diag = p.qcd_normalisation_diagnostic_row(inputs, hists, 11., 15., p.Config(blind=False))
            self.assertTrue(math.isclose(diag["ss_fit"], Function().Integral(11., 15.), rel_tol=1e-6))
            hists["OS_data"] = None
            self.assertTrue(p.qcd_normalisation_diagnostic_row(inputs, hists, 11., 15., p.Config(blind=True))["os_data_blinded"])

    def test_corrupted_common_metadata_is_rejected(self):
        _, m, _ = transfer_and_inputs()
        for mutate in (lambda x: x["transfer_statistics"]["common_transfer"].update(double_ratio=.9),
                       lambda x: x["transfer_statistics"]["common_transfer"].update(id="stale"),
                       lambda x: x.update(schema=q.QCD_STAT_SCHEMA),
                       lambda x: x["transfer_statistics"]["common_transfer"]["rows"]["2018"].update(within_2sigma=False),
                       lambda x: x["transfer_statistics"]["windows"].update(low=[6., 9.])):
            bad = copy.deepcopy(m)
            mutate(bad)
            for validator in (q.validate_qcd_stat_metadata, p.validate_qcd_stat_metadata):
                with self.assertRaises(ValueError): validator(bad)

    def test_shared_mc_component_is_not_counted_as_independent(self):
        t, _, _ = transfer_and_inputs()
        component = c.qcd_stat_components(t, 10., 20., 3.)
        other = dict(component)
        combined = c.combine_qcd_stat_components({"2016preVFP": component, "2016postVFP": other})
        self.assertAlmostEqual(combined, math.sqrt(2*component["sigma_local_stat_bound"]**2
                                                  +(2*component["sigma_mc_transfer_stat"])**2))
        with self.assertRaises(ValueError):
            c.combine_qcd_stat_components({"2016preVFP": component,
                                          "2016postVFP": dict(other, common_transfer_id=None)})
        run3 = dict(other, common_transfer_id="run3", common_transfer_group="Run3")
        self.assertAlmostEqual(c.combine_qcd_stat_components({"2016preVFP": component, "2022": run3}),
                               math.sqrt(2)*component["sigma_stat_bound"])

    def test_plotter_combines_actual_template_derivatives_with_common_covariance(self):
        years = ("2016preVFP", "2016postVFP")
        inputs = {era: transfer_and_inputs(era=era)[2] for era in years}
        def root_file(era, mode):
            m, native, basis = inputs[era]
            objects = {q.QCD_STAT_PATH: SimpleNamespace(GetString=lambda: json.dumps(m)), "OS/mass": native,
                       **{"QCDStat/"+name: h for name, h in zip(q.QCD_STAT_BASIS, basis)}}
            return SimpleNamespace(Get=lambda name: objects.get(name), IsZombie=lambda: False, Close=lambda: None)
        root = SimpleNamespace(TFile=SimpleNamespace(Open=root_file))
        cfg = p.Config(variable="dimuon_mass", divide_by_bin_width=False)
        stat = p.Uncertainty(low=[0.]*8, high=[0.]*8)
        by_year = {era: {"QCD": inputs[era][1]} for era in years}
        with patch.object(p, "process_file", side_effect=lambda cfg, era, *args, **kw: era), \
             patch.object(p, "hist_path", return_value="OS/mass"):
            result = p.add_qcd_stat_uncertainty(root, cfg, years, by_year, stat, 1.)
        per_era = {era: p.qcd_root_window_statistics(inputs[era], 11., 15.) for era in years}
        self.assertAlmostEqual(result.low[3], c.combine_qcd_stat_components(per_era))

    def test_validation_only_never_fits_or_writes_templates(self):
        args = q.build_parser().parse_args(["--mode", "ss-data", "--year", "Run2+3", "--validate-qcd-double-ratio"])
        self.assertEqual(args.qcd_transfer_method, "run-common")
        def report(root, arg, group): return c.fit_common_double_ratio(records(group), group, WINDOWS)
        with patch.object(q, "import_root", return_value=object()), \
             patch.object(q, "read_common_qcd_transfer", side_effect=report) as reader, \
             patch.object(q, "write_common_qcd_validation") as writer, \
             patch.object(q, "declare_fit_functions", side_effect=AssertionError("must not fit")), \
             patch.object(q, "save_ss_fit_anchors", side_effect=AssertionError("must not write anchors")), \
             patch.object(q, "write_ss_background_root", side_effect=AssertionError("must not write templates")):
            self.assertEqual(q.run(args), 0)
            self.assertEqual(reader.call_count, 2)
            self.assertEqual(writer.call_count, 2)
        for argv in (["--mode", "ss-data", "--qcd-transfer-validation-pmin", "nan"],
                     ["--mode", "qcd-mc", "--validate-qcd-double-ratio"],
                     ["--mode", "ss-data", "--year", "Run2", "--qcd-transfer-method", "data-low"]):
            with patch.object(q, "import_root", side_effect=AssertionError("must reject before ROOT")):
                with self.assertRaises(ValueError): q.run(q.build_parser().parse_args(argv))


if __name__ == "__main__":
    unittest.main()
