"""Physics/statistics contract checks without ROOT; run with unittest discovery."""
import bisect
from contextlib import ExitStack, redirect_stdout
import copy
import importlib.util
import json
import io
import math
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


class InputAxis(Axis):
    def FindBin(self, x): return self.FindFixBin(x)
    def GetBinWidth(self, i): return self.edges[i]-self.edges[i-1]


class InputHist(Hist):
    """Weighted input histogram sufficient for the real component loader."""
    def GetXaxis(self): return InputAxis()
    def Clone(self, name): return copy.deepcopy(self)
    def SetDirectory(self, directory): pass
    def SetName(self, name): pass
    def GetSumw2N(self): return len(self.variances)
    def Scale(self, factor):
        self.values = [x*factor for x in self.values]
        self.variances = [x*factor**2 for x in self.variances]
    def Add(self, other):
        self.values = [x+y for x, y in zip(self.values, other.values)]
        self.variances = [x+y for x, y in zip(self.variances, other.variances)]
    def Integral(self, first, last): return math.fsum(self.values[first-1:last])


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
    def test_plotter_period_scope_file_selection_and_scaling(self):
        parser = p.make_parser()
        for period in "FGH":
            cfg = p.config_from_args(parser.parse_args(["--era", "2016postVFP", "--data-period", period]))
            self.assertEqual(p.selected_period_lumi_fb(cfg), q.POSTVFP_DATA_LUMI_FB[period])
            self.assertEqual(Path(p.process_file(cfg, cfg.era, "data", -1., "", [])).name,
                             f"Skim_NIsoMuon_SingleMuon_{period}.root")
            self.assertEqual(p.plot_era_tag(cfg), f"2016postVFP_SingleMuon_{period}")
        for argv in (["--era", "2017", "--data-period", "G"],
                     ["--era", "2016postVFP", "--mc-lumi-fb", "16"],
                     ["--era", "2016postVFP", "--data-period", "F", "--period-lumi-fb", "nan"],
                     ["--era", "2016postVFP", "--data-period", "F", "--mc-lumi-fb", "0"],
                     ["--era", "2016postVFP", "--data-period", "G", "--qcd-normalisation-diagnostics"]):
            with patch.object(p, "import_root") as root, self.assertRaises(ValueError):
                p.main(argv)
            root.assert_not_called()
        cfg = p.Config(era="2016postVFP", data_period="G", period_lumi_fb=4., mc_lumi_fb=16.)
        source = InputHist([40.]*8, [9.]*8)
        with patch.object(p, "open_hist", side_effect=lambda *a, **k: source.Clone("loaded")), \
             patch.object(p, "find_signal_file", return_value="/nominal/signal.root"):
            for proc in (*p.BKG_PROCESSES, "data", "sig"):
                for suffix in (("",) if proc == "data" else ("", "Syst_JESUp")):
                    h = p.load_year_hist(None, cfg, cfg.era, proc, 20., [], suffix)
                    self.assertEqual(h.values, [40. if proc == "data" else 10.]*8)
                    self.assertEqual(h.variances, [9. if proc == "data" else 9./16.]*8)
            # Default full-era data/predictions remain unchanged.
            full_cfg = p.Config(era=cfg.era)
            self.assertEqual(p.load_year_hist(None, full_cfg, cfg.era, "tt", -1., []).values, source.values)
        self.assertEqual(source.values, [40.]*8)
        with patch.object(p, "open_hist", return_value=None), self.assertRaises(RuntimeError):
            p.load_year_hist(None, cfg, cfg.era, "data", -1., [])

    def test_plotter_period_mc_anchor_and_xsec_use_period_data(self):
        cfg = p.Config(era="2016postVFP", data_period="G", period_lumi_fb=4., mc_lumi_fb=16.,
                       qcd_method="mc", dy_method="mc", blind=True, divide_by_bin_width=False)
        # Full-era QCD=100, each other background=10; period Data=30.
        # After scaling: (30 - 4*2.5) / 25 = 0.8, including native blinded anchor.
        def opened(root, config, filename, path, warnings, **kwargs):
            value = 30. if "SingleMuon_G" in filename else 100. if "QCD_Inclusive" in filename else 10.
            return InputHist([value]*8, [value]*8)
        with patch.object(p, "open_hist", side_effect=opened), \
             patch.object(p, "rebin_hist", side_effect=lambda h, edges: h.Clone("rebinned")), \
             redirect_stdout(io.StringIO()):
            events = p.build_plot_inputs(None, cfg, "events", Axis.edges)
            xsec = p.build_plot_inputs(None, cfg, "xsec", Axis.edges)
        self.assertEqual(events.lumi_pb, 4000.)
        self.assertAlmostEqual(events.qcd_normalisation_factor, .8)
        self.assertAlmostEqual(xsec.qcd_normalisation_factor, .8)
        self.assertEqual(events.data.values, [30.]*8)
        self.assertAlmostEqual(events.data.GetBinError(1), math.sqrt(30.))
        self.assertAlmostEqual(xsec.data.GetBinContent(1), 30./4000.)
        self.assertAlmostEqual(events.bkg_total.GetBinContent(1), 30.)
        self.assertAlmostEqual(xsec.bkg_total.GetBinContent(1), 30./4000.)
        latex = MagicMock()
        p.draw_cms_labels(SimpleNamespace(TLatex=lambda: latex, kBlack=1), cfg, events.lumi_pb)
        strings = [a.args[-1] for a in latex.DrawLatex.call_args_list]
        self.assertIn("4.000 fb^{-1} (13 TeV)", strings)
        self.assertIn("2016postVFP, SingleMuon G, OS, b-jet", strings)

    def test_plotter_period_dd_statistics_retain_full_era_metadata(self):
        for method in ("run-common", "mc-double-ratio", "data-low"):
            _, _, inputs = transfer_and_inputs(method)
            cfg = p.Config(era="2016postVFP", data_period="F", period_lumi_fb=4., mc_lumi_fb=16.,
                           divide_by_bin_width=False)
            native = inputs[1]
            nominal = InputHist([native.GetBinContent(i)/4. for i in range(1, 9)], [0.]*8)
            root_file = SimpleNamespace(IsZombie=lambda: False, Close=lambda: None)
            root = SimpleNamespace(TFile=SimpleNamespace(Open=lambda *a: root_file))
            with patch.object(p, "read_qcd_stat_inputs", return_value=inputs), redirect_stdout(io.StringIO()):
                result = p.add_qcd_stat_uncertainty(root, cfg, [cfg.era], {cfg.era: {"QCD": nominal}},
                                                  p.Uncertainty([0.]*8, [0.]*8), 1.)
            expected = p.qcd_root_window_statistics(inputs, 11., 15.)
            self.assertAlmostEqual(result.low[3], expected["sigma_stat_bound"]/4.)
            # The stored full-era central is still checked rather than bypassed.
            bad = copy.deepcopy(nominal)
            bad.values[3] *= 1.1
            with patch.object(p, "read_qcd_stat_inputs", return_value=inputs), \
                 redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                p.add_qcd_stat_uncertainty(root, cfg, [cfg.era], {cfg.era: {"QCD": bad}},
                                          p.Uncertainty([0.]*8, [0.]*8), 1.)

    def test_plotter_period_dy_factorisation_and_nf_uncertainties(self):
        cfg = p.Config(era="2016postVFP", data_period="H", period_lumi_fb=4., mc_lumi_fb=16.,
                       qcd_method="mc", divide_by_bin_width=False, draw_systematics=True, strict=True)
        source = InputHist([20.]*8, [4.]*8)
        nominal = source.Clone("DY")
        nominal.Scale(.2*.25)
        def read(root, filename, path):
            if path == "DYAux/LightJetSource":
                return source.Clone("source"), None
            value = .2 if path == p.DY_AUX_NF_AMC_PATH else .22
            return SimpleNamespace(GetNbinsX=lambda: 1, GetBinContent=lambda i: value,
                                   GetBinError=lambda i: .01), None
        with patch.object(p, "read_hist", side_effect=read), \
             patch.object(p, "rebin_hist", side_effect=lambda h, edges: h.Clone("rebinned")), \
             patch.object(p, "background_detector_processes", return_value=[]), \
             patch.object(p, "background_lumi_processes", return_value=[]), \
             patch.object(p, "_generator_theory_uncertainty"), redirect_stdout(io.StringIO()) as log:
            result = p.bkg_syst_uncertainty(None, cfg, [cfg.era], {"DY": nominal},
                                          {cfg.era: {"DY": nominal}}, nominal, Axis.edges, 1., [])
        self.assertIn("DY_NF_factorisation/DY/2016postVFP: OK", log.getvalue())
        self.assertAlmostEqual(result.high[3], math.sqrt(.05**2 + .1**2))
        self.assertEqual(source.values, [20.]*8)

    def test_period_luminosities_and_invalid_scope_before_root(self):
        parser = q.build_parser()
        for period in "FGH":
            args = parser.parse_args(["--mode", "ss-data", "--year", "2016postVFP", "--data-period", period])
            diagnostic = q.data_period_diagnostic(args, q.SS_MODE)
            self.assertEqual(diagnostic.data_stem, f"Skim_NIsoMuon_SingleMuon_{period}")
            self.assertAlmostEqual(diagnostic.lumi_fb, q.POSTVFP_DATA_LUMI_FB[period])
        self.assertAlmostEqual(sum(q.POSTVFP_DATA_LUMI_FB.values()), q.POSTVFP_MC_LUMI_FB, places=8)
        self.assertNotIn(278770, q.POSTVFP_DATA_RUNS["F"])
        for argv in (["--mode", "qcd-mc", "--year", "2016postVFP", "--data-period", "G"],
                     ["--mode", "ss-data", "--year", "2016preVFP", "--data-period", "F"],
                     ["--mode", "ss-data", "--year", "Run2", "--data-period", "G"],
                     ["--mode", "ss-data", "--period-lumi-fb", "1"],
                     ["--mode", "ss-data", "--year", "2016postVFP", "--data-period", "G", "--validate-qcd-double-ratio"],
                     ["--mode", "ss-data", "--year", "2016postVFP", "--data-period", "G", "--uncertainty-diagnostics"],
                     ["--mode", "ss-data", "--year", "2016postVFP", "--data-period", "G", "--period-lumi-fb", "20"]):
            with patch.object(q, "import_root", side_effect=AssertionError("must reject before ROOT")):
                with self.assertRaises(ValueError): q.run(parser.parse_args(argv))

    def test_period_input_scales_mc_variances_and_preserves_negative_residuals(self):
        top, others = hist(80., 200., 400., 900.), hist(20., 30., 100., 120.)
        sources = {"data": InputHist([10.]*8),
                   "Skim_NIsoMuon_SingleMuon_G": InputHist([0., 10., 0., 1., 5., 0., 0., 0.]),
                   "NIsoMuon_Top": InputHist(top.values, top.variances),
                   "NIsoMuon_Others": InputHist(others.values, others.variances)}
        original = copy.deepcopy(sources)
        def source(root, path):
            return SimpleNamespace(Get=lambda name: sources[path.stem], Close=lambda: None)
        diagnostic = q.DataPeriodDiagnostic("G", 4., 16., "test")
        with patch.object(q, "open_root_file", side_effect=source) as reader, redirect_stdout(io.StringIO()):
            result, _, handles = q.build_input_histogram(object(), q.SS_MODE,
                                      [("2016postVFP", Path("/inputs"))], diagnostic)
            self.assertEqual([call.args[1].name for call in reader.call_args_list],
                             ["Skim_NIsoMuon_SingleMuon_G.root", "NIsoMuon_Top.root", "NIsoMuon_Others.root"])
            for i in range(8):
                self.assertAlmostEqual(result.values[i], sources[diagnostic.data_stem].values[i]
                                       -.25*(top.values[i]+others.values[i]))
                self.assertAlmostEqual(result.variances[i], sources[diagnostic.data_stem].variances[i]
                                       +.25**2*(top.variances[i]+others.variances[i]))
            self.assertLess(result.GetBinContent(4), 0.)
            q.close_files(handles)
        for key in sources:
            self.assertEqual(sources[key].values, original[key].values)
            self.assertEqual(sources[key].variances, original[key].variances)
        with patch.object(q, "open_root_file", side_effect=source) as reader, redirect_stdout(io.StringIO()):
            result, _, handles = q.build_input_histogram(object(), q.SS_MODE, [("2016postVFP", Path("/inputs"))])
            self.assertEqual(reader.call_args_list[0].args[1].name, "data.root")
            self.assertAlmostEqual(result.values[1], 10.-80.-20.)
            self.assertAlmostEqual(result.variances[1], 10.+400.+100.)
            q.close_files(handles)

    def test_period_workflow_never_writes_production_but_full_era_still_does(self):
        with tempfile.TemporaryDirectory() as tmp:
            for period in (None, "G"):
                argv = ["--mode", "ss-data", "--year", "2016postVFP", "--transfer-method", "era-specific"]
                if period: argv += ["--data-period", period]
                args = q.build_parser().parse_args(argv)
                density = MagicMock()
                density.GetBinContent.return_value = 10.
                fit = SimpleNamespace(function=object(), average_function=object(), fit_function=object(),
                                      result=object(), accepted=True)
                with ExitStack() as stack:
                    mocks = {}
                    for name, value in dict(import_root=object(), declare_fit_functions=None, configure_minimizer=None,
                        input_dirs=[("2016postVFP", Path(tmp))], build_input_histogram=(object(), {}, []),
                        prepare_histograms=SimpleNamespace(counts=object(), density=density),
                        build_fit_data=SimpleNamespace(data=object(), keepalive=[]), fit_all_models=([fit], [], []),
                        draw_fit_plot=(Path(tmp)/"fit.pdf", Path(tmp)/"fit.png", []), save_ss_fit_anchors=None,
                        write_ss_background_root=(Path(tmp)/"output.root", Path(tmp)/"copied.root")).items():
                        mocks[name] = stack.enter_context(patch.object(q, name, return_value=value))
                    common = stack.enter_context(patch.object(q, "read_common_qcd_transfer", side_effect=AssertionError("must not derive common transport")))
                    stack.enter_context(redirect_stdout(io.StringIO()))
                    self.assertEqual(q.run(args), 0)
                    common.assert_not_called()
                    if period:
                        mocks["save_ss_fit_anchors"].assert_not_called()
                        mocks["write_ss_background_root"].assert_not_called()
                        self.assertEqual(mocks["build_input_histogram"].call_args.args[3].period, period)
                    else:
                        mocks["save_ss_fit_anchors"].assert_called_once()
                        mocks["write_ss_background_root"].assert_called_once()

    def test_period_plot_filename_and_header_are_separate(self):
        root, density = MagicMock(), MagicMock()
        root.kBlack = 1
        density.GetNbinsX.return_value = 0
        with tempfile.TemporaryDirectory() as tmp, patch.object(q, "PLOT_DIR", Path(tmp)):
            for period in (None, "F", "G", "H"):
                argv = ["--mode", "ss-data", "--year", "2016postVFP"]
                if period: argv += ["--data-period", period]
                args = q.build_parser().parse_args(argv)
                args._data_period_diagnostic = q.data_period_diagnostic(args, q.SS_MODE)
                pdf, png, _ = q.draw_fit_plot(root, args, q.SS_MODE, "chi2", density, [], [], [])
                tag = f"-SingleMuon_{period}" if period else ""
                self.assertEqual(pdf.name, f"FitPlot_FnVariation-SS_fit_SS_Dimuon_Mass-2016postVFP{tag}-AllFits.pdf")
                self.assertEqual(png.stem, pdf.stem)
                if period:
                    root.TLatex.return_value.DrawLatex.assert_any_call(.935, .935,
                                              f"{q.POSTVFP_DATA_LUMI_FB[period]:.3f} fb^{{-1}} (13 TeV)")

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
