"""Custom OS/SS bins must preserve signed yields, Sumw2 and blinding."""
from argparse import Namespace
from array import array
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os_ss_comparison as comparison

try:
    import ROOT
except ImportError:
    ROOT = None


class QCDInputTest(unittest.TestCase):
    def test_grouped_load_uses_each_eras_filename(self):
        with TemporaryDirectory() as base_dir:
            args = Namespace(base_dir=base_dir, trigger="test_trigger", era="full",
                             mc_only=True, muon_id="POGMedium", jet_id="tight",
                             hist_name="Dilepton_Mass")
            for run2_bin, run3_bin in (("1000", "1000toInf"), ("300To470", "300to470")):
                run2_file = f"Skim_NIsoMuon_QCD_Pt-{run2_bin}_MuEnriched.root"
                run3_file = f"Skim_NIsoMuon_QCD_Pt-{run3_bin}_MuEnriched.root"
                expected = []
                for year in comparison.YEARS:
                    filename = run2_file if year in comparison.RUN2 else run3_file
                    path = Path(base_dir) / year / args.trigger / filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                    expected.append(str(path))
                for args.qcd_file in (run2_file, run3_file):
                    with self.subTest(filename=args.qcd_file), patch.object(comparison, "open_hist") as reader:
                        comparison.qcd(None, args, "OS")
                        self.assertEqual([call.args[1] for call in reader.call_args_list], expected)

    def test_run3_low_bin_and_inclusive_filename(self):
        args = Namespace(base_dir="/inputs", trigger="")
        for filename in ("Skim_NIsoMuon_QCD_Pt-15to20_MuEnriched.root",
                         "NIsoMuon_QCD_Inclusive.root", "custom.root"):
            with self.subTest(filename=filename):
                self.assertEqual(comparison.input_file(args, "2023", filename),
                                 f"/inputs/2023/{filename}")


class BinningOptionsTest(unittest.TestCase):
    def args(self, **kwargs):
        values = dict(bin_edges=None, xmin=None, xmax=None,
                      variable_binning=True, bin_width=1.)
        values.update(kwargs)
        return Namespace(**values)

    def test_defaults_and_custom_endpoints(self):
        args = self.args()
        self.assertEqual(comparison.resolve_binning(args),
                         [x for x in comparison.MASS_BINS if 5 <= x <= 80])
        args = self.args(bin_edges=[7., 11., 20., 70.])
        self.assertEqual(comparison.resolve_binning(args), args.bin_edges)
        self.assertEqual((args.xmin, args.xmax), (7., 70.))

    def test_invalid_edges_and_range_mismatch(self):
        for edges in ([5.], [5., 5., 80.], [5., 4., 80.],
                      [5., math.nan, 80.], [5., math.inf]):
            with self.subTest(edges=edges), self.assertRaises(comparison.PlotError):
                comparison.resolve_binning(self.args(bin_edges=edges))
        with self.assertRaises(comparison.PlotError):
            comparison.resolve_binning(self.args(bin_edges=[5., 20., 80.], xmin=7.))

    def test_conflicting_options_are_rejected_before_loading_inputs(self):
        with self.assertRaises(SystemExit) as exc:
            comparison.main(['--era', '2018', '--bin-edges', '5', '80',
                             '--no-variable-binning'])
        self.assertEqual(exc.exception.code, 2)


@unittest.skipIf(ROOT is None, 'PyROOT is not installed')
class CustomRebinTest(unittest.TestCase):
    def test_rebin_preserves_signed_yields_and_variances(self):
        ROOT.TH1.AddDirectory(False)
        hist = ROOT.TH1D('osss_signed', '', 4, 5., 9.)
        hist.Sumw2()
        for i, (value, error) in enumerate([(10., 2.), (-3., 4.),
                                          (5., 1.), (7., 3.)], 1):
            hist.SetBinContent(i, value)
            hist.SetBinError(i, error)
        merged = comparison.rebin(hist, [5., 7., 9.], 'osss_merged')
        self.assertEqual([merged.GetBinContent(i) for i in (1, 2)], [7., 12.])
        self.assertAlmostEqual(merged.GetBinError(1)**2, 20.)
        self.assertAlmostEqual(merged.GetBinError(2)**2, 10.)
        with self.assertRaises(comparison.PlotError):
            comparison.rebin(hist, [5., 6.5, 9.], 'osss_split')

    def test_custom_bins_crossing_blind_boundaries_are_hidden(self):
        hist = ROOT.TH1D('osss_blind', '', 5,
                         array('d', [9., 11., 13., 79., 81., 83.]))
        for i in range(1, 6):
            hist.SetBinContent(i, 10.)
            hist.SetBinError(i, 2.)
        comparison.mask_overlapping_bins(hist, 11., 80.)
        self.assertEqual([hist.GetBinContent(i) for i in range(1, 6)],
                         [10., 0., 0., 0., 10.])
        self.assertEqual([hist.GetBinError(i) for i in range(1, 6)],
                         [2., 0., 0., 0., 2.])


if __name__ == '__main__':
    unittest.main()
