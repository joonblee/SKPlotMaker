"""ROOT regression: Run-3 TH1F inputs must not round during DY matching."""
from array import array
from contextlib import redirect_stdout
import io
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dy_mc_shape_comparison as dy

try:
    import ROOT
except ImportError:
    ROOT = None


@unittest.skipIf(ROOT is None, "PyROOT is not installed")
class DYPrecisionTest(unittest.TestCase):
    def histograms(self):
        ROOT.TH1.AddDirectory(False)
        edges = array('d', [11, 13, 15, 17, 19, 22, 26, 40, 70])
        b = ROOT.TH1F('precision_b', '', 8, edges)
        l = ROOT.TH1F('precision_l', '', 8, edges)
        b.Sumw2(); l.Sumw2()
        for i, (by, ly) in enumerate(zip(
                [100, 50, 40, 25, 15, 5, 3, .478893],
                [1300, 800, 600, 400, 200, 100, 30, 7.97639]), 1):
            b.SetBinContent(i, by); l.SetBinContent(i, ly)
            b.SetBinError(i, math.sqrt(by)*1.7)
            l.SetBinError(i, math.sqrt(ly)*2.3)
        l.SetBinContent(0, -2.5); l.SetBinError(0, 3.1)
        l.SetBinContent(9, 4.7); l.SetBinError(9, 2.9)
        return b, l

    def test_float_scaling_reproduction_and_double_precision_matching(self):
        b, l = self.histograms()
        target, _ = dy.integral_and_error(b, 11, 70)
        source, _ = dy.integral_and_error(l, 11, 70)
        raw = l.Clone('precision_raw_scaled')
        raw.Scale(target/source)
        rounded, _ = dy.integral_and_error(raw, 11, 70)
        self.assertFalse(math.isclose(rounded, target, rel_tol=1e-10, abs_tol=1e-9))
        original = [l.GetBinContent(i) for i in range(10)]
        with redirect_stdout(io.StringIO()):
            scaled, factor = dy.match_lightjet(ROOT, b, l, 11, 70)
        self.assertEqual(scaled.ClassName(), 'TH1D')
        matched, _ = dy.integral_and_error(scaled, 11, 70)
        self.assertTrue(math.isclose(matched, target, rel_tol=1e-13))
        self.assertEqual(original, [l.GetBinContent(i) for i in range(10)])
        for i in range(10):
            self.assertAlmostEqual(scaled.GetBinContent(i), original[i]*factor, places=12)
            self.assertAlmostEqual(scaled.GetBinError(i), l.GetBinError(i)*factor, places=12)

    def test_conversion_preserves_variable_bins_sumw2_and_flows(self):
        _, source = self.histograms()
        converted = dy.clone_hist(source, 'precision_double')
        self.assertEqual(converted.ClassName(), 'TH1D')
        self.assertEqual(dy.axis_edges(converted), dy.axis_edges(source))
        self.assertEqual(converted.GetEntries(), source.GetEntries())
        for i in range(10):
            self.assertEqual(converted.GetBinContent(i), source.GetBinContent(i))
            self.assertEqual(converted.GetSumw2()[i], source.GetSumw2()[i])
        second = dy.clone_hist(converted, 'precision_double_again')
        for i in range(10):
            self.assertEqual(second.GetBinContent(i), converted.GetBinContent(i))
            self.assertEqual(second.GetSumw2()[i], converted.GetSumw2()[i])


if __name__ == '__main__':
    unittest.main()
