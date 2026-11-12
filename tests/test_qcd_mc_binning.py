"""MC adaptive width floor must preserve weighted yields and veto boundaries."""
from contextlib import redirect_stdout
import io
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


class MCWidthFloorTest(unittest.TestCase):
    def test_floor_rounds_up_to_existing_edges_and_merges_final_remainder(self):
        # The 0.01 GeV remainder must not survive at 9 GeV.
        original = [8., 8.05, 8.1, 8.2, 8.4, 8.6, 8.8, 8.99, 9.]
        bounded = q.bound_mc_bin_widths(original)
        self.assertEqual(bounded[0], 8.)
        self.assertEqual(bounded[-1], 9.)
        self.assertTrue(set(bounded).issubset(original))
        self.assertNotIn(8.99, bounded)
        for left, right in zip(bounded, bounded[1:]):
            self.assertGreaterEqual(right-left+1e-9, max(.1, .02*(left+right)/2.))
        # Statistical bins that are already wider than the floor stay intact.
        self.assertEqual(q.bound_mc_bin_widths([30., 32., 35., 40.]), [30., 32., 35., 40.])


@unittest.skipIf(ROOT is None, 'PyROOT is not installed')
class MCAdaptiveBinningTest(unittest.TestCase):
    def source(self):
        ROOT.TH1.AddDirectory(False)
        hist = ROOT.TH1D('mc_adaptive_input', '', 5000, 0., 100.)
        hist.Sumw2()
        for i in range(1, hist.GetNbinsX()+1):
            x = hist.GetBinCenter(i)
            value = 100. if x < 20. else (.5 if x < 60. else -1.)
            hist.SetBinContent(i, value)
            hist.SetBinError(i, 2.)
        return hist

    def test_native_alignment_signed_sumw2_and_width_bound(self):
        source = self.source()
        with redirect_stdout(io.StringIO()):
            prepared = q.prepare_histograms(ROOT, source, q.QCD_MODE, 1)
        for edge in (5., 9., 11., 80.):
            self.assertTrue(any(math.isclose(edge, x, abs_tol=1e-9) for x in prepared.edges))
        for left, right in zip(prepared.edges, prepared.edges[1:]):
            self.assertAlmostEqual(left/.02, round(left/.02), places=8)
            if 5.-1e-9 <= left and right <= 80.+1e-9:
                self.assertGreaterEqual(right-left+1e-9, max(.1, .02*(left+right)/2.))
                self.assertFalse(left < 9. < right or left < 11. < right)
            i = prepared.counts.FindBin((left+right)/2.)
            first, last = source.FindBin(left+1e-6), source.FindBin(right-1e-6)
            expected = sum(source.GetBinContent(j) for j in range(first, last+1))
            variance = sum(source.GetBinError(j)**2 for j in range(first, last+1))
            self.assertAlmostEqual(prepared.counts.GetBinContent(i), expected)
            self.assertAlmostEqual(prepared.counts.GetBinError(i)**2, variance)
        self.assertAlmostEqual(prepared.counts.Integral(), source.Integral())
        self.assertTrue(any(prepared.counts.GetBinContent(i) < 0.
                            for i in range(1, prepared.counts.GetNbinsX()+1)))

    def test_legacy_mc_and_ss_edges_are_unchanged(self):
        source = self.source()
        with redirect_stdout(io.StringIO()):
            legacy = q.prepare_histograms(ROOT, source, q.QCD_MODE, 1, mc_binning='legacy')
            expected_ss = q.make_adaptive_ss_binning(source, q.SS_MODE)
            ss = q.prepare_histograms(ROOT, source, q.SS_MODE, 1, ss_binning='adaptive',
                                      mc_min_effective_count=1000., mc_max_bin_width=1.)
        self.assertEqual(legacy.edges, q.make_variable_binning(q.QCD_MODE))
        self.assertEqual(ss.edges, expected_ss)


if __name__ == '__main__':
    unittest.main()
