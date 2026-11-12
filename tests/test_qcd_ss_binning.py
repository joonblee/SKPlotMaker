"""SS adaptive width floors preserve native signed yields and Sumw2."""
from array import array
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


@unittest.skipIf(ROOT is None, 'PyROOT is not installed')
class SSAdaptiveWidthTest(unittest.TestCase):
    def source(self):
        ROOT.TH1.AddDirectory(False)
        hist = ROOT.TH1D('ss_width_floor_input', '', 5000, 0., 100.)
        hist.Sumw2()
        for i in range(1, hist.GetNbinsX()+1):
            hist.SetBinContent(i, 100. if hist.GetBinCenter(i) < 25. else -1.)
            hist.SetBinError(i, 2.)
        return hist

    def test_production_adaptive_ss_floor_and_signed_variance(self):
        source = self.source()
        with redirect_stdout(io.StringIO()):
            prepared = q.prepare_histograms(ROOT, source, q.SS_MODE, 1, ss_binning='adaptive')
        self.assertIn(5., prepared.edges)
        self.assertIn(30., prepared.edges)
        self.assertTrue(any(prepared.counts.GetBinContent(i) < 0.
                            for i in range(1, prepared.counts.GetNbinsX()+1)))
        for left, right in zip(prepared.edges, prepared.edges[1:]):
            if 5.-1e-9 <= left and right <= 30.+1e-9:
                self.assertGreaterEqual(right-left+1e-9, max(.1, .02*(left+right)/2.))
            self.assertAlmostEqual(left/.02, round(left/.02), places=8)
            i = prepared.counts.FindBin((left+right)/2.)
            first, last = source.FindBin(left+1e-6), source.FindBin(right-1e-6)
            self.assertAlmostEqual(prepared.counts.GetBinContent(i),
                                   sum(source.GetBinContent(j) for j in range(first, last+1)))
            self.assertAlmostEqual(prepared.counts.GetBinError(i)**2, 4.*(last-first+1))
        self.assertAlmostEqual(prepared.counts.Integral(), source.Integral())

    def test_explicit_fixed_ss_binning_is_unchanged(self):
        with redirect_stdout(io.StringIO()):
            for selection in ('regular', 'legacy'):
                prepared = q.prepare_histograms(ROOT, self.source(), q.SS_MODE, 1,
                                                 ss_binning=selection)
                self.assertEqual(prepared.edges, q.make_variable_binning(q.SS_MODE, selection))

    def test_protected_edge_must_be_native(self):
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'cannot split native bins'):
            q.make_adaptive_ss_binning(self.source(), q.SS_MODE, protected_edges=(20.01,))

    def gap_source(self, gap_low, gap_high, gap_yield=0.):
        hist = ROOT.TH1D(q._NAMES.unique('ss_lookahead_gap'), '', 1000, 0., 100.)
        hist.SetDirectory(0)
        for i in range(1, 1001):
            x = hist.GetBinCenter(i)
            hist.SetBinContent(i, gap_yield if gap_low <= x < gap_high else 50.)
            hist.SetBinError(i, 1.)
        # Start with an empty native bin, so the first statistical group has
        # width 0.2 GeV and already satisfies the separate minimum-width rule.
        hist.SetBinContent(hist.FindBin(5.05), 0.)
        return hist

    def test_zero_and_negative_next_bins_are_absorbed_after_target_is_met(self):
        for gap_yield in (0., -1.):
            source = self.gap_source(5.2, 7., gap_yield)
            with redirect_stdout(io.StringIO()):
                baseline = q.make_adaptive_ss_binning(source, q.SS_MODE, 10.)
                extended = q.make_adaptive_ss_binning(source, q.SS_MODE, 10., merge_nonpositive_next=True)
            next_edge = lambda edges: next(x for x in edges if x > 5.+1e-9)
            self.assertLess(next_edge(baseline), 7.)
            self.assertAlmostEqual(next_edge(extended), 7.)
            # Rebin actual signed yields/errors: the absorbed negative bins
            # contribute with their signs and full variance, without clipping.
            first, last = source.FindBin(5.+1e-6), source.FindBin(7.-1e-6)
            self.assertAlmostEqual(sum(source.GetBinContent(i) for i in range(first, last+1)),
                                   50.+18.*gap_yield)
            counts = source.Rebin(len(extended)-1, q._NAMES.unique('ss_lookahead_counts'), array('d', extended))
            counts.SetDirectory(0)
            i = counts.FindBin(6.)
            self.assertAlmostEqual(counts.GetBinContent(i), 50.+18.*gap_yield)
            self.assertAlmostEqual(counts.GetBinError(i)**2, 20.)
            self.assertAlmostEqual(counts.Integral(), source.Integral())

    def test_nonpositive_run_stops_at_five_gev_even_if_next_bin_is_still_zero(self):
        source = self.gap_source(5.2, 12.)
        with redirect_stdout(io.StringIO()):
            edges = q.make_adaptive_ss_binning(source, q.SS_MODE, 10., merge_nonpositive_next=True)
        self.assertAlmostEqual(next(x for x in edges if x > 5.+1e-9), 10.)
        self.assertEqual(source.GetBinContent(source.FindBin(10.1)), 0.)

    def test_lookahead_stops_at_protected_boundary_before_five_gev(self):
        source = self.gap_source(18.1, 22.)
        with redirect_stdout(io.StringIO()):
            edges = q.make_adaptive_ss_binning(source, q.SS_MODE, 10.,
                                              protected_edges=(20.,), merge_nonpositive_next=True)
        self.assertIn(20., edges)
        self.assertFalse(any(18.1 <= x < 20. for x in edges))


if __name__ == '__main__':
    unittest.main()
