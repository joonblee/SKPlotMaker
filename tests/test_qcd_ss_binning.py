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

    def blank_source(self):
        hist = ROOT.TH1D(q._NAMES.unique('ss_completed_grid'), '', 1000, 0., 100.)
        hist.SetDirectory(0)
        for i in range(1, 1001):
            hist.SetBinError(i, 1.)
        return hist

    def test_whole_nonpositive_adaptive_bins_are_absorbed_with_signed_variance(self):
        for next_yield in (0., -10.):
            source = self.blank_source()
            source.SetBinContent(source.FindBin(5.05), 50.)
            source.SetBinContent(source.FindBin(6.05), next_yield)
            source.SetBinContent(source.FindBin(8.05), 20.)
            edges = q.merge_nonpositive_ss_adaptive_bins(source, [5., 6., 7., 8., 10.], 5.)
            self.assertEqual(edges, [5., 8., 10.])
            counts = source.Rebin(len(edges)-1, q._NAMES.unique('ss_completed_counts'), array('d', edges))
            counts.SetDirectory(0)
            self.assertAlmostEqual(counts.GetBinContent(1), 50.+next_yield)
            self.assertAlmostEqual(counts.GetBinError(1)**2, 30.)
            self.assertAlmostEqual(counts.Integral(0, counts.GetNbinsX()+1), source.Integral())

    def test_extra_merge_stops_before_zero_or_negative_sum(self):
        for next_yield in (-50., -51.):
            source = self.blank_source()
            source.SetBinContent(source.FindBin(5.05), 50.)
            source.SetBinContent(source.FindBin(6.05), next_yield)
            source.SetBinContent(source.FindBin(7.05), 20.)
            edges = q.merge_nonpositive_ss_adaptive_bins(source, [5., 6., 7., 8.], 5.)
            self.assertEqual(edges, [5., 6., 7., 8.])

    def test_whole_bin_extra_merge_stops_at_cap_and_protected_edges(self):
        source = self.blank_source()
        source.SetBinContent(source.FindBin(5.05), 50.)
        edges = q.merge_nonpositive_ss_adaptive_bins(source, [5., 6., 8., 10., 12., 15.], 5.)
        self.assertEqual(edges[1], 10.)
        protected = q.merge_nonpositive_ss_adaptive_bins(source, [5., 6., 8., 10., 12., 15.],
                                                        5., protected_edges=(8.,))
        self.assertEqual(protected[1], 8.)

    def test_sparse_signed_native_bins_cannot_override_target_or_width_floor(self):
        from dataclasses import replace
        source = ROOT.TH1D(q._NAMES.unique('ss_sparse_regression'), '', 5000, 0., 100.)
        source.SetDirectory(0)
        for i in range(1, 5001):
            source.SetBinContent(i, 100. if source.GetBinCenter(i) < 15. else 3. if i%2 else -2.)
            source.SetBinError(i, 1.)
        source.SetBinContent(source.FindBin(15.001), 1.)
        mode = replace(q.SS_MODE, fit_max=50.)
        with redirect_stdout(io.StringIO()):
            baseline = q.make_adaptive_ss_binning(source, mode, 10., 5.)
            edges = q.make_adaptive_ss_binning(source, mode, 10., 5., merge_nonpositive_next=True)
        self.assertTrue(set(edges).issubset(baseline))
        self.assertLessEqual(len(edges), len(baseline))
        for left, right in zip(edges, edges[1:]):
            if 5.-1e-9 <= left and right <= 50.+1e-9:
                self.assertGreaterEqual(right-left+1e-9, max(.1, .02*(left+right)/2.))


if __name__ == '__main__':
    unittest.main()
