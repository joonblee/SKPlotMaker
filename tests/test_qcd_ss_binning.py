"""SS adaptive width floors preserve native signed yields and Sumw2."""
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


if __name__ == '__main__':
    unittest.main()
