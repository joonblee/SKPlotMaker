"""Fit boundaries, SS validation region, and integrated functional envelopes."""
from contextlib import redirect_stdout
import io
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]))
import qcd_bkg_estimation as q
import qcd_ss_fit_range_comparison as ss
import qcd_fit_window_diagnostics as windows
try:
    import ROOT
except ImportError:
    ROOT=None


class Function:
    def __init__(self,f): self.f=f
    def Eval(self,x): return self.f(x)


class ReviewDiagnosticsTests(unittest.TestCase):
    def test_envelope_integral_does_not_cancel_crossing_functions(self):
        nominal, delta, models=windows.integrate_functions(
            [Function(lambda x:1.), Function(lambda x:2*x)],0.,1.,.0005)
        self.assertAlmostEqual(nominal,1.)
        self.assertAlmostEqual(delta,.5)
        self.assertAlmostEqual(models[1],1.)  # Max of integrals would incorrectly be zero.

    def test_ss_mc_sign_changes_only_region_and_veto(self):
        args=q.build_parser().parse_args(['--mode','qcd-mc','--mc-sign','SS','--mc-fit-range','5','30'])
        fit,plot=q.mc_range_modes(args,q.QCD_MODE)
        self.assertEqual(fit.region,q.SS_REGION)
        self.assertFalse(q.is_excluded(fit,9.,11.))
        self.assertTrue(q.is_excluded(q.QCD_MODE,9.,11.))
        self.assertEqual(q.mc_range_suffix(fit,plot),'-SS-Fit5to30-Plot5to80')
        args.mc_fit_range=None
        self.assertEqual(q.mc_range_modes(args,q.QCD_MODE)[0].region,q.SS_REGION)

    @unittest.skipIf(ROOT is None,'PyROOT unavailable')
    def test_native_rebin_40_fit_never_contains_40to50_events(self):
        ROOT.TH1.AddDirectory(False)
        hist=ROOT.TH1D('review_source','',1000,0.,100.)
        for i in range(1,1001):
            x=hist.GetBinCenter(i)
            hist.SetBinContent(i, 1. if x < 40. else 10000.)
            hist.SetBinError(i,1.)
        args=ss.build_parser().parse_args([])
        with redirect_stdout(io.StringIO()):
            prepared=ss.common_histograms(ROOT,hist,args)
        fit=ss.fit_histograms(ROOT,hist,prepared)
        self.assertIn(40.,fit.edges)
        self.assertNotIn(40.,prepared.edges)
        self.assertIs(ss.fit_histograms(ROOT,hist,prepared,30.),prepared)
        self.assertIs(ss.fit_histograms(ROOT,hist,prepared,50.),prepared)
        self.assertIn(20.,ss.fit_histograms(ROOT,hist,prepared,20.).edges)
        first,last=fit.counts.FindBin(30.+1e-6),fit.counts.FindBin(40.-1e-6)
        self.assertAlmostEqual(fit.counts.Integral(first,last),100.)
        for a,b in zip(fit.edges,fit.edges[1:]):
            if a >= 5. and b <= 50.:
                self.assertGreaterEqual(b-a+1e-9,max(.1,.02*(a+b)/2.))


if __name__ == '__main__': unittest.main()
