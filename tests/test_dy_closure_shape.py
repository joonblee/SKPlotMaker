"""Closure shape physics invariants and signed ROOT protocol."""
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dy_closure_shape as c
try:
    import ROOT
except ImportError:
    ROOT = None


class ShapeTests(unittest.TestCase):
    def test_yield_conserved_at_anchors_and_intermediate_or_extreme_values(self):
        nominal = [90., 10., 0.]
        up, down, audit = c.templates(nominal, [90., 13., 99.])
        self.assertEqual(up[2], 0.)
        self.assertEqual(down[2], 0.)
        self.assertAlmostEqual(up[1], 1300./103.)
        for theta in (-7., -1., -.5, 0., .35, 1., 7.):
            varied = c.morph(nominal, up, down, theta)
            self.assertAlmostEqual(sum(varied),100.)
            self.assertTrue(all(v >= 0 for v in varied))
        self.assertEqual(c.morph(nominal,up,down,0.),nominal)

    def test_physical_bound_and_zero_source_are_explicit(self):
        up, down, audit = c.templates([1.,9.,0.], [-2., 12., 55.])
        self.assertEqual(up, [0.,10.,0.])
        self.assertEqual(audit['clipped_up'],[0])
        self.assertAlmostEqual(sum(down),10.)
        self.assertEqual(c.templates([0.], [3.])[0], [0.])

    @unittest.skipIf(ROOT is None,'PyROOT unavailable')
    def test_signed_source_and_stat_errors_unchanged_and_stale_data_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'dy.root'
            f=ROOT.TFile.Open(str(path),'RECREATE');aux=f.mkdir('DYAux');aux.cd()
            source=ROOT.TH1D('LightJetSource','',5000,0.,100.);source.SetDirectory(0)
            for i in range(1,5001):
                source.SetBinContent(i, -2. if i%9 == 0 else 3.)
                source.SetBinError(i,2.)
            before=c.fingerprint(source)
            source.Write();c.write_inputs(ROOT,aux,source)
            self.assertEqual(before,c.fingerprint(source));f.Close()
            f=ROOT.TFile.Open(str(path));src,up,meta=c.read_inputs(f)
            self.assertEqual(meta['source_sha256'],before)
            i=src.FindBin(70.01)
            self.assertAlmostEqual(up.GetBinContent(i)/src.GetBinContent(i),c.SLOPE*src.GetBinCenter(i)+c.INTERCEPT)
            self.assertEqual(up.GetBinError(i),0.)
            self.assertEqual(src.GetBinError(i),2.)
            src.SetBinContent(i,99.)
            with self.assertRaisesRegex(ValueError,'stale'): c.read_inputs(f)
            f.Close()


if __name__=='__main__': unittest.main()
