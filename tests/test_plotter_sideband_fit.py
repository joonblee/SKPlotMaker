import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import plotter_sideband_fit as p

class ExportTests(unittest.TestCase):
    def test_reject_observed_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'result.json'
            path.write_text(json.dumps({'schema_version':1,'blind':False}))
            with self.assertRaisesRegex(ValueError,'blinded'):
                p.read_result(path)

    def test_reject_invalid_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'result.json'
            path.write_text(json.dumps({'schema_version':1,'blind':True,
                'observation':'nominal_background_Asimov','fit':{'status':1,'covQual':3}}))
            with self.assertRaisesRegex(ValueError,'validated'):
                p.read_result(path)

    def test_no_invented_error_band(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'result.json'
            path.write_text(json.dumps({'schema_version':1,'blind':True,
                'observation':'nominal_background_Asimov','fit':{'status':0,'covQual':3},'rows':[{}]}))
            with self.assertRaisesRegex(ValueError,'postfit'):
                p.read_result(path)

if __name__=='__main__':
    unittest.main()
