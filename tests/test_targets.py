"""Post-processing of GT-derived targets must recover the GT (sanity of sign conventions)."""
import unittest
import numpy as np
from emseg.eval.gt import relabel_2d
from emseg.eval.metrics import slice_metrics
from emseg.seg.targets import Target, postprocess, DEFAULTS, METHOD_CHANNELS
from test_metrics import synthetic


class TargetTests(unittest.TestCase):
    def setUp(self):
        self.gt = relabel_2d(synthetic(3, n=10, shape=(128, 128)))

    def test_channels_and_recovery(self):
        for method in METHOD_CHANNELS:
            t, m = Target(method)(self.gt)
            self.assertEqual(t.shape[0], METHOD_CHANNELS[method], method)
            self.assertEqual(t.shape, m.shape)
            seg = relabel_2d(postprocess(method, t, DEFAULTS[method]))
            res = slice_metrics(seg, self.gt)
            if method == "fg_cc":
                # dense partition without boundaries: CC merges everything -> large merge error expected
                self.assertGreater(res["voi_merge"], 0.5)
            else:
                self.assertLess(res["voi_sum"], 0.15, f"{method}: {res['voi_sum']}")
                self.assertGreater(res["msa"], 0.5, method)


if __name__ == "__main__":
    unittest.main()
