"""Extra EM metrics: identities and known split/merge cases."""
import unittest
import numpy as np
from elf.evaluation import rand_index
from emseg.eval.metrics import em_extra_metrics


class EMExtraTests(unittest.TestCase):
    def setUp(self):
        gt = np.zeros((40, 40), np.int64); gt[:20, :20] = 1; gt[:20, 20:] = 2; gt[20:, :] = 3
        self.gt = gt

    def test_identity(self):
        m = em_extra_metrics(self.gt.copy(), self.gt)
        for k in ("vrand_split", "vrand_merge", "vrand_f", "vinfo_split", "vinfo_merge", "vinfo_f", "dice_obj", "dice_inst", "aji"):
            self.assertAlmostEqual(m[k], 1.0, places=6, msg=k)

    def test_merge_lowers_merge_scores_only(self):
        seg = self.gt.copy(); seg[seg == 2] = 1                         # merge objects 1 and 2
        m = em_extra_metrics(seg, self.gt)
        self.assertAlmostEqual(m["vrand_split"], 1.0); self.assertLess(m["vrand_merge"], 1.0)
        self.assertAlmostEqual(m["vinfo_split"], 1.0); self.assertLess(m["vinfo_merge"], 1.0)
        self.assertLess(m["aji"], 1.0); self.assertLess(m["dice_obj"], 1.0)

    def test_split_lowers_split_scores_only(self):
        seg = self.gt.copy(); seg[20:, 20:] = 4                         # split object 3
        m = em_extra_metrics(seg, self.gt)
        self.assertLess(m["vrand_split"], 1.0); self.assertAlmostEqual(m["vrand_merge"], 1.0)
        self.assertLess(m["vinfo_split"], 1.0); self.assertAlmostEqual(m["vinfo_merge"], 1.0)
        # dice_inst: objects 1,2 perfect; object 3 best partner covers half -> dice 2*400/(800+400)
        self.assertAlmostEqual(m["dice_inst"], (1 + 1 + 800 / 1200) / 3, places=6)
        # AJI by hand: C = 400+400+400, U = 400+400+800 + unused 400
        self.assertAlmostEqual(m["aji"], 1200 / 2000, places=6)

    def test_vrand_matches_elf_arand(self):
        rng = np.random.default_rng(0); seg = self.gt.copy(); seg[rng.random(seg.shape) < 0.1] = 7
        are, _ = rand_index(seg, self.gt, ignore_gt=[0])
        self.assertAlmostEqual(em_extra_metrics(seg, self.gt)["vrand_f"], 1 - are, places=6)

    def test_gt_background_ignored(self):
        gt = self.gt.copy(); gt[:, :2] = 0; seg = gt.copy(); seg[:, :2] = 9   # prediction in ignored zone
        m = em_extra_metrics(seg, gt)
        self.assertAlmostEqual(m["aji"], 1.0); self.assertAlmostEqual(m["dice_obj"], 1.0)


if __name__ == "__main__":
    unittest.main()
