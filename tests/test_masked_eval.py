import unittest, numpy as np
from emseg.eval.evaluate import evaluate_slices, missing_mask


class MaskedEvalTests(unittest.TestCase):
    def test_missing_region_ignored(self):
        gt = np.zeros((64, 64), np.int64); gt[:, :32] = 1; gt[:, 32:] = 2
        img = np.full((64, 64), 100, np.uint8); img[:, 40:] = 0                       # right part of the image missing (>= 1024 px)
        pred = np.ones((64, 64), np.int64)                                            # predicts one object (merge)
        d0 = evaluate_slices(lambda i: gt, lambda i: pred, [0]); d1 = evaluate_slices(lambda i: gt, lambda i: pred, [0], image_fn=lambda i: img, mask_missing=0)
        d2 = evaluate_slices(lambda i: gt, lambda i: pred, [0], image_fn=lambda i: img, mask_missing=2)
        self.assertGreater(d0.voi_merge[0], 0.5); self.assertAlmostEqual(d0.voi_merge[0], d1.voi_merge[0])
        self.assertLess(d2.voi_merge[0], d0.voi_merge[0])                              # object 2 mostly ignored -> smaller merge penalty
        self.assertTrue(missing_mask(img, 2)[:, 38:].all()); self.assertFalse(missing_mask(img, 2)[:, :37].any())
        img2 = np.full((32, 32), 100, np.uint8); img2[3, 3] = 0; img2[10, 20] = 0                # isolated zero pixels are not missing data
        self.assertFalse(missing_mask(img2, 2).any())


if __name__ == "__main__":
    unittest.main()
