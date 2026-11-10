"""2x input scale: doubled offsets + sub-sampling must reproduce the mip1 targets and grid."""
import unittest
import numpy as np, torch
from emseg.seg.targets import Target, METHOD_CHANNELS, OFFSETS_2D
from emseg.seg.unet import upsample_label, upsample_image, predict_slice
from test_metrics import synthetic


class ScaleTests(unittest.TestCase):
    def test_subsampled_targets_match(self):
        lab = synthetic(2, n=10, shape=(64, 64)).astype(np.int64)
        t1, m1 = Target("affinity_mws", scale=1)(lab)
        big = upsample_label(lab, 2); self.assertEqual(big.shape, (128, 128)); self.assertTrue(np.array_equal(big[::2, ::2], lab))
        t2, m2 = Target("affinity_mws", scale=2)(big)
        self.assertEqual(Target("affinity_mws", scale=2).offsets, [[o * 2 for o in off] for off in OFFSETS_2D])
        np.testing.assert_array_equal(t2[:, ::2, ::2], t1); np.testing.assert_array_equal(m2[:, ::2, ::2], m1)

    def test_predict_slice_returns_original_grid(self):
        C = METHOD_CHANNELS["affinity_mws"]
        model = torch.nn.Conv2d(1, C, 1)
        img = np.random.default_rng(0).normal(size=(50, 70)).astype(np.float32)
        y1 = predict_slice(model, img, torch.device("cpu"), scale=1); y2 = predict_slice(model, img, torch.device("cpu"), scale=2)
        self.assertEqual(y1.shape, (C, 50, 70)); self.assertEqual(y2.shape, (C, 50, 70))
        self.assertEqual(upsample_image(img, 2).shape, (100, 140))


if __name__ == "__main__":
    unittest.main()
