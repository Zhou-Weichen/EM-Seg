"""2D LSD: analytic checks on a disk, range/background, and augmentation equivariance."""
import unittest
import numpy as np
from emseg.seg.lsd import lsd2d, augment_lsd, N_LSD
from test_metrics import synthetic


class LSDTests(unittest.TestCase):
    def test_disk(self):
        lab = np.zeros((64, 64), np.int64); yy, xx = np.mgrid[:64, :64]; lab[(yy - 32) ** 2 + (xx - 32) ** 2 <= 12 ** 2] = 1
        d = lsd2d(lab, sigma=5.0)
        self.assertEqual(d.shape, (N_LSD, 64, 64))
        self.assertTrue(np.all(d[:, lab == 0] == 0))                       # background is 0 in all channels
        self.assertTrue(np.all((d >= 0) & (d <= 1)))
        self.assertAlmostEqual(float(d[0, 32, 32]), 0.5, places=2)        # centre: zero offset
        self.assertAlmostEqual(float(d[1, 32, 32]), 0.5, places=2)
        self.assertAlmostEqual(float(d[4, 32, 32]), 0.5, places=1)        # isotropic: zero correlation
        self.assertGreater(float(d[5, 32, 32]), float(d[5, 32, 32 + 11]))  # size larger at centre than at the rim
        self.assertLess(float(d[1, 32, 32 + 11]), 0.5); self.assertGreater(float(d[1, 32, 32 - 11]), 0.5)
        self.assertLess(float(d[0, 32 + 11, 32]), 0.5); self.assertGreater(float(d[0, 32 - 11, 32]), 0.5)

    def test_augmentation_equivariance(self):
        lab = synthetic(1, n=12, shape=(96, 96)).astype(np.int64)
        d = lsd2d(lab, sigma=4.0); mask = lab > 0
        for k90, fx, fy in [(1, False, False), (2, False, False), (3, False, False), (0, True, False), (0, False, True), (1, True, True), (3, True, False)]:
            l2 = np.rot90(lab, k90)
            if fx: l2 = l2[:, ::-1]
            if fy: l2 = l2[::-1]
            ref = lsd2d(np.ascontiguousarray(l2), sigma=4.0)
            got = augment_lsd(d, mask, k90, fx, fy)
            self.assertEqual(ref.shape, got.shape, (k90, fx, fy))
            np.testing.assert_allclose(got, ref, atol=2e-3, err_msg=str((k90, fx, fy)))


if __name__ == "__main__":
    unittest.main()
