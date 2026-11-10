import unittest
import numpy as np
from emseg.eval.gt import decode_rgb, encode_rgb, relabel_2d, relabel_consecutive


class GTTests(unittest.TestCase):
    def test_rgb_roundtrip(self):
        ids = np.array([[0, 1, 255], [256, 65535, 70000]], dtype=np.uint32)
        self.assertTrue(np.array_equal(decode_rgb(encode_rgb(ids)), ids))

    def test_decode_formula(self):
        rgb = np.zeros((1, 1, 3), dtype=np.uint8); rgb[0, 0] = (1, 2, 3)
        self.assertEqual(int(decode_rgb(rgb)[0, 0]), (1 << 16) | (2 << 8) | 3)

    def test_relabel_splits_components(self):
        lab = np.zeros((5, 5), dtype=np.int64)
        lab[0, 0] = 7; lab[4, 4] = 7          # same id, two components
        lab[2, 1:4] = 3
        out = relabel_2d(lab)
        self.assertEqual(out.max(), 3)
        self.assertEqual(len(np.unique(out)), 4)     # 0 + 3 components
        self.assertNotEqual(out[0, 0], out[4, 4])
        self.assertEqual(out[2, 1], out[2, 3])
        self.assertEqual(out[1, 1], 0)

    def test_relabel_min_size(self):
        lab = np.zeros((6, 6), dtype=np.int64)
        lab[0, 0] = 1; lab[3:6, 3:6] = 2
        out = relabel_2d(lab, min_size=2)
        self.assertEqual(out[0, 0], 0)
        self.assertEqual(len(np.unique(out)), 2)
        self.assertEqual(out[4, 4], 1)

    def test_relabel_deterministic_and_connectivity(self):
        rng = np.random.default_rng(0)
        lab = rng.integers(0, 4, size=(32, 32))
        a, b = relabel_2d(lab), relabel_2d(lab)
        self.assertTrue(np.array_equal(a, b))
        diag = np.zeros((3, 3), dtype=np.int64); diag[0, 0] = diag[1, 1] = 5
        self.assertEqual(relabel_2d(diag, connectivity=1).max(), 2)
        self.assertEqual(relabel_2d(diag, connectivity=2).max(), 1)

    def test_relabel_consecutive(self):
        lab = np.array([[0, 10], [10, 30]])
        out = relabel_consecutive(lab)
        self.assertEqual(sorted(np.unique(out).tolist()), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
