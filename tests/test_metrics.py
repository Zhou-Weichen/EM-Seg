import unittest
import numpy as np
from elf.evaluation import matching as elf_matching, cremi_score
from emseg.eval.metrics import matching_stats, slice_metrics, boundary_map, dice
from emseg.eval.gt import relabel_2d


def synthetic(seed=0, n=12, shape=(96, 96)):
    """Dense random Voronoi-like partition with a little background."""
    rng = np.random.default_rng(seed)
    pts = rng.integers(0, shape[0], size=(n, 2))
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    d = (yy[..., None] - pts[:, 0]) ** 2 + (xx[..., None] - pts[:, 1]) ** 2
    lab = d.argmin(-1) + 1
    lab[:4, :4] = 0
    return lab.astype(np.int64)


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.gt = relabel_2d(synthetic(0))
        # a degraded prediction: merge two objects, split one, drop one
        pred = self.gt.copy()
        ids = np.unique(pred); ids = ids[ids > 0]
        from emseg.eval.metrics import boundary_map
        # merge two adjacent cells: find a neighbour of ids[0] across a vertical boundary
        a = pred[:, :-1]; b = pred[:, 1:]
        nb = np.unique(b[(a == ids[0]) & (b != ids[0]) & (b > 0)])
        pred[pred == nb[0]] = ids[0]                        # merge (adjacent -> stays merged after relabel)
        ids = ids[ids != nb[0]]
        big = ids[np.argmax([(pred == i).sum() for i in ids])]              # largest remaining object
        ys, xs = np.where(pred == big); order = np.lexsort((xs, ys)); half = order[len(order) // 2:]
        pred[ys[half], xs[half]] = ids.max() + 1              # split it into two halves (row-sorted)
        assert len(np.unique(pred)) == len(ids) + 2, "split did not create a new object"
        pred[pred == ids[3]] = 0                            # missing object
        self.pred = relabel_2d(pred)

    def test_perfect(self):
        m = slice_metrics(self.gt, self.gt)
        self.assertAlmostEqual(m["voi_sum"], 0.0, places=9)
        self.assertAlmostEqual(m["arand"], 0.0, places=9)
        self.assertAlmostEqual(m["msa"], 1.0, places=9)
        for k in ("f1_50", "f1_75", "ap50", "ap75", "pq50", "pq75", "dice_fg", "dice_boundary"):
            self.assertAlmostEqual(m[k], 1.0, places=9, msg=k)

    def test_degraded_direction(self):
        m = slice_metrics(self.pred, self.gt)
        self.assertGreater(m["voi_merge"], 0.0)
        self.assertGreater(m["voi_split"], 0.0)
        self.assertGreater(m["arand"], 0.0)
        self.assertLess(m["msa"], 1.0)
        self.assertLess(m["pq50"], 1.0)
        self.assertGreaterEqual(m["pq50"], m["pq75"])
        self.assertGreaterEqual(m["f1_50"], m["f1_75"])

    def test_matching_matches_elf(self):
        for thr in (0.5, 0.75):
            ref = elf_matching(self.pred, self.gt, threshold=thr, ignore_label=0)
            mine = matching_stats(self.pred, self.gt, thresholds=(thr,))
            key = f"{int(round(thr * 100))}"
            self.assertAlmostEqual(mine[f"precision{key}"], ref["precision"], places=9)
            self.assertAlmostEqual(mine[f"recall{key}"], ref["recall"], places=9)
            self.assertAlmostEqual(mine[f"f1_{key}"], ref["f1"], places=9)
            self.assertAlmostEqual(mine[f"ap{key}"], ref["segmentation_accuracy"], places=9)

    def test_voi_arand_match_elf_cremi_score(self):
        vs, vm, ar, cs = cremi_score(self.pred, self.gt, ignore_gt=(0,))
        m = slice_metrics(self.pred, self.gt)
        self.assertAlmostEqual(m["voi_split"], vs, places=9)
        self.assertAlmostEqual(m["voi_merge"], vm, places=9)
        self.assertAlmostEqual(m["arand"], ar, places=9)
        self.assertAlmostEqual(m["cremi_score"], cs, places=9)

    def test_pq_definition(self):
        # two GT objects, prediction matches one exactly and misses the other: PQ50 = 1/(1+0.5) = 2/3
        gt = np.zeros((10, 10), dtype=np.int64); gt[:5] = 1; gt[5:] = 2
        pred = np.zeros_like(gt); pred[:5] = 1
        m = matching_stats(pred, gt)
        self.assertAlmostEqual(m["pq50"], 2 / 3, places=9)
        self.assertAlmostEqual(m["ap50"], 0.5, places=9)

    def test_boundary_and_dice(self):
        lab = np.zeros((4, 4), dtype=np.int64); lab[:, :2] = 1; lab[:, 2:] = 2
        b = boundary_map(lab)
        self.assertTrue(b[:, 1].all() and b[:, 2].all() and not b[:, 0].any())
        self.assertEqual(dice(np.zeros(3, bool), np.zeros(3, bool)), 1.0)


if __name__ == "__main__":
    unittest.main()
