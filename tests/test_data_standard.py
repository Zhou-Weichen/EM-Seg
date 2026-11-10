"""Writers of the EMDATA-v2 standard (emseg.data.standard) on synthetic volumes, in a temporary EMDATA_ROOT."""
import importlib, json, os, tempfile, unittest

import h5py, numpy as np


class StandardWritersTest(unittest.TestCase):
    def test_write_volume_and_images(self):
        os.environ["EMDATA_ROOT"] = tempfile.mkdtemp()
        import emseg.data.standard as std; importlib.reload(std)
        rng = np.random.default_rng(0); raw = rng.integers(30, 220, (4, 64, 64)).astype(np.uint8)
        cell = np.zeros((4, 64, 64), np.int64); cell[:, :, :20] = 1; cell[:, :, 20:40] = 2; cell[:, :, 40:] = 3
        mito = np.zeros_like(cell); mito[:, 5:15, 5:15] = 7; mito[2] = 0
        kw = dict(resolution_nm=[30, 8, 8], modality="x", organism="x", tissue="x", source_url="x", license="x")
        s = std.write_volume_mc("t", "v", raw, {"cell": cell, "mitochondrion": mito}, **kw)
        self.assertEqual(s["n_kept"], 4, s)
        with h5py.File(std.DATA_ROOT / s["file"]) as f:
            self.assertEqual(f["labels/cell"].shape, (4, 64, 64)); self.assertEqual(f["labels/mitochondrion"][2].max(), 0)
            self.assertTrue(f["masks/mitochondrion"][2].all()); self.assertEqual(json.loads(f.attrs["annotated_classes"]), ["cell", "mitochondrion"])
        s2 = std.write_volume("t", "w", raw, mito, label_scope="organelle:mito", **kw)
        self.assertEqual(s2["n_kept"], 3); self.assertIn("no_label", s2["dropped"])
        s3 = std.write_images("t2", [("a", raw[0], {"cell": cell[0]}), ("b", raw[1], cell[1] * 0)], label_scope="dense_cells", **kw)
        self.assertEqual(s3["n_items_kept"], 1, s3)


if __name__ == "__main__":
    unittest.main()
