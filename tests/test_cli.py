import json, tempfile, unittest
from pathlib import Path
import numpy as np, tifffile
from emseg.eval import cli
from emseg.eval.evaluate import evaluate_slices, load_prediction, write_outputs


class CLITests(unittest.TestCase):
    def test_load_prediction_formats_and_evaluate(self):
        rng = np.random.default_rng(1)
        lab = rng.integers(1, 5, size=(16, 16)).astype(np.int64)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            tifffile.imwrite(d / "pred_0000.tif", lab.astype(np.uint16))
            np.save(d / "pred_0001.npy", lab)
            self.assertTrue(np.array_equal(load_prediction(d, 0), lab))
            self.assertTrue(np.array_equal(load_prediction(d, 1), lab))
            with self.assertRaises(FileNotFoundError):
                load_prediction(d, 2)
            df = evaluate_slices(lambda i: lab, lambda i: load_prediction(d, i), [0, 1])
            self.assertEqual(len(df), 2)
            self.assertTrue((df.voi_sum.abs() < 1e-9).all())
            write_outputs(df, d / "out", {"meta": 1})
            s = json.load(open(d / "out" / "summary.json"))
            self.assertEqual(s["summary"]["voi_sum"]["n"], 2)
            self.assertTrue((d / "out" / "per_slice.csv").exists())

    def test_splits_command_runs(self):
        self.assertEqual(cli.main(["splits", "--domain", "snemi"]), 0)


if __name__ == "__main__":
    unittest.main()
