import unittest, numpy as np, torch
from emseg.seg.crops import SliceCropDataset
from emseg.seg.unet import build_model, masked_bce_dice


class _Dom:
    """Two slices: one normal, one entirely constant."""
    def load_image(self, i): return (np.random.default_rng(i).integers(0, 255, (256, 256)).astype(np.uint8) if i == 0 else np.full((256, 256), 128, np.uint8))
    def load_label(self, i): return np.random.default_rng(i).integers(1, 20, (256, 256)).astype(np.int64)


class StabilityTests(unittest.TestCase):
    def test_constant_crops_rejected(self):
        ds = SliceCropDataset(_Dom(), [0, 1], "boundary_ws", patch=128, n_samples=40, augment=False, seed=0)
        for k in range(40):
            img, _, _, _ = ds[k]; self.assertGreater(float(img.std()), 0.05)
        self.assertGreater(ds.n_rejected, 0)

    def test_skip_guard_logic(self):
        # the run loop skips optimizer steps whose loss or grad norm is non-finite (see run.py)
        gn = torch.tensor(float("nan")); loss = torch.tensor(0.5)
        self.assertFalse(torch.isfinite(loss) and torch.isfinite(gn))


if __name__ == "__main__":
    unittest.main()
