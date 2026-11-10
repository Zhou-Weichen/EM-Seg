import unittest
from emseg.eval.domains.mouse30um import Mouse30um, BLANK_Z
from emseg.eval.domains.snemi import Snemi, AC3_OFFSET
from emseg.eval.domains.cremi import Cremi


class SplitTests(unittest.TestCase):
    def test_mouse_interim(self):
        s = Mouse30um.splits("interim")
        self.assertEqual((len(s["train"]), len(s["val"]), len(s["test"])), (69, 10, 20))
        for z in BLANK_Z:
            for part in s.values():
                self.assertNotIn(z, part)
        self.assertEqual(s["test"], list(range(80, 100)))

    def test_mouse_protocol(self):
        s = Mouse30um.splits("protocol")
        self.assertEqual((len(s["train"]), len(s["val"]), len(s["test"])), (79, 20, 0))

    def test_snemi_protocol(self):
        s = Snemi.splits("protocol")
        self.assertEqual((len(s["train"]), len(s["val"]), len(s["test"])), (80, 20, 100))
        self.assertTrue(all(i >= AC3_OFFSET for i in s["test"]))
        self.assertEqual(set(s["train"]) & set(s["val"]), set())

    def test_cremi_interim(self):
        for smp, ntr in (("A", 75), ("B", 75), ("C", 73)):
            s = Cremi.__new__(Cremi); s.sample = smp; s = s.splits("interim")
            self.assertEqual((len(s["train"]), len(s["val"]), len(s["test"])), (ntr, 25, 25))
        self.assertNotIn(14, s["train"]); self.assertNotIn(74, s["train"])


if __name__ == "__main__":
    unittest.main()
