"""Focused checks for uniform legacy WideField display defaults."""
import sys
from pathlib import Path
import unittest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_scripts"))

from Visual_toolkit.toolkit import Visual_toolkit, resolve_widefield_parameters
from Visual_toolkit.image_geometry import percentile_normalize


class WideFieldProfileTests(unittest.TestCase):
    def test_all_samples_use_uniform_legacy_defaults(self):
        for sample in ("251637", "252384", "252385", "252527", "252714",
                       "252790", "252383", "252718", "252970", "unknown"):
            with self.subTest(sample=sample):
                self.assertEqual(resolve_widefield_parameters(sample), (2.0, 100))

    def test_explicit_values_override_uniform_defaults(self):
        self.assertEqual(
            resolve_widefield_parameters("252384", bg_intensity=1.0, manual_threshold=150),
            (1.0, 150.0),
        )
        self.assertIsNone(Visual_toolkit.plot_widefield_context.__defaults__[1])
        self.assertIsNone(Visual_toolkit.plot_widefield_context.__defaults__[2])

    def test_public_plot_uses_legacy_clip_then_gain_rgb_formula(self):
        volume = np.array([
            [[0, 10, 40, 100], [150, 200, 500, 1000]],
            [[5, 20, 60, 120], [180, 250, 700, 2000]],
        ], dtype=np.uint16)
        mip = np.max(volume, axis=0).astype(float)
        normalized = percentile_normalize(mip)
        for sample, gain, threshold, explicit in (
            ("252384", 2.0, 100, False), ("251637", 2.0, 100, False),
            ("252790", 2.0, 100, False), ("252384", 1.0, 150, True),
        ):
            toolkit = Visual_toolkit.__new__(Visual_toolkit)
            toolkit.sample_id = sample
            captured = {}

            def capture(fig, ax, *args, **kwargs):
                captured["rgb"] = np.asarray(ax.images[0].get_array()).copy()
                plt.close(fig)

            toolkit._finalize_plot = capture
            kwargs = {"bg_intensity": gain, "manual_threshold": threshold} if explicit else {}
            toolkit.plot_widefield_context(
                volume, [0, 0, 0], [5, 5, 3], [5, 5, 0], "synthetic.swc", **kwargs
            )
            expected = np.repeat((normalized * gain)[..., None], 3, axis=2)
            mask = mip > threshold
            sig = mip[mask]
            s_norm = (sig - sig.min()) / (sig.max() - sig.min()) if sig.max() > sig.min() else np.zeros_like(sig)
            expected[mask, 0] *= 0.2
            expected[mask, 2] *= 0.2
            expected[mask, 1] = np.clip(gain + 0.2 + 0.4 * s_norm, 0, 1)
            np.testing.assert_allclose(captured["rgb"], expected, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
