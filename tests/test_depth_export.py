import tempfile
import unittest
from pathlib import Path

import h5py
import torch

from dataset.depth_export import rasterize_rgb_expected_depth, write_depth_h5


class WriteDepthH5Test(unittest.TestCase):
    def test_writes_raw_float32_depth_and_alpha_in_frame_view_order(self):
        depth = torch.tensor([[[[1.25, 2.5]], [[3.75, 5.0]]]])
        alpha = torch.tensor([[[[1.0, 0.5]], [[0.25, 0.0]]]])

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "depth.h5"
            write_depth_h5(str(output_path), depth, alpha)

            with h5py.File(output_path, "r") as output:
                self.assertEqual(output["depth"].dtype.name, "float32")
                self.assertEqual(output["alpha"].dtype.name, "float32")
                self.assertEqual(output["depth"].shape, (1, 2, 1, 2))
                self.assertTrue(
                    torch.equal(torch.from_numpy(output["depth"][...]), depth)
                )
                self.assertTrue(
                    torch.equal(torch.from_numpy(output["alpha"][...]), alpha)
                )


class RasterizeRgbExpectedDepthTest(unittest.TestCase):
    def test_requests_expected_depth_and_separates_rgb_depth_and_alpha(self):
        rendered = torch.tensor([[[[0.1, 0.2, 0.3, 4.0]]]])
        alpha = torch.tensor([[[[0.75]]]])

        def rasterizer(**kwargs):
            self.assertEqual(kwargs["render_mode"], "RGB+ED")
            return rendered, alpha, {}

        rgb, depth, rendered_alpha = rasterize_rgb_expected_depth(
            rasterizer, means="fixture"
        )

        self.assertTrue(torch.equal(rgb, rendered[..., :3]))
        self.assertTrue(torch.equal(depth, torch.tensor([[[4.0]]])))
        self.assertTrue(torch.equal(rendered_alpha, torch.tensor([[[0.75]]])))

    def test_preserves_rgb_only_rendering_when_depth_is_not_requested(self):
        rendered = torch.tensor([[[[0.1, 0.2, 0.3]]]])

        def rasterizer(**kwargs):
            self.assertEqual(kwargs["render_mode"], "RGB")
            return rendered, torch.tensor([[[[0.75]]]]), {}

        rgb, depth, rendered_alpha = rasterize_rgb_expected_depth(
            rasterizer, include_depth=False, means="fixture"
        )

        self.assertTrue(torch.equal(rgb, rendered))
        self.assertIsNone(depth)
        self.assertIsNone(rendered_alpha)
