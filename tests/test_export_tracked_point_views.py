import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scripts.export_tracked_point_views import extract_masked_frame, write_point_view


class TrackedPointViewExtractionTest(unittest.TestCase):
    def test_extract_masked_frame_intersects_masks_and_downsamples(self):
        depth = [[1.0, 1.0, 1.0, 1.0]]
        alpha = [[1.0, 1.0, 1.0, 1.0]]
        rgb = np.asarray(
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]],
            dtype=np.uint8,
        )
        camera = {
            "fx": 1.0,
            "fy": 1.0,
            "rotation": np.eye(3).tolist(),
            "position": [0.0, 0.0, 0.0],
        }

        xyz, colors = extract_masked_frame(
            depth,
            alpha,
            rgb,
            [[False, True, True, True]],
            camera,
            downsample_factor=2,
        )

        np.testing.assert_array_equal(colors, [[4, 5, 6], [10, 11, 12]])
        self.assertEqual(xyz.dtype, np.dtype("float32"))
        self.assertEqual(colors.dtype, np.dtype("uint8"))


class TrackedPointViewWriterTest(unittest.TestCase):
    def test_write_point_view_preserves_schema_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "0007.h5"
            write_point_view(
                path,
                np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
                np.asarray([[4, 5, 6]], dtype=np.uint8),
                frame=7,
                view=0,
                detected_labels=["panda", "coke"],
            )

            with h5py.File(path, "r") as output:
                self.assertEqual(output["xyz"].dtype, np.dtype("float32"))
                self.assertEqual(output["rgb"].dtype, np.dtype("uint8"))
                self.assertEqual(output["xyz"].shape, (1, 3))
                self.assertEqual(output["rgb"].shape, (1, 3))
                self.assertEqual(output.attrs["frame"], 7)
                self.assertEqual(output.attrs["view"], 0)
                self.assertEqual(output.attrs["detected_labels"], "panda,coke")


if __name__ == "__main__":
    unittest.main()
