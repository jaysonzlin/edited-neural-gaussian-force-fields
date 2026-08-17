import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
from PIL import Image

from scripts.export_tracked_point_views import (
    export_point_views,
    extract_masked_frame,
    parse_args,
    write_point_view,
)


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


class TrackedPointViewBatchTest(unittest.TestCase):
    def test_export_uses_current_frame_camera_and_writes_zero_padded_files(self):
        with tempfile.TemporaryDirectory() as directory:
            render_dir = Path(directory)
            view_dir = render_dir / "view_0"
            view_dir.mkdir()
            Image.fromarray(
                np.asarray([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
            ).save(view_dir / "0000.png")
            Image.fromarray(
                np.asarray([[[7, 8, 9], [10, 11, 12]]], dtype=np.uint8)
            ).save(view_dir / "0001.png")
            with h5py.File(render_dir / "depth.h5", "w") as depth_file:
                depth_file.create_dataset(
                    "depth", data=np.ones((2, 1, 1, 2), dtype=np.float32)
                )
                depth_file.create_dataset(
                    "alpha", data=np.ones((2, 1, 1, 2), dtype=np.float32)
                )
            (render_dir / "cameras.json").write_text(
                json.dumps(
                    [
                        {"fx": 1.0, "fy": 1.0, "rotation": np.eye(3).tolist(), "position": [0.0, 0.0, 0.0]},
                        {"fx": 1.0, "fy": 1.0, "rotation": np.eye(3).tolist(), "position": [10.0, 0.0, 0.0]},
                    ]
                )
            )
            options = SimpleNamespace(
                render_dir=render_dir,
                view=0,
                foreground_prompts="panda,ball,can,coke",
                grounding_dino_model_dir=Path("dino"),
                sam2_config=Path("sam2.yaml"),
                sam2_checkpoint=Path("sam2.pt"),
                box_threshold=0.25,
                downsample_factor=1,
            )
            box = SimpleNamespace(label="panda", xyxy=(0.0, 0.0, 1.0, 1.0))

            outputs = export_point_views(
                options,
                detector=lambda *_: [box],
                tracker=lambda *_: [
                    [np.asarray([[True, False]])],
                    [np.asarray([[False, True]])],
                ],
            )

            self.assertEqual(
                [path.name for path in outputs], ["0000.h5", "0001.h5"]
            )
            with h5py.File(outputs[1], "r") as output:
                self.assertEqual(output["xyz"].shape, (1, 3))
                self.assertEqual(output["xyz"][0, 0], 10.0)
                self.assertEqual(output.attrs["detected_labels"], "panda")

    def test_export_prints_detected_subset_and_writes_empty_files_when_none_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            render_dir = Path(directory)
            view_dir = render_dir / "view_0"
            view_dir.mkdir()
            Image.fromarray(np.asarray([[[1, 2, 3]]], dtype=np.uint8)).save(
                view_dir / "0000.png"
            )
            with h5py.File(render_dir / "depth.h5", "w") as depth_file:
                depth_file.create_dataset(
                    "depth", data=np.ones((1, 1, 1, 1), dtype=np.float32)
                )
                depth_file.create_dataset(
                    "alpha", data=np.ones((1, 1, 1, 1), dtype=np.float32)
                )
            (render_dir / "cameras.json").write_text(
                json.dumps(
                    [{"fx": 1.0, "fy": 1.0, "rotation": np.eye(3).tolist(), "position": [0.0, 0.0, 0.0]}]
                )
            )
            options = SimpleNamespace(
                render_dir=render_dir,
                view=0,
                foreground_prompts="panda,ball,can,coke",
                grounding_dino_model_dir=Path("dino"),
                sam2_config=Path("sam2.yaml"),
                sam2_checkpoint=Path("sam2.pt"),
                box_threshold=0.25,
                downsample_factor=10,
            )

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                outputs = export_point_views(
                    options,
                    detector=lambda *_: [],
                    tracker=lambda *_: self.fail("tracker must not be called"),
                )

            self.assertIn("Detected labels: []", stdout.getvalue())
            with h5py.File(outputs[0], "r") as output:
                self.assertEqual(output["xyz"].shape, (0, 3))
                self.assertEqual(output["rgb"].shape, (0, 3))


class TrackedPointViewArgumentsTest(unittest.TestCase):
    def test_defaults_match_tracked_point_view_contract(self):
        options = parse_args(
            [
                "--render-dir",
                "render",
                "--grounding-dino-model-dir",
                "dino",
                "--sam2-config",
                "sam2.yaml",
                "--sam2-checkpoint",
                "sam2.pt",
            ]
        )

        self.assertEqual(options.view, 0)
        self.assertEqual(options.foreground_prompts, "panda,ball,can,coke")
        self.assertEqual(options.downsample_factor, 10)


class TrackedPointViewDocumentationTest(unittest.TestCase):
    def test_readme_documents_tracked_point_view_export(self):
        readme = Path("README.md").read_text()

        self.assertIn("export_tracked_point_views.py", readme)
        self.assertIn("point_views", readme)

    def test_script_can_be_invoked_directly(self):
        result = subprocess.run(
            [sys.executable, "scripts/export_tracked_point_views.py", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--render-dir", result.stdout)


if __name__ == "__main__":
    unittest.main()
