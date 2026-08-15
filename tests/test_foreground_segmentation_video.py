import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.foreground_segmentation_video import (
    default_output_path,
    discover_frame_paths,
    make_progress_factory,
    overlay_masks,
    parse_args,
    normalize_video_masks,
    require_detections,
)


class ForegroundVideoArgumentsTest(unittest.TestCase):
    def test_defaults_include_coke_and_output_name(self):
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

        self.assertEqual(options.foreground_prompts, "panda,ball,can,coke")
        self.assertEqual(options.fps, 24)
        self.assertEqual(options.overlay_opacity, 0.5)
        self.assertEqual(
            default_output_path(options),
            Path("render/foreground_segmentation_view_0.mp4"),
        )

    def test_rejects_invalid_frame_range_and_opacity(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--render-dir",
                        "render",
                        "--grounding-dino-model-dir",
                        "dino",
                        "--sam2-config",
                        "sam2.yaml",
                        "--sam2-checkpoint",
                        "sam2.pt",
                        "--start-frame",
                        "3",
                        "--end-frame",
                        "2",
                    ]
                )
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--render-dir",
                        "render",
                        "--grounding-dino-model-dir",
                        "dino",
                        "--sam2-config",
                        "sam2.yaml",
                        "--sam2-checkpoint",
                        "sam2.pt",
                        "--overlay-opacity",
                        "1.1",
                    ]
                )

    def test_parses_no_progress_option(self):
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
                "--no-progress",
            ]
        )

        self.assertTrue(options.no_progress)

    def test_discovers_inclusive_contiguous_png_range(self):
        with tempfile.TemporaryDirectory() as directory:
            view_dir = Path(directory) / "view_0"
            view_dir.mkdir()
            for frame in (0, 1, 2):
                (view_dir / f"{frame:04d}.png").touch()

            paths = discover_frame_paths(Path(directory), 0, 1, 2)

        self.assertEqual(paths, [view_dir / "0001.png", view_dir / "0002.png"])


class ForegroundVideoModelBoundaryTest(unittest.TestCase):
    def test_no_initial_detection_is_actionable(self):
        with self.assertRaisesRegex(RuntimeError, "panda, ball, can, coke.*0.25"):
            require_detections([], ["panda", "ball", "can", "coke"], 0.25)

    def test_normalizes_each_propagated_object_mask(self):
        raw = {
            1: np.asarray([[1, 0]], dtype=np.float32),
            2: np.asarray([[0, 1]], dtype=np.float32),
        }

        masks = normalize_video_masks(raw, (1, 2))

        self.assertEqual(
            [mask.tolist() for mask in masks],
            [[[True, False]], [[False, True]]],
        )


class ForegroundVideoRenderTest(unittest.TestCase):
    def test_overlays_union_mask_in_green(self):
        image = np.zeros((1, 2, 3), dtype=np.uint8)
        masks = [
            np.asarray([[True, False]]),
            np.asarray([[False, True]]),
        ]

        result = overlay_masks(image, masks, 0.5)

        self.assertEqual(result.tolist(), [[[0, 128, 0], [0, 128, 0]]])

    def test_returns_original_image_when_no_masks_exist(self):
        image = np.asarray([[[10, 20, 30]]], dtype=np.uint8)

        result = overlay_masks(image, [], 0.5)

        self.assertTrue(np.array_equal(result, image))

    def test_readme_documents_video_export(self):
        readme = Path("README.md").read_text()

        self.assertIn("foreground_segmentation_video.py", readme)
        self.assertIn("--foreground-prompts panda,ball,can,coke", readme)

    def test_disabled_progress_factory_creates_a_silent_bar(self):
        bar = make_progress_factory(True)(
            total=2, desc="SAM2 propagation", leave=True
        )
        try:
            self.assertTrue(bar.disable)
            self.assertEqual(bar.total, 2)
        finally:
            bar.close()
