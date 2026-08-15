import contextlib
import io
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.depth_to_world_ply import (
    default_output_path,
    foreground_pixel_mask,
    parse_foreground_prompts,
    parse_args,
    prune_depth_mask,
    select_foreground_points,
    select_points,
    unproject_world,
)


class PointSelectionTest(unittest.TestCase):
    def test_parses_foreground_prompts(self):
        self.assertEqual(
            parse_foreground_prompts(" panda, ball ,can "),
            ["panda", "ball", "can"],
        )

    def test_selects_masked_thresholded_points_before_downsampling(self):
        points = [
            (0.0, 0.0, 0.0, 1, 1, 1),
            (1.0, 0.0, 0.0, 2, 2, 2),
            (2.0, 6.0, 0.0, 3, 3, 3),
            (3.0, 0.0, 0.0, 4, 4, 4),
        ]

        selected = select_foreground_points(
            points, [True, True, True, True], 5.0, 2
        )

        self.assertEqual(selected, [points[0], points[3]])

    def test_foreground_segmentation_requires_all_local_model_paths(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--render-dir", "render", "--foreground-segmentation"])

    def test_foreground_pixel_mask_unions_sam2_masks(self):
        options = SimpleNamespace(
            grounding_dino_model_dir=Path("dino"),
            sam2_config=Path("sam2.yaml"),
            sam2_checkpoint=Path("sam2.pt"),
            foreground_prompts="panda,ball,can",
            box_threshold=0.25,
        )

        mask = foreground_pixel_mask(
            [[(0, 0, 0), (0, 0, 0)]],
            options,
            detector=lambda *_: [(0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 2.0, 1.0)],
            segmenter=lambda *_: [[[True, False]], [[False, True]]],
        )

        self.assertEqual(mask, [[True, True]])

    def test_foreground_pixel_mask_rejects_no_detections(self):
        options = SimpleNamespace(
            grounding_dino_model_dir=Path("dino"),
            sam2_config=Path("sam2.yaml"),
            sam2_checkpoint=Path("sam2.pt"),
            foreground_prompts="panda,ball,can",
            box_threshold=0.25,
        )

        with self.assertRaisesRegex(RuntimeError, "panda, ball, can"):
            foreground_pixel_mask(
                [[(0, 0, 0)]],
                options,
                detector=lambda *_: [],
                segmenter=lambda *_: self.fail("segmenter must not be called"),
            )

    def test_segmented_default_output_name_is_distinct(self):
        options = parse_args(
            [
                "--render-dir",
                "render",
                "--foreground-segmentation",
                "--grounding-dino-model-dir",
                "dino",
                "--sam2-config",
                "sam2.yaml",
                "--sam2-checkpoint",
                "sam2.pt",
            ]
        )

        self.assertIn("foreground_grounded_sam2", default_output_path(options).name)

    def test_both_side_pruning_removes_each_endpoint_of_a_depth_jump(self):
        keep_mask = prune_depth_mask(
            depth=[[1.0, 2.0]],
            alpha=[[1.0, 1.0]],
            alpha_threshold=0.5,
            edge_threshold=0.03,
            prune_mode="both",
        )

        self.assertEqual(keep_mask, [[False, False]])

    def test_both_side_pruning_keeps_evaluating_chained_depth_jumps(self):
        keep_mask = prune_depth_mask(
            depth=[[1.0, 2.0, 1.0]],
            alpha=[[1.0, 1.0, 1.0]],
            alpha_threshold=0.5,
            edge_threshold=0.03,
            prune_mode="both",
        )

        self.assertEqual(keep_mask, [[False, False, False]])

    def test_unprojection_uses_camera_to_world_rotation_and_position(self):
        world = unproject_world(
            depth=[[2.0]],
            keep_mask=[[True]],
            fx=1.0,
            fy=1.0,
            rotation_c2w=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            position_world=[10.0, 0.0, 0.0],
        )

        self.assertEqual(world, [(9.0, -1.0, 2.0)])

    def test_downsamples_before_removing_background(self):
        points = [
            (0.0, 0.0, 0.0, 1, 2, 3),
            (1.0, 10.0, 0.0, 4, 5, 6),
            (2.0, 0.0, 0.0, 7, 8, 9),
            (3.0, 10.0, 0.0, 10, 11, 12),
            (4.0, 0.0, 0.0, 13, 14, 15),
        ]

        selected = select_points(
            points,
            downsample_factor=2,
            remove_background=True,
            background_y_threshold=5.0,
        )

        self.assertEqual(selected, [points[0], points[2], points[4]])

    def test_removes_background_before_downsampling_when_requested(self):
        points = [
            (0.0, 0.0, 0.0, 1, 2, 3),
            (1.0, 10.0, 0.0, 4, 5, 6),
            (2.0, 0.0, 0.0, 7, 8, 9),
            (3.0, 10.0, 0.0, 10, 11, 12),
            (4.0, 0.0, 0.0, 13, 14, 15),
        ]

        selected = select_points(
            points,
            downsample_factor=2,
            remove_background=True,
            background_y_threshold=5.0,
            background_first=True,
        )

        self.assertEqual(selected, [points[0], points[4]])

    def test_cli_exposes_opt_in_selection_options(self):
        options = parse_args(
            [
                "--render-dir",
                "render-output",
                "--downsample-factor",
                "10",
                "--remove-background",
                "--background-first",
                "--background-y-threshold",
                "5.0",
            ]
        )

        self.assertEqual(options.downsample_factor, 10)
        self.assertTrue(options.remove_background)
        self.assertTrue(options.background_first)
        self.assertEqual(options.background_y_threshold, 5.0)
        self.assertEqual(
            default_output_path(options).name,
            "point_cloud_0000_view_0_world_pruned_both_sides_"
            "foreground_only_downsampled_10x.ply",
        )


if __name__ == "__main__":
    unittest.main()
