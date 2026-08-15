import unittest

from scripts.depth_to_world_ply import (
    default_output_path,
    parse_args,
    prune_depth_mask,
    select_points,
    unproject_world,
)


class PointSelectionTest(unittest.TestCase):
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
