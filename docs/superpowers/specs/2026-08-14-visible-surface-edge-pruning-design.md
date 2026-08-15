# Visible-surface edge-pruning design

## Goal

Create a cleaner standard PLY from dynamic frame 0/view 0 by removing the
farther side of depth discontinuities while retaining the visible foreground
surface in world coordinates.

## Input

- `dynamic_d/3_0/0_panda_ball_can/table6/depth.h5`, datasets `depth[0, 0]`
  and `alpha[0, 0]`.
- `dynamic_d/3_0/0_panda_ball_can/table6/cameras.json`, camera 0.
- `dynamic_d/3_0/0_panda_ball_can/table6/view_0/0000.png` for RGB colors.

## Filter

Start with pixels having `alpha >= 0.5` and finite positive depth. Compare
each pixel with its four direct neighbours. When the relative depth difference
exceeds 3%, discard only the farther pixel; retain the nearer pixel. This
removes ground/background points behind a silhouette without eroding the
foreground contour.

## Output

Unproject retained pixels using camera 0's intrinsics and camera-to-world pose,
then write standard binary PLY vertices with `x`, `y`, `z`, `red`, `green`, and
`blue` fields to
`dynamic_d/3_0/0_panda_ball_can/table6/point_cloud_0000_view_0_world_pruned.ply`.
The original unpruned PLY remains unchanged.

## Verification

Validate the new PLY can be read, has the expected vertex properties, and has
fewer points than the unpruned 230,400-point source.
