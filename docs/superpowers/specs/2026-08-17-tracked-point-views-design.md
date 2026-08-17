# Tracked Point Views Design

## Goal

Export one variable-length HDF5 point cloud for every `view_0` frame in
`data/GSCollision/dynamic_d/3_0/0_panda_ball_can/table6`.

## Inputs And Outputs

- Read `view_0/0000.png` through `view_0/0099.png`, `depth.h5`, and
  `cameras.json`.
- Create `view_0/point_views/0000.h5` through `view_0/point_views/0099.h5`.
- Each output has `/xyz` as `float32[N, 3]` world coordinates and `/rgb` as
  `uint8[N, 3]` aligned RGB values. File attributes record the source frame,
  source view, and the labels detected on frame 0.

## Segmentation And Tracking

1. Run Grounding DINO on `0000.png` with `panda`, `ball`, `can`, and `coke`.
2. Print the detected-label list and continue with its detected subset; an
   empty detection result produces empty point clouds for every frame.
3. Initialize SAM2 from each detected first-frame box and propagate through
   all 100 frames.
4. Union the propagated object masks per frame before depth extraction.

## Point Extraction

For each frame, use `depth.h5[frame, 0]` and `alpha[frame, 0]`. Keep only
finite, positive depth with alpha at least `0.5`, then use the existing
two-sided relative-depth edge pruning at threshold `0.03`. Intersect the
pruned depth pixels with the union SAM2 mask, unproject with that frame's own
camera pose, preserve the corresponding RGB pixels, and retain every tenth
remaining point in row-major pixel order.

## Implementation Boundary

Add a focused exporter that reuses the existing Grounding DINO/SAM2 helpers
from `scripts/foreground_segmentation_video.py` and the pruning/unprojection
helpers from `scripts/depth_to_world_ply.py`. Do not change their public
behavior or refactor unrelated code.

## Validation

Unit-test mask intersection, frame-specific pose selection, deterministic
10x sampling, and HDF5 dataset dtypes/shapes. Run the exporter on the target
dataset and verify all 100 files exist, their frame attributes and schemas
match, and their point counts may vary by frame.
