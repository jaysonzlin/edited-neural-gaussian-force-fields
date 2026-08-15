# Background-First Point-Cloud Export

## Goal

Allow the depth-to-world PLY exporter to remove world-space background points before deterministic downsampling. Generate the table6 frame-0, view-0 output at a factor of 50 using that order.

## Interface

Add a `--background-first` boolean flag to `scripts/depth_to_world_ply.py`.

- The flag is meaningful with `--remove-background`.
- Without it, the existing order remains unchanged: downsample first, then remove background.
- With it, the exporter removes points whose `y` coordinate is at or above `--background-y-threshold`, then retains every Nth remaining point.

## Output Naming

The default output name must encode the selected order to prevent collision with the existing downsample-first output. A background-first export therefore uses:

`point_cloud_0000_view_0_world_pruned_both_sides_foreground_only_downsampled_50x.ply`

## Verification

Add a unit test with alternating foreground and background points, where background-first selection produces a different result from the default. Run the exporter against table6 with `--downsample-factor 50 --remove-background --background-first`, then read the PLY header to report its vertex count.
