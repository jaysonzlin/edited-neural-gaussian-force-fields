# Depth-to-world PLY script design

## Goal

Provide a reusable command-line script that converts one rendered depth frame
into a colored, world-space standard PLY.

## CLI

`scripts/depth_to_world_ply.py --render-dir <directory>` accepts optional
`--frame`, `--view`, `--alpha-threshold`, `--edge-threshold`, `--prune-mode`,
and `--output` arguments. Defaults are frame 0, view 0, alpha threshold 0.5,
edge threshold 0.03, and `both` pruning mode.

## Data flow

The script reads `depth.h5`, `cameras.json`, and `view_<view>/<frame>.png`
from the render directory. It retains finite positive-depth pixels satisfying
the alpha threshold, applies no pruning, farther-side pruning, or both-side
four-neighbour relative-depth pruning, then unprojects pixels through the
stored camera-to-world pose. Output vertices contain `x`, `y`, `z`, `red`,
`green`, and `blue` fields.

## Output and errors

Unless explicitly overridden, output is written inside the render directory as
`point_cloud_<frame>_view_<view>_world_<mode>.ply`. The script rejects missing
datasets/files, out-of-range frame or view indices, invalid thresholds, and
RGB/depth shape mismatches with clear errors.

## Verification

Unit tests use a synthetic fixture to verify unprojection and each pruning
mode. An end-to-end invocation against the existing table6 render output
validates the PLY can be read and has a nonzero vertex count.
