# Depth frame export design

## Goal

Export raw depth frames together with the existing RGB renders for both the
initial-scene and dynamic MPM rendering paths.

## Chosen approach

Use gsplat's `RGB+ED` render mode. Its fourth output channel is the expected
camera-space z-depth, normalized by the rendered opacity. This is the depth
value that best corresponds to the visible Gaussian-splat surface at each
pixel. The renderer will also retain alpha so consumers can identify pixels
with no rendered scene coverage.

## Output contract

When `--render_depth` is passed, each render output directory will contain a
`depth.h5` file with these datasets:

- `depth`: float32, ordered `[frame, view, height, width]`; raw expected
  camera-space z-depth with no display normalization.
- `alpha`: float32 in the same order; consumers should use it to mask empty
  pixels.

The initial-scene render is represented as one frame. Dynamic rendering
contains one entry per MPM frame. Existing `cameras.json` files remain the
camera metadata for the corresponding frame/view order. Existing RGB PNG and
MP4 outputs are unchanged.

## Scope and compatibility

`dataset/render.py` and a small testable `dataset/depth_export.py` helper
change. The renderer requests RGB and expected depth in a single rasterization
call only when the existing `--render_depth` flag is supplied; otherwise it
keeps the existing RGB-only call. The helper splits the four output channels
and writes depth only under the flag. Existing `--render_img` and
`--compile_video` behavior remains unchanged.

## Verification

Add focused regression tests for the rasterization helper's `RGB+ED` and
RGB-only contracts, plus its float32 HDF5 output. Run those tests before and
after the production change, then run the appropriate project test command or
a syntax compilation check if no full test suite is available.
