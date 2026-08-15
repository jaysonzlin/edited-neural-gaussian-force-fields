# Grounded SAM2 Segmentation Video

## Purpose

Provide a command-line script that creates an MP4 preview of foreground
segmentation across all rendered frames. It is intended to visually audit the
same Grounding DINO and SAM2 models used by the point-cloud exporter.

## Scope

Add `scripts/foreground_segmentation_video.py`. The script reads the PNG
render sequence at `<render-dir>/view_<view>/`, runs Grounding DINO once on
the first selected frame, and injects its boxes into SAM2's video predictor.
SAM2 then propagates the masks over every selected frame.

The script is a separate tool; it does not change PLY export behavior.

## Command-line interface

Required arguments:

- `--render-dir PATH`
- `--grounding-dino-model-dir PATH`
- `--sam2-config PATH`
- `--sam2-checkpoint PATH`

Optional arguments and defaults:

- `--view 0`
- `--start-frame 0`
- `--end-frame`: process through the final discovered PNG when omitted
- `--foreground-prompts panda,ball,can,coke`
- `--box-threshold 0.25`
- `--fps 24`
- `--overlay-opacity 0.5`
- `--output <render-dir>/foreground_segmentation_view_<view>.mp4`

Frames use the existing four-digit PNG convention. `--end-frame` is
inclusive, so `--start-frame 0 --end-frame 99` processes 100 frames.

## Processing and rendering

1. Discover and validate the requested contiguous frame range.
2. Load the first selected RGB frame and run offline Grounding DINO with the
   prompt list.
3. If no boxes meet the threshold, stop with an error that includes the
   prompts and threshold.
4. Initialize SAM2's video predictor from the supplied external YAML and
   checkpoint, adding each Grounding-DINO box as an object prompt on the first
   selected frame.
5. Propagate SAM2 masks across the selected frames.
6. For each RGB frame, union all object masks, alpha-blend that union in green
   over the original RGB image, and encode it into an H.264 MP4. Draw the
   Grounding-DINO boxes and prompt labels on the first output frame only.

SAM2 must receive the selected sequence in its temporary video input
directory using zero-based, consecutively numbered frame names. The output
video retains the selected-frame order and resolution.

## Errors and dependencies

The script requires CUDA, the local Grounding-DINO checkpoint directory, the
local SAM2 YAML/checkpoint, `torch`, `transformers`, `sam2`, `opencv`, and
`imageio-ffmpeg`. It must fail clearly for missing files, no CUDA, an empty or
non-contiguous frame range, invalid numeric options, failed model loading, or
no initial detections.

Hydra global state must be reset before registering the external SAM2 config
directory, matching the point-cloud exporter workaround.

## Testing

Unit tests will cover argument validation, frame-range discovery, the default
output name, default prompts (including `coke`), mask union/overlay behavior,
and the no-detection error. Heavy CUDA model inference and MP4 encoding remain
manual cluster validation steps.

