# Segmentation Video Progress Reporting

## Purpose

Make the long-running segmentation-video export observable on a cluster
terminal without changing its model or rendering output.

## Command-line interface

Add `--no-progress` to `scripts/foreground_segmentation_video.py`. Progress
bars are enabled by default and are disabled when this flag is supplied.

## Behavior

Use `tqdm` to create two sequential progress bars:

1. `SAM2 propagation` has `total=len(frame_paths)` and advances only when
   `propagate_in_video` yields a frame in the selected range. It excludes
   temporary JPEG staging, checkpoint/model construction, and DINO detection.
2. `MP4 encoding` has `total=len(frame_paths)` and advances after each RGB
   overlay frame is appended to the imageio writer.

Both bars use `leave=True` so a completed timing and rate summary remains in
the cluster log. `--no-progress` runs the same workflow with no progress-bar
output.

## Error handling and tests

The exporter must close each bar even if inference or encoding raises, using a
context manager or `try`/`finally`. Unit tests will inject a lightweight
progress-factory fake and assert the separate totals and update counts for
SAM2 propagation and MP4 encoding. No CUDA inference is required for these
tests.
