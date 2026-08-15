# Grounded SAM2 Foreground PLY Export

## Goal

Extend the depth-to-world PLY exporter so a remote CUDA machine can retain only prompt-grounded foreground pixels, below a world-space y threshold, before deterministic downsampling.

## Inputs and Model Loading

The new mode is opt-in and requires all local model paths:

- `--grounding-dino-model-dir`: offline Hugging Face directory for a Grounding DINO model and processor.
- `--sam2-config`: local SAM2 YAML configuration.
- `--sam2-checkpoint`: local SAM2 checkpoint.
- `--foreground-prompts`: comma-separated text prompts, defaulting to `panda,ball,can`.
- `--box-threshold`: Grounding DINO confidence cutoff, default `0.25`.

Models load lazily only in foreground-segmentation mode. The script requires CUDA for this mode and must not download checkpoints.

## Selection Pipeline

For the selected frame and view, Grounding DINO receives the prompt list, producing boxes at or above the score threshold. SAM2 receives each retained box and produces masks. The union of those masks is intersected with the existing valid-depth/pruning mask and with `y < --background-y-threshold`. Only then does the script apply `--downsample-factor`.

If no prompt is detected above the cutoff, the export fails and identifies the requested prompts and threshold. It does not write an empty PLY.

## Integration

Reuse the repository's installed SAM2 package and its box-prompt video predictor API for the single rendered image. Load Grounding DINO with the repository-pinned Transformers 4.49 `AutoProcessor` and `AutoModelForZeroShotObjectDetection`, with `local_files_only=True`.

The existing non-segmentation commands retain their current behavior. Segmented output defaults to a filename containing `foreground_grounded_sam2` to prevent collisions with y-threshold-only foreground PLYs.

## Verification

Unit tests use injected fake detector and segmenter callables to verify prompt parsing, threshold propagation, union-mask selection, y filtering before downsampling, no-detection failure, and lazy dependency behavior. The remote CUDA run is then invoked with local model paths and its PLY header count is checked.
