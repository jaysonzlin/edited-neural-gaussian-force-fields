# Segmentation Video Acceleration

## Purpose

Improve A100 throughput for Grounding-DINO/SAM2 video export while preserving
an explicit, observable fallback path for less capable CUDA environments.

## Command-line interface

Add `--compile`. It is disabled by default. When supplied, the exporter tries
to apply `torch.compile` to Grounding DINO and SAM2's video predictor after
they are constructed.

## Default inference configuration

For every CUDA run, use `torch.inference_mode()` and attempt BF16 autocast for
Grounding-DINO inference and SAM2 box prompting/propagation. On CUDA devices
with compute capability 8 or newer, enable TF32 matmul and cuDNN operations.

## Fallback reporting

Use the standard logger to emit one startup summary of the active settings.
Every fallback must be logged:

- If BF16 is unsupported, run that CUDA inference section in FP32 and log a
  warning naming the fallback.
- If `--compile` is supplied but `torch.compile` raises for either model, keep
  that model eager and log a warning naming the model and exception.
- If compute capability is below 8, leave TF32 disabled and log an info
  message; this is not an error.

The existing CUDA-required behavior remains unchanged. Model loading failures,
or failures that occur after a successful compilation, are not silently
recovered.

## Testing

Unit tests will inject a fake Torch module/configuration helper to confirm
accelerated A100 settings, BF16-to-FP32 fallback logging, and eager fallback
when compile raises. CUDA model execution is validated manually on the A100.
