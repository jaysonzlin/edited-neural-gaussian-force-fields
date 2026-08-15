#!/usr/bin/env python3
"""Export an MP4 preview of Grounding-DINO/SAM2 foreground segmentation."""

import argparse
from dataclasses import dataclass
import logging
import math
from pathlib import Path
import tempfile
from typing import Optional, Sequence


LOGGER = logging.getLogger(__name__)


def positive_int(value: str) -> int:
    """Parse a positive integer command-line option."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args(arguments: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse video-export options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", required=True, type=Path)
    parser.add_argument("--grounding-dino-model-dir", required=True, type=Path)
    parser.add_argument("--sam2-config", required=True, type=Path)
    parser.add_argument("--sam2-checkpoint", required=True, type=Path)
    parser.add_argument("--view", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--foreground-prompts", default="panda,ball,can,coke")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--fps", type=positive_int, default=24)
    parser.add_argument("--overlay-opacity", type=float, default=0.5)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args(arguments)
    if options.view < 0 or options.start_frame < 0:
        parser.error("--view and --start-frame must be non-negative")
    if options.end_frame is not None and options.end_frame < options.start_frame:
        parser.error("--end-frame must be at least --start-frame")
    if not math.isfinite(options.box_threshold) or options.box_threshold < 0.0:
        parser.error("--box-threshold must be finite and non-negative")
    if not math.isfinite(options.overlay_opacity) or not 0.0 <= options.overlay_opacity <= 1.0:
        parser.error("--overlay-opacity must be between 0 and 1")
    return options


def discover_frame_paths(
    render_dir: Path, view: int, start_frame: int, end_frame: Optional[int]
) -> list[Path]:
    """Return the selected inclusive, contiguous PNG frame range."""
    view_dir = render_dir / f"view_{view}"
    if not view_dir.is_dir():
        raise FileNotFoundError(f"Rendered view directory not found: {view_dir}")
    available = sorted(
        int(path.stem)
        for path in view_dir.glob("[0-9][0-9][0-9][0-9].png")
    )
    if not available:
        raise FileNotFoundError(f"No numbered PNG frames found in {view_dir}")
    final_frame = available[-1] if end_frame is None else end_frame
    paths = [
        view_dir / f"{frame:04d}.png"
        for frame in range(start_frame, final_frame + 1)
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing rendered frame: {missing[0]}")
    return paths


def default_output_path(options: argparse.Namespace) -> Path:
    """Return the conventional segmentation-video output path."""
    return options.render_dir / f"foreground_segmentation_view_{options.view}.mp4"


def make_progress_factory(disabled: bool):
    """Return a tqdm factory, optionally configured to emit no terminal output."""
    try:
        from tqdm.auto import tqdm
    except ImportError as error:
        raise RuntimeError("Progress reporting requires tqdm") from error
    return lambda **kwargs: tqdm(disable=disabled, **kwargs)


@dataclass(frozen=True)
class DetectedBox:
    """A labeled Grounding-DINO XYXY detection box."""

    label: str
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class AccelerationSettings:
    """CUDA inference settings selected for this exporter invocation."""

    use_bf16: bool
    use_tf32: bool


def configure_acceleration(torch_module) -> AccelerationSettings:
    """Enable supported CUDA acceleration and log every performance fallback."""
    use_bf16 = bool(torch_module.cuda.is_bf16_supported())
    capability = torch_module.cuda.get_device_capability()
    use_tf32 = capability[0] >= 8
    torch_module.backends.cuda.matmul.allow_tf32 = use_tf32
    torch_module.backends.cudnn.allow_tf32 = use_tf32
    if not use_bf16:
        LOGGER.warning("BF16 is unavailable; falling back to FP32 inference")
    if not use_tf32:
        LOGGER.info(
            "TF32 is unavailable on CUDA compute capability %s; leaving it disabled",
            capability,
        )
    LOGGER.info(
        "Segmentation acceleration: BF16=%s TF32=%s", use_bf16, use_tf32
    )
    return AccelerationSettings(use_bf16=use_bf16, use_tf32=use_tf32)


def maybe_compile_model(model, name: str, torch_module, enabled: bool):
    """Compile a model only when requested, retaining eager mode on failure."""
    if not enabled:
        return model
    try:
        compiled = torch_module.compile(model)
    except Exception as error:
        LOGGER.warning(
            "%s compilation failed; falling back to eager mode: %s", name, error
        )
        return model
    LOGGER.info("%s compiled successfully", name)
    return compiled


def parse_foreground_prompts(value: str) -> list[str]:
    """Parse one or more non-empty Grounding-DINO prompts."""
    prompts = [prompt.strip() for prompt in value.split(",") if prompt.strip()]
    if not prompts:
        raise ValueError("--foreground-prompts must contain at least one prompt")
    return prompts


def require_detections(
    boxes: Sequence[DetectedBox], prompts: Sequence[str], box_threshold: float
) -> None:
    """Raise an actionable error when the first frame has no detections."""
    if not boxes:
        raise RuntimeError(
            "No foreground detections for "
            f"{', '.join(prompts)} at box threshold {box_threshold}"
        )


def normalize_video_masks(object_masks, image_shape: tuple[int, int]):
    """Convert propagated SAM2 logit arrays into same-sized boolean masks."""
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Foreground segmentation requires numpy") from error
    masks = []
    for object_mask in object_masks.values():
        mask = np.asarray(object_mask) > 0.0
        if mask.shape != image_shape:
            mask = mask.squeeze()
        if mask.shape != image_shape:
            raise ValueError(
                f"SAM2 mask shape {mask.shape} does not match image shape {image_shape}"
            )
        masks.append(mask)
    return masks


def detect_initial_boxes(
    image,
    prompts,
    model_dir: Path,
    box_threshold: float,
    compile_models: bool = False,
    acceleration: Optional[AccelerationSettings] = None,
):
    """Run offline Grounding DINO on the first selected RGB frame."""
    try:
        import numpy as np
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as error:
        raise RuntimeError(
            "Foreground segmentation requires torch and transformers with Grounding DINO support"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("Foreground segmentation requires CUDA")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Grounding DINO model directory not found: {model_dir}")
    settings = acceleration or configure_acceleration(torch)

    height, width = image.shape[:2]
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        str(model_dir), local_files_only=True
    ).to("cuda")
    model = maybe_compile_model(model, "Grounding DINO", torch, compile_models)
    inputs = processor(
        images=np.asarray(image, dtype=np.uint8),
        text=". ".join(prompts) + ".",
        return_tensors="pt",
    ).to("cuda")
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=settings.use_bf16
    ):
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=box_threshold,
        target_sizes=[(height, width)],
    )[0]
    labels = results.get("text_labels", [])
    boxes = []
    for index, box in enumerate(results["boxes"]):
        label = str(labels[index]) if index < len(labels) else "foreground"
        boxes.append(DetectedBox(label, tuple(float(value) for value in box.tolist())))
    return boxes


def propagate_masks(
    frame_paths: Sequence[Path],
    boxes: Sequence[DetectedBox],
    sam2_config: Path,
    sam2_checkpoint: Path,
    progress_factory=None,
    compile_models: bool = False,
    acceleration: Optional[AccelerationSettings] = None,
):
    """Propagate first-frame boxes through a render sequence with SAM2."""
    try:
        import numpy as np
        import torch
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from PIL import Image
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as error:
        raise RuntimeError("Foreground segmentation requires the sam2 package") from error
    if not torch.cuda.is_available():
        raise RuntimeError("Foreground segmentation requires CUDA")
    if not sam2_config.is_file():
        raise FileNotFoundError(f"SAM2 config not found: {sam2_config}")
    if not sam2_checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {sam2_checkpoint}")
    if not frame_paths:
        raise ValueError("At least one rendered frame is required")
    if progress_factory is None:
        progress_factory = make_progress_factory(False)
    settings = acceleration or configure_acceleration(torch)

    with Image.open(frame_paths[0]) as first_image:
        image_shape = (first_image.height, first_image.width)
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        for frame_index, frame_path in enumerate(frame_paths):
            with Image.open(frame_path) as image:
                image.convert("RGB").save(temporary_path / f"{frame_index:05d}.jpg")

        global_hydra = GlobalHydra.instance()
        if global_hydra.is_initialized():
            global_hydra.clear()
        with initialize_config_dir(
            version_base=None, config_dir=str(sam2_config.parent.resolve())
        ):
            predictor = build_sam2_video_predictor(
                sam2_config.stem, str(sam2_checkpoint), device="cuda"
            )
        predictor = maybe_compile_model(predictor, "SAM2", torch, compile_models)
        progress = progress_factory(
            total=len(frame_paths), desc="SAM2 propagation", leave=True
        )
        try:
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=settings.use_bf16
            ):
                inference_state = predictor.init_state(video_path=str(temporary_path))
                for object_id, box in enumerate(boxes, start=1):
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=object_id,
                        box=np.asarray(box.xyxy, dtype=np.float32),
                    )

                propagated = [[] for _ in frame_paths]
                for frame_index, object_ids, mask_logits in predictor.propagate_in_video(
                    inference_state
                ):
                    if not 0 <= frame_index < len(frame_paths):
                        continue
                    masks = {
                        int(object_id): mask_logits[index].detach().cpu().numpy()
                        for index, object_id in enumerate(object_ids)
                    }
                    propagated[frame_index] = normalize_video_masks(masks, image_shape)
                    progress.update()
        finally:
            progress.close()
    return propagated


def load_rgb_frame(path: Path):
    """Load one render PNG as an RGB uint8 array."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Reading rendered images requires Pillow and numpy") from error
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def overlay_masks(image, masks, opacity: float):
    """Alpha-blend the union of foreground masks over an RGB image in green."""
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Foreground segmentation requires numpy") from error
    if not masks:
        return image.copy()
    union = np.logical_or.reduce(np.asarray(masks, dtype=bool), axis=0)
    result = image.astype(np.float32).copy()
    result[union] = (
        (1.0 - opacity) * result[union]
        + opacity * np.asarray([0, 255, 0], dtype=np.float32)
    )
    return np.rint(result).astype(np.uint8)


def annotate_initial_detections(image, boxes: Sequence[DetectedBox]):
    """Draw Grounding-DINO box/label annotations on an RGB image."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Drawing detections requires opencv-python") from error
    annotated = image.copy()
    for box in boxes:
        left, top, right, bottom = (int(round(value)) for value in box.xyxy)
        cv2.rectangle(annotated, (left, top), (right, bottom), (255, 255, 0), 2)
        cv2.putText(
            annotated,
            box.label,
            (left, max(0, top - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


def export_video(options: argparse.Namespace, progress_factory=None) -> Path:
    """Create an MP4 that overlays propagated foreground masks on RGB frames."""
    try:
        import imageio.v2 as imageio
        import torch
    except ImportError as error:
        raise RuntimeError("Writing MP4 files requires imageio and imageio-ffmpeg") from error

    frame_paths = discover_frame_paths(
        options.render_dir, options.view, options.start_frame, options.end_frame
    )
    if progress_factory is None:
        progress_factory = make_progress_factory(False)
    if not torch.cuda.is_available():
        raise RuntimeError("Foreground segmentation requires CUDA")
    acceleration = configure_acceleration(torch)
    first_frame = load_rgb_frame(frame_paths[0])
    prompts = parse_foreground_prompts(options.foreground_prompts)
    boxes = detect_initial_boxes(
        first_frame,
        prompts,
        options.grounding_dino_model_dir,
        options.box_threshold,
        options.compile,
        acceleration,
    )
    require_detections(boxes, prompts, options.box_threshold)
    masks_by_frame = propagate_masks(
        frame_paths,
        boxes,
        options.sam2_config,
        options.sam2_checkpoint,
        progress_factory,
        options.compile,
        acceleration,
    )
    output = options.output or default_output_path(options)
    output.parent.mkdir(parents=True, exist_ok=True)

    expected_shape = first_frame.shape
    progress = progress_factory(total=len(frame_paths), desc="MP4 encoding", leave=True)
    try:
        with imageio.get_writer(output, fps=options.fps, codec="libx264") as writer:
            for frame_index, (frame_path, masks) in enumerate(
                zip(frame_paths, masks_by_frame)
            ):
                image = load_rgb_frame(frame_path)
                if image.shape != expected_shape:
                    raise ValueError(
                        f"Rendered frame {frame_path} has shape {image.shape}; expected {expected_shape}"
                    )
                rendered = overlay_masks(image, masks, options.overlay_opacity)
                if frame_index == 0:
                    rendered = annotate_initial_detections(rendered, boxes)
                writer.append_data(rendered)
                progress.update()
    finally:
        progress.close()
    return output


def main() -> None:
    """Run the foreground-segmentation video exporter."""
    options = parse_args()
    output = export_video(options, make_progress_factory(options.no_progress))
    print(output)


if __name__ == "__main__":
    main()
