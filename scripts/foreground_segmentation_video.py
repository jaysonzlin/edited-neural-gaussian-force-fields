#!/usr/bin/env python3
"""Export an MP4 preview of Grounding-DINO/SAM2 foreground segmentation."""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import tempfile
from typing import Optional, Sequence


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


@dataclass(frozen=True)
class DetectedBox:
    """A labeled Grounding-DINO XYXY detection box."""

    label: str
    xyxy: tuple[float, float, float, float]


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


def detect_initial_boxes(image, prompts, model_dir: Path, box_threshold: float):
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

    height, width = image.shape[:2]
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        str(model_dir), local_files_only=True
    ).to("cuda")
    inputs = processor(
        images=np.asarray(image, dtype=np.uint8),
        text=". ".join(prompts) + ".",
        return_tensors="pt",
    ).to("cuda")
    with torch.no_grad():
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
    return propagated
