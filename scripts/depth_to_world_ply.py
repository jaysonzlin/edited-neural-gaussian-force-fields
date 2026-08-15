#!/usr/bin/env python3
"""Export a rendered depth frame as a colored world-space binary PLY."""

import argparse
import json
import math
from pathlib import Path
import struct
from typing import Iterable, List, Optional, Sequence, Tuple


PointRecord = Tuple[float, float, float, int, int, int]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args(arguments: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", required=True, type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--view", type=int, default=0)
    parser.add_argument("--alpha-threshold", type=float, default=0.5)
    parser.add_argument("--edge-threshold", type=float, default=0.03)
    parser.add_argument(
        "--prune-mode", choices=("none", "farther", "both"), default="both"
    )
    parser.add_argument(
        "--downsample-factor",
        type=positive_int,
        default=1,
        help="Keep every Nth generated point (default: 1, keep all points).",
    )
    parser.add_argument(
        "--remove-background",
        action="store_true",
        help="Remove points whose world-space y coordinate is at or beyond the cutoff.",
    )
    parser.add_argument(
        "--background-first",
        action="store_true",
        help="Remove background points before applying --downsample-factor.",
    )
    parser.add_argument(
        "--background-y-threshold",
        type=float,
        default=5.0,
        help="World-space y cutoff used by --remove-background (default: 5.0).",
    )
    parser.add_argument(
        "--foreground-segmentation",
        action="store_true",
        help="Keep only Grounding-DINO/SAM2 foreground pixels before downsampling.",
    )
    parser.add_argument("--grounding-dino-model-dir", type=Path)
    parser.add_argument("--sam2-config", type=Path)
    parser.add_argument("--sam2-checkpoint", type=Path)
    parser.add_argument(
        "--foreground-prompts",
        default="panda,ball,can",
        help="Comma-separated foreground prompts (default: panda,ball,can).",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.25,
        help="Minimum Grounding DINO box confidence (default: 0.25).",
    )
    parser.add_argument("--output", type=Path)
    options = parser.parse_args(arguments)
    if options.frame < 0 or options.view < 0:
        parser.error("--frame and --view must be non-negative")
    if not 0.0 <= options.alpha_threshold <= 1.0:
        parser.error("--alpha-threshold must be between 0 and 1")
    if options.edge_threshold < 0.0:
        parser.error("--edge-threshold must be non-negative")
    if not math.isfinite(options.background_y_threshold):
        parser.error("--background-y-threshold must be finite")
    if not math.isfinite(options.box_threshold) or options.box_threshold < 0.0:
        parser.error("--box-threshold must be finite and non-negative")
    if options.foreground_segmentation:
        for option_name in (
            "grounding_dino_model_dir",
            "sam2_config",
            "sam2_checkpoint",
        ):
            if getattr(options, option_name) is None:
                parser.error(
                    f"--foreground-segmentation requires --{option_name.replace('_', '-')}"
                )
        try:
            parse_foreground_prompts(options.foreground_prompts)
        except ValueError as error:
            parser.error(str(error))
    return options


def prune_depth_mask(
    depth: Sequence[Sequence[float]],
    alpha: Sequence[Sequence[float]],
    alpha_threshold: float,
    edge_threshold: float,
    prune_mode: str,
) -> List[List[bool]]:
    """Return which depth pixels remain after alpha and edge pruning."""
    if len(depth) != len(alpha) or any(len(row) != len(alpha[index]) for index, row in enumerate(depth)):
        raise ValueError("depth and alpha must have identical shapes")
    if prune_mode not in {"none", "farther", "both"}:
        raise ValueError("prune_mode must be one of: none, farther, both")

    keep = [
        [math.isfinite(value) and value > 0.0 and alpha[row][column] >= alpha_threshold
         for column, value in enumerate(depth_row)]
        for row, depth_row in enumerate(depth)
    ]
    if prune_mode == "none":
        return keep

    height = len(depth)
    width = len(depth[0]) if height else 0
    remove = [[False] * width for _ in range(height)]
    for row in range(height):
        for column in range(width):
            for neighbor_row, neighbor_column in ((row, column + 1), (row + 1, column)):
                if neighbor_row >= height or neighbor_column >= width:
                    continue
                if not (keep[row][column] and keep[neighbor_row][neighbor_column]):
                    continue
                first_depth = depth[row][column]
                second_depth = depth[neighbor_row][neighbor_column]
                relative_jump = abs(first_depth - second_depth) / min(first_depth, second_depth)
                if relative_jump <= edge_threshold:
                    continue
                if prune_mode == "both":
                    remove[row][column] = True
                    remove[neighbor_row][neighbor_column] = True
                elif first_depth > second_depth:
                    remove[row][column] = True
                elif second_depth > first_depth:
                    remove[neighbor_row][neighbor_column] = True
    return [
        [keep[row][column] and not remove[row][column] for column in range(width)]
        for row in range(height)
    ]


def unproject_world(
    depth: Sequence[Sequence[float]],
    keep_mask: Sequence[Sequence[bool]],
    fx: float,
    fy: float,
    rotation_c2w: Sequence[Sequence[float]],
    position_world: Sequence[float],
) -> List[Tuple[float, float, float]]:
    """Convert kept depth pixels to world coordinates using the stored camera pose."""
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")
    if len(rotation_c2w) != 3 or any(len(row) != 3 for row in rotation_c2w):
        raise ValueError("camera rotation must be a 3x3 matrix")
    if len(position_world) != 3:
        raise ValueError("camera position must have three coordinates")
    if len(depth) != len(keep_mask) or any(
        len(row) != len(keep_mask[index]) for index, row in enumerate(depth)
    ):
        raise ValueError("depth and keep mask must have identical shapes")

    height = len(depth)
    width = len(depth[0]) if height else 0
    center_x = width / 2.0
    center_y = height / 2.0
    points = []
    for row in range(height):
        for column in range(width):
            if not keep_mask[row][column]:
                continue
            distance = depth[row][column]
            camera_x = (column - center_x) * distance / fx
            camera_y = (row - center_y) * distance / fy
            camera_z = distance
            points.append(
                (
                    rotation_c2w[0][0] * camera_x
                    + rotation_c2w[0][1] * camera_y
                    + rotation_c2w[0][2] * camera_z
                    + position_world[0],
                    rotation_c2w[1][0] * camera_x
                    + rotation_c2w[1][1] * camera_y
                    + rotation_c2w[1][2] * camera_z
                    + position_world[1],
                    rotation_c2w[2][0] * camera_x
                    + rotation_c2w[2][1] * camera_y
                    + rotation_c2w[2][2] * camera_z
                    + position_world[2],
                )
            )
    return points


def select_points(
    points: Sequence[PointRecord],
    downsample_factor: int,
    remove_background: bool,
    background_y_threshold: float,
    background_first: bool = False,
) -> List[PointRecord]:
    """Select generated points in the requested downsampling/background-removal order."""
    if remove_background and background_first:
        points = [point for point in points if point[1] < background_y_threshold]
    selected = list(points[::downsample_factor])
    if remove_background and not background_first:
        selected = [point for point in selected if point[1] < background_y_threshold]
    return selected


def parse_foreground_prompts(value: str) -> List[str]:
    """Parse one or more non-empty Grounding DINO object prompts."""
    prompts = [prompt.strip() for prompt in value.split(",") if prompt.strip()]
    if not prompts:
        raise ValueError("--foreground-prompts must contain at least one prompt")
    return prompts


def select_foreground_points(
    points: Sequence[PointRecord],
    foreground_mask: Sequence[bool],
    background_y_threshold: float,
    downsample_factor: int,
) -> List[PointRecord]:
    """Keep masked foreground below the y cutoff, then deterministically downsample."""
    if len(points) != len(foreground_mask):
        raise ValueError("foreground mask must align with generated points")
    candidates = [
        point
        for point, is_foreground in zip(points, foreground_mask)
        if is_foreground and point[1] < background_y_threshold
    ]
    return candidates[::downsample_factor]


def ground_foreground_boxes(
    image: Sequence[Sequence[Tuple[int, int, int]]],
    prompts: Sequence[str],
    model_dir: Path,
    box_threshold: float,
) -> List[Tuple[float, float, float, float]]:
    """Run an offline Grounding DINO model and return XYXY boxes."""
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

    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        str(model_dir), local_files_only=True
    ).to("cuda").eval()
    height = len(image)
    width = len(image[0]) if height else 0
    text = ". ".join(prompts) + "."
    inputs = processor(
        images=np.asarray(image, dtype=np.uint8), text=text, return_tensors="pt"
    ).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=box_threshold,
        target_sizes=[(height, width)],
    )
    return [tuple(float(value) for value in box.tolist()) for box in results[0]["boxes"]]


def segment_boxes(
    image: Sequence[Sequence[Tuple[int, int, int]]],
    boxes: Sequence[Tuple[float, float, float, float]],
    sam2_config: Path,
    sam2_checkpoint: Path,
) -> List[List[List[bool]]]:
    """Convert Grounding DINO boxes into same-size binary SAM2 masks."""
    try:
        import numpy as np
        import torch
        from hydra import initialize_config_dir
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as error:
        raise RuntimeError("Foreground segmentation requires the sam2 package") from error
    if not torch.cuda.is_available():
        raise RuntimeError("Foreground segmentation requires CUDA")
    if not sam2_config.is_file():
        raise FileNotFoundError(f"SAM2 config not found: {sam2_config}")
    if not sam2_checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {sam2_checkpoint}")

    with initialize_config_dir(
        version_base=None, config_dir=str(sam2_config.parent.resolve())
    ):
        model = build_sam2(
            sam2_config.stem, str(sam2_checkpoint), device="cuda"
        )
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(np.asarray(image, dtype=np.uint8))
    masks = []
    for box in boxes:
        mask, _, _ = predictor.predict(
            box=np.asarray(box, dtype=np.float32), multimask_output=False
        )
        masks.append(mask[0].astype(bool).tolist())
    return masks


def foreground_pixel_mask(
    image: Sequence[Sequence[Tuple[int, int, int]]],
    options: argparse.Namespace,
    *,
    detector=ground_foreground_boxes,
    segmenter=segment_boxes,
) -> List[List[bool]]:
    """Return the union of SAM2 masks for detected foreground prompts."""
    prompts = parse_foreground_prompts(options.foreground_prompts)
    boxes = detector(
        image,
        prompts,
        options.grounding_dino_model_dir,
        options.box_threshold,
    )
    if not boxes:
        raise RuntimeError(
            "No foreground detections for "
            f"{', '.join(prompts)} at box threshold {options.box_threshold}"
        )
    masks = segmenter(image, boxes, options.sam2_config, options.sam2_checkpoint)
    height = len(image)
    width = len(image[0]) if height else 0
    union = [[False] * width for _ in range(height)]
    for mask in masks:
        if len(mask) != height or any(len(row) != width for row in mask):
            raise ValueError("SAM2 mask dimensions must match the render image")
        for row in range(height):
            for column in range(width):
                union[row][column] = union[row][column] or bool(mask[row][column])
    return union


def load_depth_and_alpha(render_dir: Path, frame: int, view: int) -> Tuple[list, list]:
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("Reading depth.h5 requires h5py. Install project requirements.") from error

    depth_path = render_dir / "depth.h5"
    if not depth_path.is_file():
        raise FileNotFoundError(f"Depth file not found: {depth_path}")
    with h5py.File(depth_path, "r") as depth_file:
        if "depth" not in depth_file or "alpha" not in depth_file:
            raise ValueError("depth.h5 must contain 'depth' and 'alpha' datasets")
        depth_data = depth_file["depth"]
        alpha_data = depth_file["alpha"]
        if depth_data.shape != alpha_data.shape or len(depth_data.shape) != 4:
            raise ValueError("depth and alpha must have matching [frame, view, height, width] shapes")
        frame_count, view_count, _, _ = depth_data.shape
        if frame >= frame_count or view >= view_count:
            raise IndexError(
                f"Requested frame {frame}, view {view}; available frames 0-{frame_count - 1}, "
                f"views 0-{view_count - 1}"
            )
        return depth_data[frame, view].tolist(), alpha_data[frame, view].tolist()


def load_rgb(render_dir: Path, frame: int, view: int, width: int, height: int) -> List[List[Tuple[int, int, int]]]:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Reading rendered images requires Pillow. Install project requirements.") from error

    image_path = render_dir / f"view_{view}" / f"{frame:04d}.png"
    if not image_path.is_file():
        raise FileNotFoundError(f"Rendered image not found: {image_path}")
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        if rgb.size != (width, height):
            raise ValueError(
                f"RGB image shape {rgb.size} does not match depth shape {(width, height)}"
            )
        pixels = list(rgb.getdata())
    return [pixels[row * width : (row + 1) * width] for row in range(height)]


def write_ply(path: Path, points: Iterable[PointRecord]) -> None:
    records = list(points)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(records)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode()
    with path.open("wb") as output:
        output.write(header)
        for point in records:
            output.write(struct.pack("<fffBBB", *point))


def default_output_path(options: argparse.Namespace) -> Path:
    suffix = {"none": "world", "farther": "world_pruned", "both": "world_pruned_both_sides"}[options.prune_mode]
    name = f"point_cloud_{options.frame:04d}_view_{options.view}_{suffix}"
    if options.foreground_segmentation:
        name += "_foreground_grounded_sam2"
        if options.downsample_factor > 1:
            name += f"_downsampled_{options.downsample_factor}x"
    elif options.remove_background and options.background_first:
        name += "_foreground_only"
        if options.downsample_factor > 1:
            name += f"_downsampled_{options.downsample_factor}x"
    else:
        if options.downsample_factor > 1:
            name += f"_downsampled_{options.downsample_factor}x"
        if options.remove_background:
            name += "_foreground_only"
    return options.render_dir / f"{name}.ply"


def export_point_cloud(options: argparse.Namespace) -> Path:
    depth, alpha = load_depth_and_alpha(options.render_dir, options.frame, options.view)
    height = len(depth)
    width = len(depth[0]) if height else 0
    if not height or not width:
        raise ValueError("depth frame must not be empty")
    if any(len(row) != width for row in depth):
        raise ValueError("depth frame rows must have a consistent width")

    camera_path = options.render_dir / "cameras.json"
    if not camera_path.is_file():
        raise FileNotFoundError(f"Camera file not found: {camera_path}")
    cameras = json.loads(camera_path.read_text())
    if options.view >= len(cameras):
        raise IndexError(f"Requested view {options.view}; cameras.json has {len(cameras)} cameras")
    camera = cameras[options.view]

    keep_mask = prune_depth_mask(
        depth,
        alpha,
        options.alpha_threshold,
        options.edge_threshold,
        options.prune_mode,
    )
    coordinates = unproject_world(
        depth,
        keep_mask,
        float(camera["fx"]),
        float(camera["fy"]),
        camera["rotation"],
        camera["position"],
    )
    rgb = load_rgb(options.render_dir, options.frame, options.view, width, height)
    colors = [
        rgb[row][column]
        for row in range(height)
        for column in range(width)
        if keep_mask[row][column]
    ]
    generated = [(*coordinate, *color) for coordinate, color in zip(coordinates, colors)]
    if options.foreground_segmentation:
        mask = foreground_pixel_mask(rgb, options)
        aligned_mask = [
            mask[row][column]
            for row in range(height)
            for column in range(width)
            if keep_mask[row][column]
        ]
        selected = select_foreground_points(
            generated,
            aligned_mask,
            options.background_y_threshold,
            options.downsample_factor,
        )
    else:
        selected = select_points(
            generated,
            options.downsample_factor,
            options.remove_background,
            options.background_y_threshold,
            options.background_first,
        )
    output = options.output or default_output_path(options)
    write_ply(output, selected)
    return output


def main(arguments: Optional[Sequence[str]] = None) -> None:
    options = parse_args(arguments)
    output = export_point_cloud(options)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
