#!/usr/bin/env python3
"""Export tracked foreground point clouds into per-frame HDF5 files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Optional, Sequence

import h5py
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.depth_to_world_ply import prune_depth_mask, unproject_world
from scripts.foreground_segmentation_video import (
    detect_initial_boxes,
    discover_frame_paths,
    load_rgb_frame,
    parse_foreground_prompts,
    propagate_masks,
)


def positive_int(value: str) -> int:
    """Parse a strictly positive command-line integer."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args(arguments: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse tracked point-view export options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", required=True, type=Path)
    parser.add_argument("--grounding-dino-model-dir", required=True, type=Path)
    parser.add_argument("--sam2-config", required=True, type=Path)
    parser.add_argument("--sam2-checkpoint", required=True, type=Path)
    parser.add_argument("--view", type=int, default=0)
    parser.add_argument("--foreground-prompts", default="panda,ball,can,coke")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--alpha-threshold", type=float, default=0.5)
    parser.add_argument("--edge-threshold", type=float, default=0.03)
    parser.add_argument("--downsample-factor", type=positive_int, default=10)
    options = parser.parse_args(arguments)
    if options.view < 0:
        parser.error("--view must be non-negative")
    if not 0.0 <= options.alpha_threshold <= 1.0:
        parser.error("--alpha-threshold must be between 0 and 1")
    if not math.isfinite(options.edge_threshold) or options.edge_threshold < 0.0:
        parser.error("--edge-threshold must be finite and non-negative")
    if not math.isfinite(options.box_threshold) or options.box_threshold < 0.0:
        parser.error("--box-threshold must be finite and non-negative")
    parse_foreground_prompts(options.foreground_prompts)
    return options


def extract_masked_frame(
    depth: Sequence[Sequence[float]],
    alpha: Sequence[Sequence[float]],
    rgb: np.ndarray,
    tracked_mask: Sequence[Sequence[bool]],
    camera: Mapping[str, object],
    *,
    alpha_threshold: float = 0.5,
    edge_threshold: float = 0.03,
    downsample_factor: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pruned, tracked, world-space points and aligned colors."""
    if downsample_factor < 1:
        raise ValueError("downsample_factor must be at least 1")

    depth_array = np.asarray(depth, dtype=np.float32)
    alpha_array = np.asarray(alpha, dtype=np.float32)
    mask_array = np.asarray(tracked_mask, dtype=bool)
    rgb_array = np.asarray(rgb, dtype=np.uint8)
    if depth_array.ndim != 2 or alpha_array.shape != depth_array.shape:
        raise ValueError("depth and alpha must be matching two-dimensional arrays")
    if mask_array.shape != depth_array.shape:
        raise ValueError("tracked_mask must match the depth shape")
    if rgb_array.shape != (*depth_array.shape, 3):
        raise ValueError("rgb must have shape [height, width, 3]")

    pruned_mask = np.asarray(
        prune_depth_mask(
            depth_array.tolist(),
            alpha_array.tolist(),
            alpha_threshold,
            edge_threshold,
            "both",
        ),
        dtype=bool,
    )
    keep_mask = pruned_mask & mask_array
    coordinates = unproject_world(
        depth_array.tolist(),
        keep_mask.tolist(),
        float(camera["fx"]),
        float(camera["fy"]),
        camera["rotation"],
        camera["position"],
    )
    xyz = np.asarray(coordinates, dtype=np.float32).reshape((-1, 3))
    colors = rgb_array[keep_mask].reshape((-1, 3))
    return xyz[::downsample_factor], colors[::downsample_factor]


def write_point_view(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    *,
    frame: int,
    view: int,
    detected_labels: Sequence[str],
) -> None:
    """Write one variable-length point cloud with its source metadata."""
    xyz_array = np.asarray(xyz, dtype=np.float32)
    rgb_array = np.asarray(rgb, dtype=np.uint8)
    if xyz_array.ndim != 2 or xyz_array.shape[1:] != (3,):
        raise ValueError("xyz must have shape [point, 3]")
    if rgb_array.shape != xyz_array.shape:
        raise ValueError("rgb must have the same shape as xyz")

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as output:
        output.create_dataset("xyz", data=xyz_array)
        output.create_dataset("rgb", data=rgb_array)
        output.attrs["frame"] = frame
        output.attrs["view"] = view
        output.attrs["detected_labels"] = ",".join(detected_labels)


def union_tracked_masks(
    masks: Sequence[np.ndarray], image_shape: tuple[int, int]
) -> np.ndarray:
    """Return the boolean union of a frame's propagated object masks."""
    if not masks:
        return np.zeros(image_shape, dtype=bool)
    normalized = np.asarray(masks, dtype=bool)
    if normalized.ndim != 3 or normalized.shape[1:] != image_shape:
        raise ValueError(
            f"tracked masks must have shape [object, {image_shape[0]}, {image_shape[1]}]"
        )
    return np.logical_or.reduce(normalized, axis=0)


def export_point_views(
    options,
    *,
    detector=detect_initial_boxes,
    tracker=propagate_masks,
) -> list[Path]:
    """Export one tracked foreground point cloud for each rendered frame."""
    frame_paths = discover_frame_paths(options.render_dir, options.view, 0, None)
    first_frame = load_rgb_frame(frame_paths[0])
    image_shape = first_frame.shape[:2]
    prompts = parse_foreground_prompts(options.foreground_prompts)
    boxes = detector(
        first_frame,
        prompts,
        options.grounding_dino_model_dir,
        options.box_threshold,
    )
    detected_labels = [box.label for box in boxes]
    print(f"Detected labels: {detected_labels}")
    masks_by_frame = (
        tracker(
            frame_paths,
            boxes,
            options.sam2_config,
            options.sam2_checkpoint,
        )
        if boxes
        else [[] for _ in frame_paths]
    )
    if len(masks_by_frame) != len(frame_paths):
        raise ValueError("SAM2 masks must contain one entry per rendered frame")

    camera_path = options.render_dir / "cameras.json"
    if not camera_path.is_file():
        raise FileNotFoundError(f"Camera file not found: {camera_path}")
    cameras = json.loads(camera_path.read_text())
    if len(cameras) < len(frame_paths):
        raise ValueError("cameras.json must contain one camera pose per frame")

    depth_path = options.render_dir / "depth.h5"
    if not depth_path.is_file():
        raise FileNotFoundError(f"Depth file not found: {depth_path}")
    output_dir = options.render_dir / f"view_{options.view}" / "point_views"
    outputs = []
    with h5py.File(depth_path, "r") as depth_file:
        if "depth" not in depth_file or "alpha" not in depth_file:
            raise ValueError("depth.h5 must contain 'depth' and 'alpha' datasets")
        depth_data = depth_file["depth"]
        alpha_data = depth_file["alpha"]
        expected_shape = (len(frame_paths), options.view + 1, *image_shape)
        if depth_data.shape != alpha_data.shape or depth_data.shape[:2] != expected_shape[:2]:
            raise ValueError("depth and alpha must match the selected frame and view range")
        if depth_data.shape[2:] != image_shape:
            raise ValueError("depth images must match the rendered RGB dimensions")

        for frame_index, frame_path in enumerate(frame_paths):
            rgb = load_rgb_frame(frame_path)
            if rgb.shape != first_frame.shape:
                raise ValueError(
                    f"Rendered frame {frame_path} has shape {rgb.shape}; expected {first_frame.shape}"
                )
            tracked_mask = union_tracked_masks(masks_by_frame[frame_index], image_shape)
            xyz, colors = extract_masked_frame(
                depth_data[frame_index, options.view],
                alpha_data[frame_index, options.view],
                rgb,
                tracked_mask,
                cameras[frame_index],
                alpha_threshold=getattr(options, "alpha_threshold", 0.5),
                edge_threshold=getattr(options, "edge_threshold", 0.03),
                downsample_factor=options.downsample_factor,
            )
            output = output_dir / f"{frame_path.stem}.h5"
            write_point_view(
                output,
                xyz,
                colors,
                frame=frame_index,
                view=options.view,
                detected_labels=detected_labels,
            )
            outputs.append(output)
    return outputs


def main(arguments: Optional[Sequence[str]] = None) -> None:
    """Run the tracked point-view exporter."""
    outputs = export_point_views(parse_args(arguments))
    print(f"Wrote {len(outputs)} point-view files")


if __name__ == "__main__":
    main()
