#!/usr/bin/env python3
"""Export tracked foreground point clouds into per-frame HDF5 files."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np

from scripts.depth_to_world_ply import prune_depth_mask, unproject_world


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
