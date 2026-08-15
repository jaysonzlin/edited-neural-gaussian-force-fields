#!/usr/bin/env python3
"""Export an MP4 preview of Grounding-DINO/SAM2 foreground segmentation."""

import argparse
import math
from pathlib import Path
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
