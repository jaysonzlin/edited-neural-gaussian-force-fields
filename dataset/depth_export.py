from typing import Any, Callable, Optional, Tuple

import h5py
import torch


def rasterize_rgb_expected_depth(
    rasterizer: Callable[..., Tuple[torch.Tensor, torch.Tensor, Any]],
    include_depth: bool = True,
    **kwargs: Any,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    render_mode = "RGB+ED" if include_depth else "RGB"
    rendering, alpha, _ = rasterizer(**kwargs, render_mode=render_mode)
    if not include_depth:
        return rendering, None, None
    return rendering[..., :3], rendering[..., 3], alpha[..., 0]


def write_depth_h5(path: str, depth: torch.Tensor, alpha: torch.Tensor) -> None:
    with h5py.File(path, "w") as output:
        output.create_dataset(
            "depth", data=depth.detach().cpu().float().numpy(), compression="gzip"
        )
        output.create_dataset(
            "alpha", data=alpha.detach().cpu().float().numpy(), compression="gzip"
        )
