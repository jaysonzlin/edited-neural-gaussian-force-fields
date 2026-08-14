from typing import Any, Callable, Tuple

import h5py
import torch


def rasterize_rgb_expected_depth(
    rasterizer: Callable[..., Tuple[torch.Tensor, torch.Tensor, Any]], **kwargs: Any
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rendering, alpha, _ = rasterizer(**kwargs, render_mode="RGB+ED")
    return rendering[..., :3], rendering[..., 3], alpha[..., 0]


def write_depth_h5(path: str, depth: torch.Tensor, alpha: torch.Tensor) -> None:
    with h5py.File(path, "w") as output:
        output.create_dataset(
            "depth", data=depth.detach().cpu().float().numpy(), compression="gzip"
        )
        output.create_dataset(
            "alpha", data=alpha.detach().cpu().float().numpy(), compression="gzip"
        )
