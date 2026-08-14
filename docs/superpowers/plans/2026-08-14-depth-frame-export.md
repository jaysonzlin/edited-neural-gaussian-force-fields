# Depth Frame Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export raw expected camera-space depth and alpha alongside RGB output for initial and dynamic Gaussian/MPM renders.

**Architecture:** A small depth module wraps rasterization to request gsplat `RGB+ED` and return separate RGB, expected-depth, and alpha tensors. Its HDF5 writer accepts canonical float32 tensors ordered `[frame, view, height, width]`. Both rendering paths use the wrapper and pass depth plus alpha to the writer only when `--render_depth` is set.

**Tech Stack:** Python, PyTorch, h5py, gsplat, unittest.

## Global Constraints

- Use `RGB+ED`, not accumulated `RGB+D` depth.
- Preserve raw float32 depth and alpha; never normalize the HDF5 product.
- Preserve current RGB PNG and MP4 behavior.

---

### Task 1: Add the testable HDF5 writer

**Files:**

- Create: `dataset/depth_export.py`
- Create: `tests/__init__.py`
- Create: `tests/test_depth_export.py`

**Interfaces:**

- Produces `rasterize_rgb_expected_depth(rasterizer: Callable[..., tuple], **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]` and `write_depth_h5(path: str, depth: torch.Tensor, alpha: torch.Tensor) -> None`.
- Accepts two `[frame, view, height, width]` tensors and writes float32 datasets named `depth` and `alpha`.

- [ ] **Step 1: Write the failing test**

```python
class WriteDepthH5Test(unittest.TestCase):
    def test_writes_raw_float32_depth_and_alpha_in_frame_view_order(self):
        depth = torch.tensor([[[[1.25, 2.5]], [[3.75, 5.0]]]])
        alpha = torch.tensor([[[[1.0, 0.5]], [[0.25, 0.0]]]])
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "depth.h5"
            write_depth_h5(str(output_path), depth, alpha)
            with h5py.File(output_path, "r") as output:
                self.assertEqual(output["depth"].dtype.name, "float32")
                self.assertEqual(output["alpha"].dtype.name, "float32")
                self.assertEqual(output["depth"].shape, (1, 2, 1, 2))
                self.assertTrue(torch.equal(torch.from_numpy(output["depth"][...]), depth))


class RasterizeRgbExpectedDepthTest(unittest.TestCase):
    def test_requests_expected_depth_and_separates_rgb_depth_and_alpha(self):
        rendered = torch.tensor([[[[0.1, 0.2, 0.3, 4.0]]]])
        alpha = torch.tensor([[[[0.75]]]])

        def rasterizer(**kwargs):
            self.assertEqual(kwargs["render_mode"], "RGB+ED")
            return rendered, alpha, {}

        rgb, depth, rendered_alpha = rasterize_rgb_expected_depth(rasterizer, means="fixture")

        self.assertTrue(torch.equal(rgb, rendered[..., :3]))
        self.assertTrue(torch.equal(depth, torch.tensor([[[4.0]]])))
        self.assertTrue(torch.equal(rendered_alpha, torch.tensor([[[0.75]]])))
```

- [ ] **Step 2: Run it and confirm red**

Run: `python -m unittest tests.test_depth_export.WriteDepthH5Test.test_writes_raw_float32_depth_and_alpha_in_frame_view_order -v`

Expected: `ModuleNotFoundError: No module named 'dataset.depth_export'`.

- [ ] **Step 3: Add the minimal writer**

```python
def rasterize_rgb_expected_depth(rasterizer, **kwargs):
    rendering, alpha, _ = rasterizer(**kwargs, render_mode="RGB+ED")
    return rendering[..., :3], rendering[..., 3], alpha[..., 0]


def write_depth_h5(path: str, depth: torch.Tensor, alpha: torch.Tensor) -> None:
    with h5py.File(path, "w") as output:
        output.create_dataset("depth", data=depth.detach().cpu().float().numpy(), compression="gzip")
        output.create_dataset("alpha", data=alpha.detach().cpu().float().numpy(), compression="gzip")
```

- [ ] **Step 4: Run it and confirm green**

Run: `python -m unittest tests.test_depth_export.WriteDepthH5Test.test_writes_raw_float32_depth_and_alpha_in_frame_view_order -v`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add dataset/depth_export.py tests/__init__.py tests/test_depth_export.py && git commit -m "feat: add raw depth HDF5 writer"`

### Task 2: Render expected depth in both paths

**Files:**

- Modify: `dataset/render.py:26,332-379,474-506,512-547`
- Modify: `tests/test_depth_export.py`

**Interfaces:**

- Consumes `write_depth_h5(path, depth, alpha)`.
- Produces `depth.h5` with `depth` and `alpha` arranged `[frame, view, height, width]`; initial output has one frame and dynamic output has one MPM frame per entry.

- [ ] **Step 1: Implement the rendering change**

```python
rendering_img, rendering_depth, rendering_alpha = rasterize_rgb_expected_depth(
    rasterization, means=pos, ...
)
```

In the initial path, concatenate all views and call the writer with `.squeeze(-1).unsqueeze(0)` depth and alpha. In the dynamic path, collect CPU float32 depth/alpha into `[frame, view, height, width]` buffers and call the writer after the frame loop. Continue using `rendering_img` for all existing RGB code.

- [ ] **Step 2: Run focused tests and syntax check**

Run: `python -m unittest tests.test_depth_export -v && python -m py_compile dataset/render.py dataset/depth_export.py tests/test_depth_export.py`

Expected: PASS and exit code 0.

- [ ] **Step 3: Commit**

Run: `git add dataset/render.py tests/test_depth_export.py && git commit -m "feat: export expected depth frames with RGB renders"`

### Task 3: Verify delivery

**Files:**

- Verify: `dataset/render.py`, `dataset/depth_export.py`, and `tests/test_depth_export.py`

- [ ] **Step 1: Run the focused test suite**

Run: `python -m unittest tests.test_depth_export -v`

Expected: all tests pass.

- [ ] **Step 2: Inspect the final diff**

Run: `git diff HEAD~2..HEAD --check && git status --short`

Expected: no whitespace errors and no uncommitted implementation files.

- [ ] **Step 3: Match the design**

Verify both paths use `RGB+ED`; files contain raw float32 `depth` and `alpha`; initial output has one frame; dynamic output has one record per MPM frame; and RGB code still receives only the first three channels.
