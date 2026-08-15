# Visible-Surface Edge-Pruning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a pruned, colored, world-space PLY for dynamic frame 0/view 0.

**Architecture:** Read depth, alpha, camera, and RGB inputs from the existing dynamic-render output. Apply a four-neighbour 3% relative-depth discontinuity test that removes only the farther pixel, then unproject the remaining pixel centres with the camera-to-world pose and write a binary standard PLY.

**Tech Stack:** Python, NumPy, h5py, OpenCV, plyfile.

## Global Constraints

- Start with finite positive-depth pixels where `alpha >= 0.5`.
- For each four-neighbour pair with relative depth difference above `0.03`, discard only the farther pixel.
- Retain world-space positions and RGB from frame 0/view 0.
- Write `dynamic_d/3_0/0_panda_ball_can/table6/point_cloud_0000_view_0_world_pruned.ply` without modifying the unpruned PLY.

---

### Task 1: Produce and validate the pruned PLY

**Files:**

- Create: `dynamic_d/3_0/0_panda_ball_can/table6/point_cloud_0000_view_0_world_pruned.ply`

**Interfaces:**

- Consumes `depth.h5["depth"][0, 0]`, `depth.h5["alpha"][0, 0]`, `cameras.json[0]`, and `view_0/0000.png`.
- Produces a binary PLY with `x`, `y`, `z`, `red`, `green`, and `blue` vertex properties.

- [ ] **Step 1: Write a failing output-contract check**

```python
from pathlib import Path

assert Path("dynamic_d/3_0/0_panda_ball_can/table6/point_cloud_0000_view_0_world_pruned.ply").exists()
```

- [ ] **Step 2: Run the check to confirm red**

Run: `python -c 'from pathlib import Path; assert Path("dynamic_d/3_0/0_panda_ball_can/table6/point_cloud_0000_view_0_world_pruned.ply").exists()'`

Expected: `AssertionError` because the pruned PLY does not yet exist.

- [ ] **Step 3: Generate the PLY**

Use a one-off Python export that masks `alpha < 0.5`, compares horizontal and vertical neighbour pairs using `abs(d1 - d2) / min(d1, d2) > 0.03`, marks only the farther endpoint for removal, unprojects with `point_world = point_camera @ rotation.T + position`, and writes RGB vertex fields.

- [ ] **Step 4: Validate the artifact**

Run a `plyfile.PlyData.read` check that asserts the six required properties and verifies the vertex count is greater than zero and less than 230,400.

- [ ] **Step 5: Commit the plan documentation only**

Run: `git add docs/superpowers/plans/2026-08-14-visible-surface-edge-pruning.md && git commit -m "docs: add edge pruning plan"`
