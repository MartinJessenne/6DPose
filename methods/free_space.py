"""
Free-space (visibility) consistency check for 6D pose estimation hypotheses.

A depth sensor measuring a range `d` along a ray asserts that the space between
the sensor origin and `d` is empty. A candidate pose that places solid cart
structure inside that observed free space contradicts a physical measurement and
can be rejected -- no descriptor, no normals, and no symmetry assumption
required. This is what breaks the 180-degree tie that distance-counting scores
cannot see: a cart facing the wrong way puts its body between the camera and the
face the camera actually measured.

Two properties worth stating because they are easy to get backwards:

* **Absence of a measurement is not evidence of a violation.** A pixel with no
  valid range (sensor hole, beyond depth_trunc, masked out) contributes to
  neither counter. A hypothesis that throws geometry into unmeasured space
  therefore scores 0 violations and reads as *clean*. This is why the context is
  built from the full-frame UNMASKED depth: the masked crop only knows free
  space on the cart's own silhouette, which is exactly where a 90-degree-rotated
  body does not project.
* **Occlusion is safe by construction.** An occluder makes Z_meas smaller, and a
  violation requires Z_pred < Z_meas - margin, so geometry hidden behind
  something else is never counted as violating.

The context/evaluate split exists so this can be called inside a RANSAC loop:
the extrinsic inversion and the depth conversion happen once per frame, and the
per-hypothesis cost is one matrix multiply, a divide and a gather.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from pipeline import MaskedImageFrame


@dataclass(frozen=True)
class VisibilityContext:
    """
    Per-frame constants for the free-space test, precomputed once.

    Attributes:
        depth: (H, W) full-frame unmasked depth in meters. 0 / non-finite means
            "not observed", never "observed as empty".
        fx, fy, cx, cy: ORIGINAL (uncropped) camera intrinsics. Because `depth`
            is full-frame, projected pixel coordinates index it directly; there
            is no crop offset to subtract.
        T_cam_robot: 4x4 robot-frame-to-camera-frame transform, i.e. the inverse
            of the extrinsic, inverted once here rather than per hypothesis.
    """

    depth: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    T_cam_robot: np.ndarray

    @classmethod
    def from_frame(
        cls, frame: "MaskedImageFrame", T_robot_camera: np.ndarray
    ) -> "VisibilityContext | None":
        """
        Builds a context from a pipeline frame, or returns None when the frame
        cannot support the test (no frame, no extrinsic, no full-frame depth).

        Returning None rather than raising keeps the free-space check optional at
        every call site: callers treat None as "no visibility evidence available"
        and fall back to their other criteria.
        """
        if frame is None or T_robot_camera is None:
            return None

        depth = getattr(frame, "depth_full", None)
        if depth is None:
            return None

        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()
        depth = np.ascontiguousarray(depth, dtype=np.float32)
        if depth.ndim != 2 or depth.size == 0:
            return None

        return cls(
            depth=depth,
            fx=float(frame.camera.fx),
            fy=float(frame.camera.fy),
            cx=float(frame.camera.cx),
            cy=float(frame.camera.cy),
            T_cam_robot=np.linalg.inv(np.asarray(T_robot_camera, dtype=np.float64)),
        )


def count_violations(
    ctx: VisibilityContext,
    points_cad: np.ndarray,
    T: np.ndarray,
    margin: float,
) -> tuple[int, int, float]:
    """
    Counts free-space violations for one pose hypothesis.

    Args:
        ctx: Per-frame visibility context.
        points_cad: (N, 3) model points in the CAD frame. These should be sampled
            from the FULL cart, not the front slab: the body is the geometry that
            lands in observed free space when the cart is placed the wrong way
            round, so a slab-only sample makes the test blind to the very failure
            it exists to catch.
        T: 4x4 candidate pose, CAD frame -> robot frame.
        margin: Depth tolerance in meters. A model point counts as violating only
            when it is in front of the measured surface by more than this.

    Returns:
        (n_violations, n_observed, violation_ratio). n_observed is the number of
        model points that landed on a pixel carrying a valid range -- the sample
        size behind the ratio, and the reason a caller can tell "clean" from
        "no evidence".
    """
    pts = np.asarray(points_cad, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points_cad must have shape (N, 3), got {pts.shape}")
    if pts.shape[0] == 0:
        return 0, 0, 0.0

    T = np.asarray(T, dtype=np.float64)

    # CAD -> robot -> camera, composed once so the points are transformed once.
    T_cam_cad = ctx.T_cam_robot @ T
    P_cam = pts @ T_cam_cad[:3, :3].T + T_cam_cad[:3, 3]

    X, Y, Z = P_cam[:, 0], P_cam[:, 1], P_cam[:, 2]

    in_front = Z > 1e-3
    if not np.any(in_front):
        return 0, 0, 0.0
    X, Y, Z = X[in_front], Y[in_front], Z[in_front]

    # Pinhole projection into the full frame: no crop offset, since ctx.depth is
    # the uncropped image and the intrinsics are the original ones.
    u = np.round(ctx.fx * X / Z + ctx.cx).astype(np.int64)
    v = np.round(ctx.fy * Y / Z + ctx.cy).astype(np.int64)

    h, w = ctx.depth.shape
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.any(in_bounds):
        return 0, 0, 0.0

    Z_pred = Z[in_bounds]
    Z_meas = ctx.depth[v[in_bounds], u[in_bounds]]

    observed = (Z_meas > 0.0) & np.isfinite(Z_meas)
    n_observed = int(observed.sum())
    if n_observed == 0:
        return 0, 0, 0.0

    n_violations = int((observed & (Z_pred < Z_meas - margin)).sum())
    return n_violations, n_observed, float(n_violations / n_observed)
