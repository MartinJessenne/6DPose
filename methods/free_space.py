"""
Free-space (visibility) consistency check for 6D pose estimation hypotheses.

A depth sensor measuring distance `d` asserts that the space between the sensor
origin and `d` is empty (free space). Candidate pose hypotheses that place solid
cart structure (drawbar, front face, frame) inside observed free space violate
this physical constraint.

This module projects 3D CAD model points into the sensor's cropped depth image
and counts free-space violations to disambiguate 180-degree flipped poses.
"""

import numpy as np
import torch
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from pipeline import MaskedImageFrame


def compute_free_space_violations(
    model_points: np.ndarray,
    T: np.ndarray,
    T_robot_camera: np.ndarray,
    frame: "MaskedImageFrame",
    margin: float = 0.03,
) -> Tuple[int, int, float]:
    """
    Computes free-space consistency violations for a candidate pose hypothesis T.

    Transforms model points (in robot base link frame) back into camera coordinates using
    T_robot_camera, then projects into the cropped depth image of `frame`. A violation occurs
    when a model point's predicted camera depth Z_pred is strictly in front of the observed
    surface depth Z_meas by more than `margin` meters:
        Z_pred < Z_meas - margin

    Args:
        model_points (np.ndarray): (N, 3) 3D points sampled from the CAD model.
        T (np.ndarray): 4x4 homogeneous transformation matrix in robot base link frame.
        T_robot_camera (np.ndarray): 4x4 extrinsic transform from camera to robot base link.
        frame (MaskedImageFrame): Frame containing depth crop, crop offsets, and camera model.
        margin (float): Violation margin in meters (default: 0.03 m = 3 cm).

    Returns:
        Tuple[int, int, float]: (n_violations, n_observed, violation_ratio)
    """
    if model_points is None or len(model_points) == 0:
        return 0, 0, 0.0

    # Ensure model_points is a 2D numpy array of shape (N, 3)
    pts = np.asarray(model_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"model_points must be of shape (N, 3), got shape {pts.shape}")

    # 1. Transform model points from CAD local frame to robot base frame
    # P_robot = P_model @ R^T + t
    P_robot = pts @ T[:3, :3].T + T[:3, 3]

    # 2. Transform from robot base frame to camera frame
    # T_cam_robot = inv(T_robot_camera)
    T_cam_robot = np.linalg.inv(np.asarray(T_robot_camera, dtype=np.float64))
    P_cam = P_robot @ T_cam_robot[:3, :3].T + T_cam_robot[:3, 3]

    X, Y, Z = P_cam[:, 0], P_cam[:, 1], P_cam[:, 2]

    # Filter points in front of the camera (Z > 1 mm)
    valid_z = Z > 1e-3
    if not np.any(valid_z):
        return 0, 0, 0.0

    X, Y, Z = X[valid_z], Y[valid_z], Z[valid_z]

    # 3. Project to full-image pixel coordinates (u, v) using camera intrinsics
    fx = float(frame.camera.fx)
    fy = float(frame.camera.fy)
    cx = float(frame.camera.cx)
    cy = float(frame.camera.cy)

    u_full = (fx * X / Z) + cx
    v_full = (fy * Y / Z) + cy

    # Shift by crop bounding box offsets (xmin, ymin)
    u_crop = np.round(u_full - frame.xmin).astype(int)
    v_crop = np.round(v_full - frame.ymin).astype(int)

    # 4. Filter points within cropped depth image boundaries
    depth_data = frame.depth
    if isinstance(depth_data, torch.Tensor):
        depth_np = depth_data.detach().cpu().numpy()
    else:
        depth_np = np.asarray(depth_data)

    h_crop, w_crop = depth_np.shape

    in_bounds = (u_crop >= 0) & (u_crop < w_crop) & (v_crop >= 0) & (v_crop < h_crop)
    if not np.any(in_bounds):
        return 0, 0, 0.0

    u_valid = u_crop[in_bounds]
    v_valid = v_crop[in_bounds]
    Z_pred = Z[in_bounds]

    # 5. Lookup measured depth from sensor crop
    Z_meas = depth_np[v_valid, u_valid]

    # Valid observed points: measured depth > 0 and finite
    observed_mask = (Z_meas > 0.0) & np.isfinite(Z_meas)
    n_observed = int(observed_mask.sum())
    if n_observed == 0:
        return 0, 0, 0.0

    # Violation: model predicted point depth is in front of measured depth by > margin
    violations = observed_mask & (Z_pred < Z_meas - margin)
    n_violations = int(violations.sum())
    ratio = float(n_violations / n_observed)

    return n_violations, n_observed, ratio
