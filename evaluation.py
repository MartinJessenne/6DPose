# =====================================================================
# 2. CORE EVALUATION PIPELINE
# =====================================================================
from __future__ import annotations

import logging
import time

import numpy as np
import open3d as o3d
from datasets import Dataset

from methods.base import BasePoseEstimator
from metrics import FrameRecord, GROSS_YAW_DEG, PoseErrorMetrics, extract_pose_errors
from pipeline import (
    Camera,
    compute_ground_truth_pose,
    instance_detected,
    process_and_reconstruct,
)


def evaluate_pipeline(
    dataset: Dataset,
    model,
    camera: Camera,
    estimator: BasePoseEstimator,
    sample_indices: list[int],
    meshes: dict[str, o3d.geometry.TriangleMesh],
    depth_trunc: float = 3.0,
) -> tuple[list[PoseErrorMetrics], list[float], int, int, list[FrameRecord]]:
    """
    Evaluates the pose estimation pipeline on a set of sample indices.

    Returns:
        tuple containing:
            - error_metrics (list[PoseErrorMetrics]): Decomposed errors for successfully matched samples.
            - times (list[float]): Pose estimation duration in seconds for each successful sample.
            - detection_failures (int): Count of samples where YOLO failed to detect a cart.
            - pose_failures (int): Count of samples where pose estimation failed.
            - frame_records (list[FrameRecord]): Per-frame outcome + failure cause +
              flip-disambiguation diagnostics for EVERY sample, for offline auditing.
    """
    error_metrics = []
    times = []
    frame_records = []
    detection_failures = 0
    pose_failures = 0

    def record_failure(sample_idx: int, cart_type: str | None, outcome: str, reason: str) -> None:
        frame_records.append(
            FrameRecord(
                sample_idx=int(sample_idx),
                cart_type=cart_type,
                outcome=outcome,
                failure_reason=reason,
            )
        )

    for sample_idx in sample_indices:
        row = dataset[int(sample_idx)]
        img = row["rgb"]
        depth_bytes = row["depth"]

        # 1. Run YOLO detection
        # We execute the 2D instance segmentation model. If no target cart instance
        # is detected, we cannot proceed with point cloud reconstruction or pose
        # registration, so we log it as a detection failure and continue to the next sample.
        result = model(img, retina_masks=True, verbose=False)
        if not instance_detected(result):
            detection_failures += 1
            record_failure(sample_idx, None, "not_detected", "yolo_no_instance")
            continue

        # 2. Segment and Reconstruct Point Cloud
        try:
            cart_type, pcd, frame = process_and_reconstruct(
                img, depth_bytes, result, camera, depth_trunc=depth_trunc, return_frame=True
            )
        except Exception:
            logging.exception(f"PointCloud processing failed for index {sample_idx}")
            pose_failures += 1
            record_failure(sample_idx, None, "abstained", "pointcloud_error")
            continue

        # Retrieve preloaded mesh
        cad_mesh = meshes.get(cart_type)
        if cad_mesh is None:
            logging.error(f"CAD mesh not found for cart type '{cart_type}' (sample {sample_idx})")
            pose_failures += 1
            record_failure(sample_idx, cart_type, "abstained", "mesh_missing")
            continue

        # 3. Perform 6D Pose Estimation with timing
        start_time = time.time()
        try:
            T_final = estimator.estimate_pose(pcd, cad_mesh, cart_type=cart_type, frame=frame)
            if T_final is None:
                # The estimator names WHICH check rejected the frame (see
                # RansacResult.reason -> RansacEstimator._last_failure_reason).
                # Estimators that aren't instrumented fall back to "estimator_none".
                reason = getattr(estimator, "_last_failure_reason", None) or "estimator_none"
                logging.error(f"Pose estimator abstained on index {sample_idx}: {reason}")
                pose_failures += 1
                record_failure(sample_idx, cart_type, "abstained", reason)
                continue
        except Exception:
            logging.exception(f"Pose estimator raised an exception for index {sample_idx}")
            pose_failures += 1
            record_failure(sample_idx, cart_type, "abstained", "estimator_exception")
            continue
        elapsed_time = time.time() - start_time

        # 4. Calculate Ground Truth pose and compare
        extrinsic = getattr(estimator, "extrinsic", None)
        if extrinsic is None:
            raise ValueError(
                "Estimator must have an extrinsic camera-to-robot transform configured."
            )

        try:
            T_world_camera = np.asarray(row["camera_view_transform"]).reshape(4, 4).T
            T_world_cart = np.asarray(row["bbox_3d_transform"][0]).reshape(4, 4).T
            T_ground_truth = compute_ground_truth_pose(
                T_world_camera, T_world_cart, T_robot_camera=extrinsic
            )
            metrics = extract_pose_errors(T_final, T_ground_truth)
            error_metrics.append(metrics)
            times.append(elapsed_time)

            diagnostics = getattr(estimator, "_last_diagnostics", None) or {}
            frame_records.append(
                FrameRecord(
                    sample_idx=int(sample_idx),
                    cart_type=cart_type,
                    outcome="good" if abs(metrics.yaw) <= GROSS_YAW_DEG else "gross_yaw",
                    # Kept alongside `outcome` on purpose: `flipped` is the
                    # near-180 degree failure mode specifically, while a
                    # "gross_yaw" outcome also catches merely-imprecise poses
                    # in the 15-90 degree band. Separating them is how we tell
                    # "still flipping" from "converging badly".
                    flipped=abs(metrics.yaw) > 90.0,
                    trans_xy=metrics.trans_xy,
                    yaw_err=metrics.yaw,
                    fitness=diagnostics.get("fitness"),
                    inlier_rmse=diagnostics.get("inlier_rmse"),
                    front_face_angle_deg=diagnostics.get("front_face_angle_deg"),
                    icp_yaw_delta_deg=diagnostics.get("icp_yaw_delta_deg"),
                )
            )
        except Exception:
            logging.exception(f"Error metric extraction failed for index {sample_idx}")
            pose_failures += 1
            record_failure(sample_idx, cart_type, "abstained", "metric_error")
            continue

    return error_metrics, times, detection_failures, pose_failures, frame_records


def draw_eval_indices(total_samples: int, size: int, seed: int) -> list[int]:
    """
    Draws a deterministic evaluation subset from the dataset.

    Uses the modern Generator API exclusively (np.random.default_rng) so a
    given seed selects the same frames regardless of call site. Sweep and
    single-run evaluation used to draw from two different PRNG streams
    (default_rng's PCG64 vs. the legacy np.random.seed/np.random.choice
    global MT19937 state) -- same seed value, silently different frames,
    which made cross-run comparisons (e.g. an A/B between two estimators)
    untrustworthy unless both happened to use the same code path.
    """
    rng = np.random.default_rng(seed)
    return rng.choice(total_samples, min(size, total_samples), replace=False).tolist()


def derive_internal_seeds(base_seed: int, n: int, salt: int = 0) -> list[int]:
    """
    Derives n distinct, reproducible estimator-internal RANSAC seeds from
    base_seed (+ an optional salt, e.g. an Optuna trial number, so different
    trials/calls don't accidentally share the same repeat seeds).

    Deliberately NOT exposed as an Optuna trial.suggest_int(...) search
    dimension: a pure-noise dimension has no smooth relationship to the
    objective and would mislead TPE/NSGA-II's surrogate model. Instead, the
    caller evaluates across these seeds and pools the results, making the
    search (or a final report) robust to seed luck instead of resting on a
    single, possibly-lucky draw.
    """
    rng = np.random.default_rng([base_seed, salt])
    return [int(s) for s in rng.integers(0, 2**31 - 1, size=n)]
