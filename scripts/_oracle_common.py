"""Shared, additive-only helpers for scripts/export_oracle_results.py and
scripts/export_seed_spread.py.

Both scripts need the same thing: the 18 committed fixture frames, each
turned into a scene point cloud built from the GROUND-TRUTH mask (not YOLO)
via the exact same crop -> mask -> RGBD -> point-cloud path
evaluate_pipeline() uses in production (pipeline.process_and_reconstruct),
so the only thing under test is the estimator's geometry, not the detector.

Nothing here re-derives pipeline.py's math: crop_and_mask_inputs and
point_cloud_processing are imported and called exactly as evaluation.py
calls them, with the GT mask/bbox substituted for YOLO's.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cli_config import CameraConfig
from pipeline import (
    Camera,
    MaskedImageFrame,
    compute_ground_truth_pose,
    crop_and_mask_inputs,
    load_cad_meshes,
    load_parquet_dataset,
    point_cloud_processing,
)
from scripts.export_cpp_fixtures import decode_depth, decode_mask_bool

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
SPLITS = ("test", "validation", "train")


@dataclass
class Frame:
    split: str
    row_index: int
    cart_type: str
    row: dict


def load_all_frames() -> list[Frame]:
    """All 18 committed fixture frames, in the same split/row order as
    tests/fixtures/manifest.json (verified 1:1 against it, see
    export_cpp_fixtures.main -- same assertion repeated here since this is
    an independent entry point)."""
    manifest = json.load(open(FIXTURES_DIR / "manifest.json"))
    frames = []
    for split in SPLITS:
        ds = load_parquet_dataset(dataset_path=str(FIXTURES_DIR), test_glob=f"data/{split}-*.parquet")
        prov = manifest[split]["frames"]
        assert len(ds) == len(prov), f"{split}: {len(ds)} rows but manifest lists {len(prov)}"
        for i, row in enumerate(ds):
            assert row["bbox_3d_class_name"][0] == prov[i]["cart_type"]
            frames.append(Frame(split=split, row_index=int(prov[i]["index"]), cart_type=row["bbox_3d_class_name"][0], row=row))
    return frames


def ground_truth_pose(row: dict, extrinsic: np.ndarray) -> np.ndarray:
    t_world_camera = np.asarray(row["camera_view_transform"], dtype=float).reshape(4, 4).T
    t_world_cart = np.asarray(row["bbox_3d_transform"][0], dtype=float).reshape(4, 4).T
    return compute_ground_truth_pose(t_world_camera, t_world_cart, extrinsic)


def build_gt_masked_pcd(row: dict, camera: Camera, depth_trunc: float):
    """Scene point cloud in CAMERA frame, built from the frame's own
    ground-truth mask -- the same crop_and_mask_inputs -> point_cloud_processing
    path pipeline.process_and_reconstruct() runs, with YOLO's (class, bbox,
    mask) replaced by the ground-truth ones.

    Returns (pcd, n_mask_px). pcd may be empty if the mask is empty; callers
    must handle that the way evaluate_pipeline treats an empty scene cloud
    (RansacEstimator.estimate_pose returns None, reason "empty_scene_cloud").
    """
    depth, width, height = decode_depth(row)
    mask_bool = decode_mask_bool(row, width, height)

    ys, xs = np.nonzero(mask_bool)
    n_mask_px = int(mask_bool.sum())
    if n_mask_px == 0:
        bbox = [0, 0, width, height]
    else:
        # Tight bounding box of the GT mask, playing the role YOLO's box plays
        # in process_and_reconstruct -- crop_and_mask_inputs only uses it to
        # shrink the working array and shift cx/cy; every pixel outside the
        # mask is blacked out regardless of how loose the box is.
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

    rgb_np = np.array(row["rgb"])[..., :3].copy()  # RGBA -> RGB, matches process_and_reconstruct's [H,W,3]
    orig_img_tensor = torch.from_numpy(rgb_np)
    depth_tensor = torch.from_numpy(depth.copy())
    mask_tensor = torch.from_numpy(mask_bool)

    frame = crop_and_mask_inputs(
        orig_img=orig_img_tensor,
        mask=mask_tensor,
        depth_tensor=depth_tensor,
        bbox=bbox,
        camera=camera,
    )
    pcd = point_cloud_processing(frame, depth_trunc=depth_trunc)
    return pcd, n_mask_px


def build_camera_and_sensor():
    cfg = CameraConfig()
    camera = Camera(fx=cfg.fx, fy=cfg.fy, cx=cfg.cx, cy=cfg.cy)
    return cfg, camera, cfg.sensor
