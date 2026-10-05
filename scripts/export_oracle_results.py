"""Task 2: run the frozen VSACSe2Estimator("tuned-vis-gate") oracle on all 18
committed fixture frames, fed the GROUND-TRUTH mask instead of YOLO, and
record what it produces as oracle_results.json.

This isolates the geometry pipeline (FPFH + PROSAC/MSAC + front-face gate +
SE(2) ICP) from the 2D detector, which is the right thing to diff a C++ port
against: a detector mismatch would be a completely different bug class from
a registration mismatch.

ADDITIVE ONLY -- imports pipeline.py / cli_config.py / methods/, never edits
them. See scripts/_oracle_common.py for the shared GT-mask -> scene-pcd path.

Run: PYTHONPATH=/home/martin/6DPose /home/martin/6DPose/.venv/bin/python scripts/export_oracle_results.py
"""

from __future__ import annotations

import dataclasses
import json
import time

import numpy as np
import tyro

from cli_config import VSACSe2ProfileSelect
from metrics import extract_pose_errors
from methods.vsac_se2 import VSACSe2Estimator
from pipeline import load_cad_meshes
from scripts._oracle_common import build_camera_and_sensor, build_gt_masked_pcd, load_all_frames

OUT_PATH = "/home/martin/martinjessenne/nxtbot_cart_pose/test/fixtures/oracle_results.json"

# The exact frozen constants from the task, reproduced here ONLY as an
# assertion target -- the values actually used come from constructing the
# profile the same way `main.py model:vsac3dof model.profile:tuned-vis-gate`
# does (see build_profile below), never hand-set.
EXPECTED = dict(
    depth_trunc=5.5,
    voxel_size=0.02,
    ransac_max_iterations=46940,
    ransac_confidence=0.999,
    rho=0.108351160884956,
    z_gate_threshold=0.34896831026780845,
    edge_length_tolerance=0.14,
    z_offset=0.01,
    front_crop_aspect=2.0,
    front_face_max_angle_deg=60.0,
    hoppe_normal_orientation=True,
    icp_visibility_cull=True,
    icp_max_correspondence_distance=0.15,
    icp_max_iterations=100,
    icp_gnc_mu_shrink=1.4,
    normal_consistency=False,
)


def build_profile():
    """Constructs the `tuned-vis-gate` profile exactly as `main.py` does for
    `model:vsac3dof model.profile:tuned-vis-gate`: tyro resolves the CLI
    subcommand token against VSACSe2ProfileSelect's Union-of-subcommands
    annotation in cli_config.py, which is where the profile's actual default
    values live (tyro.conf.subcommand(name="tuned-vis-gate", default=...)).
    No field is hand-set here -- hand-setting them would silently drift from
    cli_config.py the next time that profile is retuned.
    """
    profile = tyro.cli(VSACSe2ProfileSelect, args=["tuned-vis-gate"])
    got = dataclasses.asdict(profile.params)
    got["depth_trunc"] = profile.depth_trunc
    mismatches = {k: (v, got[k]) for k, v in EXPECTED.items() if got[k] != v}
    if mismatches:
        raise AssertionError(f"tuned-vis-gate profile drifted from the task's frozen constants: {mismatches}")
    return profile


def wrap180(deg: float) -> float:
    return float(np.degrees(np.arctan2(np.sin(np.radians(deg)), np.cos(np.radians(deg)))))


def run_one(estimator: VSACSe2Estimator, frame, camera, depth_trunc, meshes, extrinsic) -> dict:
    from scripts._oracle_common import ground_truth_pose

    row = frame.row
    T_gt = ground_truth_pose(row, extrinsic)
    cad_mesh = meshes[frame.cart_type]

    pcd, n_mask_px = build_gt_masked_pcd(row, camera, depth_trunc)

    result = {
        "split": frame.split,
        "row_index": frame.row_index,
        "cart_type": frame.cart_type,
        "n_mask_px": n_mask_px,
        "T_gt": T_gt.tolist(),
    }

    try:
        T_final = estimator.estimate_pose(pcd, cad_mesh, cart_type=frame.cart_type)
    except Exception as exc:
        result.update(success=False, abstention_reason=f"estimator_exception: {exc!r}", T_est=None)
        return result

    if T_final is None:
        reason = getattr(estimator, "_last_failure_reason", None) or "estimator_none"
        result.update(success=False, abstention_reason=reason, T_est=None)
        return result

    diagnostics = getattr(estimator, "_last_diagnostics", None) or {}
    metrics = extract_pose_errors(T_final, T_gt)

    result.update(
        success=True,
        abstention_reason=None,
        T_est=T_final.tolist(),
        effective_inlier_fraction=diagnostics.get("icp_effective_inlier_fraction"),
        robust_rmse=diagnostics.get("icp_robust_rmse"),
        median_kernel_scale=diagnostics.get("icp_median_kernel_scale"),
        trans_xy_error_m=metrics.trans_xy,
        yaw_error_deg=wrap180(metrics.yaw),
    )
    return result


def main():
    profile = build_profile()
    print("tuned-vis-gate params:", dataclasses.asdict(profile.params))
    print("depth_trunc:", profile.depth_trunc)

    cfg, camera, sensor = build_camera_and_sensor()
    extrinsic = np.array(cfg.extrinsic, dtype=float)
    meshes = load_cad_meshes()
    frames = load_all_frames()

    estimator = VSACSe2Estimator(params=profile.params, sensor=sensor)
    for cart_type, mesh in meshes.items():
        estimator.prepare(mesh, cart_type)

    results = []
    t0 = time.time()
    for fr in frames:
        t_frame0 = time.time()
        res = run_one(estimator, fr, camera, profile.depth_trunc, meshes, extrinsic)
        res["latency_s"] = time.time() - t_frame0
        results.append(res)
        status = "OK" if res["success"] else f"ABSTAIN({res['abstention_reason']})"
        extra = (
            f"trans_xy={res.get('trans_xy_error_m'):.4f}m yaw={res.get('yaw_error_deg'):.3f}deg"
            if res["success"]
            else ""
        )
        print(
            f"[{fr.split}/{fr.row_index}] {fr.cart_type:9s} {status:28s} "
            f"{res['latency_s']:.2f}s {extra}"
        )

    total = time.time() - t0
    n_success = sum(r["success"] for r in results)
    print(f"\n{n_success}/{len(results)} succeeded. Total wall-clock: {total:.1f}s")

    out = {
        "profile": "tuned-vis-gate",
        "params": dataclasses.asdict(profile.params),
        "depth_trunc": profile.depth_trunc,
        "camera": {"fx": cfg.fx, "fy": cfg.fy, "cx": cfg.cx, "cy": cfg.cy},
        "T_robot_camera": extrinsic.tolist(),
        "results": results,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
