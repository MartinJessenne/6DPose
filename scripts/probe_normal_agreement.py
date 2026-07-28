"""
Is the E1 normal-agreement term live, and does it point the right way?

Why this exists
---------------
E1 adds `n_model . n_scene > 0` to score_msac to make the score asymmetric under
the 180-degree flip. Whether that WORKS is an A/B question; whether the signal
EXISTS at all is a physics question, and it is much cheaper to answer.

The front-face gate taught this lesson the expensive way: a mechanism can be
correctly implemented, log plausible numbers, and be measuring nothing. Its
rejection-ratio instrumentation (methods/vsac_se2.py) exists for exactly that,
and this is the same check for E1.

For each fixture, at the GROUND-TRUTH pose and at its 180-degree twin (rotated
about the model centroid, so both poses occupy the same region), report:

  n_pos     positional inliers -- what score_msac counts today
  keep      fraction of those that ALSO pass the normal test

Read the two `keep` columns together:

  keep_gt high, keep_flip low   the term separates the twins. E1 should work.
  both high                     INERT. Normals agree either way -- the term
                                removes nothing and E1 is a no-op by construction.
  both low                      the sign convention disagrees between the clouds;
                                the term is destroying correct inliers.
  keep_gt low, keep_flip high   inverted somewhere. Do not trust an E1 sweep.

Note n_pos itself is the flip-blindness this whole roadmap is about: if
n_pos_gt ~= n_pos_flip, positional scoring genuinely cannot tell them apart, and
that is the defect E1 targets.

Usage
-----
    uv run scripts/probe_normal_agreement.py model:vsac3dof model.profile:tuned
"""

import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import tyro
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cli_config import LocalEvalArgs  # noqa: E402
from methods.base import prepare_scene_point_cloud  # noqa: E402
from methods.vsac_se2 import score_msac  # noqa: E402
from pipeline import (  # noqa: E402
    Camera,
    compute_ground_truth_pose,
    instance_detected,
    load_cad_meshes,
    load_hf_model,
    load_parquet_dataset,
    process_and_reconstruct,
)

SPLITS = ("test", "validation", "train")


def flip_about_centroid(T: np.ndarray, model_points: np.ndarray) -> np.ndarray:
    """T composed with a 180-degree yaw about the model's own centroid.

    About the CENTROID, not the origin: the CAD origin sits at the towing face,
    so rotating there would translate the cart by its own length and the two
    hypotheses would not compete for the same space. The centroid keeps them
    overlapping, which is the case the scorer actually has to separate.
    """
    c = model_points.mean(axis=0)
    R = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    F = np.eye(4)
    F[:3, :3] = R
    F[:3, 3] = c - R @ c
    return T @ F


def main() -> int:
    args = tyro.cli(LocalEvalArgs)
    fixtures = Path(args.fixtures_path)
    if args.o3d_seed is not None:
        o3d.utility.random.seed(args.o3d_seed)

    extrinsic = np.array(args.camera.extrinsic, dtype=np.float64)
    camera = Camera(fx=args.camera.fx, fy=args.camera.fy, cx=args.camera.cx, cy=args.camera.cy)

    model = load_hf_model(
        local_model_path=args.yolo.local_path, repo_id=args.yolo.repo, filename=args.yolo.file
    )
    meshes = load_cad_meshes()
    estimator = args.model.ESTIMATOR_CLS(params=args.model.profile.params, extrinsic=extrinsic)
    for cart_type, mesh in meshes.items():
        estimator.prepare(mesh, cart_type)

    voxel = estimator.params.voxel_size
    tau = voxel * 1.5
    print(f"voxel_size={voxel}  tau={tau:.3f}")
    print(f"\n{'split':11} {'cart':9} {'n_pos_gt':>9} {'keep_gt':>8} "
          f"{'n_pos_flip':>11} {'keep_flip':>10}")
    print("-" * 62)

    keep_gts, keep_flips, pos_ratio = [], [], []
    splits = SPLITS if args.split == "all" else (args.split,)
    for split in splits:
        ds = load_parquet_dataset(dataset_path=str(fixtures), test_glob=f"data/{split}-*.parquet")
        for i in range(len(ds)):
            row = ds[i]
            result = model(row["rgb"], retina_masks=True, verbose=False)
            if not instance_detected(result):
                continue
            cart_type, pcd, _ = process_and_reconstruct(
                row["rgb"], row["depth"], result, camera,
                depth_trunc=args.model.profile.depth_trunc, return_frame=True,
            )
            t_wc = np.asarray(row["camera_view_transform"], dtype=float).reshape(4, 4).T
            t_wk = np.asarray(row["bbox_3d_transform"][0], dtype=float).reshape(4, 4).T
            T_gt = compute_ground_truth_pose(t_wc, t_wk, extrinsic)

            prep = estimator._PREPARATION_CACHE[
                (type(estimator).__name__, cart_type, estimator._get_prep_params_key())
            ]
            model_down = prep["model_down"]
            model_pts = np.asarray(model_down.points)
            model_nrm = np.asarray(model_down.normals)

            # Identical scene path to the estimator (methods/ransac.py), P0 fix
            # included -- otherwise this would probe a convention the pipeline
            # does not use.
            scene = prepare_scene_point_cloud(pcd, extrinsic)
            scene_down = scene.voxel_down_sample(voxel)
            scene_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30)
            )
            scene_down.orient_normals_towards_camera_location(camera_location=extrinsic[:3, 3])
            scene_pts = np.asarray(scene_down.points)
            scene_nrm = np.asarray(scene_down.normals)
            tree = cKDTree(scene_pts)

            cells = []
            for T in (T_gt, flip_about_centroid(T_gt, model_pts)):
                pts = model_pts @ T[:3, :3].T + T[:3, 3]
                nrm = model_nrm @ T[:3, :3].T
                _, _, mask_pos, _ = score_msac(pts, tree, tau=tau)
                _, _, mask_nrm, _ = score_msac(
                    pts, tree, tau=tau, transformed_model_normals=nrm, scene_normals=scene_nrm
                )
                n_pos = int(mask_pos.sum())
                cells.append((n_pos, int(mask_nrm.sum()) / n_pos if n_pos else float("nan")))

            (n_gt, k_gt), (n_fl, k_fl) = cells
            keep_gts.append(k_gt)
            keep_flips.append(k_fl)
            if n_gt:
                pos_ratio.append(n_fl / n_gt)
            print(f"{split:11} {cart_type:9} {n_gt:9d} {k_gt:8.3f} {n_fl:11d} {k_fl:10.3f}")

    print("-" * 62)
    kg = np.array(keep_gts, dtype=float)
    kf = np.array(keep_flips, dtype=float)
    print(f"keep at ground truth   median {np.nanmedian(kg):.3f}")
    print(f"keep at 180-deg twin   median {np.nanmedian(kf):.3f}")
    print(f"separation (gt - flip) median {np.nanmedian(kg - kf):+.3f}")
    print(f"positional inliers, flip/gt ratio  median {np.median(pos_ratio):.3f}"
          "   (~1.0 = positional scoring is blind, which is the premise)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
