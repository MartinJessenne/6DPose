"""
Measures whether the global registration stage is even *capable* of finding the
right pose, separately from whether it does.

Why this exists
---------------
The front-face gate turned gross-yaw errors into abstentions without producing a
single extra correct pose, and on some frames rejected 100% of thousands of
candidates. That is only explicable two ways: either RANSAC is not sampling long
enough, or the correspondences it samples from do not contain the answer. Those
demand opposite fixes, and guessing between them is how the last three mechanisms
were built.

Four numbers per frame decide it:

  n_mutual     mutual FPFH matches (the raw retrieval result)
  n_gated      what survives the z-consistency gate -- the sampling pool
  w            fraction of that pool that is CORRECT, judged against the
               ground-truth pose. This is the whole ballgame: a 2-point minimal
               solver needs two correct correspondences at once, so the chance
               per iteration is w^2 and the iterations needed for 99.9%
               confidence is log(0.001)/log(1 - w^2). At w=0.10 that is ~460; at
               w=0.02 it is ~11,500; at w=0.005 it is ~184,000.
  gt_fitness   fraction of model points landing within tau of a scene point when
               posed at ground truth. The ceiling. If this is low the answer is
               not in the scene cloud at all and no sampler can find it --
               segmentation, occlusion or depth truncation, not RANSAC.

Usage
-----
    uv run scripts/diagnose_fpfh.py model:vsac3dof model.profile:default
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
from methods.constrained_ransac import match_correspondences_fpfh  # noqa: E402
from methods.ransac3dof import front_face_angle_deg, front_face_from_mesh  # noqa: E402
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


def required_iterations(w: float, confidence: float = 0.999) -> float:
    """Iterations for `confidence` of drawing two correct correspondences at once."""
    if w <= 0.0:
        return float("inf")
    if w >= 1.0:
        return 1.0
    return float(np.log(1.0 - confidence) / np.log1p(-(w * w)))


def main() -> int:
    args = tyro.cli(LocalEvalArgs)
    fixtures = Path(args.fixtures_path)
    if args.o3d_seed is not None:
        o3d.utility.random.seed(args.o3d_seed)

    extrinsic = np.array(args.camera.extrinsic, dtype=np.float64)
    camera = Camera(fx=args.camera.fx, fy=args.camera.fy, cx=args.camera.cx, cy=args.camera.cy)
    cam_xy = extrinsic[:3, 3][:2]

    model = load_hf_model(
        local_model_path=args.yolo.local_path, repo_id=args.yolo.repo, filename=args.yolo.file
    )
    meshes = load_cad_meshes()
    estimator = args.model.ESTIMATOR_CLS(params=args.model.profile.params, extrinsic=extrinsic)
    for cart_type, mesh in meshes.items():
        estimator.prepare(mesh, cart_type)

    voxel = estimator.params.voxel_size
    tau = voxel * 1.5
    budget = estimator.params.ransac_max_iterations

    # Three tolerances, because "correct" is not one question.
    #
    #   tau = 1.5*voxel  -- the SCORER's threshold (methods/vsac_se2.py:460), so
    #       it is the right bar for "would MSAC have counted this an inlier".
    #       But it SCALES WITH THE PRESET: 3cm at voxel 0.02, 9cm at 0.06. The
    #       original run of this script compared presets on it and read the
    #       difference as a property of the preset, which it is not.
    #   TAU_ABS = 0.05 -- fixed, so preset rows are finally comparable.
    #   tau_icp = icp_max_correspondence_distance -- the REACHABILITY bar. The
    #       global stage does not have to be right, it has to land inside ICP's
    #       basin. A hypothesis 6cm off is "incorrect" at 3cm and recoverable.
    #       This is the column that reconciles this script with W&B xadco0h8,
    #       which scores yaw_median 1.10deg on 1500 frames at a preset this
    #       script previously measured at 0/18 frames carrying >= 2.
    TAU_ABS = 0.05
    tau_icp = estimator.params.icp_max_correspondence_distance

    print(
        f"voxel_size={voxel}  tau_scorer={tau:.3f}  tau_abs={TAU_ABS:.3f}  "
        f"tau_icp={tau_icp:.3f}  ransac_max_iterations={budget}"
    )
    print(
        f"\n{'frame':>5} {'cart':9} {'gt_ang':>7} {'n_scene':>7} {'n_gate':>6} "
        f"{'ok_scr':>6} {'ok_abs':>6} {'ok_icp':>6} "
        f"{'w_scr':>7} {'w_icp':>7} {'iter_scr':>10} {'iter_icp':>10} {'gt_fit':>7}"
    )
    print("-" * 110)

    rows = []
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
            mesh = meshes[cart_type]
            t_wc = np.asarray(row["camera_view_transform"], dtype=float).reshape(4, 4).T
            t_wk = np.asarray(row["bbox_3d_transform"][0], dtype=float).reshape(4, 4).T
            T_gt = compute_ground_truth_pose(t_wc, t_wk, extrinsic)

            prep = estimator._PREPARATION_CACHE[
                (type(estimator).__name__, cart_type, estimator._get_prep_params_key())
            ]
            model_down = prep["model_down"]
            model_fpfh = np.asarray(prep["model_fpfh"].data).T
            model_pts = np.asarray(model_down.points)

            # Reproduces RansacEstimator.estimate_pose's scene path exactly
            # (methods/ransac.py:139-191), so these numbers describe the real
            # pipeline rather than an approximation of it.
            scene = prepare_scene_point_cloud(pcd, extrinsic)
            scene_down = scene.voxel_down_sample(voxel)
            scene_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30)
            )
            scene_fpfh = np.asarray(
                o3d.pipelines.registration.compute_fpfh_feature(
                    scene_down,
                    o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100),
                ).data
            ).T
            scene_pts = np.asarray(scene_down.points)

            corr = match_correspondences_fpfh(model_fpfh, scene_fpfh)
            n_mutual = len(corr)

            z_off = -float(np.asarray(mesh.vertices)[:, 2].min())
            if n_mutual:
                dz = np.abs(
                    scene_pts[corr[:, 1], 2] - model_pts[corr[:, 0], 2] - z_off
                )
                gated = corr[dz < estimator.params.z_gate_threshold]
            else:
                gated = corr
            n_gated = len(gated)

            # A correspondence is CORRECT when the ground-truth pose actually maps
            # its model point onto its matched scene point. This is the quantity
            # the sampler's success probability is a function of.
            if n_gated:
                posed = model_pts[gated[:, 0]] @ T_gt[:3, :3].T + T_gt[:3, 3]
                resid = np.linalg.norm(posed - scene_pts[gated[:, 1]], axis=1)
                n_ok = int((resid < tau).sum())
                n_ok_abs = int((resid < TAU_ABS).sum())
                n_ok_icp = int((resid < tau_icp).sum())
            else:
                n_ok = n_ok_abs = n_ok_icp = 0
            w = n_ok / n_gated if n_gated else 0.0
            w_icp = n_ok_icp / n_gated if n_gated else 0.0

            # Ceiling: how much of the model the scene can support at all.
            posed_all = model_pts @ T_gt[:3, :3].T + T_gt[:3, 3]
            d, _ = cKDTree(scene_pts).query(posed_all, k=1)
            gt_fit = float((d < tau).mean())

            gt_ang = front_face_angle_deg(T_gt, front_face_from_mesh(mesh), cam_xy)
            need = required_iterations(w)
            need_icp = required_iterations(w_icp)
            rows.append((w, need, gt_fit, n_gated, n_ok, n_ok_abs, n_ok_icp, w_icp, need_icp))
            print(
                f"{i:>5} {cart_type:9} {gt_ang:7.1f} {len(scene_pts):7d} {n_gated:6d} "
                f"{n_ok:6d} {n_ok_abs:6d} {n_ok_icp:6d} {w:7.3f} {w_icp:7.3f} "
                f"{need:10,.0f} {need_icp:10,.0f} {gt_fit:7.3f}"
            )

    if not rows:
        print("no frames evaluated")
        return 1

    w_all = np.array([r[0] for r in rows])
    need_all = np.array([r[1] for r in rows])
    fit_all = np.array([r[2] for r in rows])
    gated_all = np.array([r[3] for r in rows])
    ok_all = np.array([r[4] for r in rows])
    ok_abs_all = np.array([r[5] for r in rows])
    ok_icp_all = np.array([r[6] for r in rows])
    w_icp_all = np.array([r[7] for r in rows])
    need_icp_all = np.array([r[8] for r in rows])

    print("-" * 110)
    print(f"frames                     {len(rows)}")
    print(f"n_gated       median/min   {np.median(gated_all):.0f} / {gated_all.min():.0f}")
    print(f"gt_fitness    median/min   {np.median(fit_all):.3f} / {fit_all.min():.3f}")
    print(f"budget                     {budget:,}")

    # The decisive quantity is frames carrying >= 2 correct correspondences: the
    # SE(2) minimal solver consumes two, so below that the pose is unreachable at
    # any budget. Reported at all three tolerances, because which one you pick
    # changes the answer -- and that sensitivity is itself the result.
    print(f"\n{'tolerance':<14} {'metres':>7} {'n_ok med':>9} {'w med':>7} "
          f"{'iters med':>11} {'frames >=2':>11} {'over budget':>12}")
    for label, metres, ok, ww, nd in (
        ("scorer 1.5v", tau, ok_all, w_all, need_all),
        ("absolute", TAU_ABS, ok_abs_all, None, None),
        ("ICP basin", tau_icp, ok_icp_all, w_icp_all, need_icp_all),
    ):
        w_s = f"{np.median(ww):7.3f}" if ww is not None else f"{'-':>7}"
        n_s = f"{np.median(nd):11,.0f}" if nd is not None else f"{'-':>11}"
        ob_s = f"{int((nd > budget).sum()):5d} /{len(rows):3d}" if nd is not None else f"{'-':>12}"
        print(f"{label:<14} {metres:7.3f} {np.median(ok):9.0f} {w_s} {n_s} "
              f"{int((ok >= 2).sum()):5d} /{len(rows):3d} {ob_s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
