"""
Runs the full pose pipeline over the committed fixtures and prints the headline
metrics. This is the pre-push gate.

Why this exists
---------------
Three separate flip-disambiguation mechanisms shipped, each improved a headline
rate, and each turned out to have done so by converting flips into abstentions --
every rate with a success denominator improves when you empty it. All three were
detectable on a handful of frames. None was detected, because the only way to run
the real pipeline was a multi-hour remote sweep.

18 committed frames (tests/fixtures, see scripts/fetch_test_samples.py) make that
loop seconds long and local. Nothing goes to a remote sweep until it passes here.

Read gross_yaw_rate and abstention_rate TOGETHER. A large gross-yaw improvement
paid for with abstentions is the failure mode this script exists to catch, which
is why both are printed side by side with the partition invariant.

Usage
-----
    uv run scripts/local_eval.py model:vsac3dof model.profile:default
    uv run scripts/local_eval.py --split test model:ransac3dof model.profile:default
    uv run scripts/local_eval.py --param-overrides front_face_max_angle_deg=60.0 \
        model:vsac3dof model.profile:default

Flags go BEFORE the model: token. tyro applies each argument to the subcommand
that precedes it, so `... model:vsac3dof model.profile:default --csv out.csv`
fails with "unrecognized argument" -- the flag is being offered to the profile
subcommand, which has no such field.
"""

import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import tyro

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark import (  # noqa: E402
    compute_trial_metrics,
    evaluate_pipeline,
    resolve_param_overrides,
    write_frame_records_csv,
)
from cli_config import LocalEvalArgs  # noqa: E402
from pipeline import Camera, load_cad_meshes, load_hf_model, load_parquet_dataset  # noqa: E402

SPLITS = ("test", "validation", "train")


def main() -> int:
    args = tyro.cli(LocalEvalArgs)

    fixtures = Path(args.fixtures_path)
    if not (fixtures / "data").exists():
        raise SystemExit(
            f"No fixtures at {fixtures / 'data'}. Run: uv run scripts/fetch_test_samples.py"
        )

    # Open3D's global RNG, not numpy's: prepare() builds the model cloud with
    # sample_points_uniformly, which ignores the estimator's own seed. See
    # LocalEvalArgs.o3d_seed for the measured spread when this is left unset.
    if args.o3d_seed is not None:
        o3d.utility.random.seed(args.o3d_seed)

    extrinsic = np.array(args.camera.extrinsic, dtype=np.float64)
    camera = Camera(fx=args.camera.fx, fy=args.camera.fy, cx=args.camera.cx, cy=args.camera.cy)
    estimator_cls = args.model.ESTIMATOR_CLS

    # Same validation path as the sweep: unknown names raise rather than warn,
    # so a typo cannot quietly turn a treatment arm into a second control run.
    overrides = resolve_param_overrides(estimator_cls, extrinsic, args.param_overrides)
    params = type(args.model.profile.params)(
        **{**vars(args.model.profile.params), **overrides}
    )

    print(f"model={type(args.model).__name__}  split={args.split}  overrides={overrides or '{}'}")
    model = load_hf_model(
        local_model_path=args.yolo.local_path, repo_id=args.yolo.repo, filename=args.yolo.file
    )
    meshes = load_cad_meshes()

    splits = SPLITS if args.split == "all" else (args.split,)
    all_errors, all_times, all_records = [], [], []
    det_failures = pose_failures = 0
    t0 = time.perf_counter()

    for split in splits:
        dataset = load_parquet_dataset(
            dataset_path=str(fixtures), test_glob=f"data/{split}-*.parquet"
        )
        estimator = estimator_cls(params=params, extrinsic=extrinsic)
        # Pre-prepare every mesh, exactly as benchmark.py does before its own
        # evaluate_pipeline call. Without this the first frame of each cart type
        # pays FPFH preparation inside the timed region, so p95_latency_s here
        # would not be comparable to the sweep's.
        for cart_type, mesh in meshes.items():
            estimator.prepare(mesh, cart_type)

        errors, times, det, pose, records = evaluate_pipeline(
            dataset=dataset,
            model=model,
            camera=camera,
            estimator=estimator,
            sample_indices=list(range(len(dataset))),
            meshes=meshes,
            depth_trunc=args.model.profile.depth_trunc,
        )
        all_errors += errors
        all_times += times
        all_records += records
        det_failures += det
        pose_failures += pose
        print(
            f"  {split:11s} n={len(dataset)}  matched={len(errors)} "
            f"det_fail={det} pose_fail={pose}"
        )

    m = compute_trial_metrics(all_errors, all_times, det_failures, pose_failures)

    print("\n" + "=" * 58)
    print(f"{'pose_ar':<22} {m.pose_ar:.4f}      (higher is better)")
    print(f"{'gross_yaw_rate':<22} {m.gross_yaw_rate:.4f}")
    print(f"{'abstention_rate':<22} {m.abstention_rate:.4f}   <-- read WITH gross_yaw")
    print(f"{'good_rate':<22} {m.good_rate:.4f}")
    print("-" * 58)
    print(f"{'detection_failure':<22} {m.detection_failure_rate:.4f}")
    print(f"{'p95_latency_s':<22} {m.p95_latency_s:.4f}")
    print(f"{'trans_xy_p50':<22} {m.trans_xy_p50}")
    print(f"{'yaw_p50':<22} {m.yaw_p50}")
    print(f"{'n_eval / n_attempted':<22} {m.n_eval} / {m.n_attempted}")
    print("=" * 58)
    print(
        "partition: good + gross + abstain = "
        f"{m.good_rate:.3f} + {m.gross_yaw_rate:.3f} + {m.abstention_rate:.3f} = "
        f"{m.good_rate + m.gross_yaw_rate + m.abstention_rate:.3f}"
    )
    print(f"wall clock: {time.perf_counter() - t0:.1f}s")

    if args.csv:
        write_frame_records_csv(Path(args.csv), all_records)
        print(f"per-frame CSV: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
