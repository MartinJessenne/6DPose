"""
6D Pose Estimation Benchmark & Hyperparameter Tuning Suite.

This script evaluates and optimizes 6D Pose Estimation algorithms (such as PPF and RANSAC)
against ground truth labels from Isaac Sim. It runs in two distinct modes:

Usage Modes:
------------
1. Default Benchmark Evaluation (--sweep not passed, or --no-sweep)
   Evaluates a chosen method on a random subset of the test split using its default/optimized
   hyperparameters, printing a detailed performance report (Mean/Median Translation & Rotation
   errors, Match Success Rate, and Mean Execution time).

   Example (PPF):
     uv run benchmark.py --eval-size 30 model:ppf model.profile:default

   Example (RANSAC):
     uv run benchmark.py --eval-size 30 model:ransac model.profile:default

2. Hyperparameter Sweep Optimization (--sweep)
   Launches a Multi-Objective Bayesian Optimization sweep using Optuna to find the Pareto Front
   of optimal accuracy vs. speed trade-offs. The search space is dynamically configured based
   on the selected method, and results are persisted in a local SQLite database for visualization.

   Example (PPF Sweep):
     uv run benchmark.py --sweep --name PPF_Sweep --trials 50 --eval-size 30 model:ppf model.profile:default

   Example (RANSAC Sweep):
     uv run benchmark.py --sweep --name RANSAC_Sweep --trials 50 --eval-size 30 model:ransac model.profile:default

CLI Configuration Overrides:
----------------------------
  model:ppf|ransac|ransac3dof   The 6D pose estimation method to run (required, no default --
                                see docs/explanation/tyro_cli_config.md for why).
  model.profile:<name>          Tuning profile for the chosen method (required); run
                                `uv run benchmark.py model:<algo> --help` to list them.
  --sweep / --no-sweep          Flag to execute the Optuna hyperparameter sweep (default: no-sweep).
  --name NAME                   Name for this run -- the Optuna study (-> 'optuna_<NAME>.db')
                                in sweep mode, or this benchmark's W&B run name otherwise.
  --trials NUM                  Number of optimization trials to execute (default: 30).
  --eval-size NUM                Number of validation samples to evaluate per trial/benchmark (default: 20).
  --seed NUM                    Optional fixed seed to ensure reproducibility.
  --model.profile.params.<field> <value>   Override a single hyperparameter of the chosen profile.
  --model.profile.depth-trunc <value>      Override the chosen profile's depth truncation.
"""

import dataclasses
import os

import numpy as np
import open3d as o3d
import tyro

import wandb
from cli_config import BenchmarkArgs
from evaluation import derive_internal_seeds, draw_eval_indices, evaluate_pipeline
from metrics import (
    GROSS_YAW_DEG,
    FrameRecord,
    PoseErrorMetrics,
    compute_average_recall,
    compute_trial_metrics,
    extract_pose_errors,
    finite_or_none,
    write_frame_records_csv,
)
from pipeline import (
    Camera,
    load_cad_meshes,
    load_hf_model,
    load_parquet_dataset,
)
from reporting import log_input_artifacts
from run_config import resolve_param_overrides
from sweep import run_parameter_sweep


# =====================================================================
# 4. CLI ENTRY POINT
# =====================================================================
# tyro.cli(BenchmarkArgs) builds the parser directly from the BenchmarkArgs
# dataclass (cli_config.py) -- no external config file, no dynamic string
# resolution. `args.model` is already a concrete, type-checked *Preset
# instance by the time we get here: `.ESTIMATOR_CLS` is the real class object
# and `.profile.params`/`.profile.depth_trunc` are the chosen profile's
# values. See docs/explanation/tyro_cli_config.md for the full picture.
def main():
    args = tyro.cli(BenchmarkArgs)

    # Load model, camera, and dataset
    print("Loading pipeline assets...")
    model = load_hf_model(
        local_model_path=args.yolo.local_path, repo_id=args.yolo.repo, filename=args.yolo.file
    )
    camera = Camera(fx=args.camera.fx, fy=args.camera.fy, cx=args.camera.cx, cy=args.camera.cy)
    dataset = load_parquet_dataset(dataset_path=args.dataset.path, test_glob=args.dataset.test_glob)

    estimator_cls = args.model.ESTIMATOR_CLS
    # Whether this estimator's params dataclass has a 'seed' field to vary at
    # all (only Ransac3DoFParams and its subclasses do today) -- --n-seeds is
    # a no-op, not a crash, for estimators with no such field.
    supports_seed = "seed" in {f.name for f in dataclasses.fields(args.model.profile.params)}

    # Hoist CAD mesh loading
    meshes = load_cad_meshes()

    # Camera extrinsic is always present now (CameraConfig has a real default,
    # not an optional YAML key), so no None-fallback is needed here anymore.
    extrinsic = np.array(args.camera.extrinsic, dtype=np.float64)

    if args.sweep:
        run_parameter_sweep(
            dataset=dataset,
            model=model,
            camera=camera,
            study_name=args.name,
            estimator_cls=estimator_cls,
            sweep_size=args.eval_size,
            n_trials=args.trials,
            meshes=meshes,
            yolo_cfg=args.yolo,
            dataset_cfg=args.dataset,
            extrinsic=extrinsic,
            seed=args.seed,
            n_seeds=args.n_seeds,
            supports_seed=supports_seed,
            dump_frames=args.dump_frames,
            param_overrides=args.param_overrides,
        )

    else:
        # Default Evaluation mode
        seed = args.seed
        if seed is None:
            seed = int(np.random.SeedSequence().entropy % (2**31 - 1))

        np.random.seed(seed)
        o3d.utility.random.seed(seed)

        total_samples = len(dataset)
        eval_indices = draw_eval_indices(total_samples, args.eval_size, seed)

        if args.n_seeds > 1 and not supports_seed:
            print(
                f"Note: {type(args.model.profile.params).__name__} has no 'seed' field; "
                "--n-seeds > 1 has no effect for this estimator, running once."
            )
        effective_n_seeds = args.n_seeds if supports_seed else 1
        internal_seeds = derive_internal_seeds(seed, effective_n_seeds) if supports_seed else [None]

        print(
            f"Evaluating '{estimator_cls.__name__}' parameters on {len(eval_indices)} test samples..."
        )
        print(f"Seed: {seed}")
        if effective_n_seeds > 1:
            print(f"Pooled over {effective_n_seeds} internal RANSAC seeds: {internal_seeds}")
        print(f"Indices: {eval_indices}\n")

        # Checking if --param-overrides has been wrongly used in this branch
        if args.param_overrides:
            raise ValueError(
                "--param-overrides is only valid in sweep mode; it has no effect in evaluation mode. Set profile params directly instead, e.g. --model.profile.params.icp-visibility-cull"
            )

        wandb_config = {
            **dataclasses.asdict(args.model.profile.params),
            "depth_trunc": args.model.profile.depth_trunc,
            "eval_size": args.eval_size,
            "seed": seed,
            "n_seeds": effective_n_seeds,
        }
        with wandb.init(
            project="6dpose",
            name=args.name,
            group=estimator_cls.__name__,
            job_type="benchmark",
            config=wandb_config,
        ) as run:
            log_input_artifacts(run, args.yolo, args.dataset)

            # Pool results across internal seeds (a no-op loop of length 1 when
            # effective_n_seeds == 1) instead of trusting a single draw -- every
            # downstream statistic below then runs once over the pooled data.
            error_metrics: list[PoseErrorMetrics] = []
            times: list[float] = []
            frame_records: list[FrameRecord] = []
            det_failed = 0
            pose_failed = 0
            for seed_i in internal_seeds:
                params_i = (
                    dataclasses.replace(args.model.profile.params, seed=seed_i)
                    if seed_i is not None
                    else args.model.profile.params
                )
                # Directly construct the chosen preset's estimator -- no
                # _target_ string resolution needed, args.model.ESTIMATOR_CLS
                # is already the concrete class.
                estimator = estimator_cls(params=params_i, extrinsic=extrinsic)
                for cart_type, mesh in meshes.items():
                    estimator.prepare(mesh, cart_type)

                em, t, df, pf, fr = evaluate_pipeline(
                    dataset,
                    model,
                    camera,
                    estimator,
                    eval_indices,
                    meshes,
                    depth_trunc=args.model.profile.depth_trunc,
                )
                error_metrics.extend(em)
                times.extend(t)
                det_failed += df
                pose_failed += pf
                frame_records.extend(fr)

            if args.dump_frames:
                frames_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "benchmark_runs"
                )
                write_frame_records_csv(
                    os.path.join(frames_dir, f"{args.name}_frames.csv"), frame_records
                )

            m = compute_trial_metrics(error_metrics, times, det_failed, pose_failed)

            print("\n" + "=" * 58)
            print("BENCHMARK REPORT")
            print("=" * 58)
            print(f"Evaluated {m.n_eval} samples; {m.n_attempted} reached the estimator.")
            print("")
            print("THE FIVE:")
            print(f"  pose_ar                 {m.pose_ar:.4f}   (accuracy, higher is better)")
            print(
                f"  p95_latency_s           {m.p95_latency_s:.4f}   (speed, lower is better)"
                if np.isfinite(m.p95_latency_s)
                else "  p95_latency_s           n/a      (no successful estimation)"
            )
            print(
                f"  gross_yaw_rate          {m.gross_yaw_rate:.4f}   (|yaw| > {GROSS_YAW_DEG:g}°)"
            )
            print(f"  abstention_rate         {m.abstention_rate:.4f}   (no pose returned)")
            print(f"  detection_failure_rate  {m.detection_failure_rate:.4f}   (YOLO, upstream)")
            print("")
            # good + gross + abstention == 1.0 by construction; printing the sum
            # makes that visible rather than something you have to trust.
            print(
                f"  partition check: good {m.good_rate:.4f} + gross {m.gross_yaw_rate:.4f} "
                f"+ abstained {m.abstention_rate:.4f} = "
                f"{m.good_rate + m.gross_yaw_rate + m.abstention_rate:.4f}"
            )

            log_payload = {
                "pose_ar": m.pose_ar,
                "p95_latency_s": finite_or_none(m.p95_latency_s),
                "gross_yaw_rate": m.gross_yaw_rate,
                "abstention_rate": m.abstention_rate,
                "detection_failure_rate": m.detection_failure_rate,
                "diag/good_rate": m.good_rate,
                "diag/n_attempted": m.n_attempted,
                "diag/n_eval": m.n_eval,
                "diag/trans_xy_p50": m.trans_xy_p50,
                "diag/yaw_p50": m.yaw_p50,
            }

            if m.trans_xy_p50 is not None:
                print("")
                print(f"CONDITIONAL ON GOOD SAMPLES ONLY (|yaw| <= {GROSS_YAW_DEG:g}°):")
                print(f"  median XY translation error  {m.trans_xy_p50:.4f} m")
                print(f"  median yaw error             {m.yaw_p50:.2f}°")
                print("  (readable magnitudes -- NOT comparable across configs with")
                print("   different abstention rates, since the conditioning set differs)")

            # Abstention causes -- the point of the per-frame records. Answers
            # "did the estimator starve at the FPFH stage or reject its own
            # candidates?", which a single pose_failures count cannot.
            failures = [r for r in frame_records if r.outcome != "good"]
            if failures:
                by_reason: dict[str, int] = {}
                for r in failures:
                    # gross_yaw frames produced a pose, so they carry no
                    # abstention cause -- don't render a "/None" suffix for them.
                    key = f"{r.outcome}/{r.failure_reason}" if r.failure_reason else r.outcome
                    by_reason[key] = by_reason.get(key, 0) + 1
                print("")
                print("FAILURE BREAKDOWN:")
                for key, count in sorted(by_reason.items(), key=lambda kv: -kv[1]):
                    print(f"  {key:<34} {count:>4}  ({count / m.n_eval * 100:.1f}% of evaluated)")
                    log_payload[f"diag/failures/{key.replace('/', '_')}"] = count

            # True ~180 degree flips, separated from merely-imprecise poses in
            # the 15-90 degree band: same headline bucket, different root cause.
            flipped = [r for r in frame_records if r.flipped]
            if m.n_attempted:
                flip_share = len(flipped) / m.n_attempted
                print("")
                print(f"  of which true flips (|yaw| > 90°): {len(flipped)} ({flip_share:.4f})")
                log_payload["diag/flip_share"] = flip_share

            run.log(log_payload)
            print("=" * 58)


if __name__ == "__main__":
    main()
