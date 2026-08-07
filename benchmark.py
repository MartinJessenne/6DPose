"""Layer 2a: Single Benchmark Evaluation Orchestrator.

Evaluates a chosen 6D pose estimation method on a specified subset of the test split
using its default or profile hyperparameters, printing a detailed performance report.
"""

import contextlib
import dataclasses
import os

import numpy as np

import wandb
from cli_config import EvalArgs
from evaluation import derive_internal_seeds, draw_eval_indices, evaluate_pipeline
from metrics import (
    GROSS_YAW_DEG,
    FrameRecord,
    PoseErrorMetrics,
    compute_trial_metrics,
    write_frame_records_csv,
)
from pipeline import Camera
from reporting import log_frame_records, log_input_artifacts


def run_benchmark_eval(
    dataset,
    model,
    camera: Camera,
    meshes: dict,
    cfg: EvalArgs,
):
    """Orchestrates a single benchmark evaluation run across dataset samples."""
    seed = cfg.resolved_seed
    estimator_cls = cfg.estimator_cls
    sensor = cfg.camera.sensor

    total_samples = len(dataset)
    eval_indices = draw_eval_indices(total_samples, cfg.eval_size, seed)

    effective_n_seeds = cfg.n_seeds
    internal_seeds = derive_internal_seeds(seed, effective_n_seeds)

    print(
        f"Evaluating '{estimator_cls.__name__}' parameters on {len(eval_indices)} test samples..."
    )
    print(f"Seed: {seed}")
    if effective_n_seeds > 1:
        print(f"Pooled over {effective_n_seeds} internal RANSAC seeds: {internal_seeds}")
    print(f"Indices: {eval_indices}\n")

    if cfg.resolved_overrides:
        print(f"Parameter overrides applied: {cfg.resolved_overrides}")

    base_estimator = estimator_cls.build(
        profile_params=cfg.model.profile.params,
        overrides=cfg.overrides,
        sensor=sensor,
    )

    if not cfg.use_wandb:
        run_ctx = contextlib.nullcontext()
    else:
        wandb_config = {
            **dataclasses.asdict(base_estimator.params),
            "depth_trunc": cfg.depth_trunc,
            "eval_size": cfg.eval_size,
            "seed": seed,
            "n_seeds": effective_n_seeds,
        }
        run_ctx = wandb.init(
            project="6dpose",
            name=cfg.name,
            group=estimator_cls.__name__,
            job_type="benchmark",
            config=wandb_config,
        )

    with run_ctx as run:
        if run is not None:
            log_input_artifacts(run, cfg.yolo, cfg.dataset)

        error_metrics: list[PoseErrorMetrics] = []
        times: list[float] = []
        frame_records: list[FrameRecord] = []
        det_failed = 0
        pose_failed = 0
        for seed_i in internal_seeds:
            params_i = (
                dataclasses.replace(base_estimator.params, seed=seed_i)
                if seed_i is not None
                else base_estimator.params
            )
            estimator = estimator_cls(params=params_i, sensor=sensor)
            for cart_type, mesh in meshes.items():
                estimator.prepare(mesh, cart_type)

            em, t, df, pf, fr = evaluate_pipeline(
                dataset,
                model,
                camera,
                estimator,
                eval_indices,
                meshes,
                depth_trunc=cfg.depth_trunc,
            )
            error_metrics.extend(em)
            times.extend(t)
            det_failed += df
            pose_failed += pf
            frame_records.extend(fr)

        if cfg.dump_frames:
            frames_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_runs")
            frames_csv = os.path.join(frames_dir, f"{cfg.name}_frames.csv")
            write_frame_records_csv(frames_csv, frame_records)
            log_frame_records(run, cfg.name, frames_csv, frame_records)

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
        print(f"  gross_yaw_rate          {m.gross_yaw_rate:.4f}   (|yaw| > {GROSS_YAW_DEG:g}°)")
        print(f"  abstention_rate         {m.abstention_rate:.4f}   (no pose returned)")
        print(f"  detection_failure_rate  {m.detection_failure_rate:.4f}   (YOLO, upstream)")
        print("")
        print(
            f"  partition check: good {m.good_rate:.4f} + gross {m.gross_yaw_rate:.4f} "
            f"+ abstained {m.abstention_rate:.4f} = "
            f"{m.good_rate + m.gross_yaw_rate + m.abstention_rate:.4f}"
        )

        log_payload = {
            "pose_ar": m.pose_ar,
            "p95_latency_s": m.p95_latency_s if np.isfinite(m.p95_latency_s) else None,
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

        failures = [r for r in frame_records if r.outcome != "good"]
        if failures:
            by_reason: dict[str, int] = {}
            for r in failures:
                key = f"{r.outcome}/{r.failure_reason}" if r.failure_reason else r.outcome
                by_reason[key] = by_reason.get(key, 0) + 1
            print("")
            print("FAILURE BREAKDOWN:")
            for key, count in sorted(by_reason.items(), key=lambda kv: -kv[1]):
                print(f"  {key:<34} {count:>4}  ({count / m.n_eval * 100:.1f}% of evaluated)")
                log_payload[f"diag/failures/{key.replace('/', '_')}"] = count

        flipped = [r for r in frame_records if r.flipped]
        if m.n_attempted:
            flip_share = len(flipped) / m.n_attempted
            print("")
            print(f"  of which true flips (|yaw| > 90°): {len(flipped)} ({flip_share:.4f})")
            log_payload["diag/flip_share"] = flip_share

        if run is not None:
            run.log(log_payload)
        print("=" * 58)
