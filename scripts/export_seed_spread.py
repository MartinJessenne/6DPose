"""Task 3: measure the RNG tolerance a C++ port should be diffed against.

The C++ RANSAC will not reproduce the Python bit-for-bit -- different
languages draw different sample sequences from "the same" seed, and even the
SAME seed in Python moves the sampler's internal state differently between
platforms/versions. So the per-frame pass/fail threshold for a Python-vs-C++
diff must be MEASURED as the natural spread the Python oracle already shows
across ITS OWN seeds, not guessed.

Method: re-run VSACSe2Estimator("tuned-vis-gate") over all 18 fixture frames
for N different values of VSACSe2Params.seed (the only RNG entry point --
np.random.default_rng(self.params.seed) in methods/vsac_se2.py's
_global_registration). For each frame, over the seeds where the estimator
succeeded, compute the spread of the estimated planar translation (x, y) and
yaw around their own per-frame median. Frames whose pose flips >90 degrees
between seeds, or whose success/abstention outcome changes across seeds, are
flagged unstable and excluded from the pooled tolerance (see module
docstring rationale in the report).

CHECKPOINTED: this is the expensive step (18 frames x 20 seeds x up to 46940
RANSAC iterations, ~20 minutes total). Every single frame's result is
written to CHECKPOINT_PATH as soon as it's computed, and re-running this
script skips whatever the checkpoint already has -- so it is safe to run it
repeatedly (e.g. across several sub-10-minute foreground calls) until it
reports DONE, rather than needing one uninterrupted ~20 minute process.

ADDITIVE ONLY. Run:
    PYTHONPATH=/home/martin/6DPose /home/martin/6DPose/.venv/bin/python -u scripts/export_seed_spread.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import time

import numpy as np

from evaluation import derive_internal_seeds
from methods.vsac_se2 import VSACSe2Estimator
from pipeline import load_cad_meshes
from scripts._oracle_common import build_camera_and_sensor, build_gt_masked_pcd, load_all_frames
from scripts.export_oracle_results import build_profile

OUT_PATH = "/home/martin/martinjessenne/nxtbot_cart_pose/test/fixtures/oracle_seed_spread.json"
# NOTE: originally pointed at a session-scoped /tmp dir from a prior killed run,
# which does not survive across sessions. Repointed at the durable checkpoint
# that already lives under the fixtures dir (git-tracked-adjacent, not /tmp) so
# resumption actually finds the 6.9 completed seeds instead of starting over.
CHECKPOINT_PATH = (
    "/home/martin/martinjessenne/nxtbot_cart_pose/test/fixtures/seed_spread_checkpoint.json"
)

N_SEEDS = 20
BASE_SEED = 0  # arbitrary but fixed, for reproducibility of which 20 seeds get used
FLIP_THRESHOLD_DEG = 90.0  # matches metrics.py's own flipped = abs(yaw_err) > 90.0 convention
PER_FRAME_TIME_BUDGET_S = 180.0
# Remaining work after the checkpointed 6.9/20 seeds is ~236 frame-instances at a
# measured mean 3.9s/frame from the checkpoint's own latency_s field (~920s /
# ~15min). Raised from the original 480s so one background invocation can clear
# the whole remaining sweep instead of stopping partway through.
WALL_CLOCK_BUDGET_S = 3000.0  # stop issuing new work after this long in ONE process invocation


def wrap180(deg: float) -> float:
    return float(np.degrees(np.arctan2(np.sin(np.radians(deg)), np.cos(np.radians(deg)))))


def circular_center_deg(angles_deg: np.ndarray) -> float:
    rad = np.radians(angles_deg)
    return float(np.degrees(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))))


def circular_median_deg(angles_deg: np.ndarray) -> float:
    """Median of a (presumed unimodal) cluster of angles, robust to where the
    cluster sits relative to the +-180 wrap boundary: re-center on the
    circular mean, take an ordinary median of the wrapped residuals, add the
    center back. Only meaningful for a tight, non-multimodal cluster --
    exactly the 'stable' case this script separates out before calling it."""
    center = circular_center_deg(angles_deg)
    shifted = np.array([wrap180(a - center) for a in angles_deg])
    return wrap180(center + float(np.median(shifted)))


def pose_translation_yaw(T: np.ndarray) -> tuple[np.ndarray, float]:
    xy = T[:2, 3].copy()
    yaw = float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))
    return xy, yaw


def max_pairwise_circular_spread(angles_deg: np.ndarray) -> float:
    if len(angles_deg) < 2:
        return 0.0
    diffs = [abs(wrap180(a - b)) for i, a in enumerate(angles_deg) for b in angles_deg[i + 1 :]]
    return float(max(diffs))


def frame_key(fr) -> str:
    return f"{fr.split}/{fr.row_index}"


def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint: dict) -> None:
    """Atomic write: a crash/kill mid-write can never corrupt the checkpoint
    that resumption reads, since os.replace is a single filesystem rename."""
    tmp_path = CHECKPOINT_PATH + ".tmp"
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(tmp_path, "w") as f:
        json.dump(checkpoint, f)
    os.replace(tmp_path, CHECKPOINT_PATH)


def run_pending(seeds, frames, profile, camera, sensor, meshes) -> bool:
    """Fills in any (seed, frame) cells missing from the checkpoint. Returns
    True once every cell is present (i.e. the sweep is fully done)."""
    checkpoint = load_checkpoint()
    t_start = time.time()

    for seed_i in seeds:
        seed_key = str(seed_i)
        checkpoint.setdefault(seed_key, {})
        missing_frames = [fr for fr in frames if frame_key(fr) not in checkpoint[seed_key]]
        if not missing_frames:
            continue

        if time.time() - t_start > WALL_CLOCK_BUDGET_S:
            print(f"Wall-clock budget ({WALL_CLOCK_BUDGET_S:.0f}s) reached for this invocation; "
                  f"stopping early with progress saved. Re-run to continue.")
            return False

        params_i = dataclasses.replace(profile.params, seed=seed_i)
        estimator = VSACSe2Estimator(params=params_i, sensor=sensor)
        for cart_type, mesh in meshes.items():
            estimator.prepare(mesh, cart_type)

        for fr in missing_frames:
            t0 = time.time()
            pcd, _n_mask = build_gt_masked_pcd(fr.row, camera, profile.depth_trunc)
            try:
                T_est = estimator.estimate_pose(pcd, meshes[fr.cart_type], cart_type=fr.cart_type)
            except Exception as exc:
                T_est = None
                reason = f"estimator_exception: {exc!r}"
            else:
                reason = None if T_est is not None else (
                    getattr(estimator, "_last_failure_reason", None) or "estimator_none"
                )
            elapsed = time.time() - t0

            if T_est is not None:
                xy, yaw = pose_translation_yaw(T_est)
                record = {"success": True, "reason": None, "xy": xy.tolist(), "yaw": yaw, "latency_s": elapsed}
            else:
                record = {"success": False, "reason": reason, "xy": None, "yaw": None, "latency_s": elapsed}

            checkpoint[seed_key][frame_key(fr)] = record
            save_checkpoint(checkpoint)

            if elapsed > PER_FRAME_TIME_BUDGET_S:
                print(f"WARNING: frame {frame_key(fr)} seed {seed_i} took {elapsed:.1f}s")

            print(f"  seed={seed_i} {frame_key(fr):20s} {'OK' if T_est is not None else 'ABSTAIN'} {elapsed:.2f}s "
                  f"[{time.time() - t_start:.0f}s elapsed this run]")

            if time.time() - t_start > WALL_CLOCK_BUDGET_S:
                print(f"Wall-clock budget reached mid-seed; stopping with progress saved. Re-run to continue.")
                return False

    return True


def aggregate_and_write(seeds, frames):
    checkpoint = load_checkpoint()

    frame_reports = []
    pooled_trans_dev = []
    pooled_yaw_dev = []
    unstable_frames = []

    for fr in frames:
        key = frame_key(fr)
        per_seed = {s: checkpoint[str(s)][key] for s in seeds}
        successes = {s: v for s, v in per_seed.items() if v["success"]}
        n_success = len(successes)
        n_total = len(per_seed)

        success_churn = n_success != 0 and n_success != n_total
        yaws = np.array([v["yaw"] for v in successes.values()]) if successes else np.array([])
        flip_spread = max_pairwise_circular_spread(yaws) if len(yaws) >= 2 else 0.0
        is_flip_unstable = flip_spread > FLIP_THRESHOLD_DEG
        unstable = success_churn or is_flip_unstable or n_success < 2

        report = {
            "split": fr.split,
            "row_index": fr.row_index,
            "cart_type": fr.cart_type,
            "n_success": n_success,
            "n_total": n_total,
            "unstable": bool(unstable),
            "unstable_reason": (
                "success_churn" if success_churn
                else "yaw_flip" if is_flip_unstable
                else "insufficient_successes" if n_success < 2
                else None
            ),
            "max_pairwise_yaw_spread_deg": flip_spread,
            "per_seed": {str(s): per_seed[s] for s in seeds},
        }

        if not unstable:
            xy_arr = np.array([v["xy"] for v in successes.values()])
            median_xy = np.median(xy_arr, axis=0)
            trans_dev = np.linalg.norm(xy_arr - median_xy[None, :], axis=1)

            median_yaw = circular_median_deg(yaws)
            yaw_dev = np.array([abs(wrap180(y - median_yaw)) for y in yaws])

            report.update(
                median_xy_m=median_xy.tolist(),
                median_yaw_deg=median_yaw,
                trans_dev_median_m=float(np.median(trans_dev)),
                trans_dev_p95_m=float(np.percentile(trans_dev, 95)),
                trans_dev_max_m=float(np.max(trans_dev)),
                yaw_dev_median_deg=float(np.median(yaw_dev)),
                yaw_dev_p95_deg=float(np.percentile(yaw_dev, 95)),
                yaw_dev_max_deg=float(np.max(yaw_dev)),
            )
            pooled_trans_dev.extend(trans_dev.tolist())
            pooled_yaw_dev.extend(yaw_dev.tolist())
        else:
            unstable_frames.append(f"{fr.split}/{fr.row_index} ({fr.cart_type})")

        frame_reports.append(report)
        status = "UNSTABLE:" + str(report["unstable_reason"]) if unstable else (
            f"trans_dev p95={report['trans_dev_p95_m']:.4f}m yaw_dev p95={report['yaw_dev_p95_deg']:.3f}deg"
        )
        print(f"[{fr.split}/{fr.row_index}] {fr.cart_type:9s} n_success={n_success}/{n_total} {status}")

    pooled_trans_dev = np.array(pooled_trans_dev)
    pooled_yaw_dev = np.array(pooled_yaw_dev)

    pooled = {
        "n_stable_frames": len(frames) - len(unstable_frames),
        "n_unstable_frames": len(unstable_frames),
        "unstable_frames": unstable_frames,
        "trans_dev_median_m": float(np.median(pooled_trans_dev)) if len(pooled_trans_dev) else None,
        "trans_dev_p95_m": float(np.percentile(pooled_trans_dev, 95)) if len(pooled_trans_dev) else None,
        "trans_dev_max_m": float(np.max(pooled_trans_dev)) if len(pooled_trans_dev) else None,
        "yaw_dev_median_deg": float(np.median(pooled_yaw_dev)) if len(pooled_yaw_dev) else None,
        "yaw_dev_p95_deg": float(np.percentile(pooled_yaw_dev, 95)) if len(pooled_yaw_dev) else None,
        "yaw_dev_max_deg": float(np.max(pooled_yaw_dev)) if len(pooled_yaw_dev) else None,
    }

    print("\nPOOLED (stable frames only):")
    print(json.dumps(pooled, indent=2))

    out = {
        "profile": "tuned-vis-gate",
        "n_seeds": N_SEEDS,
        "base_seed": BASE_SEED,
        "seeds": seeds,
        "flip_threshold_deg": FLIP_THRESHOLD_DEG,
        "pooled": pooled,
        "frames": frame_reports,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


def main():
    profile = build_profile()
    cfg, camera, sensor = build_camera_and_sensor()
    meshes = load_cad_meshes()
    frames = load_all_frames()

    seeds = derive_internal_seeds(BASE_SEED, N_SEEDS)
    print(f"Using {N_SEEDS} internal seeds derived from base_seed={BASE_SEED}: {seeds}")

    done = run_pending(seeds, frames, profile, camera, sensor, meshes)
    if not done:
        print("NOT DONE -- re-run this script to continue from the checkpoint.")
        return

    print("All seeds x frames computed. Aggregating.")
    aggregate_and_write(seeds, frames)
    print("DONE")


if __name__ == "__main__":
    main()
