"""
Answers one question: why does the 18-frame local harness read good_rate 0.111
while the sweep's best config reads 0.725?

Why this exists
---------------
Those two numbers have been quoted side by side as if comparable, and the gap
recorded as an open item (30.04.4's validity caveats). Two hypotheses, and they
demand different responses:

  H1 PRESET. Local runs used VSAC profile "default" (voxel_size 0.06). Every good
     trial in sweep s3zi4564 sits at 0.02 -- median good_rate by voxel is
     0.681 at 0.02 against 0.304 at 0.06. If this is the whole story, the harness
     is sound and was simply pointed at an untuned configuration.

  H2 SAMPLE. tests/fixtures/manifest.json is a 3 carts x 3 splits x 2 bearings
     factorial, and the two bearings are ~0 deg and ~44 deg. The production cone
     is +-45 deg. So HALF the fixture set sits within 3 deg of the hardest
     viewpoint that exists, while a random draw from the sweep is roughly uniform
     in bearing. If this is the story, the two numbers were never comparable and
     the fix is to stop comparing them.

These are separable: H1 predicts the gap closes when the same config runs on the
fixtures; H2 predicts a large split between the near-frontal and cone-edge
buckets that survives any config. Both can be true.

Reports the outcome partition (good / gross_yaw / abstained / not_detected --
FrameRecord.outcome mirrors TrialMetrics' rates exactly, see its docstring)
overall and split by bearing bucket, plus per cart type and per cargo state.

Usage
-----
    uv run scripts/reconcile_harness.py model:vsac3dof model.profile:tuned
    uv run scripts/reconcile_harness.py model:vsac3dof model.profile:default
"""

import json
import sys
from collections import Counter
from pathlib import Path

import open3d as o3d
import tyro

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark import evaluate_pipeline, resolve_param_overrides  # noqa: E402
from cli_config import LocalEvalArgs  # noqa: E402
from pipeline import Camera, load_cad_meshes, load_hf_model, load_parquet_dataset  # noqa: E402

SPLITS = ("test", "validation", "train")

# Confirmed by Martin against scripts/inspect_fixtures.py's contact sheet:
# "all leanflow occurrences have boxes except validation[65], which is a bare
# frame. test[63] also has a box."  Keyed by (split, manifest index).
# Cargo is geometry present in the scene cloud and absent from the CAD, so it is
# unexplained under any scene->model objective -- worth tracking as its own axis.
LOADED = {
    ("test", 63),
    ("test", 77),
    ("validation", 90),
    ("train", 40),
    ("train", 61),
}
BARE_LEANFLOW = {("validation", 65)}

# The cone edge, per E03: ground-truth |phi| reaches 48.79 deg over 560 poses.
BEARING_SPLIT_DEG = 20.0


def partition(outcomes: list[str]) -> str:
    """good/gross/abstain rates over frames the estimator was actually handed."""
    c = Counter(outcomes)
    attempted = len(outcomes) - c["not_detected"]
    if attempted == 0:
        return f"n=0 attempted (all {c['not_detected']} undetected)"
    return (
        f"n={attempted:2d}  good {c['good'] / attempted:.3f}  "
        f"gross_yaw {c['gross_yaw'] / attempted:.3f}  "
        f"abstain {c['abstained'] / attempted:.3f}"
        + (f"  [+{c['not_detected']} undetected]" if c["not_detected"] else "")
    )


def main() -> int:
    args = tyro.cli(LocalEvalArgs)
    fixtures = Path(args.fixtures_path)
    if args.o3d_seed is not None:
        o3d.utility.random.seed(args.o3d_seed)

    manifest = json.loads((fixtures / "manifest.json").read_text())
    sensor = args.camera.sensor
    camera = Camera(fx=args.camera.fx, fy=args.camera.fy, cx=args.camera.cx, cy=args.camera.cy)
    estimator_cls = args.model.ESTIMATOR_CLS

    overrides = resolve_param_overrides(estimator_cls, args.param_overrides)
    params = type(args.model.profile.params)(**{**vars(args.model.profile.params), **overrides})

    print(
        f"profile params: voxel_size={params.voxel_size} "
        f"front_crop_aspect={getattr(params, 'front_crop_aspect', None)} "
        f"z_gate={params.z_gate_threshold} iters={params.ransac_max_iterations} "
        f"depth_trunc={args.model.profile.depth_trunc}"
    )
    print(f"overrides: {overrides or '{}'}\n")

    model = load_hf_model(
        local_model_path=args.yolo.local_path, repo_id=args.yolo.repo, filename=args.yolo.file
    )
    meshes = load_cad_meshes()

    rows = []  # (split, manifest_idx, cart, bearing, range, loaded, outcome)
    for split in SPLITS:
        dataset = load_parquet_dataset(
            dataset_path=str(fixtures), test_glob=f"data/{split}-*.parquet"
        )
        estimator = estimator_cls(params=params, sensor=sensor)
        for cart_type, mesh in meshes.items():
            estimator.prepare(mesh, cart_type)

        _, _, _, _, records = evaluate_pipeline(
            dataset=dataset,
            model=model,
            camera=camera,
            estimator=estimator,
            sample_indices=list(range(len(dataset))),
            meshes=meshes,
            depth_trunc=args.model.profile.depth_trunc,
        )
        # The parquet rows are the manifest's frames in manifest order, so
        # record.sample_idx (a row index) indexes straight into the entry list.
        entries = manifest[split]["frames"]
        for rec in records:
            e = entries[rec.sample_idx]
            key = (split, e["index"])
            rows.append(
                (
                    split,
                    e["index"],
                    e["cart_type"],
                    e["bearing_deg"],
                    e["range_m"],
                    key in LOADED,
                    rec.outcome,
                )
            )

    print(f"{'split':11} {'idx':>4} {'cart':9} {'bearing':>8} {'range':>6} {'cargo':>6}  outcome")
    print("-" * 66)
    for r in sorted(rows, key=lambda r: abs(r[3])):
        print(
            f"{r[0]:11} {r[1]:4d} {r[2]:9} {r[3]:+8.2f} {r[4]:6.2f} "
            f"{'boxes' if r[5] else 'bare':>6}  {r[6]}"
        )

    print("\n" + "=" * 66)
    print("OVERALL          ", partition([r[6] for r in rows]))

    print("\n--- H2: bearing bucket (the fixture stratification hypothesis) ---")
    near = [r[6] for r in rows if abs(r[3]) < BEARING_SPLIT_DEG]
    edge = [r[6] for r in rows if abs(r[3]) >= BEARING_SPLIT_DEG]
    print(f"  near-frontal <{BEARING_SPLIT_DEG:.0f}deg  ", partition(near))
    print(f"  cone edge   >={BEARING_SPLIT_DEG:.0f}deg  ", partition(edge))

    print("\n--- cart type ---")
    for cart in sorted({r[2] for r in rows}):
        print(f"  {cart:10}", partition([r[6] for r in rows if r[2] == cart]))

    print("\n--- cargo (scene geometry absent from the CAD) ---")
    print("  loaded    ", partition([r[6] for r in rows if r[5]]))
    print("  bare      ", partition([r[6] for r in rows if not r[5]]))
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
