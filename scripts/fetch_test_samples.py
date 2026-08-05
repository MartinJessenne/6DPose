# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "duckdb",
#   "datasets",
#   "huggingface-hub",
#   "numpy",
#   "tyro",
# ]
# ///
"""
Fetches a small, bearing-stratified sample of `UItraviolet/industrial_cart` into
`tests/fixtures/`, so the full pipeline can be run and scored locally.

Why this exists
---------------
The dataset is 159 shards / 45 GB, and the machine that writes this code has no
copy of it. Every regression so far was found only after a multi-hour remote
sweep, because there was no way to run the real pipeline against real frames
first. 18 committed frames close that gap.

Two phases, because selection and extraction want different tools:

1. SELECT with duckdb over `hf://`. Reading only the three small pose/label
   columns scans a whole 294 MB shard in ~3 s -- columnar pushdown means the rgb
   and depth blobs are never transferred. Bearings are computed here, so the
   choice of frames costs almost no bandwidth.

2. EXTRACT with `datasets`, NOT duckdb. `load_dataset(...).select(...).to_parquet()`
   preserves the HF *feature* metadata; a duckdb `COPY TO` would write `rgb` as a
   bare STRUCT(bytes, path), and `load_parquet_dataset` would then hand YOLO a
   dict instead of an image. This is the only reason phase 2 is not also duckdb.

Stratification
--------------
Frames are picked to span the bearing range (angle between the cart's outward
front-face arrow and the direction to the camera), 2 per cart type per split, one
from the low half and one from the high half of the observed range.

That axis is deliberate: a front-face gate accidentally written against a fixed
base_link axis, rather than against the direction to the camera, passes on
centred carts and only fails on laterally-offset ones. Without high-bearing
frames in the fixture set, that bug ships.

CLI args use tyro, per the project convention, but the dataclass is local rather
than in cli_config.py: this script runs on its own PEP 723 dependency set, and
importing cli_config would drag in open3d, torch and the whole estimator stack
just to parse two flags.

Usage
-----
    hf auth login                       # needs read access to the dataset
    uv run scripts/fetch_test_samples.py --dry-run
    uv run scripts/fetch_test_samples.py
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import tyro
from datasets import load_dataset
from huggingface_hub import get_token

REPO = "UItraviolet/industrial_cart"
HF_BASE = f"hf://datasets/{REPO}/data/"

# One shard per split is plenty: each holds 93-94 frames spanning every cart type
# and the full bearing range, and each shard we touch costs a ~294 MB download in
# phase 2.
SHARDS = {
    "test": "test-00000-of-00016.parquet",
    "validation": "validation-00000-of-00016.parquet",
    "train": "train-00000-of-00127.parquet",
}

CART_TYPES = ("colruyt", "leanflow", "picanol")

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "tests" / "fixtures" / "data"

# USD (Z-back, Y-up) -> OpenCV (Z-forward, Y-down). Mirrors
# pipeline.compute_ground_truth_pose, which is not imported here because this
# script runs under its own PEP 723 dependency set (no open3d, no torch).
T_USD_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def load_extrinsic() -> np.ndarray:
    """Camera-to-robot-base extrinsic, read from cli_config without importing it.

    cli_config pulls in tyro and the estimator stack; this script deliberately
    runs on a minimal dependency set, so the literal is parsed out of the source
    instead. Kept honest by asserting the shape and the rotation's orthonormality.
    """
    import ast

    src = (REPO_ROOT / "cli_config.py").read_text()
    marker = "extrinsic: tuple[tuple[float, float, float, float], ...] = "
    start = src.index(marker) + len(marker)
    depth, end = 0, start
    for i in range(start, len(src)):
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    ext = np.array(ast.literal_eval(src[start:end]), dtype=float)
    assert ext.shape == (4, 4), f"extrinsic must be 4x4, got {ext.shape}"
    assert np.allclose(ext[:3, :3] @ ext[:3, :3].T, np.eye(3), atol=1e-6), (
        "extrinsic rotation block is not orthonormal"
    )
    return ext


def ground_truth_pose(view_transform, bbox_transform, extrinsic) -> np.ndarray:
    """T_robot_cart. Same chain as pipeline.compute_ground_truth_pose.

    Both source transforms arrive row-major flattened and are transposed, matching
    benchmark.py's own reading of these columns.
    """
    t_world_camera = np.asarray(view_transform, dtype=float).reshape(4, 4).T
    t_world_cart = np.asarray(bbox_transform, dtype=float).reshape(4, 4).T
    return extrinsic @ T_USD_TO_CV @ t_world_camera @ t_world_cart


def front_face_bearing_deg(pose, camera_xy) -> float:
    """Signed angle (deg) from the cart's outward front-face arrow to the camera.

    The arrow is +x in the CAD frame for every cart in this fleet (origin at the
    bottom centre of the towing face). Under an SE(2) pose the rotation is about
    Z, so the arrow's z-component stays identically zero and the XY computation is
    exact rather than a projection. 0 deg means facing the camera head-on.
    """
    r2 = pose[:2, :2]
    n2 = r2 @ np.array([1.0, 0.0])  # rotated arrow, unit
    a2 = pose[:2, 3]  # anchor sits at the CAD origin
    v = camera_xy - a2
    det = n2[0] * v[1] - n2[1] * v[0]  # 2-D cross product
    return float(np.degrees(np.arctan2(det, float(n2 @ v))))


def scan_split(con, shard, extrinsic) -> list[dict]:
    """Phase 1: bearings for every frame in one shard, metadata columns only."""
    url = HF_BASE + shard
    rows = con.execute(
        f"SELECT camera_view_transform, bbox_3d_transform, bbox_3d_class_name FROM '{url}'"  # noqa: S608
    ).fetchall()

    camera_xy = extrinsic[:3, 3][:2]
    out = []
    for idx, (view, bbox, names) in enumerate(rows):
        # One cart per frame throughout this dataset; skip anything unexpected
        # rather than silently taking the first of several.
        if not names or len(names) != 1:
            continue
        pose = ground_truth_pose(view, bbox[0], extrinsic)
        out.append(
            {
                "index": idx,
                "cart_type": names[0],
                "bearing_deg": round(front_face_bearing_deg(pose, camera_xy), 2),
                "range_m": round(float(np.linalg.norm(camera_xy - pose[:2, 3])), 3),
            }
        )
    return out


def stratify(candidates: list[dict], per_cart: int) -> list[dict]:
    """Per cart type, take the frames closest to head-on and closest to the edge.

    Sorting by |bearing| and taking from both ends guarantees the fixture set
    spans the range instead of clustering wherever the shard happens to be dense.
    """
    picked = []
    for cart in CART_TYPES:
        pool = sorted(
            (c for c in candidates if c["cart_type"] == cart),
            key=lambda c: abs(c["bearing_deg"]),
        )
        if len(pool) < per_cart:
            raise RuntimeError(f"only {len(pool)} '{cart}' frames in this shard, need {per_cart}")
        half = per_cart // 2
        picked += pool[:half] + pool[-(per_cart - half) :]
    return picked


@dataclass(frozen=True)
class FetchArgs:
    """Selects and downloads the bearing-stratified fixture frames."""

    # Phase 1 only: print the chosen frames without downloading any shard. Phase 1
    # reads three small columns per shard (~3 s, no bulk transfer), so this is the
    # cheap way to review the selection before committing to ~880 MB.
    dry_run: bool = False
    # Frames per cart type per split. 2 gives one near-head-on and one near-edge
    # frame per cart, which is the point of the stratification.
    per_cart_per_split: int = 2


def main() -> int:
    args = tyro.cli(FetchArgs)

    token = get_token()
    if not token:
        print("No HF token. Run `hf auth login` first.", file=sys.stderr)
        return 1

    extrinsic = load_extrinsic()
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")
    con.execute(f"CREATE SECRET hf_tok (TYPE HUGGINGFACE, TOKEN '{token}')")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for split, shard in SHARDS.items():
        print(f"\n=== {split} ({shard}) ===")
        print("  phase 1: scanning bearings (metadata columns only)...")
        candidates = scan_split(con, shard, extrinsic)
        picked = stratify(candidates, args.per_cart_per_split)
        picked.sort(key=lambda c: c["index"])

        for c in picked:
            print(
                f"    row {c['index']:>3}  {c['cart_type']:<9} "
                f"bearing {c['bearing_deg']:>7.2f}°  range {c['range_m']:.2f} m"
            )
        manifest[split] = {"source_shard": shard, "frames": picked}

        if args.dry_run:
            continue

        # Phase 2: `datasets`, to keep the HF feature metadata intact.
        print(f"  phase 2: extracting {len(picked)} rows (downloads ~294 MB, cached)...")
        # verification_mode=no_checks: the repo declares train/validation/test, and
        # loading a single shard trips ExpectedMoreSplitsError otherwise. We are
        # deliberately taking one split at a time.
        ds = load_dataset(
            REPO,
            data_files={split: f"data/{shard}"},
            split=split,
            token=token,
            verification_mode="no_checks",
        )
        subset = ds.select([c["index"] for c in picked])
        out_path = OUT_DIR / f"{split}-00000-of-00001.parquet"
        subset.to_parquet(out_path)
        print(f"  wrote {out_path.relative_to(REPO_ROOT)} ({out_path.stat().st_size / 1e6:.1f} MB)")

    if args.dry_run:
        print("\n(dry run: nothing written)")
        return 0

    (OUT_DIR.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_readme(manifest)
    print(f"\nDone. Fixtures in {OUT_DIR.relative_to(REPO_ROOT)}")
    return 0


def write_readme(manifest: dict) -> None:
    lines = [
        "# Test fixtures",
        "",
        f"18 frames sampled from [`{REPO}`](https://huggingface.co/datasets/{REPO}),",
        "committed so the full pipeline can be run and scored without network or credentials.",
        "",
        "Regenerate with `uv run scripts/fetch_test_samples.py`.",
        "",
        "## Why these frames",
        "",
        "6 per split, 2 per cart type, chosen to span the **bearing** range -- the angle",
        "between the cart's outward front-face arrow and the direction to the camera.",
        "A front-face gate wrongly written against a fixed base_link axis (instead of the",
        "direction to the camera) passes on head-on carts and fails only on angled ones,",
        "so both extremes must be present.",
        "",
        "Ground-truth bearings in this dataset span roughly ±45°; flipped poses sit at",
        "≥135°. See the plan's threshold discussion.",
        "",
        "## Provenance",
        "",
        "| split | source shard | row | cart | bearing | range |",
        "|---|---|---|---|---|---|",
    ]
    for split, info in manifest.items():
        for f in info["frames"]:
            lines.append(
                f"| {split} | `{info['source_shard']}` | {f['index']} | {f['cart_type']} "
                f"| {f['bearing_deg']:.2f}° | {f['range_m']:.2f} m |"
            )
    lines.append("")
    (OUT_DIR.parent / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
