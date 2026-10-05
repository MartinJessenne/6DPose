"""Export the 18 committed real fixture frames (tests/fixtures/) to a single
self-describing binary file that a C++ port can load without any Python
dependency, plus a JSON manifest for humans.

ADDITIVE ONLY: this script imports pipeline.compute_ground_truth_pose and
cli_config.CameraConfig rather than re-deriving the USD -> OpenCV ->
robot-frame conversion or the intrinsics -- see the module docstrings there
for why that chain is exactly the step a port gets silently wrong.

Binary layout (little-endian), written verbatim per the task spec:

    "CFX1"                    4 bytes magic
    n_frames                  uint32
    per frame:
      split_len               uint32
      split                   char[split_len]        e.g. "test"
      row_index                uint32                 index within the shard
      cart_len                 uint32
      cart_type                char[cart_len]         "colruyt" | "picanol" | "leanflow"
      width, height            uint32, uint32
      depth                    float32[width*height]  METRES, 0.0 = invalid/no return
      mask                     uint8[width*height]    255 = this cart, 0 = everything else
      fx, fy, cx, cy           float64 x4
      T_robot_camera           float64[16]            row-major
      T_gt                     float64[16]            row-major, CAD -> robot base

Run: /home/martin/6DPose/.venv/bin/python scripts/export_cpp_fixtures.py
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from cli_config import CameraConfig
from pipeline import compute_ground_truth_pose, load_parquet_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
OUT_DIR = Path("/home/martin/martinjessenne/nxtbot_cart_pose/test/fixtures")

SPLITS = ("test", "validation", "train")

# Raw depth includes genuine far-background returns (open scene geometry
# behind/around the cart, up to ~342 m measured on frame test[0]) that carry
# no information any profile in cli_config.py uses -- the largest depth_trunc
# across every VSAC/RANSAC/PPF profile is 6.2 m. Values beyond this bound are
# remapped to 0.0 (the format's own "invalid/no return" sentinel) purely to
# keep the exported buffer within a sane working volume; nothing inside the
# scene of interest is ever this far away. Chosen well above every
# depth_trunc in cli_config.py so no in-range point is ever touched.
DEPTH_MAX_M = 20.0


def decode_depth(row: dict) -> np.ndarray:
    """Raw depth blob -> (H, W) float32 array, in METRES, 0.0 = invalid.

    Reuses the exact decode pipeline.py's process_and_reconstruct() uses:
    `np.frombuffer(depth_bytes, np.float32).reshape((height, width))`, no
    scaling applied. depth_resolution is stored [height, width] (verified:
    len(depth_bytes) // 4 == depth_resolution[0] * depth_resolution[1], and
    equals camera_resolution's [width, height] transposed).

    Values beyond DEPTH_MAX_M are zeroed -- see the constant's comment.
    """
    height, width = int(row["depth_resolution"][0]), int(row["depth_resolution"][1])
    depth = np.frombuffer(row["depth"], dtype=np.float32).reshape((height, width)).copy()
    depth[depth > DEPTH_MAX_M] = 0.0
    return depth, width, height


def decode_mask_bool(row: dict, width: int, height: int) -> np.ndarray:
    """Ground-truth boolean mask (H, W) for the single target cart in this frame.

    semantic_labels is a JSON string mapping the semantic id (as a string
    key) to {"class": <cart type name>}, e.g. {"0": {"class": "picanol"}}.
    The `semantic` PNG is RGBA, one flat color per semantic id, background
    pixels are exactly (0, 0, 0, 0) (fully transparent black).

    Every one of the 18 committed fixture frames carries exactly ONE
    semantic id -- the target cart named by bbox_3d_class_name[0] -- so
    "not background" and "the target cart" are the same predicate here.
    Asserted rather than assumed, since a future fixture regen could add
    clutter/occluder instances and silently break that identity.

    Shared by scripts/export_oracle_results.py and scripts/export_seed_spread.py
    so the C++ fixture mask and the Python oracle's input mask are pixel-for-
    pixel the same array, not two independent re-derivations.
    """
    labels = json.loads(row["semantic_labels"])
    if len(labels) != 1:
        raise ValueError(
            f"Expected exactly one semantic id per fixture frame, got {labels}. "
            "The mask-from-alpha shortcut assumes a single labeled instance; "
            "resolve via bbox_3d_semantic_id[0] and a per-id color table instead."
        )
    (sem_id_str, entry) = next(iter(labels.items()))
    if entry["class"] != row["bbox_3d_class_name"][0]:
        raise ValueError(
            f"semantic_labels class {entry['class']!r} != "
            f"bbox_3d_class_name[0] {row['bbox_3d_class_name'][0]!r}"
        )
    if int(sem_id_str) != int(row["bbox_3d_semantic_id"][0]):
        raise ValueError("semantic id key does not match bbox_3d_semantic_id[0]")

    sem_arr = np.array(row["semantic"])  # (H, W, 4) RGBA
    if sem_arr.shape[:2] != (height, width):
        raise ValueError(f"semantic PNG shape {sem_arr.shape[:2]} != depth shape {(height, width)}")
    return sem_arr[..., 3] > 0


def export_frame(split: str, row_index: int, row: dict, extrinsic: np.ndarray, camera: CameraConfig):
    depth, width, height = decode_depth(row)
    mask = decode_mask_bool(row, width, height).astype(np.uint8) * 255

    t_world_camera = np.asarray(row["camera_view_transform"], dtype=float).reshape(4, 4).T
    t_world_cart = np.asarray(row["bbox_3d_transform"][0], dtype=float).reshape(4, 4).T
    T_gt = compute_ground_truth_pose(t_world_camera, t_world_cart, extrinsic)

    cart_type = row["bbox_3d_class_name"][0]

    # Per-frame intrinsics derivable from camera_focal_length / camera_aperture
    # / camera_resolution (USD camera convention: fx = f / aperture_h * res_w),
    # reported for the mismatch check but NOT what gets written: no code path
    # in pipeline.py / benchmark.py / inspect_pose.py ever reads
    # camera_focal_length, camera_aperture or camera_projection (confirmed by
    # grep) -- every estimator run is fed CameraConfig's fixed fx/fy/cx/cy.
    # The export must match what the oracle in Task 2 actually uses.
    derived_fx = row["camera_focal_length"] / row["camera_aperture"][0] * row["camera_resolution"][0]
    derived_fy = row["camera_focal_length"] / row["camera_aperture"][1] * row["camera_resolution"][1]

    return {
        "split": split,
        "row_index": row_index,
        "cart_type": cart_type,
        "width": width,
        "height": height,
        "depth": depth,
        "mask": mask,
        "fx": camera.fx,
        "fy": camera.fy,
        "cx": camera.cx,
        "cy": camera.cy,
        "T_robot_camera": extrinsic,
        "T_gt": T_gt,
        "derived_fx": float(derived_fx),
        "derived_fy": float(derived_fy),
        "n_mask_px": int(np.count_nonzero(mask)),
    }


def write_bin(frames: list[dict], out_path: Path) -> None:
    with open(out_path, "wb") as f:
        f.write(b"CFX1")
        f.write(struct.pack("<I", len(frames)))
        for fr in frames:
            split_b = fr["split"].encode("utf-8")
            f.write(struct.pack("<I", len(split_b)))
            f.write(split_b)
            f.write(struct.pack("<I", fr["row_index"]))
            cart_b = fr["cart_type"].encode("utf-8")
            f.write(struct.pack("<I", len(cart_b)))
            f.write(cart_b)
            f.write(struct.pack("<II", fr["width"], fr["height"]))
            f.write(fr["depth"].astype("<f4").tobytes())
            f.write(fr["mask"].astype("<u1").tobytes())
            f.write(struct.pack("<dddd", fr["fx"], fr["fy"], fr["cx"], fr["cy"]))
            f.write(fr["T_robot_camera"].astype("<f8").reshape(-1).tobytes())
            f.write(fr["T_gt"].astype("<f8").reshape(-1).tobytes())


def verify_bin(path: Path, expected: list[dict]) -> None:
    """Read the file back with an independent decoder and cross-check against
    the in-memory frames that were used to write it."""
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"CFX1", magic
        (n_frames,) = struct.unpack("<I", f.read(4))
        assert n_frames == len(expected), (n_frames, len(expected))

        for fr in expected:
            (split_len,) = struct.unpack("<I", f.read(4))
            split = f.read(split_len).decode("utf-8")
            assert split == fr["split"], (split, fr["split"])
            (row_index,) = struct.unpack("<I", f.read(4))
            assert row_index == fr["row_index"]
            (cart_len,) = struct.unpack("<I", f.read(4))
            cart_type = f.read(cart_len).decode("utf-8")
            assert cart_type == fr["cart_type"]
            width, height = struct.unpack("<II", f.read(8))
            assert (width, height) == (fr["width"], fr["height"])
            n_px = width * height
            depth = np.frombuffer(f.read(4 * n_px), dtype="<f4").reshape(height, width)
            mask = np.frombuffer(f.read(n_px), dtype="<u1").reshape(height, width)
            fx, fy, cx, cy = struct.unpack("<dddd", f.read(32))
            T_rc = np.frombuffer(f.read(128), dtype="<f8").reshape(4, 4)
            T_gt = np.frombuffer(f.read(128), dtype="<f8").reshape(4, 4)

            # Byte-exact round trip for the arrays.
            assert np.array_equal(depth, fr["depth"].astype("<f4"))
            assert np.array_equal(mask, fr["mask"])
            assert (fx, fy, cx, cy) == (fr["fx"], fr["fy"], fr["cx"], fr["cy"])
            assert np.allclose(T_rc, fr["T_robot_camera"])
            assert np.allclose(T_gt, fr["T_gt"])

            # Physical sanity.
            finite = np.isfinite(depth)
            assert finite.all(), "depth must be finite everywhere (0.0 marks invalid, not NaN/Inf)"
            assert depth.min() >= 0.0 and depth.max() <= 20.0, (depth.min(), depth.max())
            assert set(np.unique(mask).tolist()) <= {0, 255}, "mask must be binary 0/255"
            assert np.array_equal(T_gt[3, :], [0.0, 0.0, 0.0, 1.0]), T_gt[3, :]
            R = T_gt[:3, :3]
            assert np.allclose(R.T @ R, np.eye(3), atol=1e-6), "T_gt rotation not orthonormal"
            assert abs(np.linalg.det(R) - 1.0) < 1e-6, "T_gt rotation not proper (det != 1)"

        trailing = f.read()
        assert trailing == b"", f"{len(trailing)} unexpected trailing bytes"

    print(f"Verified {n_frames} frames in {path} ({path.stat().st_size} bytes).")


def main():
    if not (FIXTURES_DIR / "data").exists():
        raise SystemExit(f"Fixtures missing at {FIXTURES_DIR}; nothing to export.")

    manifest_src = json.load(open(FIXTURES_DIR / "manifest.json"))
    camera = CameraConfig()
    extrinsic = np.array(camera.extrinsic, dtype=float)

    all_frames = []
    fixture_manifest_entries = []
    intrinsics_mismatches = []

    for split in SPLITS:
        ds = load_parquet_dataset(dataset_path=str(FIXTURES_DIR), test_glob=f"data/{split}-*.parquet")
        prov = manifest_src[split]["frames"]
        assert len(ds) == len(prov), f"{split}: {len(ds)} rows but manifest lists {len(prov)}"

        for i, row in enumerate(ds):
            prov_entry = prov[i]
            assert row["bbox_3d_class_name"][0] == prov_entry["cart_type"], (
                f"{split}[{i}]: dataset cart {row['bbox_3d_class_name'][0]!r} != "
                f"manifest cart {prov_entry['cart_type']!r}"
            )
            row_index = int(prov_entry["index"])

            fr = export_frame(split, row_index, row, extrinsic, camera)
            all_frames.append(fr)

            mismatch_fx = abs(fr["derived_fx"] - camera.fx)
            mismatch_fy = abs(fr["derived_fy"] - camera.fy)
            if mismatch_fx > 1.0 or mismatch_fy > 1.0:
                intrinsics_mismatches.append(
                    (split, row_index, fr["derived_fx"], fr["derived_fy"])
                )

            fixture_manifest_entries.append(
                {
                    "split": split,
                    "row_index": row_index,
                    "cart_type": fr["cart_type"],
                    "bearing_deg": prov_entry["bearing_deg"],
                    "range_m": prov_entry["range_m"],
                    "width": fr["width"],
                    "height": fr["height"],
                    "n_mask_px": fr["n_mask_px"],
                }
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bin_path = OUT_DIR / "fixtures.bin"
    write_bin(all_frames, bin_path)
    verify_bin(bin_path, all_frames)

    manifest_out = {
        "camera": {"fx": camera.fx, "fy": camera.fy, "cx": camera.cx, "cy": camera.cy},
        "T_robot_camera": extrinsic.tolist(),
        "frames": fixture_manifest_entries,
    }
    manifest_path = OUT_DIR / "fixtures_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_out, f, indent=2)

    print(f"Wrote {len(all_frames)} frames to {bin_path}")
    print(f"Wrote manifest to {manifest_path}")
    if intrinsics_mismatches:
        print(
            f"Per-frame derived intrinsics differ from CameraConfig by >1px for "
            f"{len(intrinsics_mismatches)}/{len(all_frames)} frames (export uses "
            f"CameraConfig's fixed values regardless -- see export_frame docstring). "
            f"Example: {intrinsics_mismatches[0]}"
        )
    else:
        print("Per-frame derived intrinsics agree with CameraConfig within 1px.")


if __name__ == "__main__":
    main()
