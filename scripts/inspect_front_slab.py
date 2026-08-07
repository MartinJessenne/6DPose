"""
Opens a window showing the front slab of all three carts at a chosen aspect ratio.

This is exactly what the pipeline registers against: the same crop_front_face
call Ransac3DoFEstimator makes, so what you see is what FPFH and ICP see.

Two cuts happen, and only one of them is a knob:

  HORIZONTAL (depth)  keeps geometry within `depth` of the mesh's +x extreme,
                      where depth = l_y / aspect. Since the slab's width is l_y,
                      `aspect` IS the plan-view aspect ratio -- 1.0 asks for a
                      square, and a square slab has a genuine 90-degree minimum
                      in the registration objective. This is the knob.
  VERTICAL (height)   drops everything below 0.16 m off the floor, to exclude
                      the wheels and casters: their steering angle is fixed in
                      the CAD but arbitrary in reality, so they are a spurious
                      registration cue. Fixed, matching crop_front_face's
                      default -- not exposed here, because changing it would
                      stop this showing what the pipeline actually uses.

Both are true plane slices via trimesh, re-triangulated at the exact cut plane,
not bounding-box filters.

Usage
-----
    uv run scripts/inspect_front_slab.py                # aspect 2.0, the default
    uv run scripts/inspect_front_slab.py --aspect 1.0   # the old square default
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import tyro

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from methods.ransac3dof import crop_front_face  # noqa: E402

# Gap between carts in the viewer, in metres. Layout only -- it has no bearing
# on the geometry, it just stops the slabs overlapping on screen.
_LAYOUT_GAP = 0.4

# One colour per cart, in load order, so the three stay distinguishable.
_COLOURS = [
    [0.85, 0.28, 0.28],
    [0.28, 0.55, 0.85],
    [0.35, 0.72, 0.40],
]


@dataclass
class Args:
    """Inspect the pipeline's front-slab crop at a given aspect ratio."""

    aspect: float = 2.0
    """Plan-view aspect ratio of the slab. depth = l_y / aspect."""

    meshes_dir: Path = REPO_ROOT / "meshes"
    """Directory holding the cart .ply files."""


def _extent(values: np.ndarray) -> float:
    """Peak-to-peak extent of a coordinate column, in metres."""
    return float(values.max() - values.min())


def main(args: Args) -> None:
    mesh_paths = sorted(args.meshes_dir.glob("*.ply"))
    if not mesh_paths:
        raise SystemExit(f"No .ply meshes found in {args.meshes_dir}")

    geometries = []
    y_cursor = 0.0

    for i, path in enumerate(mesh_paths):
        mesh = o3d.io.read_triangle_mesh(str(path))
        if not mesh.has_triangles():
            print(f"{path.stem}: no triangles, skipping")
            continue

        l_y = _extent(np.asarray(mesh.vertices)[:, 1])
        slab = crop_front_face(mesh, depth=l_y / args.aspect)

        sv = np.asarray(slab.vertices)
        if len(sv) == 0:
            print(f"{path.stem}: empty slab at aspect {args.aspect}, skipping")
            continue

        sx, sy, sz = (_extent(sv[:, k]) for k in range(3))
        ratio = max(sx, sy) / min(sx, sy)
        flag = "   <-- near-square, degenerate in yaw" if ratio < 1.15 else ""
        print(f"{path.stem:<10} slab {sx:.3f} x {sy:.3f} x {sz:.3f} m   ratio {ratio:.3f}{flag}")

        # Lay the carts out along +y so they sit side by side rather than on top
        # of one another. Shifts the copy on screen only; nothing is measured
        # after this point.
        slab.translate((0.0, y_cursor - sv[:, 1].min(), 0.0))
        y_cursor += sy + _LAYOUT_GAP

        # Without normals Open3D renders the surface flat and unreadable.
        slab.compute_vertex_normals()
        slab.paint_uniform_color(_COLOURS[i % len(_COLOURS)])
        geometries.append(slab)

    if not geometries:
        raise SystemExit("Nothing to show.")

    # Axis triad at the origin: +x (red) is the towing-face direction the crop
    # measures from, so it tells you at a glance which way the slab faces.
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Front slabs — aspect {args.aspect:g}",
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
