"""Layer 1: Entry Point & Asset Resolution for 6DPose Pipeline.

Resolves CLI command options via tyro, loads heavy assets (YOLO model, Camera intrinsics,
Parquet dataset, CAD meshes) ONCE in memory, and dispatches to the requested strategy:
  - eval  -> benchmark.run_benchmark_eval
  - sweep -> sweep.run_parameter_sweep
"""

import numpy as np
import open3d as o3d
import tyro

from benchmark import run_benchmark_eval
from cli_config import Command, SweepArgs
from pipeline import (
    Camera,
    load_cad_meshes,
    load_hf_model,
    load_parquet_dataset,
)
from sweep import run_parameter_sweep


def main():
    args = tyro.cli(Command)

    np.random.seed(args.resolved_seed)
    if args.o3d_seed is not None:
        o3d.utility.random.seed(args.o3d_seed)

    if isinstance(args, SweepArgs):
        print(f"Search space ({len(args.search_space)} free parameters):")
        for name, rng in args.search_space.items():
            print(f"  {name:32} {rng}")

    # Load heavy assets ONCE in memory
    print("Loading pipeline assets...")
    model = load_hf_model(
        local_model_path=args.yolo.local_path, repo_id=args.yolo.repo, filename=args.yolo.file
    )
    camera = Camera(fx=args.camera.fx, fy=args.camera.fy, cx=args.camera.cx, cy=args.camera.cy)
    dataset = load_parquet_dataset(dataset_path=args.dataset.path, test_glob=args.dataset.test_glob)
    meshes = load_cad_meshes()

    # Dispatch to Strategy (Layer 2a benchmark.py vs Layer 2b sweep.py)
    if isinstance(args, SweepArgs):
        run_parameter_sweep(
            dataset=dataset,
            model=model,
            camera=camera,
            meshes=meshes,
            cfg=args,
        )
    else:
        run_benchmark_eval(
            dataset=dataset,
            model=model,
            camera=camera,
            meshes=meshes,
            cfg=args,
        )


if __name__ == "__main__":
    main()
