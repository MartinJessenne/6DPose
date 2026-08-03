# =====================================================================
# 3. OPTUNA SWEEP STUDY (MULTI-OBJECTIVE OPTIMIZATION)
# =====================================================================
import glob
import os

import wandb


def log_input_artifacts(run, yolo_cfg, dataset_cfg):
    """Register the YOLO weights and dataset as W&B *reference* artifacts.

    checksum=True records a content hash of each referenced file -- no bytes are
    uploaded, so an unchanged model/dataset produces no new version while any
    modification mints one. That version bump is the "was this input touched?"
    signal for the cross-commit report.
    """
    model_art = wandb.Artifact(
        "yolo-detector", type="model", metadata={"hf_repo": yolo_cfg.repo, "hf_file": yolo_cfg.file}
    )
    model_art.add_reference(f"file://{os.path.abspath(yolo_cfg.local_path)}", checksum=True)
    run.log_artifact(model_art)

    dataset_art = wandb.Artifact(
        "dataset",
        type="dataset",
        metadata={
            "path": dataset_cfg.path,
            "train_glob": dataset_cfg.train_glob,
            "val_glob": dataset_cfg.val_glob,
            "test_glob": dataset_cfg.test_glob,
        },
    )
    for glob_pattern in (dataset_cfg.train_glob, dataset_cfg.val_glob, dataset_cfg.test_glob):
        for shard in sorted(glob.glob(glob_pattern)):
            dataset_art.add_reference(f"file://{os.path.abspath(shard)}", checksum=True)
    run.log_artifact(dataset_art)
