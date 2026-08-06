# =====================================================================
# 3. OPTUNA SWEEP STUDY (MULTI-OBJECTIVE OPTIMIZATION)
# =====================================================================
import dataclasses
import glob
import os
import re

import wandb
from metrics import FrameRecord

# W&B artifact names admit only [A-Za-z0-9_.-]; run names come from --name /
# --study-name and are free text, so "E1 control" would raise at log_artifact
# time -- after the whole evaluation has been paid for.
_ARTIFACT_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")


def _artifact_name(name: str) -> str:
    return _ARTIFACT_UNSAFE.sub("-", name)


def log_input_artifacts(run, yolo_cfg, dataset_cfg):
    """Register the YOLO weights and dataset as W&B *reference* artifacts.

    checksum=True records a content hash of each referenced file -- no bytes are
    uploaded, so an unchanged model/dataset produces no new version while any
    modification mints one. That version bump is the "was this input touched?"
    signal for the cross-commit report.
    """
    if run is None:
        return
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


def log_frame_records(run, name: str, csv_path: str, records: list[FrameRecord]) -> None:
    """Publish one evaluation's per-frame records to W&B, as both a Table and an Artifact.

    Both, because they answer different questions. The Table is the browsable
    one -- sortable, filterable and groupable in the UI, and comparable across
    runs -- so "which cart do the failures land on?" becomes a group-by instead
    of a script. The Artifact is the durable one: versioned, content-addressed
    (an identical re-log mints no new version), and fetched back with
    `run.use_artifact("frames-<name>:latest").download()`.

    `add_file`, NOT `add_reference` -- the opposite of log_input_artifacts
    above, and the distinction is the whole point of this function.
    add_reference records a checksum and uploads ZERO bytes, which is right for
    dataset shards that already live somewhere permanent. Outputs have no such
    home: a reference to a path on a rented GPU box is a dangling pointer the
    moment the instance is destroyed, which is how the 05-08 ablation came
    within one `scp` of losing its only copy.

    The local CSV is still written and still the thing scripts/mcnemar_arms.py
    consumes -- this uploads that same file rather than replacing it, so the
    analysis path is identical whether the CSV came from a local run or from
    `use_artifact(...).download()`.
    """
    if run is None or not records:
        return

    fields = [f.name for f in dataclasses.fields(FrameRecord)]
    run.log(
        {
            f"frames/{name}": wandb.Table(
                columns=fields,
                data=[[getattr(r, f) for f in fields] for r in records],
            )
        }
    )

    artifact = wandb.Artifact(
        f"frames-{_artifact_name(name)}",
        type="frame-records",
        metadata={"n_records": len(records)},
    )
    artifact.add_file(csv_path)
    run.log_artifact(artifact)


def log_frame_records_dir(run, name: str, directory: str) -> None:
    """Publish a sweep's whole per-trial frames directory as ONE artifact.

    One artifact rather than one per trial: a 60-trial sweep would otherwise
    mint 60 artifact objects for what is conceptually a single output of a
    single run, and the artifact browser stops being readable at exactly the
    point it starts being useful. add_dir keeps every trial_N.csv individually
    downloadable inside that one version.

    No Table here. The per-trial headline metrics are already logged at
    step=trial.number and collected in `pareto_table`; a pooled frames Table
    would be ~60x the rows with no trial column to group by, so it would carry
    less information than the artifact it duplicates.
    """
    if run is None or not os.path.isdir(directory):
        return

    artifact = wandb.Artifact(
        f"frames-{_artifact_name(name)}",
        type="frame-records",
        metadata={"n_trials": len(glob.glob(os.path.join(directory, "*.csv")))},
    )
    artifact.add_dir(directory)
    run.log_artifact(artifact)
