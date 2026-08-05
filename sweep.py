import dataclasses
import os

import numpy as np
import open3d as o3d
import optuna
import plotly.graph_objects as go

import wandb
from evaluation import derive_internal_seeds, draw_eval_indices, evaluate_pipeline
from methods.base import BaseParams, BasePoseEstimator
from metrics import (
    FrameRecord,
    PoseErrorMetrics,
    compute_trial_metrics,
    finite_or_none,
    write_frame_records_csv,
)
from reporting import log_input_artifacts
from run_config import resolve_param_overrides

# Set logging level for Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_parameter_sweep(
    dataset,
    model,
    camera,
    study_name,
    estimator_cls: type[BasePoseEstimator],
    profile_params: BaseParams,
    sweep_size: int,
    n_trials: int,
    meshes: dict[str, o3d.geometry.TriangleMesh],
    yolo_cfg,
    dataset_cfg,
    extrinsic: np.ndarray | None = None,
    seed: int | None = None,
    n_seeds: int = 1,
    dump_frames: bool = True,
    param_overrides: dict | None = None,
):
    """
    Launches a Multi-Objective Bayesian Optimization sweep using Optuna
    to find the Pareto Front of optimal accuracy vs. speed trade-offs.

    n_seeds: number of estimator-internal RANSAC seeds each trial is pooled
    over (see derive_internal_seeds). dump_frames: write each trial's per-frame
    CSV (see FrameRecord) under sweeps/<study_name>_frames/.
    param_overrides: estimator params pinned for every trial, declaring the arm
    (see BenchmarkArgs.param_overrides). Validated here, before any compute, so a
    typo cannot turn a treatment arm into a silent second copy of the control.
    """
    resolved_overrides = resolve_param_overrides(estimator_cls, param_overrides)
    if resolved_overrides:
        print(f"Parameter overrides pinned for every trial: {resolved_overrides}")

    # Stable, run-independent storage location: restarting the same command
    # after a crash (or on a new instance with the file restored) resumes the
    # study instead of starting a fresh DB in a new Hydra timestamped dir.
    project_root = os.path.dirname(os.path.abspath(__file__))
    sweep_dir = os.path.join(project_root, "sweeps")
    os.makedirs(sweep_dir, exist_ok=True)
    db_name = os.path.join(sweep_dir, f"optuna_{study_name}.db")
    db_url = f"sqlite:///{db_name}"

    study = optuna.create_study(
        study_name=study_name,
        storage=db_url,
        # Maximize pose_ar, minimize p95 latency. NOTE this flipped from
        # ["minimize", "minimize"] when the objective stopped being the
        # redundant `accuracy_score = 1 - AR` and became pose_ar itself.
        # Optuna refuses to load a study whose directions changed, so studies
        # created before that switch cannot be resumed -- use a fresh --name.
        directions=["maximize", "minimize"],
        load_if_exists=True,
    )

    # Retrieve run attributes from the Optuna study's metadata to check if we are resuming
    existing_seed = study.user_attrs.get("seed")
    existing_eval_size = study.user_attrs.get("eval_size")
    existing_indices = study.user_attrs.get("sweep_indices")

    total_samples = len(dataset)

    # CASE 1: Resuming an existing study that has proper validation metadata.
    if existing_seed is not None:
        # Integrity Guard: Ensure the requested sweep size matches what was already evaluated.
        if existing_eval_size != sweep_size:
            raise ValueError(
                f"Validation size mismatch: DB has eval_size={existing_eval_size}, "
                f"but sweep requested eval_size={sweep_size}."
            )
        # Integrity Guard: If a seed was explicitly passed, verify it matches the stored seed.
        if seed is not None and seed != existing_seed:
            raise ValueError(
                f"Seed mismatch: DB has seed={existing_seed}, but sweep requested seed={seed}."
            )
        # Load the existing seed and sample indices to ensure evaluation is on the exact same validation subset.
        seed = existing_seed
        sweep_indices = existing_indices
        print(f"Resuming existing study. Loaded seed={seed}, eval_size={sweep_size}")

    # CASE 2: Fresh study initialization.
    else:
        # Generate a seed if not explicitly provided.
        # We retrieve a high-entropy random integer from the OS (via SeedSequence().entropy)
        # and cast it to fit within a standard 31-bit integer range (modulo 2**31 - 1)
        # to make sure it is a valid seed for all downstream library generators.
        if seed is None:
            seed = int(np.random.SeedSequence().entropy % (2**31 - 1))
        # Draw indices deterministically using a generator seeded with our selected seed
        sweep_indices = draw_eval_indices(total_samples, sweep_size, seed)

        # Persist attributes in the study so future runs can resume with the same setup
        study.set_user_attr("seed", seed)
        study.set_user_attr("eval_size", sweep_size)
        study.set_user_attr("sweep_indices", sweep_indices)
        print(f"Created new study. Seed={seed}, eval_size={sweep_size}")

    print(f"Sweep validation indices: {sweep_indices}\n")

    # Seed global random number generators
    np.random.seed(seed)
    o3d.utility.random.seed(seed)

    # ONE W&B run for this whole sweep (1 CLI execution <-> 1 run), not one per
    # trial -- a 200-trial sweep would otherwise flood the workspace with 200
    # separate run pages. Each trial logs into the SAME run at step=trial.number,
    # so every metric/param gets a real per-trial history (a genuine trend line
    # in the W&B UI, not the single-point "history" a one-shot log produces).
    with wandb.init(
        project="6dpose",
        name=study_name,
        group=estimator_cls.__name__,
        job_type="sweep",
        tags=[study_name],
        config={"eval_size": sweep_size, "n_trials": n_trials, "seed": seed},
    ) as run:
        log_input_artifacts(run, yolo_cfg, dataset_cfg)

        def objective(trial: optuna.Trial) -> tuple[float, float]:
            # 1. Suggest global parameters
            depth_trunc = trial.suggest_float("depth_trunc", 2.0, 7.0, step=0.1)

            # 3. Evaluate across effective_n_seeds estimator-internal RANSAC
            # seeds and pool the resulting frames together (rather than
            # sampling `seed` as its own Optuna dimension -- see
            # derive_internal_seeds), so the search is robust to seed luck
            # instead of resting on a single, possibly-lucky draw.
            trial_seeds = derive_internal_seeds(seed, n_seeds, salt=trial.number)

            error_metrics: list[PoseErrorMetrics] = []
            times: list[float] = []
            frame_records: list[FrameRecord] = []
            det_failed = 0
            pose_failed = 0
            gross_yaw_rate_per_seed = []

            trial_params = estimator_cls.params_cls.sample_optuna(
                trial, base=profile_params, fixed=resolved_overrides
            ).with_overrides(**resolved_overrides)

            for seed_i in trial_seeds:
                params_i = (
                    dataclasses.replace(trial_params, seed=seed_i)
                    if seed_i is not None
                    else trial_params
                )
                trial_estimator = estimator_cls(params=params_i, extrinsic=extrinsic)

                # Offline CAD mesh preparation (voxelization, normals, and FPFH/PPF
                # database generation). Cached per (class, cart_type, voxel_size,
                # front_crop_aspect) -- seed isn't part of that key, so repeats hit
                # the cache instead of recomputing, and offline prep costs stay off
                # the timed online pose estimation latency metric either way.
                for cart_type, mesh in meshes.items():
                    trial_estimator.prepare(mesh, cart_type)

                em, t, df, pf, fr = evaluate_pipeline(
                    dataset,
                    model,
                    camera,
                    trial_estimator,
                    sweep_indices,
                    meshes,
                    depth_trunc=depth_trunc,
                )
                error_metrics.extend(em)
                times.extend(t)
                det_failed += df
                pose_failed += pf
                frame_records.extend(fr)
                # Per-seed rate uses the same n_attempted denominator as the
                # pooled metric, so the spread is comparable to the mean.
                gross_yaw_rate_per_seed.append(compute_trial_metrics(em, t, df, pf).gross_yaw_rate)

            if dump_frames:
                write_frame_records_csv(
                    os.path.join(sweep_dir, f"{study_name}_frames", f"trial_{trial.number}.csv"),
                    frame_records,
                )

            # The five headline metrics over the pooled (all-seeds) samples.
            # p95 latency covers successful estimations only: abstentions are
            # already counted against objective 1 via the n_attempted
            # denominator, and charging them a fabricated latency too would
            # corrupt the latency objective.
            m = compute_trial_metrics(error_metrics, times, det_failed, pose_failed)

            # Optuna's own record, independent of W&B.
            trial.set_user_attr("pose_ar", m.pose_ar)
            trial.set_user_attr("p95_latency_s", m.p95_latency_s)
            trial.set_user_attr("gross_yaw_rate", m.gross_yaw_rate)
            trial.set_user_attr("abstention_rate", m.abstention_rate)
            trial.set_user_attr("detection_failure_rate", m.detection_failure_rate)
            trial.set_user_attr("n_attempted", m.n_attempted)
            # Per-seed breakdown (not just the pooled mean) so a later analysis
            # can check how seed-sensitive a config is -- see analyze_sweep.py.
            trial.set_user_attr("n_seeds", len(trial_seeds))
            trial.set_user_attr("trial_seeds", trial_seeds)
            trial.set_user_attr("gross_yaw_rate_per_seed", gross_yaw_rate_per_seed)

            # Log params + metrics together, indexed by trial number -- this is
            # what makes each key's W&B history a real per-trial trend line
            # instead of a single point.
            run.log(
                {
                    **dataclasses.asdict(trial_params),
                    "depth_trunc": depth_trunc,
                    # --- the five ---
                    "pose_ar": m.pose_ar,
                    "p95_latency_s": finite_or_none(m.p95_latency_s),
                    "gross_yaw_rate": m.gross_yaw_rate,
                    "abstention_rate": m.abstention_rate,
                    "detection_failure_rate": m.detection_failure_rate,
                    # --- diagnostics, namespaced so they can never be mistaken
                    # for the headline five ---
                    "diag/good_rate": m.good_rate,
                    "diag/n_attempted": m.n_attempted,
                    "diag/trans_xy_p50": m.trans_xy_p50,
                    "diag/yaw_p50": m.yaw_p50,
                    "diag/gross_yaw_rate_std": float(np.std(gross_yaw_rate_per_seed))
                    if len(trial_seeds) > 1
                    else 0.0,
                },
                step=trial.number,
            )

            if m.abstention_rate == 1.0:
                raise optuna.TrialPruned(
                    f"Trial {trial.number} abstained on every frame, skipping."
                )

            return m.pose_ar, m.p95_latency_s

        print(f"Sweep results are being saved to SQLite database: '{db_name}'")

        # n_trials is a TOTAL target for the study, not an increment: a crashed
        # sweep restarted with the same command only runs the remaining trials.
        # Trials left in RUNNING state by a crash are not counted as finished.
        finished = sum(1 for t in study.trials if t.state.is_finished())
        remaining = max(0, n_trials - finished)
        if finished:
            print(f"Resuming: {finished} finished trials in study, running {remaining} more.")
        try:
            if remaining > 0:
                study.optimize(objective, n_trials=remaining)
        finally:
            # Build the Pareto-front scatter from every COMPLETE trial in the
            # study (not just ones run in this process) -- reads straight from
            # Optuna's persistent SQLite storage, so a resumed sweep's chart is
            # always the complete picture, and an interrupted sweep (Ctrl+C
            # mid-study.optimize) still gets a chart for whatever finished
            # before the interrupt, since `finally` runs either way.
            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if completed:
                param_names = list(completed[0].params.keys())

                # Sortable/filterable raw table -- keeps the per-trial data
                # queryable in W&B alongside the chart (and is the searchable
                # companion to the frontier plot's per-point hover).
                columns = [
                    "trial_number",
                    *param_names,
                    "pose_ar",
                    "p95_latency_s",
                    "gross_yaw_rate",
                    "abstention_rate",
                    "diag_n_attempted",
                ]
                rows = [
                    [
                        t.number,
                        *[t.params.get(name) for name in param_names],
                        t.values[0],
                        t.values[1],
                        t.user_attrs.get("gross_yaw_rate"),
                        t.user_attrs.get("abstention_rate"),
                        t.user_attrs.get("n_attempted"),
                    ]
                    for t in completed
                ]

                # Split trials into Pareto-optimal (study.best_trials, the
                # non-dominated set) vs. everything else (dominated).
                best_numbers = {t.number for t in study.best_trials}
                pareto = [t for t in completed if t.number in best_numbers]
                dominated = [t for t in completed if t.number not in best_numbers]

                run.log(
                    {
                        "pareto_front": build_pareto_figure(
                            pareto, dominated, param_names, study_name
                        ),
                        "pareto_table": wandb.Table(columns=columns, data=rows),
                    }
                )

    print("\n" + "=" * 50)
    print("SWEEP COMPLETE (PARETO FRONT FINDINGS)")
    print("=" * 50)
    print(f"Found {len(study.best_trials)} optimal trade-off trials on the Pareto Front:")
    for _, trial in enumerate(study.best_trials):
        print(f"\n[Trial {trial.number}]")
        print(f"  - pose_ar:  {trial.values[0]:.4f} (↑ better)")
        print(f"  - p95 Execution Time:    {trial.values[1]:.4f}s (↓ faster)")
        print("  - Hyperparameters:")
        for name, val in trial.params.items():
            print(f"    * {name}: {val}")
    print("=" * 50)


def build_pareto_figure(pareto, dominated, param_names, study_name):
    """Build the interactive Pareto-front scatter logged to W&B at sweep end.

    x = p95 latency (objective 2, minimized), y = pose_ar (objective 1, MAXIMIZED),
    so the bottom-right corner is best. Dominated trials form a recessive
    gray field, Pareto-optimal trials (study.best_trials) are highlighted in blue and
    connected by the frontier line. Every point's hover carries its trial number and
    full hyperparameter set, so any point on the frontier is traceable straight back
    to the iteration and config that produced it -- which a bare wandb.plot.scatter
    (fixed tooltip, x/y only) cannot do.
    """
    BLUE, GRAY, GRID, AXIS = "#2a78d6", "#898781", "#e1e0d9", "#c3c2b7"
    INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

    # customdata columns, in order:
    # trial_number, *params, gross_yaw_rate, abstention_rate.
    # Both failure rates ride along in the tooltip so a frontier point that looks
    # accurate can be checked for how much it simply declined to answer.
    def customdata(trials):
        return [
            [t.number]
            + [t.params.get(name) for name in param_names]
            + [t.user_attrs.get("gross_yaw_rate"), t.user_attrs.get("abstention_rate")]
            for t in trials
        ]

    hover_lines = [
        "<b>Trial %{customdata[0]}</b>",
        "pose_ar: %{y:.4f}",
        "p95 latency: %{x:.3f}s",
    ]
    for i, name in enumerate(param_names, start=1):
        hover_lines.append(f"{name}: %{{customdata[{i}]}}")
    hover_lines.append(f"gross_yaw_rate: %{{customdata[{len(param_names) + 1}]:.3f}}")
    hover_lines.append(f"abstention_rate: %{{customdata[{len(param_names) + 2}]:.3f}}")
    hovertemplate = "<br>".join(hover_lines) + "<extra></extra>"

    fig = go.Figure()

    # Dominated trials -- recessive gray field, drawn first (underneath).
    fig.add_trace(
        go.Scatter(
            x=[t.values[1] for t in dominated],
            y=[t.values[0] for t in dominated],
            mode="markers",
            name="Dominated trials",
            marker=dict(color=GRAY, size=8, opacity=0.55),
            customdata=customdata(dominated),
            hovertemplate=hovertemplate,
        )
    )

    # Pareto frontier -- straight line through the optimal trials sorted by latency.
    pareto_sorted = sorted(pareto, key=lambda t: t.values[1])
    fig.add_trace(
        go.Scatter(
            x=[t.values[1] for t in pareto_sorted],
            y=[t.values[0] for t in pareto_sorted],
            mode="lines",
            name="Pareto frontier",
            line=dict(color=BLUE, width=2),
            hoverinfo="skip",
        )
    )

    # Pareto-optimal trials -- highlighted, drawn on top.
    fig.add_trace(
        go.Scatter(
            x=[t.values[1] for t in pareto],
            y=[t.values[0] for t in pareto],
            mode="markers",
            name="Pareto-optimal",
            marker=dict(color=BLUE, size=12, line=dict(color=SURFACE, width=1.5)),
            customdata=customdata(pareto),
            hovertemplate=hovertemplate,
        )
    )

    fig.update_layout(
        title=dict(text=f"Pareto Front — {study_name}", font=dict(size=18, color=INK)),
        template="plotly_white",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=INK2, size=13),
        xaxis=dict(
            title="p95 latency (s)  →  slower",
            gridcolor=GRID,
            zeroline=False,
            linecolor=AXIS,
            ticks="outside",
            tickcolor=AXIS,
        ),
        yaxis=dict(
            title="pose_ar  →  better",
            gridcolor=GRID,
            zeroline=False,
            linecolor=AXIS,
            ticks="outside",
            tickcolor=AXIS,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=GRID),
        margin=dict(l=70, r=30, t=70, b=90),
        annotations=[
            dict(
                text="← better (fast & accurate)",
                x=0.01,
                y=0.98,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color=GRAY, size=12),
            )
        ],
    )
    return fig
