"""Layer 2b: Hyperparameter Sweep Orchestrator.

Launches a Multi-Objective Bayesian Optimization sweep using Optuna to find the
Pareto Front of optimal accuracy vs. speed trade-offs across algorithm parameters.
"""

import contextlib
import dataclasses
import os

import numpy as np
import open3d as o3d
import optuna
import plotly.graph_objects as go

import wandb
from cli_config import SweepArgs
from evaluation import derive_internal_seeds, draw_eval_indices, evaluate_pipeline
from metrics import (
    FrameRecord,
    PoseErrorMetrics,
    compute_trial_metrics,
    finite_or_none,
    write_frame_records_csv,
)
from reporting import log_frame_records_dir, log_input_artifacts

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_parameter_sweep(
    dataset,
    model,
    camera,
    meshes: dict[str, o3d.geometry.TriangleMesh],
    cfg: SweepArgs,
):
    """Launches Optuna Multi-Objective Bayesian Optimization sweep."""
    resolved_overrides = cfg.resolved_overrides
    if resolved_overrides:
        print(f"Parameter overrides pinned for every trial: {resolved_overrides}")

    project_root = os.path.dirname(os.path.abspath(__file__))
    sweep_dir = os.path.join(project_root, "sweeps")
    os.makedirs(sweep_dir, exist_ok=True)
    db_name = os.path.join(sweep_dir, f"optuna_{cfg.name}.db")
    db_url = f"sqlite:///{db_name}"

    study = optuna.create_study(
        study_name=cfg.name,
        storage=db_url,
        directions=["maximize", "minimize"],
        load_if_exists=True,
    )

    existing_seed = study.user_attrs.get("seed")
    existing_eval_size = study.user_attrs.get("eval_size")
    existing_indices = study.user_attrs.get("sweep_indices")

    total_samples = len(dataset)

    if existing_seed is not None:
        if existing_eval_size != cfg.eval_size:
            raise ValueError(
                f"Validation size mismatch: DB has eval_size={existing_eval_size}, "
                f"but sweep requested eval_size={cfg.eval_size}."
            )
        if cfg.seed is not None and cfg.seed != existing_seed:
            raise ValueError(
                f"Seed mismatch: DB has seed={existing_seed}, but sweep requested seed={cfg.seed}."
            )
        seed = existing_seed
        sweep_indices = existing_indices
        print(f"Resuming existing study. Loaded seed={seed}, eval_size={cfg.eval_size}")

    else:
        seed = cfg.resolved_seed
        sweep_indices = draw_eval_indices(total_samples, cfg.eval_size, seed)

        study.set_user_attr("seed", seed)
        study.set_user_attr("eval_size", cfg.eval_size)
        study.set_user_attr("sweep_indices", sweep_indices)
        print(f"Created new study. Seed={seed}, eval_size={cfg.eval_size}")

    print(f"Sweep validation indices: {sweep_indices}\n")

    np.random.seed(seed)
    o3d.utility.random.seed(seed)

    if not cfg.use_wandb:
        run_ctx = contextlib.nullcontext()
    else:
        run_ctx = wandb.init(
            project="6dpose",
            name=cfg.name,
            group=cfg.estimator_cls.__name__,
            job_type="sweep",
            tags=[cfg.name],
            config={"eval_size": cfg.eval_size, "n_trials": cfg.trials, "seed": seed},
        )

    with run_ctx as run:
        log_input_artifacts(run, cfg.yolo, cfg.dataset)

        def objective(trial: optuna.Trial) -> tuple[float, float]:
            trial_seeds = derive_internal_seeds(seed, cfg.n_seeds, salt=trial.number)

            error_metrics: list[PoseErrorMetrics] = []
            times: list[float] = []
            frame_records: list[FrameRecord] = []
            det_failed = 0
            pose_failed = 0
            gross_yaw_rate_per_seed = []

            trial_params = cfg.estimator_cls.params_cls.sample_optuna(
                trial, base=cfg.resolved_params, fixed=resolved_overrides
            ).with_overrides(**resolved_overrides)

            for seed_i in trial_seeds:
                params_i = (
                    dataclasses.replace(trial_params, seed=seed_i)
                    if seed_i is not None
                    else trial_params
                )
                trial_estimator = cfg.estimator_cls(params=params_i, sensor=cfg.camera.sensor)

                for cart_type, mesh in meshes.items():
                    trial_estimator.prepare(mesh, cart_type)

                em, t, df, pf, fr = evaluate_pipeline(
                    dataset,
                    model,
                    camera,
                    trial_estimator,
                    sweep_indices,
                    meshes,
                    depth_trunc=cfg.depth_trunc,
                )
                error_metrics.extend(em)
                times.extend(t)
                det_failed += df
                pose_failed += pf
                frame_records.extend(fr)
                gross_yaw_rate_per_seed.append(compute_trial_metrics(em, t, df, pf).gross_yaw_rate)

            if cfg.dump_frames:
                write_frame_records_csv(
                    os.path.join(sweep_dir, f"{cfg.name}_frames", f"trial_{trial.number}.csv"),
                    frame_records,
                )

            m = compute_trial_metrics(error_metrics, times, det_failed, pose_failed)

            trial.set_user_attr("pose_ar", m.pose_ar)
            trial.set_user_attr("p95_latency_s", m.p95_latency_s)
            trial.set_user_attr("gross_yaw_rate", m.gross_yaw_rate)
            trial.set_user_attr("abstention_rate", m.abstention_rate)
            trial.set_user_attr("detection_failure_rate", m.detection_failure_rate)
            trial.set_user_attr("n_attempted", m.n_attempted)
            trial.set_user_attr("n_seeds", len(trial_seeds))
            trial.set_user_attr("trial_seeds", trial_seeds)
            trial.set_user_attr("gross_yaw_rate_per_seed", gross_yaw_rate_per_seed)

            if run is not None:
                run.log(
                    {
                        **dataclasses.asdict(trial_params),
                        "depth_trunc": cfg.depth_trunc,
                        "pose_ar": m.pose_ar,
                        "p95_latency_s": finite_or_none(m.p95_latency_s),
                        "gross_yaw_rate": m.gross_yaw_rate,
                        "abstention_rate": m.abstention_rate,
                        "detection_failure_rate": m.detection_failure_rate,
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

        finished = sum(1 for t in study.trials if t.state.is_finished())
        remaining = max(0, cfg.trials - finished)
        if finished:
            print(f"Resuming: {finished} finished trials in study, running {remaining} more.")
        try:
            if remaining > 0:
                study.optimize(objective, n_trials=remaining)
        finally:
            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if completed and run is not None:
                param_names = list(completed[0].params.keys())
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

                best_numbers = {t.number for t in study.best_trials}
                pareto = [t for t in completed if t.number in best_numbers]
                dominated = [t for t in completed if t.number not in best_numbers]

                run.log(
                    {
                        "pareto_front": build_pareto_figure(
                            pareto, dominated, param_names, cfg.name
                        ),
                        "pareto_table": wandb.Table(columns=columns, data=rows),
                    }
                )

            if cfg.dump_frames:
                log_frame_records_dir(run, cfg.name, os.path.join(sweep_dir, f"{cfg.name}_frames"))

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
    """Build the interactive Pareto-front scatter logged to W&B at sweep end."""
    BLUE, GRAY, GRID, AXIS = "#2a78d6", "#898781", "#e1e0d9", "#c3c2b7"
    INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

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
