"""
Post-sweep diagnostic: does gross_yaw_rate already correlate with accuracy loss?

Checks the hypothesis behind keeping gross_yaw_rate a passive (logged, not
optimized) metric in benchmark.py's sweep objective: a grossly mis-oriented
pose fails every BOP-style AR threshold by construction, so maximizing pose_ar
should already be near-equivalent to minimizing gross_yaw_rate wherever the
2-objective sweep actually searches -- which would make a dedicated 3rd Optuna
objective (or constraint) unnecessary. This script answers that empirically per
study instead of assuming it either way.

If accuracy loss and gross_yaw_rate are strongly positively correlated on the
Pareto front (both bad together, both good together) -- no further sweep
changes are needed. If not -- some region of the search trades orientation
quality against accuracy in a way the 2-objective search can't see, and it's
worth adding an Optuna constraints_func (NSGA-II) capping gross_yaw_rate on a
follow-up sweep, rather than a full 3rd objective (which would dilute search
budget across a 3-way Pareto front instead of the current 2D one).

A DECOUPLED result is itself diagnostic. Under the pre-2026-07-25 metrics the
correlation was +0.84 for Ransac3DoF but only +0.11 (n.s.) for VSAC -- because
the old flip_rate divided by the SUCCESS count, so an estimator could suppress
it just by abstaining. abstention_rate is reported alongside for exactly that
reason: if the correlation is weak, check whether abstention is what is
absorbing the difference.

Usage:
    uv run analyze_sweep.py <study_name> [--plot report.html]

Reads the same sweeps/optuna_<study_name>.db SQLite study that
run_parameter_sweep (benchmark.py) writes to.
"""

import argparse
import os

import numpy as np
import optuna
import plotly.graph_objects as go
from scipy import stats


def load_trial_data(study_name: str) -> tuple[optuna.Study, list[optuna.trial.FrozenTrial]]:
    project_root = os.path.dirname(os.path.abspath(__file__))
    db_name = os.path.join(project_root, "sweeps", f"optuna_{study_name}.db")
    db_url = f"sqlite:///{db_name}"
    study = optuna.load_study(study_name=study_name, storage=db_url)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    return study, completed


def correlate_gross_yaw_vs_accuracy(
    trials: list[optuna.trial.FrozenTrial],
) -> tuple[float, float]:
    """
    Spearman rank correlation between accuracy loss and gross_yaw_rate (a
    user_attr, never optimized directly).

    values[0] is pose_ar, which the sweep MAXIMIZES, so accuracy loss is
    1 - values[0]: both quantities then point the same way (bigger = worse) and
    a positive rho means "they move together", as the docstring above assumes.

    Spearman rather than Pearson: the hypothesis under test ("both move
    together") is a monotonic-association claim, not a linear one.
    """
    accuracy_loss = np.array([1.0 - t.values[0] for t in trials])
    gross_yaw = np.array([t.user_attrs.get("gross_yaw_rate", np.nan) for t in trials])
    valid = ~np.isnan(gross_yaw)
    rho, p_value = stats.spearmanr(accuracy_loss[valid], gross_yaw[valid])
    return float(rho), float(p_value)


def build_scatter_figure(study: optuna.Study, trials: list[optuna.trial.FrozenTrial], title: str):
    """
    gross_yaw_rate vs accuracy_loss scatter, Pareto-optimal trials highlighted --
    mirrors the color convention benchmark.py's own build_pareto_figure
    already established (blue = Pareto-optimal, gray = dominated), so a
    reader sees the same visual language in both places.
    """
    BLUE, GRAY, GRID, AXIS = "#2a78d6", "#898781", "#e1e0d9", "#c3c2b7"
    INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

    pareto_numbers = {t.number for t in study.best_trials}
    pareto = [t for t in trials if t.number in pareto_numbers]
    dominated = [t for t in trials if t.number not in pareto_numbers]

    fig = go.Figure()
    for group, color, name in [(dominated, GRAY, "Dominated"), (pareto, BLUE, "Pareto-optimal")]:
        if not group:
            continue
        fig.add_trace(
            go.Scatter(
                x=[1.0 - t.values[0] for t in group],
                y=[t.user_attrs.get("gross_yaw_rate") for t in group],
                mode="markers",
                name=name,
                marker=dict(size=8, color=color, line=dict(width=1, color=SURFACE)),
                # Abstention rides along in the tooltip: a point sitting at low
                # gross_yaw_rate AND high accuracy loss is almost always a trial
                # that abstained its way out of trouble.
                customdata=[[t.number, t.user_attrs.get("abstention_rate")] for t in group],
                hovertemplate=(
                    "<b>Trial %{customdata[0]}</b><br>"
                    "accuracy loss (1 - pose_ar): %{x:.4f}<br>"
                    "gross_yaw_rate: %{y:.4f}<br>"
                    "abstention_rate: %{customdata[1]:.4f}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="accuracy loss (1 - pose_ar)",
        yaxis_title="gross_yaw_rate",
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=dict(color=INK),
        xaxis=dict(gridcolor=GRID, linecolor=AXIS),
        yaxis=dict(gridcolor=GRID, linecolor=AXIS),
        legend=dict(font=dict(color=INK2)),
    )
    return fig


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("study_name", help="Optuna study name (matches --name used with --sweep)")
    parser.add_argument("--plot", default=None, help="Optional path to save an HTML scatter plot")
    args = parser.parse_args()

    study, trials = load_trial_data(args.study_name)
    if not trials:
        print(f"No completed trials found in study '{args.study_name}'.")
        return

    pareto_numbers = {t.number for t in study.best_trials}
    pareto_trials = [t for t in trials if t.number in pareto_numbers]

    rho_all, p_all = correlate_gross_yaw_vs_accuracy(trials)
    print(
        f"Study: {args.study_name}  "
        f"({len(trials)} completed trials, {len(pareto_trials)} Pareto-optimal)"
    )
    print(
        f"Spearman(accuracy_loss, gross_yaw_rate) -- all trials:   "
        f"rho={rho_all:+.3f}  p={p_all:.4f}"
    )

    # Abstention context: without it, a weak correlation is uninterpretable.
    abstentions = np.array(
        [t.user_attrs.get("abstention_rate", np.nan) for t in trials], dtype=float
    )
    abstentions = abstentions[~np.isnan(abstentions)]
    if abstentions.size:
        print(
            f"abstention_rate -- median={np.median(abstentions):.3f}  max={abstentions.max():.3f}"
        )

    if len(pareto_trials) < 3:
        print(
            "Pareto front has fewer than 3 trials -- not enough to estimate a "
            "separate correlation yet; run more trials."
        )
    else:
        rho_pareto, p_pareto = correlate_gross_yaw_vs_accuracy(pareto_trials)
        print(
            f"Spearman(accuracy_loss, gross_yaw_rate) -- Pareto front: "
            f"rho={rho_pareto:+.3f}  p={p_pareto:.4f}"
        )
        print()
        if rho_pareto > 0.5 and p_pareto < 0.05:
            print(
                "Strong positive correlation on the Pareto front: accuracy loss and "
                "gross_yaw_rate move together there, so maximizing pose_ar already pushes "
                "gross_yaw_rate down too. No dedicated objective/constraint needed yet."
            )
        else:
            print(
                "Correlation on the Pareto front is weak or not significant: some configs trade "
                "orientation quality against accuracy in ways the 2-objective search can't see. "
                "First check the abstention_rate line above -- a high median means the estimator "
                "is declining to answer on the hard frames rather than getting them wrong, which "
                "suppresses gross_yaw_rate without improving pose_ar. If abstention is low, "
                "consider an Optuna constraints_func (NSGA-II sampler) capping gross_yaw_rate on "
                "a follow-up sweep, rather than a full 3rd objective."
            )

    if args.plot:
        fig = build_scatter_figure(
            study, trials, title=f"{args.study_name}: gross_yaw_rate vs accuracy loss"
        )
        fig.write_html(args.plot)
        print(f"\nScatter saved to {args.plot}")


if __name__ == "__main__":
    main()
