import ast
import dataclasses

import numpy as np

from methods.base import BasePoseEstimator

# tyro hands override values through as raw strings; ast.literal_eval wants the
# Python spellings, so these three are title-cased before parsing.
_BOOLS = {"true", "false", "none"}


def resolve_param_overrides(
    estimator_cls: type[BasePoseEstimator],
    extrinsic: np.ndarray,
    overrides: dict | None,
) -> dict:
    """
    Validates and type-coerces CLI parameter overrides against the estimator's
    own params dataclass.

    An unknown field name raises rather than warning. The whole reason this
    mechanism exists is that a sweep silently ignored parameters set on the
    command line; replacing one silent no-op with another (a warning nobody reads
    in a 200-trial log) would leave the same failure available -- an arm that
    reports it is testing something while running the control.

    Values arrive as strings from tyro and are parsed as Python literals, so
    `front_face_max_angle_deg 60.0`, `voxel_size 0.04`, `z_offset None`,
    `icp_refine_ladder "(0.05,0.02,0.01)"` (note the syntax for tuple values)
    all land as the right type; anything unparseable is kept as the raw string.
    """
    if not overrides:
        return {}

    probe_params = estimator_cls(extrinsic=extrinsic).params
    valid_fields = {f.name for f in dataclasses.fields(probe_params)}

    resolved = {}
    for name, raw in overrides.items():
        if name not in valid_fields:
            raise ValueError(
                f"Unknown parameter override '{name}' for {estimator_cls.__name__}. "
                f"Available: {sorted(valid_fields)}"
            )
        if isinstance(raw, str):
            try:
                resolved[name] = ast.literal_eval(raw.capitalize() if raw in _BOOLS else raw)
            except (ValueError, SyntaxError):
                resolved[name] = raw
        else:
            resolved[name] = raw
    return resolved
