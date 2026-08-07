"""
The result contract shared by every registration stage.

Global registration (constrained_ransac, vsac_se2) and local refinement
(se2_icp) all return the same thing: a pose, how much of the model it explains,
how tightly, and -- on abstention -- why not. That contract belongs to none of
them in particular, so it lives here.

It used to live in methods/constrained_ransac.py under the name RansacResult,
which forced se2_icp to import from the RANSAC module to describe an ICP result.
ICP is not RANSAC; the dependency pointed backwards and the name was wrong at
half its call sites.

Deliberately free of open3d and of every solver in this package, so any stage
can import it without dragging a solver along.
"""

import numpy as np


class RegistrationResult:
    """
    Result container returned by any registration stage.

    API-compatible with the open3d registration result, so the fields read the
    same at a call site that used to consume one
    (T_init = result.transformation ; result.fitness).

    Attributes:
        transformation (np.ndarray): 4x4 homogeneous transformation matrix
            representing the estimated 6D/SE(2) pose of the CAD model in the
            scene frame (base_link).
        fitness (float): Fraction of downsampled model points that fall within
            `distance_threshold` of a scene point after applying `transformation`
            (in range [0.0, 1.0]).
        inlier_rmse (float): Root Mean Square Error (RMSE) of Euclidean distances
            for all inlier model points (points within `distance_threshold`).
        reason (str | None): On abstention (fitness == 0.0), WHICH check rejected
            the frame -- one of ABSTENTION_REASONS. Every abstention path used to
            collapse into the same anonymous `fitness == 0.0`, which made
            "the FPFH stage found no correspondences" indistinguishable from
            "every candidate pose was rejected" in the sweep logs. None on
            success.
    """

    def __init__(
        self,
        transformation: np.ndarray,
        fitness: float,
        inlier_rmse: float,
        reason: str | None = None,
    ):
        self.transformation = transformation
        self.fitness = fitness
        self.inlier_rmse = inlier_rmse
        self.reason = reason


# Abstention causes reported by the registration stage (RegistrationResult.reason).
# Kept as a flat tuple of plain strings rather than an Enum so they survive the
# trip into a CSV column / W&B table unchanged.
ABSTENTION_REASONS = (
    "fpfh_insufficient",  # fewer than DOF mutual FPFH correspondences
    "z_gate_insufficient",  # z-gate left fewer than DOF correspondences
    "no_inliers",  # no candidate pose scored a single inlier
    # VSAC only, and deliberately separate from "no_inliers": the pose WAS
    # scored, the Poisson null gate then vetoed it. Disappears if that gate is
    # ablated -- which is exactly why it needs its own bucket while it exists.
    # Emitted only by the ablated Poisson null gate. Kept in the list because
    # sweeps run before the ablation (e.g. study VSAC_NullOn_M2) have frame CSVs
    # full of it, and a reader of those files needs the name to mean something.
    "null_rejected",
    # VSAC only: every candidate that fit was vetoed by FrontFaceGate for pointing
    # its towing face away from the camera. Separate from "no_inliers" because the
    # two demand opposite responses -- no_inliers means the scene gave nothing to
    # work with, this means the gate refused what it was given, and if it appears
    # in bulk the gate is misconfigured (see the rejection-ratio log in vsac_se2).
    "orientation_rejected",
    "empty_scene_cloud",  # segmentation/reprojection produced no scene points
    "registration",  # abstained without naming a cause (estimator not yet instrumented)
)
