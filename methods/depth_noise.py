"""
The depth sensor: where it sits, and how badly it lies at a given range.

See docs/explanation/gnc_refinement.md for the noise derivation and why the
bound is per-point rather than a constant.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, eq=False)
class DepthSensor:
    """
    A stereo depth camera: its pose in the robot base frame, and its noise model.

    Replaces the `extrinsic: np.ndarray | None = None` that every estimator used
    to take -- optional in the signature, mandatory in the body.

    eq=False because a generated __eq__ would compare ndarrays with ==, which
    returns an array rather than a bool.

    Attributes:
        fx: Horizontal focal length, pixels.
        baseline: Stereo baseline, meters.
        sigma_disparity: Sub-pixel disparity matching accuracy, pixels. Lives
            here rather than on the estimator's params because params are the
            sweepable surface, and this is a measurable sensor property.
        T_robot_camera: 4x4 rigid transform, camera frame to robot base frame.
    """

    fx: float
    baseline: float
    sigma_disparity: float
    T_robot_camera: np.ndarray

    T_camera_robot: np.ndarray = field(init=False)

    def __post_init__(self):
        if self.fx <= 0:
            raise ValueError(f"fx must be positive, got {self.fx}")
        if self.baseline <= 0:
            raise ValueError(f"baseline must be positive, got {self.baseline}")
        if self.sigma_disparity <= 0:
            raise ValueError(f"sigma_disparity must be positive, got {self.sigma_disparity}")

        T = np.asarray(self.T_robot_camera, dtype=float)
        if T.shape != (4, 4):
            raise ValueError(f"T_robot_camera must be 4x4, got {T.shape}")

        # object.__setattr__ is how a frozen dataclass assigns derived fields.
        object.__setattr__(self, "T_robot_camera", T)
        # Analytic rigid inverse: for [R t; 0 1] the inverse is [R^T, -R^T t].
        R, t = T[:3, :3], T[:3, 3]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t
        object.__setattr__(self, "T_camera_robot", T_inv)

    @property
    def camera_origin(self) -> np.ndarray:
        """(3,) camera centre in the robot base frame."""
        return self.T_robot_camera[:3, 3]

    def depth(self, points_robot: np.ndarray) -> np.ndarray:
        """
        (N,) camera-frame depth: the coordinate ALONG THE OPTICAL AXIS, not the
        Euclidean range from the camera centre. Only the third row of the inverse
        transform is needed.
        """
        pts = np.asarray(points_robot, dtype=float)
        return pts @ self.T_camera_robot[2, :3] + self.T_camera_robot[2, 3]

    def noise_bound(self, points_robot: np.ndarray) -> np.ndarray:
        """
        (N,) one-sigma depth uncertainty in meters, sigma_z = z^2 sigma_d / (f B).

        This is `c_bar` in Yang et al. (RA-L 2020) -- the scale at which the
        robust kernel stops treating a residual as noise.

        Raises:
            ValueError: on non-positive camera-frame depth. Scene points come
                from a depth image, so z > 0 by construction; a violation means
                the cloud is in the wrong frame or T_robot_camera is inverted.
                Clipping would produce a zero-width kernel that silently drives
                every weight to zero.
        """
        z = self.depth(points_robot)
        if z.size and z.min() <= 0.0:
            raise ValueError(
                f"DepthSensor.noise_bound got a point at camera-frame depth "
                f"{z.min():.4g} m (must be > 0). The cloud is probably in the "
                "wrong frame, or T_robot_camera is inverted."
            )
        return z**2 * self.sigma_disparity / (self.fx * self.baseline)
