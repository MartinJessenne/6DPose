"""
Shared construction helpers for the SE(2) test suites.

Production code builds a planar pose in two steps -- se2_matrix to get the 3x3
SE(2) matrix, se2_to_se3 to embed it at a height -- because the two halves are
independently useful there (the ICP feeds se2_to_se3 an increment straight from
the exponential map, never an angle). Tests almost always want both at once and
start from an angle, so they get one shorthand rather than repeating the pair at
every call site.
"""

import numpy as np

from methods.depth_noise import DepthSensor
from methods.se2_lie_utils import se2_matrix, se2_to_se3

# Camera looking along robot +x from `standoff` behind the origin. Camera frame
# is x-right / y-down / z-forward (OpenCV), robot is x-forward / y-left / z-up,
# so the columns below are the images of the camera axes in robot coordinates:
# cam x -> -y, cam y -> -z, cam z -> +x.
_R_ROBOT_CAMERA = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)

# The rig's real intrinsics (cli_config.CameraConfig), so a synthetic fixture
# sitting at the default 3 m standoff gets c_bar = 3^2 * 0.1 / 60.80 = 0.0148 m
# -- the constant these suites used before the noise model became per-point.
_FX = 639.99768
_BASELINE = 0.095


def se2_pose(theta: float, t_xy, z: float = 0.0) -> np.ndarray:
    """Builds a 4x4 planar pose from a yaw angle, an (x, y) translation and a height."""
    return se2_to_se3(se2_matrix(theta, np.asarray(t_xy, dtype=float)), z)


def depth_sensor(standoff: float = 3.0, sigma_disparity: float = 0.1) -> DepthSensor:
    """
    A DepthSensor viewing the origin from `standoff` meters along robot -x.

    Fixtures in these suites are built around the origin with an extent of about
    a meter, so the default puts every scene point at a camera-frame depth of
    2-4 m: comfortably positive (noise_bound rejects points behind the camera)
    and in the range the rig actually operates at.
    """
    T = np.eye(4)
    T[:3, :3] = _R_ROBOT_CAMERA
    T[:3, 3] = [-standoff, 0.0, 0.0]
    return DepthSensor(
        fx=_FX, baseline=_BASELINE, sigma_disparity=sigma_disparity, T_robot_camera=T
    )


def unweighted_sensor() -> DepthSensor:
    """
    The hard-truncation control arm: a sensor so noisy that nothing can be an
    outlier.

    At sigma_disparity = 1e6 px every c_bar is enormous, so mu_init =
    2 max(r/c_bar)^2 lands below 1 (the schedule collapses to a flat mu = 1) and
    every Geman-McClure weight is 1 to within float noise. That is exactly plain
    least squares inside the KD-tree radius -- the behaviour that shipped before
    GNC -- expressed through the treatment's own code path rather than a
    separate branch that could rot.
    """
    return depth_sensor(sigma_disparity=1e6)
