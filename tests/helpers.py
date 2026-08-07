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

from methods.se2_lie_utils import se2_matrix, se2_to_se3


def se2_pose(theta: float, t_xy, z: float = 0.0) -> np.ndarray:
    """Builds a 4x4 planar pose from a yaw angle, an (x, y) translation and a height."""
    return se2_to_se3(se2_matrix(theta, np.asarray(t_xy, dtype=float)), z)
