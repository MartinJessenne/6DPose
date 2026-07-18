"""
SE(2) utilities for planar pose estimation (ground-bounded cart: x, y, theta).

Provides:
  1. The 2-point minimal solver used by the SE(2)-constrained RANSAC
     (replaces the 3-point Kabsch/SVD solver needed in SE(3)).
  2. The se(2) exponential/logarithm maps (hat/vee, exp/log), intended for
     composing Gauss-Newton increments in a constrained ICP without leaving
     the group (naive addition of (theta, x, y) increments is wrong for
     large rotations).
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. RANSAC minimal solver: 2 correspondences -> (theta, t)
# ---------------------------------------------------------------------------

def minimal_solver_se2(p1, p2, q1, q2):
    """
    Solves the planar rigid transform (R(theta), t) such that
    q_i ~= R(theta) @ p_i + t, from 2 correspondences.

    Args:
        p1, p2: model points projected on the (x, y) plane -- shape (2,)
        q1, q2: corresponding scene points on the (x, y) plane -- shape (2,)

    Returns:
        (theta, t) with t of shape (2,).
    """
    p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    q1, q2 = np.asarray(q1, dtype=float), np.asarray(q2, dtype=float)

    v_p = p2 - p1
    v_q = q2 - q1

    theta = np.arctan2(v_q[1], v_q[0]) - np.arctan2(v_p[1], v_p[0])

    R = so2_exp(theta)
    # Anchor the translation on the centroids of both correspondences so
    # measurement noise on the two points is averaged instead of loaded
    # entirely onto the first one.
    t = 0.5 * (q1 + q2) - R @ (0.5 * (p1 + p2))
    return theta, t


def so2_exp(theta):
    """Exponential map so(2) -> SO(2): the 2x2 rotation matrix R(theta)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])


# ---------------------------------------------------------------------------
# 2. se(2) Lie algebra: hat / vee, exp / log
#    xi = (omega, vx, vy)  <->  xi_hat = 3x3 generator matrix
# ---------------------------------------------------------------------------

def se2_hat(xi):
    """xi = (omega, vx, vy) -> 3x3 generator matrix (element of se(2))."""
    omega, vx, vy = xi
    return np.array([
        [0.0,   -omega, vx],
        [omega,  0.0,   vy],
        [0.0,    0.0,   0.0],
    ])


def se2_vee(xi_hat):
    """Inverse of hat: 3x3 generator matrix -> vector (omega, vx, vy)."""
    omega = xi_hat[1, 0]
    vx, vy = xi_hat[0, 2], xi_hat[1, 2]
    return np.array([omega, vx, vy])


def _V_matrix(omega, eps=1e-8):
    """
    Left Jacobian V coupling rotation and translation in se2_exp.
    V -> I as omega -> 0 (straight-line motion, no arc correction).
    """
    if abs(omega) < eps:
        # First-order Taylor expansion: avoids division by zero, V ~ I
        return np.eye(2) + 0.5 * omega * np.array([[0, -1], [1, 0]])

    s, c = np.sin(omega), np.cos(omega)
    A = np.array([[0, -1], [1, 0]])
    return (s / omega) * np.eye(2) + ((1 - c) / omega) * A


def se2_exp(xi):
    """
    Exact exponential map: xi = (omega, vx, vy) -> 3x3 homogeneous matrix
    T = [[R, t], [0, 1]] in SE(2).

    Use this to compose a (delta_theta, delta_x, delta_y) increment from a
    constrained Gauss-Newton step without ever leaving the group.
    """
    omega, vx, vy = xi
    R = so2_exp(omega)
    V = _V_matrix(omega)
    t = V @ np.array([vx, vy])

    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2] = t
    return T


def se2_log(T, eps=1e-8):
    """
    Logarithm map: SE(2) homogeneous matrix -> xi = (omega, vx, vy).
    Inverse of se2_exp -- useful to compute an increment between two poses.
    """
    R = T[:2, :2]
    t = T[:2, 2]
    omega = np.arctan2(R[1, 0], R[0, 0])
    V = _V_matrix(omega, eps)
    v = np.linalg.solve(V, t)
    return np.array([omega, v[0], v[1]])


def se2_matrix(theta, t):
    """Builds the 3x3 homogeneous matrix from (theta, t=(x, y))."""
    T = np.eye(3)
    T[:2, :2] = so2_exp(theta)
    T[:2, 2] = t
    return T
