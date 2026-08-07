"""
SE(2) utilities for planar pose estimation (ground-bounded cart: x, y, theta).

Provides:
  1. The 2-point minimal solver used by the SE(2)-constrained RANSAC
     (replaces the 3-point Kabsch/SVD solver needed in SE(3)).
  2. The se(2) exponential/logarithm maps (hat/vee, exp/log), intended for
     composing Gauss-Newton increments in a constrained ICP without leaving
     the group (naive addition of (theta, x, y) increments is wrong for
     large rotations).
  3. The bridge between the two representations this project uses: the 3x3
     homogeneous SE(2) matrix that the maths is written in, and the 4x4
     SE(3)-embedded matrix that every downstream consumer expects.

Section 3 is the module's reason to be the single home for all of this. A
planar pose has exactly three numbers in it, but the pipeline passes it around
as a 4x4. Every place that crosses that boundary by hand is a place that can
disagree with the others -- and before this module owned the crossing, four
of them had. Anything that converts, composes, or projects a planar pose
belongs here; nothing that knows what a residual is does.
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
    return np.array([[c, -s], [s, c]])


# ---------------------------------------------------------------------------
# 2. se(2) Lie algebra: hat / vee, exp / log
#    xi = (omega, vx, vy)  <->  xi_hat = 3x3 generator matrix
# ---------------------------------------------------------------------------


def se2_hat(xi):
    """xi = (omega, vx, vy) -> 3x3 generator matrix (element of se(2))."""
    omega, vx, vy = xi
    return np.array(
        [
            [0.0, -omega, vx],
            [omega, 0.0, vy],
            [0.0, 0.0, 0.0],
        ]
    )


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


def se2_exp_about(xi, center):
    """
    Exponential map with the rotation applied about `center` rather than about
    the origin: the increment that spins the object in place.

    This is the conjugation of exp(xi) by a pure translation to `center`,

        Delta_c = Trans(c) . exp(xi) . Trans(c)^-1

    which expands to the same rotation block with the translation column
    c - R c + t. Setting x = c in the result gives y = c + t: the centre point
    only translates, it never swings.

    Why it exists -- and what it is NOT worth, which was measured rather than
    assumed:

    The Gauss-Newton Jacobian of a planar fit has one rotation column, of size
    |p_i - c|, and two translation columns of size ~1 (unit normals). Rotating
    about the world origin makes the rotation column scale with the object's
    DISTANCE FROM THE ORIGIN (2-3 m for a cart seen from the robot base) rather
    than with the object's own RADIUS (~0.3 m). The columns also lose their
    independence: to first order R(w) p - p = w (-p_y, p_x), so once |p| dwarfs
    the object's extent that displacement is nearly identical for every point,
    and a displacement identical everywhere IS a translation. Both effects show
    up as conditioning. Measured on the corner fixture in tests/test_se2.py:

        standoff    cond(J^T J) centred    origin-centred
        0.36 m            3.1                 2.5
        3.6 m             3.1                 3.3e2
        36 m              3.1                 2.5e6

    Centring on the centroid holds it at ~3 at any range; the origin-centred
    form degrades without bound. The centroid is the sharpest choice because
    for a common normal n the rotation/translation cross term of J^T J is
    n_y n_x sum_i (p_x,i - c_x) - n_x^2 sum_i (p_y,i - c_y), and both sums
    vanish identically by the definition of a centroid. Real normals vary, so
    only their fluctuation about the mean survives -- small.

    What this does NOT buy, contrary to what the previous version of this code
    implied: better convergence. Recentring is an invertible reparameterisation
    of the tangent space, and Gauss-Newton with an exact linear solve is
    equivariant under such a reparameterisation -- the composed pose increment
    is the same group element either way. Measured at 3.6 m standoff, one step
    computed both ways agrees to 1e-14, and full ICP runs agree to the last
    digit of the reported yaw. The converged fixed point is defined by
    J^T r = 0, which does not know where the rotation centre was.

    So this is insurance, not a fix, and it is worth keeping only because it is
    free. It starts to matter when equivariance breaks or precision runs out:
    under Levenberg-Marquardt damping (a lambda I term is NOT invariant to
    reparameterisation, so the centre would then steer the trajectory), in
    float32, or at ranges well beyond this workcell. Do not cite it to explain
    a convergence result -- it cannot cause one.

    Args:
        xi: (omega, vx, vy), an element of the Lie algebra se(2). A velocity,
            not a pose.
        center: (2,) the point to rotate about, in the same frame as the pose
            this increment will be composed onto.

    Returns:
        3x3 homogeneous SE(2) matrix.
    """
    center = np.asarray(center, dtype=float)
    T = se2_exp(xi)
    T[:2, 2] += center - T[:2, :2] @ center
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


# ---------------------------------------------------------------------------
# 3. Representation bridge: 3x3 SE(2)  <->  4x4 SE(3), z-up
# ---------------------------------------------------------------------------


def se2_to_se3(T2: np.ndarray, z: float = 0.0) -> np.ndarray:
    """
    Embeds a 3x3 homogeneous SE(2) matrix into a 4x4 SE(3) matrix at height `z`,
    with roll and pitch exactly zero.

    The third row and column of the result are whatever the 4x4 identity
    already provides, and are never written. That is what makes the output
    EXACTLY planar rather than approximately planar: there is no arithmetic on
    the vertical axis to accumulate rounding error, so no amount of iteration
    can drift the pose off the ground plane and no re-projection is needed
    afterwards.

    Takes a matrix rather than (theta, t) so that it composes with everything
    upstream of it: se2_exp for a Gauss-Newton increment, se2_exp_about for a
    recentred one, se2_matrix when a caller genuinely starts from an angle.
    Building the rotation from an angle in here instead would duplicate
    so2_exp, which is how this project ended up with four hand-rolled copies of
    this embedding in the first place.

    Args:
        T2: 3x3 homogeneous SE(2) matrix.
        z: Height of the plane, in metres. Leave at 0.0 for an INCREMENT --
           an increment must not translate vertically, or the ground constraint
           is broken by the very operation meant to preserve it. Pass the
           object's z_offset when building a POSE.

    Returns:
        4x4 SE(3) matrix.
    """
    T = np.eye(4)
    T[:2, :2] = T2[:2, :2]
    T[:2, 3] = T2[:2, 2]
    T[2, 3] = z
    return T


def project_to_se2(T: np.ndarray, z_offset: float = 0.0) -> np.ndarray:
    """
    Projects an arbitrary SE(3) 4x4 matrix onto the SE(2) ground-plane manifold:
    nearest planar yaw, roll = pitch = 0, z pinned to `z_offset`.

    The yaw is the Frobenius-nearest rotation about +Z, i.e. the theta
    minimising ||R - R_z(theta)||_F. Writing that out, the minimiser is
        theta = atan2(R_10 - R_01, R_00 + R_11)
    -- the antisymmetric part of the upper-left 2x2 block over its symmetric
    part. This is the 2D case of the usual SVD orthogonalisation, and it is
    closed-form precisely because SO(2) is one-dimensional.

    Caller trace:
    - Ransac3DoFEstimator._project_pose, as a final safety net guaranteeing the
      planar invariants hold on the returned pose.
    """
    R = T[:3, :3]
    theta = np.arctan2(R[1, 0] - R[0, 1], R[0, 0] + R[1, 1])
    return se2_to_se3(se2_matrix(theta, T[:2, 3]), z_offset)
