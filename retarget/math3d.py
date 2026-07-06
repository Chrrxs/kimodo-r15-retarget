"""Minimal pure-Python 3D math helpers for the BVH -> Roblox R15 retargeter.

Conventions (read this before touching anything below):

- Quaternions are tuples ``(w, x, y, z)`` and are kept unit-length.
- ``quat_mul(a, b)`` is the Hamilton product. When rotating COLUMN vectors the
  product applies ``b`` first, then ``a``:  ``R(quat_mul(a, b)) == R(a) @ R(b)``.
- Rotation matrices are row-major tuples of three rows, acting on column
  vectors (``v' = M @ v``).  This matches scipy's ``Rotation.as_matrix()``
  convention that the kimodo BVH pipeline uses.
- BVH Euler channels are INTRINSIC rotations applied in channel order.
  Channels ``Zrotation Yrotation Xrotation`` therefore mean
  ``R = Rz @ Ry @ Rx``  (equivalently ``q = qz * qy * qx``), exactly what
  scipy's ``Rotation.from_euler("ZYX", ...)`` produces (uppercase = intrinsic),
  which is what kimodo uses to read its own BVH exports.
- Angles are radians internally; degrees appear only at the BVH/CLI boundary.

No external dependencies: everything is plain tuples + ``math``.
"""

import math

# --------------------------------------------------------------------------
# Vectors
# --------------------------------------------------------------------------


def vec_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def vec_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec_cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vec_length(a):
    return math.sqrt(vec_dot(a, a))


def vec_normalize(a):
    n = vec_length(a)
    if n < 1e-12:
        raise ValueError("cannot normalize near-zero vector %r" % (a,))
    return (a[0] / n, a[1] / n, a[2] / n)


# --------------------------------------------------------------------------
# Quaternions (w, x, y, z)
# --------------------------------------------------------------------------

QUAT_IDENTITY = (1.0, 0.0, 0.0, 0.0)


def quat_mul(a, b):
    """Hamilton product a*b (rotation b applied first, then a)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_conj(q):
    """Conjugate == inverse for unit quaternions."""
    return (q[0], -q[1], -q[2], -q[3])


def quat_normalize(q):
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n < 1e-12:
        raise ValueError("cannot normalize near-zero quaternion %r" % (q,))
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def quat_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


def quat_from_axis_angle(axis, angle_rad):
    axis = vec_normalize(axis)
    h = 0.5 * angle_rad
    s = math.sin(h)
    return (math.cos(h), axis[0] * s, axis[1] * s, axis[2] * s)


_AXIS_VECTORS = {
    "X": (1.0, 0.0, 0.0),
    "Y": (0.0, 1.0, 0.0),
    "Z": (0.0, 0.0, 1.0),
}


def quat_from_bvh_euler(order, degrees):
    """Quaternion from BVH Euler channels.

    ``order`` is a string of axis letters in CHANNEL ORDER (e.g. ``"ZYX"`` for
    channels ``Zrotation Yrotation Xrotation``); ``degrees`` are the channel
    values in the same order.  Intrinsic composition: q = q0 * q1 * q2.
    """
    q = QUAT_IDENTITY
    for letter, deg in zip(order.upper(), degrees):
        q = quat_mul(q, quat_from_axis_angle(_AXIS_VECTORS[letter], math.radians(deg)))
    return q


def quat_rotate(q, v):
    """Rotate vector v by unit quaternion q (v' = q v q^-1)."""
    w, x, y, z = q
    # t = 2 * cross(q.xyz, v)
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])
    # v' = v + w*t + cross(q.xyz, t)
    return (
        v[0] + w * tx + (y * tz - z * ty),
        v[1] + w * ty + (z * tx - x * tz),
        v[2] + w * tz + (x * ty - y * tx),
    )


def quat_to_mat3(q):
    """Row-major 3x3 rotation matrix ((r00,r01,r02),(r10,..),(r20,..))."""
    w, x, y, z = quat_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def mat3_to_quat(m):
    """Quaternion from a row-major 3x3 rotation matrix (Shepperd's method)."""
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        return quat_normalize(
            (0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
        )
    if m[0][0] >= m[1][1] and m[0][0] >= m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        return quat_normalize(
            ((m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
        )
    if m[1][1] >= m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        return quat_normalize(
            ((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s)
        )
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
    return quat_normalize(
        ((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s)
    )


def quat_between(u, v):
    """Minimal rotation q with quat_rotate(q, u) == v (u, v unit-ish vectors).

    Uses the half-vector construction; for antiparallel inputs picks an
    arbitrary perpendicular 180-degree axis.  NOTE: the twist about the target
    vector is unconstrained by the inputs -- the minimal rotation is a
    canonical but not unique choice (relevant for arm-twist retargeting).
    """
    u = vec_normalize(u)
    v = vec_normalize(v)
    d = vec_dot(u, v)
    if d < -1.0 + 1e-9:
        # 180 degrees: rotate about any axis perpendicular to u.
        # Cross with the world axis least aligned with u for stability.
        ref = (1.0, 0.0, 0.0) if abs(u[0]) < 0.9 else (0.0, 1.0, 0.0)
        axis = vec_normalize(vec_cross(u, ref))
        return (0.0, axis[0], axis[1], axis[2])
    c = vec_cross(u, v)
    return quat_normalize((1.0 + d, c[0], c[1], c[2]))


def quat_angle(q):
    """Rotation angle of q in radians, in [0, pi]."""
    w = q[0]
    s = math.sqrt(q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    return 2.0 * math.atan2(s, abs(w))


def quat_angle_between(a, b):
    """Angle in radians of the relative rotation between two unit quaternions."""
    return quat_angle(quat_mul(quat_conj(a), b))


def quat_slerp(a, b, t):
    """Spherical linear interpolation from a (t=0) to b (t=1), shortest arc."""
    d = quat_dot(a, b)
    if d < 0.0:  # hemisphere alignment for shortest path
        b = (-b[0], -b[1], -b[2], -b[3])
        d = -d
    if d > 1.0 - 1e-9:
        # Nearly identical: nlerp to avoid division by ~0.
        q = tuple(a[i] + t * (b[i] - a[i]) for i in range(4))
        return quat_normalize(q)
    theta = math.acos(max(-1.0, min(1.0, d)))
    s = math.sin(theta)
    wa = math.sin((1.0 - t) * theta) / s
    wb = math.sin(t * theta) / s
    return quat_normalize(tuple(wa * a[i] + wb * b[i] for i in range(4)))


def lerp_vec(a, b, t):
    return tuple(a[i] + t * (b[i] - a[i]) for i in range(3))
