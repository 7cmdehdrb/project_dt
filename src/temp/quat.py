import numpy as np


def transform_euler_to_quat(x, y, z):
    """
    Convert Euler angles (in radians) to a quaternion.
    The input is expected to be in the order of roll (X), pitch (Y), yaw (Z).
    """
    tx = x
    tz = z
    ty = y

    roll, pitch, yaw = tx, ty, tz

    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


def setup(db: og.Database):
    pass


def cleanup(db: og.Database):
    pass


def compute(db: og.Database):
    x = db.inputs.x
    y = db.inputs.y
    z = db.inputs.z

    quat = transform_euler_to_quat(x, y, z)
    db.outputs.quaternion = quat  # Output the quaternion as [w, x, y
    return True
