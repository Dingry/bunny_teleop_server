import numpy as np
from pytransform3d import rotations, coordinates

MANO2ROBOT = np.array(
    [
        [-1, 0, 0],
        [0, 0, -1],
        [0, -1, 0],
    ]
)

OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)

OPERATOR2AVP_RIGHT = OPERATOR2MANO_RIGHT

OPERATOR2AVP_LEFT = np.array(
    [
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ]
)


def project_average_rotation(quat_list: np.ndarray):
    gravity_dir = np.array([0, 0, -1])

    last_quat = quat_list[-1, :]
    last_mat = rotations.matrix_from_quaternion(last_quat)
    gravity_quantity = gravity_dir @ last_mat  # (3, )
    max_gravity_axis = np.argmax(np.abs(gravity_quantity))
    same_direction = gravity_quantity[max_gravity_axis] > 0

    next_axis = (max_gravity_axis + 1) % 3
    next_next_axis = (max_gravity_axis + 2) % 3
    angles = []
    for i in range(quat_list.shape[0]):
        next_dir = rotations.matrix_from_quaternion(quat_list[i])[:3, next_axis]
        next_dir[2] = 0  # Projection to non gravity direction
        next_dir_angle = coordinates.spherical_from_cartesian(next_dir)[2]
        angles.append(next_dir_angle)

    angle = np.mean(angles)
    final_mat = np.zeros([3, 3])
    final_mat[:3, max_gravity_axis] = gravity_dir * same_direction
    final_mat[:3, next_axis] = [np.cos(angle), np.sin(angle), 0]
    final_mat[:3, next_next_axis] = np.cross(
        final_mat[:3, max_gravity_axis], final_mat[:3, next_axis]
    )
    return rotations.quaternion_from_matrix(final_mat, strict_check=True)


class LPFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.y = None
        self.is_init = False

    def next(self, x):
        if not self.is_init:
            self.y = x
            self.is_init = True
            return self.y.copy()
        self.y = self.y + self.alpha * (x - self.y)
        return self.y.copy()

    def reset(self):
        self.y = None
        self.is_init = False


class LPRotationFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.is_init = False

        self.y = None

    def next(self, x: np.ndarray):
        assert x.shape == (4,)

        if not self.is_init:
            self.y = x
            self.is_init = True
            return self.y.copy()

        self.y = rotations.quaternion_slerp(self.y, x, self.alpha, shortest_path=True)
        return self.y.copy()

    def reset(self):
        self.y = None
        self.is_init = False
