from abc import abstractmethod

import numpy as np


class TeleopVisualizerBase:
    def __init__(
        self,
        operator_name,
        robot_urdf_path: str,
        is_right_hand=True,
    ):
        self.hand_type = "right" if is_right_hand else "left"
        self.operator_name = operator_name

        # Load robot
        try:
            import pinocchio as pin
        except ImportError:
            raise RuntimeError(
                f"To use web visualization in the teleop server, "
                f"please install the pinocchio package by \n"
                f"pip install pin \n"
                f"More information: https://pypi.org/project/pin/"
            )

        self.model: pin.Model = pin.buildModelFromUrdf(robot_urdf_path)
        self.data: pin.Data = self.model.createData()
        self.neutral_qpos = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, self.neutral_qpos)
        pin.updateFramePlacements(self.model, self.data)

        self.index_client2viz = None

    # -------------------------------------------------------------------------- #
    # Robot specific method
    # -------------------------------------------------------------------------- #
    def get_robot_joint_names(self):
        names = list(self.model.names)
        return names[1:]

    def set_joint_index_mapping(self, index_client2viz):
        self.index_client2viz = index_client2viz

    def _update_robot_kinematics(self, qpos: np.ndarray):
        import pinocchio as pin

        qpos_viz = qpos[self.index_client2viz]
        pin.forwardKinematics(self.model, self.data, qpos_viz)
        pin.updateFramePlacements(self.model, self.data)

    # -------------------------------------------------------------------------- #
    # Abstract method interface
    # -------------------------------------------------------------------------- #

    @abstractmethod
    def init_robot_base_pose(self, robot_base_pose_vec: np.ndarray):
        pass

    @abstractmethod
    def create_init_frame(self, init_frame_pose_vec: np.ndarray):
        pass

    @abstractmethod
    def update_init_viz(self, opacity: float):
        pass

    @abstractmethod
    def create_ee_target(
        self, ee_pose_vec_in_base: np.ndarray, robot_base_pose_vec: np.ndarray
    ):
        pass

    @abstractmethod
    def update_ee_target(self, ee_pose_vec_in_base: np.ndarray):
        pass

    @abstractmethod
    def update_robot(self, qpos: np.ndarray):
        pass
