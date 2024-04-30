import numpy as np
from pytransform3d import transformations as pt
from sim_web_visualizer.base_visualizer_client import MeshCatVisualizerBase

from dvp_teleop_server.communication.visualizer_base import TeleopVisualizerBase


class TeleopWebVisualizer(TeleopVisualizerBase):
    def __init__(
        self,
        viz: MeshCatVisualizerBase,
        operator_name,
        robot_urdf_path: str,
        is_right_hand=True,
    ):
        super().__init__(
            operator_name=operator_name,
            is_right_hand=is_right_hand,
            robot_urdf_path=robot_urdf_path,
        )

        # Load different visualizer
        self.viz = viz
        self.operator_viz = self.viz.viz[f"/Teleop/{operator_name}/{self.hand_type}"]
        self.operator_viz.delete()
        self.robot_viz = self.viz.viz[f"/Robot/{self.hand_type}"]
        self.robot_viz.delete()
        self.ee_target_viz = None

        # Load robot
        robot_resources = self.viz.dry_load_asset(
            str(robot_urdf_path), collapse_fixed_joints=False
        )
        self.viz.load_asset_resources(robot_resources, self.robot_viz.path.lower())

        # Initialization
        for i, frame in enumerate(self.model.frames):
            self.robot_viz[frame.name].set_transform(self.data.oMf[i].homogeneous)

    def init_robot_base_pose(self, robot_base_pose_vec: np.ndarray):
        robot_base_mat = pt.transform_from_pq(robot_base_pose_vec)
        self.robot_viz.set_transform(robot_base_mat)

    def create_init_frame(self, init_frame_pose: np.ndarray):
        self.operator_viz.delete()
        if init_frame_pose is not None:
            self.viz.create_coordinate_axis(
                pose_mat=init_frame_pose,
                root_path=f"{self.operator_viz.path.lower()}/init",
                opacity=1,
                scale=0.2,
            )
            self.operator_viz["init"].set_property("opacity", 0.5)

    def update_init_viz(self, opacity: float):
        self.operator_viz["init"].set_property("opacity", opacity)

    def create_ee_target(
        self, ee_pose_vec_in_base: np.ndarray, robot_base_pose_vec: np.ndarray
    ):
        # Remove initialization marker
        self.viz.viz[f"{self.operator_viz.path.lower()}/init"].delete()

        # Set the robot base frame pose`
        robot_base_mat = pt.transform_from_pq(robot_base_pose_vec)
        self.operator_viz.set_transform(robot_base_mat)

        # Create new ee pose
        ee_pose_mat = pt.transform_from_pq(ee_pose_vec_in_base)
        path = self.operator_viz[f"{self.hand_type}_ee_target"].path.lower()
        self.ee_target_viz = self.viz.viz[path]
        self.viz.create_coordinate_axis(
            ee_pose_mat,
            path,
            scale=0.1,
            opacity=0.7,
            sphere_radius=0.02,
        )

    def update_ee_target(self, ee_pose_vec_in_base: np.ndarray):
        ee_pose_mat = pt.transform_from_pq(ee_pose_vec_in_base)
        self.ee_target_viz.set_transform(ee_pose_mat)

    def update_robot(self, qpos: np.ndarray):
        self._update_robot_kinematics(qpos)
        # TODO: code below is too slow
        for i, frame in enumerate(self.model.frames):
            self.robot_viz[frame.name].set_transform(self.data.oMf[i].homogeneous)
