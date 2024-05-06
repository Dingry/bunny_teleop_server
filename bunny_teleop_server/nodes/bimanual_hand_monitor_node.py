from copy import deepcopy
from threading import Lock
from typing import Tuple

import numpy as np
import rclpy
from bunny_teleop.init_config import BimanualAlignmentMode
from geometry_msgs.msg import Pose
from hand_msgs.msg import BimanualHandDetection
from pytransform3d import rotations
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from bunny_teleop_server.utils.robot_utils import project_average_rotation


class BimanualMonitorNode(Node):
    def __init__(
        self,
        detection_topic_name: str,
        need_init=True,
        verbose=False,
    ):
        super().__init__("hand_monitor")
        self.verbose = verbose

        # Detection results subscriber
        self._detection_sub_group = MutuallyExclusiveCallbackGroup()
        self.detection_sub = self.create_subscription(
            BimanualHandDetection,
            f"{detection_topic_name}/results",
            self.on_hand_detection,
            qos_profile=10,
            callback_group=self._detection_sub_group,
        )

        # Initialization settings. These settings will be overridden by InitializationConfig from client
        self.align_gravity_dir = True
        self.bimanual_alignment_mode: BimanualAlignmentMode = (
            BimanualAlignmentMode.ALIGN_CENTER
        )

        # Initial calibration
        self._init_left_hand_wrist_pose_list = []
        self._init_right_hand_wrist_pose_list = []
        self.init_process = 0
        self.initialized = not need_init
        self.ready_to_reinit = False

        # Initialization frame information in the humand world
        # Note: for single hand alignment mode (currently all modes except ALIGN_SEPARATELY),
        # these two values will be same. For ALIGN_SEPARATELY, both value are meaningful
        self.calibrated_init_pos = (np.zeros(3), np.zeros(3))
        self.calibrated_init_rot = (np.eye(3), np.zeros(3))

        # Multi-thread variables for wrist pose and hand joints
        # Wrist pose is in the operator frame defined during initialization and hand joints in local wrist frame
        # Note that we always convert the joint representation in the publisher to mediapipe style,
        # no matter whether we are using Apple Vision Pro with a different joint representation
        self.hand_pose_lock = Lock()
        self.last_wrist_pose_in_init_frame = [
            np.array([0, 0, 0, 1, 0, 0, 0]),
            np.array([0, 0, 0, 1, 0, 0, 0]),
        ]
        self.last_hand_joints = [np.zeros([21, 3]), np.zeros([21, 3])]

    def on_hand_detection(self, data: BimanualHandDetection):
        """
        Callback function when receive hand detection results from hand detection node
        At hand monitor level, this function update the wrist pose and hand joints.

        :param data: BimanualHandDetection message data from detection node
        """
        is_success = data.detected
        if not is_success:
            return
        left_global_pose = data.left_wrist_pose
        right_global_pose = data.right_wrist_pose
        left_pos, left_quat = ros_pose_to_pos_quat(left_global_pose)
        right_pos, right_quat = ros_pose_to_pos_quat(right_global_pose)
        left_hand_joints = np.reshape(data.left_joints, [21, 3])
        right_hand_joints = np.reshape(data.right_joints, [21, 3])

        if not self.initialized:
            self._init_calibration(
                np.concatenate([left_pos, left_quat]),
                left_hand_joints,
                np.concatenate([right_pos, right_quat]),
                right_hand_joints,
            )

            if not self.initialized:
                return

        left_local_pos, left_local_quat = self._compute_pose_in_init_frame(
            left_pos, left_quat, hand_num=0
        )
        right_local_pos, right_local_quat = self._compute_pose_in_init_frame(
            right_pos, right_quat, hand_num=1
        )

        # Compare the wrist between this and last steps
        # If the hand wrist motion is very large between two detection results, abort it
        with self.hand_pose_lock:
            last_hand_wrist_pose = self.last_wrist_pose_in_init_frame.copy()
        last_left_wrist_pose, last_right_wrist_pose = (
            last_hand_wrist_pose[0],
            last_hand_wrist_pose[1],
        )
        far_away_threshold = 0.4
        not_too_far_away = (
            np.linalg.norm(last_left_wrist_pose[:3] - left_local_pos)
            < far_away_threshold
            and np.linalg.norm(last_right_wrist_pose[:3] - right_local_pos)
            < far_away_threshold
        )

        # If it is the first time to receive a wrist pose after init, no need to check distance
        just_started = (
            np.sum(np.abs(last_left_wrist_pose[:3])) < 1e-3
            and np.sum(np.abs(last_right_wrist_pose[:3])) < 1e-3
        )

        if not_too_far_away or just_started:
            with self.hand_pose_lock:
                self.last_wrist_pose_in_init_frame[0] = np.concatenate(
                    [left_local_pos, left_local_quat]
                )
                self.last_wrist_pose_in_init_frame[1] = np.concatenate(
                    [right_local_pos, right_local_quat]
                )

                self.last_hand_joints[0] = left_hand_joints
                self.last_hand_joints[1] = right_hand_joints

    def _compute_pose_in_init_frame(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        hand_num,
    ) -> Tuple[np.ndarray, np.ndarray]:
        operator_rot = rotations.matrix_from_quaternion(quat)
        local_operator_rot = self.calibrated_init_rot[hand_num].T @ operator_rot
        local_operator_pos = (
            self.calibrated_init_rot[hand_num].T @ pos
            - self.calibrated_init_rot[hand_num].T @ self.calibrated_init_pos[hand_num]
        )
        local_operator_quat = rotations.quaternion_from_matrix(local_operator_rot)
        return local_operator_pos, local_operator_quat

    @staticmethod
    def _compute_hand_joint_angles(joints: np.ndarray):
        tip_index = np.array([4, 8, 12, 16, 20])
        palm_bone_index = np.array([1, 5, 9, 13, 17])
        root = joints[0:1, :]
        tips = joints[tip_index]
        palm_bone = joints[palm_bone_index]
        tip_vec = tips - root
        tip_vec = tip_vec / np.linalg.norm(tip_vec, axis=1, keepdims=True)
        palm_vec = palm_bone - root
        palm_vec = palm_vec / np.linalg.norm(palm_vec, axis=1, keepdims=True)
        angle = np.arccos(np.clip(np.sum(tip_vec * palm_vec, axis=1), -1.0, 1.0))
        return angle

    def _compute_init_frame(
        self, init_frame_pos: np.ndarray, init_frame_quat: np.ndarray
    ):
        num_data = init_frame_pos.shape[0]
        weight = (np.arange(num_data) + 1) / np.sum(np.arange(num_data) + 1)

        if self.align_gravity_dir:
            calibrated_quat = project_average_rotation(init_frame_quat)
        else:
            calibrated_quat = init_frame_quat[-1]
        print(
            f"Initialization rotation matrix: \n {rotations.matrix_from_quaternion(calibrated_quat)}"
        )
        calibrated_wrist_pos = np.sum(weight[:, None] * init_frame_pos, axis=0).astype(
            np.float32
        )
        calibrated_wrist_rot = rotations.matrix_from_quaternion(calibrated_quat)

        return calibrated_wrist_pos, calibrated_wrist_rot

    def _init_calibration(
        self,
        left_hand_wrist_pose: np.ndarray,
        left_hand_joints: np.ndarray,
        right_hand_wrist_pose: np.ndarray,
        right_hand_joints: np.ndarray,
    ):
        init_begin = self.init_process > 0
        if init_begin:
            # Check whether the current hand pose is good to continue initialization
            left_hand_continue_init = self._is_hand_pose_good_for_init(
                left_hand_wrist_pose,
                left_hand_joints,
                self._init_left_hand_wrist_pose_list[-1],
            )
            right_hand_continue_init = self._is_hand_pose_good_for_init(
                right_hand_wrist_pose,
                right_hand_joints,
                self._init_right_hand_wrist_pose_list[-1],
            )
            continue_init = left_hand_continue_init and right_hand_continue_init

            # Stop initialization process and clear data if not continue init
            if not continue_init:
                self._init_left_hand_wrist_pose_list.clear()
                self._init_right_hand_wrist_pose_list.clear()
                self.init_process = 0

        # Update initialization buffer to track shape and root pose
        self.init_process += 0.005
        self._init_left_hand_wrist_pose_list.append(left_hand_wrist_pose)
        self._init_right_hand_wrist_pose_list.append(right_hand_wrist_pose)

        # When initialization is finished, compute the initialization results
        if self.init_process >= 1:
            left_init_collect_data = np.stack(self._init_left_hand_wrist_pose_list)
            right_init_collect_data = np.stack(self._init_right_hand_wrist_pose_list)

            # Compute init_frame_pos based on bimanual alignment mode
            if self.bimanual_alignment_mode == BimanualAlignmentMode.ALIGN_CENTER:
                init_frame_pos = (
                    left_init_collect_data[:, :3] + right_init_collect_data[:, :3]
                ) / 2
                # TODO: better method to handle orientation
                init_frame_quat = right_init_collect_data[:, 3:7]
            elif self.bimanual_alignment_mode == BimanualAlignmentMode.ALIGN_LEFT:
                init_frame_pos = left_init_collect_data[:, :3]
                init_frame_quat = left_init_collect_data[:, 3:7]
            elif self.bimanual_alignment_mode == BimanualAlignmentMode.ALIGN_RIGHT:
                init_frame_pos = right_init_collect_data[:, :3]
                init_frame_quat = right_init_collect_data[:, 3:7]
            elif self.bimanual_alignment_mode == BimanualAlignmentMode.ALIGN_SEPARATELY:
                init_frame_pos = (
                    left_init_collect_data[:, :3],
                    right_init_collect_data[:, :3],
                )
                init_frame_quat = (
                    left_init_collect_data[:, 3:7],
                    right_init_collect_data[:, 3:7],
                )
            else:
                raise NotImplementedError

            # Single frame alignment mode
            if self.bimanual_alignment_mode != BimanualAlignmentMode.ALIGN_SEPARATELY:
                calibrated_wrist_pos, calibrated_wrist_rot = self._compute_init_frame(
                    init_frame_pos, init_frame_quat
                )
                self.calibrated_init_pos = (calibrated_wrist_pos, calibrated_wrist_pos)
                self.calibrated_init_rot = (calibrated_wrist_rot, calibrated_wrist_rot)
            # Double frame alignment mode
            else:
                left_calibrated_pos, left_calibrated_rot = self._compute_init_frame(
                    init_frame_pos[0], init_frame_quat[0]
                )
                right_calibrated_pos, right_calibrated_rot = self._compute_init_frame(
                    init_frame_pos[1], init_frame_quat[1]
                )

                self.calibrated_init_pos = (left_calibrated_pos, right_calibrated_pos)
                self.calibrated_init_rot = (left_calibrated_rot, right_calibrated_rot)

            self.initialized = True

    def _is_hand_pose_good_for_init(
        self,
        hand_wrist_pose: np.ndarray,
        hand_joints: np.ndarray,
        last_hand_wrist_pose: np.ndarray,
    ):
        # Check whether the hand wrist is moving during initialization
        not_far_away_threshold = 0.05
        hand_not_far_away = (
            np.linalg.norm(hand_wrist_pose[:3] - last_hand_wrist_pose[:3])
            < not_far_away_threshold
        )

        # Check whether the fingers are in a flatten way
        flat_threshold = (0.01, np.deg2rad(15))
        finger_angles = self._compute_hand_joint_angles(hand_joints)
        hand_flat_spread = (
            flat_threshold[0] < np.mean(finger_angles) < flat_threshold[1]
        )

        return hand_not_far_away and hand_flat_spread

    def prepare_reinit(self):
        self.initialized = False
        self.init_process = 0
        self._init_left_hand_wrist_pose_list.clear()
        self._init_right_hand_wrist_pose_list.clear()
        self.calibrated_init_pos = (np.zeros(3), np.zeros(3))
        self.calibrated_init_rot = (np.eye(3), np.zeros(3))

    def get_last_wrist_poses(self):
        with self.hand_pose_lock:
            return deepcopy(self.last_wrist_pose_in_init_frame)

    def get_last_hand_joints(self):
        with self.hand_pose_lock:
            return deepcopy(self.last_hand_joints)


def run_hand_node(node: Node, thread_num=2):
    # The spin function will use a single-thread executor by default. We can not use that
    # The multi-thread executor is used instead
    # ref: https://docs.ros.org/en/humble/Concepts/About-Executors.html
    executor = MultiThreadedExecutor(thread_num)
    try:
        node.get_logger().info(
            "Beginning bimanual hand monitor node, shut down with CTRL-C"
        )
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.\n")
    node.destroy_node()
    rclpy.shutdown()


def ros_pose_to_pos_quat(msg: Pose):
    pos = msg.position
    pos = np.array([pos.x, pos.y, pos.z])
    quat = msg.orientation
    quat = np.array([quat.w, quat.x, quat.y, quat.z])
    return pos, quat


def main():
    rclpy.init(args=None)
    teleop_node = BimanualMonitorNode(
        detection_topic_name="/visionpro/bimanual_hand_detection",
        verbose=True,
    )
    run_hand_node(teleop_node)


if __name__ == "__main__":
    main()
