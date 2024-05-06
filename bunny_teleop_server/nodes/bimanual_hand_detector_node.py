#!/usr/bin/env python3
import argparse
from typing import Dict

import grpc
import numpy as np
import rclpy
from avp_stream.grpc_msg import handtracking_pb2, handtracking_pb2_grpc
from avp_stream.utils.grpc_utils import process_matrices, process_matrix
from geometry_msgs.msg import Pose, Quaternion, Point
from hand_msgs.msg import BimanualHandDetection
from pytransform3d import rotations
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray

from bunny_teleop_server.utils.viz_utils import (
    draw_mark_array_points,
    draw_mark_array_lines,
    HAND_CONNECTIONS,
)
from bunny_teleop_server.utils.robot_utils import OPERATOR2AVP_RIGHT, OPERATOR2AVP_LEFT


def three_mat_mul(left_rot: np.ndarray, mat: np.ndarray, right_rot: np.ndarray):
    result = np.eye(4)
    rotation = left_rot @ mat[:3, :3] @ right_rot
    pos = left_rot @ mat[:3, 3]
    result[:3, :3] = rotation
    result[:3, 3] = pos
    return result


def rotate_head(R, degrees=-90):
    # Convert degrees to radians
    theta = np.radians(degrees)
    # Create the rotation matrix for rotating around the x-axis
    R_x = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1],
        ]
    )
    R_rotated = R @ R_x
    return R_rotated


def two_mat_batch_mul(batch_mat: np.ndarray, left_rot: np.ndarray):
    result = np.tile(np.eye(4), [batch_mat.shape[0], 1, 1])
    result[:, :3, :3] = np.matmul(left_rot[None, ...], batch_mat[:, :3, :3])
    result[:, :3, 3] = batch_mat[:, :3, 3] @ left_rot.T
    return result


def pose_np2ros(wrist_mat: np.ndarray):
    quat = rotations.quaternion_from_matrix(wrist_mat[:3, :3], strict_check=False)
    pos = wrist_mat[:3, 3]
    quat_xyzw = Quaternion(x=quat[1], y=quat[2], z=quat[3], w=quat[0])
    point = Point(x=pos[0], y=pos[1], z=pos[2])
    pose = Pose(position=point, orientation=quat_xyzw)
    return pose


def joint_avp2hand(finger_mat: np.ndarray):
    finger_index = np.array(
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]
    )
    finger_mat = finger_mat[finger_index]
    return finger_mat


class VisionProHandDetector(Node):
    def __init__(
        self,
        vision_pro_name,
        vision_pro_ip,
        verbose=False,
        visualize_3d_detection=False,
    ):
        super().__init__(node_name=f"{vision_pro_name}_hand_detector")
        self.visionpro_name = vision_pro_name
        self.prefix = f"/{vision_pro_name}/bimanual_hand_detection"

        # Create publisher
        self.detection_pub = self.create_publisher(
            BimanualHandDetection, f"{self.prefix}/results", 10
        )

        # Create cache variable
        # Note: The currrent Vision pro streaming does not include RGB data
        self.latest_color_frame = np.zeros([0, 0, 3])
        self.logger = self.get_logger()
        self.clock = self.get_clock()
        self.last_time = self.clock.now()

        # Debug
        self.verbose = verbose
        self.visualize_3d_detection = visualize_3d_detection

        # Visualization of 3D detection results
        if self.visualize_3d_detection:
            self.avp_hand_skeleton_pub = self.create_publisher(
                MarkerArray, f"{self.prefix}/vision_pro_skeleton", 5
            )

        # Setup avp communication
        self.avp_address = vision_pro_ip
        self.latest_transformation = None
        self.axis_transform = np.array(
            [[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64
        )

        # Camera information
        self.base_frame_name = f"{vision_pro_name}_base_frame"

    def start(self):
        self.get_logger().info(
            f"Beginning VisionPro hand detector node with name: {self.visionpro_name}. "
            f"Waiting for initial data from {self.avp_address}."
        )
        self.stream()

        while self.latest_transformation is None:
            pass

        self.get_logger().info(f"Initial data received. VisionPro node is working now.")

    def stream(self):
        # Ref: https://github.com/Improbable-AI/VisionProTeleop
        request = handtracking_pb2.HandUpdate()
        try:
            with grpc.insecure_channel(f"{self.avp_address}:12345") as channel:
                stub = handtracking_pb2_grpc.HandTrackingServiceStub(channel)
                responses = stub.StreamHandUpdates(request)
                for response in responses:
                    left_joints = process_matrices(
                        response.left_hand.skeleton.jointMatrices
                    )
                    right_joints = process_matrices(
                        response.right_hand.skeleton.jointMatrices
                    )
                    left_joints = two_mat_batch_mul(left_joints, OPERATOR2AVP_LEFT.T)
                    right_joints = two_mat_batch_mul(right_joints, OPERATOR2AVP_RIGHT.T)

                    transformations = {
                        "left_wrist": three_mat_mul(
                            self.axis_transform,
                            process_matrix(response.left_hand.wristMatrix)[0],
                            OPERATOR2AVP_LEFT,
                        ),
                        "right_wrist": three_mat_mul(
                            self.axis_transform,
                            process_matrix(response.right_hand.wristMatrix)[0],
                            OPERATOR2AVP_RIGHT,
                        ),
                        "left_fingers": left_joints,
                        "right_fingers": right_joints,
                        "head": rotate_head(
                            three_mat_mul(
                                self.axis_transform,
                                process_matrix(response.Head)[0],
                                np.eye(3),
                            )
                        ),
                    }
                    self.latest_transformation = transformations
                    self.publish_vision_pro_message()

        except Exception as e:
            self.get_logger().error(e)

    def publish_vision_pro_message(self):
        transformation = self.latest_transformation
        msg = BimanualHandDetection()
        now = self.clock.now()
        msg.header.stamp = now.to_msg()
        try:
            self.fill_detection_msg(msg, transformation)

            # Update time stamp and duration
            now = self.clock.now()
            msg.duration = (self.last_time - now).to_msg()
            self.last_time = now
            self.detection_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(e)

    def fill_detection_msg(
        self, message: BimanualHandDetection, transformation: Dict[str, np.ndarray]
    ):
        ros_head_pose = pose_np2ros(transformation["head"])
        ros_right_hand_pose = pose_np2ros(transformation["right_wrist"])
        ros_left_hand_pose = pose_np2ros(transformation["left_wrist"])
        right_joint_pose = joint_avp2hand(transformation["right_fingers"]).astype(
            np.float32
        )
        left_joint_pose = joint_avp2hand(transformation["left_fingers"]).astype(
            np.float32
        )
        message.detected = True
        message.head_pose = ros_head_pose
        message.right_wrist_pose = ros_right_hand_pose
        message.left_wrist_pose = ros_left_hand_pose
        message.right_joints = right_joint_pose[:, :3, 3].flatten()
        message.left_joints = left_joint_pose[:, :3, 3].flatten()

        if self.visualize_3d_detection:
            left_wrist_rot = transformation["left_wrist"][:3, :3]
            left_wrist_trans = transformation["left_wrist"][:3, 3][None, ...]
            left_keypoints = (
                left_joint_pose[:, :3, 3] @ left_wrist_rot.T + left_wrist_trans
            )

            right_wrist_rot = transformation["right_wrist"][:3, :3]
            right_wrist_trans = transformation["right_wrist"][:3, 3][None, ...]
            right_keypoints = (
                right_joint_pose[:, :3, 3] @ right_wrist_rot.T + right_wrist_trans
            )
            self.publish_hand_skeleton(left_keypoints, right_keypoints)

        return message

    def publish_hand_skeleton(
        self, left_keypoint_3d: np.ndarray, right_keypoint_3d: np.ndarray
    ):
        stamp = self.clock.now().to_msg()
        color = np.array([1, 0, 0, 1], dtype=float)
        ns = "visionpro"
        ns_id = 1
        pub = self.avp_hand_skeleton_pub
        if pub.get_subscription_count() == 0:
            return

        msg = MarkerArray()
        for keypoints_3d in [right_keypoint_3d, left_keypoint_3d]:
            msg = draw_mark_array_points(
                msg,
                keypoints_3d,
                f"{ns}_points",
                self.base_frame_name,
                stamp,
                color,
                0.01,
                ns_id=ns_id,
                id_offset=0,
                lifetime=100,
            )
            msg = draw_mark_array_lines(
                msg,
                keypoints_3d,
                HAND_CONNECTIONS,
                f"{ns}_lines",
                self.base_frame_name,
                stamp,
                np.array([0.12, 0.12, 0.12, 1]),
                0.005,
                ns_id=ns_id,
                id_offset=21,
                lifetime=100,
            )
            ns_id += 1
        pub.publish(msg)

    def close(self):
        self.stream_thread.join()
        self.get_logger().info(
            f"Close VisionPro hand detector node with name: {self.visionpro_name}."
        )
        super().close()


def main():
    # Parse camera name
    description = """\
    Run a hand detection node for a Apple Vision Pro.
    Need to call this function multiple times if you want to use multiple camera.
    --------------------------------
    Example: python3 visionpro_detector_node.py --visionpro-name=my_avp --avp_ip="127.0.0.1"
    """
    parser = argparse.ArgumentParser(description)
    parser.add_argument(
        "--visionpro-name",
        required=True,
        type=str,
        help="Name of the Apple Vision Pro, current not important but reserved for future multiple AVP settings.",
    )
    parser.add_argument(
        "--avp-ip",
        required=True,
        type=str,
        help=f"IP address of Apple Vision Pro in a local network.",
    )
    parser.add_argument(
        "--ros-args",
        required=False,
        action="store_true",
        help="Do not use this one, just a dirty hack to be compatible with ROS2 node action.",
    )
    args = parser.parse_args()

    # Create ROS Node
    rclpy.init(args=None)
    avp_name = args.visionpro_name

    # Camera signal from ROS topic
    hand_detector_node = VisionProHandDetector(
        vision_pro_name=avp_name,
        vision_pro_ip=args.avp_ip,
        verbose=True,
        visualize_3d_detection=True,
    )
    try:
        hand_detector_node.start()
    except KeyboardInterrupt:
        hand_detector_node.get_logger().info("Keyboard interrupt, shutting down.\n")

    hand_detector_node.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
