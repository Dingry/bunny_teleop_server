import time
from concurrent import futures
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Optional, Dict, List, Tuple

import grpc
import numpy as np

from teleop.communication import avp_visualizer_pb2_grpc, avp_visualizer_pb2
from teleop.communication.visualizer_base import TeleopVisualizerBase
from teleop.utils.comminication_config import CommunicationConfig


@dataclass
class SceneUpdateData:
    actor_data: Dict[str, np.ndarray]  # Quaterion in xyzw convention

    def _actor_data_to_grpc_msg(self):
        actor_poses = {}
        for name, pose in self.actor_data.items():
            pos = avp_visualizer_pb2.Pos(pose[0], pose[1], pose[2])
            quat = avp_visualizer_pb2.Quat(w=pose[6], x=pose[3], y=pose[4], z=pose[5])
            pose = avp_visualizer_pb2.Pose(pos, quat)
            actor_poses[name] = pose

        return actor_poses

    def to_grpc_msg(self):
        actor_msg = self._actor_data_to_grpc_msg()
        return avp_visualizer_pb2.UpdateSceneRes(
            actor_poses=actor_msg, articulation_poses={}
        )


class StreamingServiceServicer(avp_visualizer_pb2_grpc.UpdateSceneServiceServicer):
    def __init__(self, queue: Queue):
        super().__init__()
        self.queue = queue

    def UpdateScene(
        self, request: avp_visualizer_pb2.UpdateSceneReq, context: grpc.ServicerContext
    ):
        print(
            f"\033[92m", f"Received request from client: {request.info}", f" \033[00m"
        )
        try:
            while True:
                time.sleep(0.01)
                data: SceneUpdateData = self.queue.get(timeout=100)
                yield data.to_grpc_msg()

        except KeyboardInterrupt:
            context.set_code(grpc.StatusCode.OK)
            context.set_details("Stream completed")
        except Empty:
            print(f"Timeout error for vision pro visualizer.")
            context.set_code(grpc.StatusCode.CANCELLED)


class TeleopVisionProVisualizer(TeleopVisualizerBase):
    def __init__(
        self,
        queue: Queue,
        operator_name,
        robot_urdf_path: str,
        is_right_hand=True,
    ):
        super().__init__(
            operator_name=operator_name,
            is_right_hand=is_right_hand,
            robot_urdf_path=robot_urdf_path,
        )

        self.queue = queue

    def init_robot_base_pose(self, robot_base_pose_vec: np.ndarray):
        pass

    def create_init_frame(self, init_frame_pose_vec: np.ndarray):
        pass

    def update_init_viz(self, opacity: float):
        pass

    def create_ee_target(
        self, ee_pose_vec_in_base: np.ndarray, robot_base_pose_vec: np.ndarray
    ):
        pass

    def update_ee_target(self, ee_pose_vec_in_base: np.ndarray):
        pass

    def update_robot(self, qpos: np.ndarray):
        import pinocchio as pin

        self._update_robot_kinematics(qpos)
        actor_poses = {}
        for i, frame in enumerate(self.model.frames):
            actor_poses[frame.name] = pin.SE3ToXYZQUAT(self.data.oMf[i])

        self.queue.put(actor_poses)


def setup_vision_pro_visualizer(
    config: CommunicationConfig,
) -> Tuple[List[TeleopVisionProVisualizer], grpc.Server]:
    num_robot = len(config.viz_urdf_paths)
    if num_robot > 2:
        raise ValueError(f"More than two URDF path provided, currently not supported")
    if num_robot <= 0:
        raise ValueError(
            f"No robot URDF provided, which is required by visualizer.\n"
            f" You do not need to setup comm_cfg if you do not need a visualizer."
        )
    if config.viz_type != "vision_pro":
        raise ValueError(
            f"Only vision_pro visualizer can be handled by vision pro visualizer, but get: {config.viz_type}"
        )

    queue = Queue(maxsize=1000)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    avp_visualizer_pb2_grpc.add_UpdateSceneServiceServicer_to_server(
        StreamingServiceServicer(queue), server
    )

    visualizers = []
    is_right = False
    for urdf in config.viz_urdf_paths:
        visualizer = TeleopVisionProVisualizer(
            queue,
            operator_name=config.operator_name,
            robot_urdf_path=urdf,
            is_right_hand=is_right,
        )
        visualizers.append(visualizer)
        is_right = True

    server.add_insecure_port(f"[::]:{config.viz_port}")
    server.start()
    print(
        "\033[34m",
        f"Vision Pro Visualizer started with {num_robot} robot.",
        "\033[39m",
    )

    return visualizers, server
