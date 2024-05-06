import argparse
from pathlib import Path

import rclpy
import yaml
from dex_retargeting.retargeting_config import RetargetingConfig
from sim_web_visualizer.base_visualizer_client import MeshCatVisualizerBase

# from bunny_teleop.communication.vision_pro_visualizer import (
#     setup_vision_pro_visualizer,
# )
from bunny_teleop_server.communication.web_visualizer import TeleopWebVisualizer
from bunny_teleop_server.nodes.bimanual_hand_monitor_node import run_hand_node
from bunny_teleop_server.nodes.bimanual_teleop_server_node import BimanualRobotTeleopNode
from bunny_teleop_server.utils.camera_config import get_camera_config
from bunny_teleop_server.utils.comminication_config import (
    CommunicationConfig,
    get_communication_config,
)
from bunny_teleop_server.utils.motion_control_config import MotionControlConfig


def parse_args():
    description = """\
        Read the camera config and launch the camera driver and/or hand detector.
        --------------------------------
        Example: python3 script/visual_module.launch.py --cfg example/example_camera_config.yaml
        """
    parser = argparse.ArgumentParser(description)
    parser.add_argument(
        "--kinematics-cfg",
        "-k",
        required=True,
        help="File path to the kinematics config.",
    )
    parser.add_argument(
        "--camera-cfg", "-c", required=True, help="File path to the camera config."
    )
    parser.add_argument(
        "--communication-cfg",
        "-comm",
        required=False,
        default=None,
        help="File path to the communication config.",
    )
    parser.add_argument(
        "--teleop-host",
        "-th",
        required=False,
        default="localhost",
        type=str,
        help="The teleoperation server host address. "
        "Can be the ip address of the machine to run this file.",
    )
    parser.add_argument(
        "--teleop-port",
        "-tp",
        required=False,
        default=5500,
        type=int,
        help="The visualization server host port.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    kinematics_config = args.kinematics_cfg
    camera_config = args.camera_cfg
    comm_config = args.communication_cfg

    # Kinematics and camera config
    camera_config = get_camera_config(camera_config)
    kinematics_path = Path(kinematics_config)
    if not Path(kinematics_config).is_absolute():
        kinematics_path = kinematics_path.absolute()
    with kinematics_path.open("r") as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)
        left_config = yaml_config["left"]
        right_config = yaml_config["right"]

    # Set retargeting directory
    root_dir = kinematics_path.parent.parent.parent.resolve().absolute()
    urdf_dir = (root_dir / "assets/urdf").resolve()
    RetargetingConfig.set_default_urdf_dir(urdf_dir)
    MotionControlConfig.set_default_robot_config_dir(
        (root_dir / "configs/robot_config").resolve()
    )
    CommunicationConfig.set_default_asset_dir(urdf_dir)

    # Build motion control
    motion_controls = []
    retargetings = []
    bimanual_control_configs = []
    for i, cfg_dict in enumerate([left_config, right_config]):
        control_config = MotionControlConfig.from_dict(cfg_dict["control"])
        bimanual_control_configs.append(control_config)
        motion_control = control_config.build() if control_config is not None else None
        motion_controls.append(motion_control)

        retargeting_config = RetargetingConfig.from_dict(cfg_dict["retargeting"])
        retargeting = retargeting_config.build()
        retargetings.append(retargeting)

    # Parse camera config
    detection_topic_names = []
    hand_type = []
    for cam_cfg in camera_config.configs:
        cam_name = cam_cfg.name
        detection_topic_names.append(f"/{cam_name}/bimanual_hand_detection")
        if len(hand_type) == 0:
            hand_type.append(cam_cfg.hand_type)
        else:
            if hand_type[0] != cam_cfg.hand_type:
                raise ValueError(
                    f"Multiple cameras should have the same hand type in a single config."
                )

    # Parse communication config
    if comm_config is not None:
        comm_cfg = get_communication_config(comm_config)
        if len(comm_cfg.viz_urdf_paths) != 2:
            raise ValueError(
                f"Two urdf path is required in the communication path to visualize the robot"
            )
        if comm_cfg.viz_type == "sim_web_visualizer":
            from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
            start_zmq_server_as_subprocess()

            viz = MeshCatVisualizerBase(port=comm_cfg.viz_port, host=comm_cfg.viz_host)
            left_robot_viz = TeleopWebVisualizer(
                viz,
                comm_cfg.operator_name,
                is_right_hand=False,
                robot_urdf_path=comm_cfg.viz_urdf_paths[0],
            )
            right_robot_viz = TeleopWebVisualizer(
                viz,
                comm_cfg.operator_name,
                is_right_hand=True,
                robot_urdf_path=comm_cfg.viz_urdf_paths[1],
            )
            robot_viz = (left_robot_viz, right_robot_viz)
        elif comm_cfg.viz_type == "vision_pro":
            raise NotImplementedError
            # avp_visualizers, grpc_server = setup_vision_pro_visualizer(comm_cfg)
            # robot_viz = (avp_visualizers[0], avp_visualizers[1])
        else:
            raise NotImplementedError
    else:
        robot_viz = None

    # Build teleportation node
    rclpy.init(args=None)
    teleop_node = BimanualRobotTeleopNode(
        detection_topic_name=detection_topic_names[0],
        need_init=True,
        retargeting_optimizers=(retargetings[0], retargetings[1]),
        motion_controls=(motion_controls[0], motion_controls[1]),
        robot_viz=robot_viz,
        disable_orientation_control=tuple(
            cfg.disable_orientation_control for cfg in bimanual_control_configs
        ),
        motion_scaling_factor=tuple(
            cfg.motion_scaling_factor for cfg in bimanual_control_configs
        ),
        low_pass_smoothing_wrist=tuple(
            cfg.low_pass_alpha for cfg in bimanual_control_configs
        ),
        verbose=True,
        teleop_port=args.teleop_port,
        teleop_host=args.teleop_host,
    )

    thread_num = 5
    run_hand_node(teleop_node, thread_num)


if __name__ == "__main__":
    main()
