import argparse
from pathlib import Path

from launch import LaunchDescription, LaunchService
from launch_ros.actions import Node

from dvp_teleop_server.utils.camera_config import get_camera_config, CameraConfig


def parse_args():
    description = """\
        Read the camera config and launch the bimanual hand detector.
        --------------------------------
        Example: python3 script/bimanual_visual_module.launch.py --cfg example/visionpro_config.yaml
        """
    parser = argparse.ArgumentParser(description)
    parser.add_argument(
        "--cfg", "-c", required=True, help="File path to the camera config."
    )
    args = parser.parse_args()
    return args


def launch_detector(cfg: CameraConfig):
    launch_list = []
    executable = Path(__file__).parent.parent / "nodes/bimanual_hand_detector_node.py"
    executable = executable.absolute()
    for config in cfg.configs:
        arguments = dict(
            visionpro_name=config.name,
        )
        if config.type.lower() != "visionpro":
            raise ValueError(
                f"Current bimanual visual module launch file is only compatible with camera type: visionpro"
                f"However, the current camera type is {config.type.lower()}"
            )
        else:
            arguments["avp_ip"] = config.avp_ip

        argument_list = [
            f"--{k.replace('_', '-')}={v}"
            for k, v in arguments.items()
            if type(v) is not bool
        ]
        argument_list += [f"--{k}" for k, v in arguments.items() if v is True]
        node = Node(
            executable=str(executable),
            arguments=argument_list,
            output={
                "stdout": "screen",
                "stderr": "screen",
            },
        )
        launch_list.append(node)
    return launch_list


def generate_launch_description():
    args = parse_args()
    cfg = get_camera_config(args.cfg)
    launch_list = launch_detector(cfg)

    return LaunchDescription(launch_list)


def main():
    desc = generate_launch_description()
    service = LaunchService()
    service.include_launch_description(desc)
    service.run()


if __name__ == "__main__":
    main()
