from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class SingleCameraConfig:
    name: str
    type: str
    serial_num: str = ""
    hand_type: str = "right_hand"
    depth: bool = False
    model: str = "default_model"
    web_port: Optional[int] = None
    avp_ip: Optional[str] = None

    _TYPE = ["realsense", "kinect", "apple", "webstream", "visionpro"]
    _HAND = ["right_hand", "left_hand"]

    def __post_init__(self):
        if self.type not in self._TYPE:
            raise ValueError(f"Camera type must be one of {self._TYPE}")
        if self.hand_type not in self._HAND:
            raise ValueError(f"Hand type must be one of {self._HAND}")
        if self.type == "webstream" and self.web_port is None:
            raise ValueError(
                f"Web_port should be specified when using webstream camera type"
            )


@dataclass
class CameraConfig:
    configs: List[SingleCameraConfig]

    def __post_init__(self):
        # If more than one camera are the same type, e.g. two RealSense camera,
        # then both of them should have a unique serial number.
        type_dict = {}
        for config in self.configs:
            cam_type = config.type.lower()

            if cam_type not in type_dict:
                type_dict[cam_type] = [(config.name, config.serial_num is not None)]
            else:
                type_dict[cam_type].append((config.name, config.serial_num is not None))

        for cam_type, name_list in type_dict.items():
            if len(name_list) > 1:
                has_serial = [cam[1] for cam in name_list]
                if not all(has_serial):
                    names = [cam[0] for cam in name_list]
                    no_serial_names = [cam[0] for cam in name_list if not cam[1]]
                    raise ValueError(
                        f"Camera {names} are all {cam_type}. But {no_serial_names} "
                        f"do not specify serial number."
                        f"Note: you have to specify the serial number of each camera "
                        f"if two of them are the same type, e.g. Realsense"
                    )

    @classmethod
    def from_file(cls, config_path):
        path = Path(config_path)
        if not path.is_absolute():
            path = path.absolute()

        configs = []
        with path.open("r") as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            camera_config = yaml_config["cameras"]

            for cfg in camera_config:
                name = list(cfg.keys())[0]
                serial_num = (
                    cfg[name].pop("serial_num") if "serial_num" in cfg[name] else ""
                )
                config = SingleCameraConfig(
                    **cfg[name], serial_num=serial_num, name=name
                )
                configs.append(config)

        return CameraConfig(configs=configs)


def get_camera_config(config_path) -> CameraConfig:
    config = CameraConfig.from_file(config_path)
    return config
