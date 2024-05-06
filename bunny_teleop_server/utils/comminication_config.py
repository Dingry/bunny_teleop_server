from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import yaml


@dataclass
class CommunicationConfig:
    # Web visualizer config
    viz_type: str
    viz_host: str
    viz_port: int
    viz_urdf_paths: List[str]

    operator_name: str

    _VIZ_TYPE = ["vision_pro", "sim_web_visualizer"]
    _DEFAULT_ASSET_DIR = "./"

    def __post_init__(self):
        self.viz_type = self.viz_type.lower()
        if self.viz_type not in self._VIZ_TYPE:
            raise ValueError(f"Camera type must be one of {self._VIZ_TYPE}")
        if len(self.viz_urdf_paths) > 2:
            raise ValueError(
                f"We can only accept 0, 1, or 2 urdf paths in the config but given: {len(self.viz_urdf_paths)}"
            )

        for i, path in enumerate(self.viz_urdf_paths):
            # URDF path check
            new_asset_path = Path(path)
            if not new_asset_path.is_absolute():
                new_asset_path = self._DEFAULT_ASSET_DIR / new_asset_path
                new_asset_path = new_asset_path.absolute()
            if not new_asset_path.exists():
                raise ValueError(f"URDF path {new_asset_path} does not exist")
            self.viz_urdf_paths[i] = str(new_asset_path)

    @classmethod
    def set_default_asset_dir(cls, asset_dir: Union[str, Path]):
        path = Path(asset_dir)
        if not path.exists():
            raise ValueError(f"URDF dir {asset_dir} not exists.")
        cls._DEFAULT_ASSET_DIR = asset_dir

    @classmethod
    def from_file(cls, config_path):
        path = Path(config_path)
        if not path.is_absolute():
            path = path.absolute()

        with path.open("r") as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            cfg = yaml_config["communication"]
            if cfg is None:
                return None
            else:
                config = CommunicationConfig(**cfg)
                return config

    @classmethod
    def from_dict(cls, cfg: Dict):
        if cfg is None:
            return None
        else:
            config = CommunicationConfig(**cfg)
            return config


def get_communication_config(config_path) -> CommunicationConfig:
    config = CommunicationConfig.from_file(config_path)
    return config


if __name__ == "__main__":
    # Path below is relative to this file
    test_config = get_communication_config(
        "../../assets/config/camera_config/single_d455_config.yml"
    )
    print(test_config)
