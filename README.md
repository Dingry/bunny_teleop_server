

### Start

```shell
docker run -it --net host --name dvp_teleop_server -v ~/project:/project dvp_teleop_server /bin/bash
ln -s /project/DexVisionPro/teleop_server/ ./dvp_teleop_server
cd /server/dvp_teleop_server/ros && colcon clean workspace -y && colcon build
pip3 install /project/DexVisionPro/teleop_client/dist/dvp_teleop-0.0.1-py3-none-any.whl
pip3 install numpy --upgrade
pip3 install . 
```

TODO:

The retargeting code now are old. Integrate it with new retargeting code.