FROM ros:humble

# Language setup
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

WORKDIR /server

# Install common packages
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
  python3-dev \
  python3-pip \
  curl \
  git \
  wget \
  vim \
  unzip

# Add necessary environment variable to bashrc
RUN echo '\
source /opt/ros/humble/setup.bash \n\
if [ -d "/server/bunny_teleop_server" ] \n\
then \n\
  source /server/bunny_teleop_server/ros/install/local_setup.bash \n\
fi \n' >> $HOME/.bashrc


RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu \
    git+https://github.com/colcon/colcon-clean && \
    rm -r /root/.cache

# Git clone and install
RUN cd /server && git clone https://github.com/Dingry/bunny_teleop_server && chmod +x /server/bunny_teleop_server/bunny_teleop_server/nodes/bimanual_hand_detector_node.py

RUN cd bunny_teleop_server && pip install . --use-deprecated legacy-resolver && pip install numpy==1.23.0 && chmod +x /usr/local/lib/python3.10/dist-packages/bunny_teleop_server/nodes/bimanual_hand_detector_node.py

# Build ROS and chmod the file
RUN /bin/bash -c ". /opt/ros/humble/setup.bash && cd /server/bunny_teleop_server/ros && colcon clean workspace -y && colcon build"

