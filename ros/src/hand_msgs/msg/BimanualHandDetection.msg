std_msgs/Header header
bool detected
geometry_msgs/Pose left_wrist_pose
geometry_msgs/Pose right_wrist_pose
geometry_msgs/Pose head_pose
float32[63] left_joints # [21, 3] row major
float32[63] right_joints # [21, 3] row major
builtin_interfaces/Duration duration
