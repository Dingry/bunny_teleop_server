from typing import Iterable

from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3, Pose
from rclpy.duration import Duration
from visualization_msgs.msg import Marker, MarkerArray

HAND_CONNECTIONS = [
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
]


def draw_mark_array_lines(
    msg: MarkerArray,
    points,
    skeletons,
    ns,
    frame_id,
    stamp,
    rgba,
    scale=0.01,
    ns_id=0,
    id_offset=0,
    lifetime=0.25,
):
    points = points.astype(float)
    if not isinstance(rgba[0], Iterable):
        rgba = [rgba] * len(skeletons)

    for sk_id, sk in enumerate(skeletons):
        marker = Marker(type=Marker.LINE_STRIP, action=Marker.MODIFY)

        # Header
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = ns_id * 100 + sk_id + id_offset
        marker.lifetime = Duration(seconds=lifetime).to_msg()

        # Geometry
        marker.scale.x = scale
        marker.color.r = rgba[sk_id][0]
        marker.color.g = rgba[sk_id][1]
        marker.color.b = rgba[sk_id][2]
        marker.color.a = rgba[sk_id][3]

        for point_ind in sk:
            marker.points.append(
                Point(
                    x=points[point_ind, 0],
                    y=points[point_ind, 1],
                    z=points[point_ind, 2],
                )
            )

        msg.markers.append(marker)
    return msg


def draw_mark_array_points(
    msg: MarkerArray,
    points,
    ns,
    frame_id,
    stamp,
    rgba,
    scale=0.01,
    ns_id=0,
    id_offset=0,
    lifetime=0.25,
):
    points = points.astype(float)
    if not isinstance(rgba[0], Iterable):
        rgba = [rgba] * len(points)

    for point_id, point in enumerate(points):
        marker = Marker(type=Marker.SPHERE, action=Marker.MODIFY)
        # Header
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = ns_id * 100 + point_id + id_offset
        marker.lifetime = Duration(seconds=lifetime).to_msg()

        # Geometry
        marker.pose = Pose(position=Point(x=point[0], y=point[1], z=point[2]))
        marker.scale = Vector3(x=scale, y=scale, z=scale)
        marker.color.r = rgba[point_id][0]
        marker.color.g = rgba[point_id][1]
        marker.color.b = rgba[point_id][2]
        marker.color.a = rgba[point_id][3]

        msg.markers.append(marker)
    return msg
