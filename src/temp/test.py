# Standard library
import os
import csv
from typing import Tuple, Optional

# Third-party
import numpy as np
import trimesh

# ROS2 core
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, qos_profile_system_default

# ROS2 messages
from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *
from moveit_msgs.msg import *
from shape_msgs.msg import *

# TF
from tf2_ros import *

# ROS2 MoveIt
from moveit2_commander import *
from rotutils import *

# ROS2 package index
from ament_index_python.packages import get_package_share_directory


import numpy as np


def _resolve_resource_path(url_or_path: str) -> str:
    """
    Supports:
      - Absolute/relative filesystem paths
      - package://<pkg_name>/<subpath> (ROS package resource)
    """
    if url_or_path.startswith("package://"):
        # package://pkg_name/sub/dir/model.stl
        rest = url_or_path[len("package://") :]
        parts = rest.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid package URL: {url_or_path}")
        pkg, subpath = parts
        base = get_package_share_directory(pkg)
        return os.path.join(base, subpath)
    return url_or_path


def _as_shape_msgs_mesh(tm: trimesh.Trimesh, scale: Tuple[float, float, float]) -> Mesh:
    """
    Convert a trimesh.Trimesh to shape_msgs/Mesh with optional per-axis scale.
    Ensures triangles only.
    """
    # Force triangulation if faces have more than 3 verts
    if tm.faces.shape[1] != 3:
        tm = trimesh.Trimesh(vertices=tm.vertices, faces=tm.faces, process=True)

    # Apply scale
    S = np.asarray(scale, dtype=np.float64).reshape(1, 3)
    verts = np.asarray(tm.vertices, dtype=np.float64) * S
    faces = np.asarray(tm.faces, dtype=np.int32)

    print(verts.shape, faces.shape)

    msg = Mesh()
    for vx, vy, vz in verts:
        msg.vertices.append(Point(x=float(vx), y=float(vy), z=float(vz)))
    for i0, i1, i2 in faces:
        tri = MeshTriangle(vertex_indices=[int(i0), int(i1), int(i2)])
        msg.triangles.append(tri)

    return msg


def load_mesh_as_collision_object(
    url_or_path: str,
    *,
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> CollisionObject:
    """
    Read an STL/OBJ (and most formats trimesh supports) and return a moveit_msgs/CollisionObject.

    Args:
      url_or_path: filesystem path or 'package://<pkg>/<subpath>'
      object_id: unique ID for the object in the planning scene
      frame_id: TF frame in which the pose is expressed
      pose: geometry_msgs/Pose of the mesh origin; if None, identity
      scale: per-axis scale applied to mesh vertices

    Returns:
      moveit_msgs.msg.CollisionObject with meshes[0] and mesh_poses[0] set, operation=ADD
    """
    path = _resolve_resource_path(url_or_path)

    # Load with trimesh (force mesh; will merge if the file has multiple sub-meshes)
    tm = trimesh.load(path, force="mesh")
    if isinstance(tm, trimesh.Scene):
        # If it's a Scene, combine into a single mesh in scene coordinates
        tm = trimesh.util.concatenate(tuple(g for g in tm.geometry.values()))
    if not isinstance(tm, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type from file: {path}")

    mesh_msg = _as_shape_msgs_mesh(tm, scale)

    return mesh_msg


class TestNode(Node):
    def __init__(self):
        super().__init__("mesh_test_node")

        self._mesh_msg = load_mesh_as_collision_object(
            "/home/min/7cmdehdrb/project_dt/src/Buckle.stl"
        )

        self._mesh_publisher = self.create_publisher(
            Mesh, "topic", qos_profile=qos_profile_system_default
        )

    def run(self):
        self._mesh_publisher.publish(self._mesh_msg)
        self.get_logger().info("Published mesh message.")


def main(args=None):
    rclpy.init(args=args)

    import threading

    node = TestNode()

    th = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    th.start()

    rate = node.create_rate(5.0)
    while rclpy.ok():
        node.run()
        rate.sleep()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
