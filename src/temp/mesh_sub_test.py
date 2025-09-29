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
from custom_msgs.msg import MeshInstance

# TF
from tf2_ros import *

# ROS2 MoveIt
from moveit2_commander import *
from rotutils import *

# ROS2 package index
from ament_index_python.packages import get_package_share_directory


import numpy as np


class TestNode(Node):
    def __init__(self):
        super().__init__("mesh_test_node")

        self._get_planning_scene_manager = GetPlanningScene_ServiceManager(node=self)
        self._apply_planning_scene_manager = ApplyPlanningScene_ServiceManager(
            node=self
        )

        self._co: CollisionObject = None
        self._mesh_sub = self.create_subscription(
            Mesh,
            "/ur5e",
            self._mesh_callback,
            qos_profile_system_default,
        )

        self._co2: CollisionObject = None
        self._mesh_sub = self.create_subscription(
            Mesh,
            "/ConveyorBelt_LineA1",
            self._mesh_callback2,
            qos_profile_system_default,
        )

    def _mesh_callback(self, msg: Mesh):
        self.get_logger().info(
            f"Received mesh with {len(msg.vertices)} vertices and {len(msg.triangles)} triangles."
        )

        co = CollisionObject()
        co.id = "test_mesh"
        co.header.frame_id = "ur5e"
        co.meshes.append(msg)
        co.mesh_poses.append(
            Pose(
                position=Point(x=0.5, y=0.3, z=0.0),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        )
        co.operation = CollisionObject.ADD

        self._co = co

    def _mesh_callback2(self, msg: Mesh):
        self.get_logger().info(
            f"Received mesh with {len(msg.vertices)} vertices and {len(msg.triangles)} triangles."
        )

        co = CollisionObject()
        co.id = "test_mesh2"
        co.header.frame_id = "ConveyorBelt_LineA1"
        co.meshes.append(msg)
        co.mesh_poses.append(
            Pose(
                position=Point(x=0.5, y=0.3, z=0.0),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        )
        co.operation = CollisionObject.ADD

        self._co2 = co

    def update_planning_scene(self):
        self._planning_scene = self._get_planning_scene_manager.run()

        collision_objects = []
        if self._co is not None:
            collision_objects.append(self._co)
        if self._co2 is not None:
            collision_objects.append(self._co2)

        self._apply_planning_scene_manager.run(
            collision_objects=collision_objects,
            scene=self._planning_scene,
        )


def main(args=None):
    rclpy.init(args=args)

    import threading

    node = TestNode()

    th = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    th.start()

    rate = node.create_rate(5.0)
    while rclpy.ok():
        node.update_planning_scene()
        rate.sleep()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
