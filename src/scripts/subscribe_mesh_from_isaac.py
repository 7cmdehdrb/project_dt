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

        self._planning_scene: PlanningScene = None
        self._get_planning_scene_manager = GetPlanningScene_ServiceManager(node=self)
        self._apply_planning_scene_manager = ApplyPlanningScene_ServiceManager(
            node=self
        )

        self._instances = {}

        self.create_subscription(
            MeshInstance,
            "/mesh_instance3",
            self._mesh_callback,
            qos_profile_system_default,
        )
        self.create_subscription(
            MeshInstance,
            "/mesh_instance4",
            self._mesh_callback,
            qos_profile_system_default,
        )
        self.create_subscription(
            MeshInstance,
            "/mesh_instance5",
            self._mesh_callback,
            qos_profile_system_default,
        )
        self.create_subscription(
            MeshInstance,
            "/mesh_instance6",
            self._mesh_callback,
            qos_profile_system_default,
        )

    def _mesh_callback(self, msg: MeshInstance):
        print(f"Received mesh instance: {msg.id}")

        co = CollisionObject()
        co.id = msg.id
        co.header.frame_id = "world"
        co.meshes.append(msg.mesh)
        co.mesh_poses.append(msg.poses)
        co.operation = CollisionObject.ADD

        if len(msg.mesh.triangles) == 0 or len(msg.mesh.vertices) == 0:
            self.get_logger().warn(
                f"Mesh instance {msg.id} has no triangles or vertices. Ignoring."
            )
            return None

        self._instances[msg.id] = co
        print(f"Total instances: {len(self._instances)}")

    def update_planning_scene(self):

        self._planning_scene = self._get_planning_scene_manager.run()

        if len(self._instances) == 0:
            self.get_logger().warn("No mesh instances received yet.")
            return

        self._apply_planning_scene_manager.run(
            collision_objects=list(self._instances.values()),
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
