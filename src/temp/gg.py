import sys

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, qos_profile_system_default
from sensor_msgs.msg import *
from std_msgs.msg import *
from geometry_msgs.msg import *
from moveit_msgs.msg import *
from trajectory_msgs.msg import *
from moveit2_commander import *

import numpy as np
from rotutils import euler_from_quaternion


class TestNode(Node):
    def __init__(self):
        super().__init__("test_node")

        self._fk_manager = FK_ServiceManager(node=self)
        self._ik_manager = IK_ServiceManager(node=self)
        
        self._get_planning_scene_manager = GetPlanningScene_ServiceManager(node=self)
        self._catesian_path_manager = CartesianPath_ServiceManager(node=self)
        self._kinematic_path_manager = KinematicPath_ServiceManager(node=self)
        self._execute_trajectory_manager = ExecuteTrajectory_ServiceManager(node=self)

        self._joint_states_pub = self.create_publisher(JointState, "/ur_joint_command", qos_profile=qos_profile_system_default)
        self._joint_states_sub = self.create_subscription(JointState, "/ur_joint_states", callback=self.joint_states_callback, qos_profile=qos_profile_system_default)
        
        self._target_pose_sub = self.create_subscription(PoseStamped, "/target_pose", callback=self.target_pose_callback, qos_profile=qos_profile_system_default)
        
        self._execute_duration = 0.0
        
        self._joint_state_msg: JointState = None
        self._target_pose_msg: PoseStamped = None
        
        self._traj: RobotTrajectory = None
        
        self._last_time = self.get_clock().now().to_msg()

        # self.timer = self.create_timer(1.0, self.timer_callback)

    def target_pose_callback(self, msg: PoseStamped):
        if self._target_pose_msg is None:
            self._target_pose_msg = msg
            self._execute_duration = 0.0
            self._traj = None
            return # First time set
            
        last_pos = np.array([
            self._target_pose_msg.pose.position.x,
            self._target_pose_msg.pose.position.y,
            self._target_pose_msg.pose.position.z,
        ])
        current_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])
        
        dist = np.linalg.norm(current_pos - last_pos)
        if dist > 0.01:
            print(f"Target pose changed: {current_pos}")        
            self._target_pose_msg = msg
            self._execute_duration = 0.0
            self._traj = None
            

    def joint_states_callback(self, msg: JointState):
        self._joint_state_msg = msg
        
                
    def planning(self) -> bool:
        if self._joint_state_msg is None or self._target_pose_msg is None:
            self.get_logger().warn("No JointState or Target Pose")
            return False

        try:
            # goal_robot_state: RobotState = self._ik_manager.run(
            #     pose_stamped=PoseStamped(
            #         header=Header(frame_id="base_link", stamp=Time().to_msg()),
            #         pose=self._target_pose_msg.pose
            #     ),
            #     joint_states=self._joint_state_msg,
            #     end_effector="tool0",
            #     avoid_collisions=False
            # )
            
            # if goal_robot_state is None:
            #     raise Exception("No IK solution")
            
            # goal_constraints: Constraints = self._kinematic_path_manager.get_goal_constraint(
            #     goal_joint_states=goal_robot_state.joint_state,
            #     tolerance=0.1
            # )
            
            traj: RobotTrajectory = self._catesian_path_manager.run(
                header=Header(frame_id="base_link", stamp=self.get_clock().now().to_msg()),
                waypoints=[self._target_pose_msg.pose],
                joint_states=self._joint_state_msg,
                end_effector="tool0"
            )
            
            # traj: RobotTrajectory = self._kinematic_path_manager.run(
            #     goal_constraints=[goal_constraints],
            #     path_constraints=None,
            #     joint_states=self._joint_state_msg,
            #     num_planning_attempts=100,
            #     allowed_planning_time=1.0,
            #     max_velocity_scaling_factor=1.0,
            #     max_acceleration_scaling_factor=1.0
            # )
                        
            if traj is None:
                raise Exception("No Trajectory")
            
            scaled_traj: RobotTrajectory = self._execute_trajectory_manager.scale_trajectory(
                trajectory=traj,
                scale_factor=1.0
            )
        
        except Exception as e:
            self.get_logger().warn(f"Planning failed: {e}")
            return False
        
        self._last_time = self.get_clock().now().to_msg()
        
        self._traj = scaled_traj
        return True
        
    def execute(self, traj: RobotTrajectory, dt: float) -> JointState | None:
        # 1. Get execution duration
        duration = self._execute_duration

        currnet_time = self.get_clock().now().to_msg()
        time_diff = Time.from_msg(currnet_time) - Time.from_msg(self._last_time)
        
        self._last_time = currnet_time
        self._execute_duration += time_diff.nanoseconds * 1e-9
        
        # print(f"Execute duration: {self._execute_duration}, dt: {dt}, time_diff: {time_diff.nanoseconds * 1e-9}")

        # 2. Extract joint positions and velocities at the given duration
        jt = traj.joint_trajectory
        if not jt.points:
            self.get_logger().warn("Trajectory has no points")
            return None

        # Find the closest point in time
        target_point = None
        for pt in jt.points:
            pt: JointTrajectoryPoint
            pt_time = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            if pt_time >= duration:
                target_point = pt
                break
            
        if target_point is None:
            target_point = jt.points[-1]
            
        target_point = jt.points[-1]
            
        # print(target_point)

        joint_state = JointState(
            header=Header(frame_id="base_link", stamp=self.get_clock().now().to_msg()),
            name=jt.joint_names,
            position=list(target_point.positions),
            velocity=list(target_point.velocities) if target_point.velocities else [],
        )
        
        self._joint_states_pub.publish(joint_state)

        # 4. Return joint state
        return joint_state

    def run(self):
        if self._traj is None:
            self.planning()
            print(f"Planning... {self._traj is not None}")
        else:
            self.execute(traj=self._traj, dt=0.0)        




def main():
    rclpy.init(args=None)

    test_node = TestNode()
    
    import threading
    import time
    
    th = threading.Thread(target=rclpy.spin, args=(test_node,), daemon=True)
    th.start()
    
    r = test_node.create_rate(100.0)
    while rclpy.ok():
        test_node.run()
        r.sleep()

    test_node.destroy_node()
    rclpy.shutdown()
        
main()