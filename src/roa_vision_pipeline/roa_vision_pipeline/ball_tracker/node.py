#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState

from vision_interfaces.msg import Robocupvision
from vision_interfaces.msg import PanTiltMsgs
from dynamixel_rdk_msgs.msg import DynamixelPanTiltMsgs
from std_msgs.msg import Bool

from roa_vision_pipeline.ball_tracker.params import (
    declare_ball_tracker_params,
    load_ball_tracker_params,
)
from roa_vision_pipeline.ball_tracker.controller import BallTrackerController


class BallTrackerNode(Node):
    def __init__(self):
        super().__init__("ball_tracker_node")

        declare_ball_tracker_params(self)
        self.params = load_ball_tracker_params(self)

        self.controller = BallTrackerController(self.params)

        self.vision_sub = self.create_subscription(Robocupvision, self.params.vision_topic, self.vision_callback, 1,)
        self.pan_zero_sub = self.create_subscription(Bool, self.params.master2vision_topic, self.pan_zero_callback, 1,)

        self.ball_lost_pub = self.create_publisher(PanTiltMsgs, self.params.ball_lost_topic, 1,)
        self.pantilt_pub = self.create_publisher(DynamixelPanTiltMsgs, self.params.pantilt_topic, 1,)
        self.joint_state_pub = self.create_publisher(JointState, self.params.joint_state_topic, 1,)

        self.timer = self.create_timer(1.0 / self.params.rate_hz, self.tick,)

        self.pan_zeroed = False

        self.get_logger().info("BallTrackerNode initialized.")
        self.get_logger().info(f"rate_hz: {self.params.rate_hz}")
        self.get_logger().info(
            f"pan_id: {self.params.pan_id}, "
            f"pan_max_deg: {self.params.pan_max_deg}, "
            f"pan_min_deg: {self.params.pan_min_deg}"
        )
        self.get_logger().info(
            f"tilt_id: {self.params.tilt_id}, "
            f"tilt_max_deg: {self.params.tilt_max_deg}, "
            f"tilt_min_deg: {self.params.tilt_min_deg}"
        )
        self.get_logger().info(
            f"scan_pan_speed: {self.params.scan_pan_speed_deg_s} deg/s, "
            f"scan_tilt_speed: {self.params.scan_tilt_speed_deg_s} deg/s"
        )
        self.get_logger().info(
            f"img_w: {self.params.img_w}, img_h: {self.params.img_h}"
        )
        self.get_logger().info(
            f"pan_dir: {self.params.pan_dir}, "
            f"tilt_dir: {self.params.tilt_dir}"
        )
        self.get_logger().info(
            f"joint names: "
            f"{self.params.torso_joint_name}, "
            f"{self.params.pan_joint_name}, "
            f"{self.params.tilt_joint_name}"
        )

    def vision_callback(self, msg: Robocupvision):
        if msg.ball_d == 0 and msg.ball_x == 0 and msg.ball_y == 0:
            self.controller.update_vision(ball_seen=False)
            return

        self.controller.update_vision(
            ball_seen=True,
            ball_x=float(msg.ball_x),
            ball_y=float(msg.ball_y),
            ball_cam_x=float(msg.ball_cam_x) * 0.1,
            ball_cam_y=-float(msg.ball_cam_y) * 0.1,
        )
    
    def pan_zero_callback(self, msg: bool):
        self.pan_zeroed = msg.data

    def tick(self):
        pan_deg, tilt_deg, lost_ball_right, lost_ball_left, log_msg = self.controller.tick()

        if log_msg is not None:
            self.get_logger().info(log_msg)
        
        if self.params.pan_zero_flag and self.pan_zeroed:
            pan_deg = 0.0

        self.publish_pantilt(
            pan_deg=pan_deg,
            tilt_deg=tilt_deg,
        )

        self.publish_joint_state(
            pan_deg=pan_deg,
            tilt_deg=tilt_deg,
        )

        self.publish_ball_lost(
            lost_ball_right=lost_ball_right, 
            lost_ball_left=lost_ball_left
        )

    def publish_pantilt(self, pan_deg: float, tilt_deg: float):
        msg = DynamixelPanTiltMsgs()

        msg.pan_id = self.params.pan_id
        msg.pan_goal_position = float(pan_deg)
        msg.pan_profile_acceleration = 0.0
        msg.pan_profile_velocity = 0.0

        msg.tilt_id = self.params.tilt_id
        msg.tilt_goal_position = float(tilt_deg)
        msg.tilt_profile_acceleration = 0.0
        msg.tilt_profile_velocity = 0.0

        self.pantilt_pub.publish(msg)

    def publish_joint_state(self, pan_deg: float, tilt_deg: float):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.name = [
            self.params.torso_joint_name,
            self.params.pan_joint_name,
            self.params.tilt_joint_name,
        ]

        msg.position = [
            0.0,
            math.radians((-1) * pan_deg),
            math.radians((-1) * tilt_deg),
        ]

        self.joint_state_pub.publish(msg)
    
    def publish_ball_lost(self, lost_ball_right: bool, lost_ball_left: bool):
        msg = PanTiltMsgs()

        msg.lost_pan_right = lost_ball_right
        msg.lost_pan_left = lost_ball_left

        self.ball_lost_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BallTrackerNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
