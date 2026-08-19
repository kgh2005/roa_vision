#!/usr/bin/env python3

import threading

import numpy as np
import math

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge

import tf2_ros
import tf2_geometry_msgs

from vision_interfaces.msg import BoundingBox
from vision_interfaces.msg import Robocupvision, Robocupvisionfeature

from robocup_zedm.refiner.params import (
    declare_refiner_params,
    load_refiner_params,
)
from robocup_zedm.refiner.refiner import Refiner


class RefinerNode(Node):
    def __init__(self):
        super().__init__("refiner_node")

        declare_refiner_params(self)
        self.params = load_refiner_params(self)

        self.refiner = Refiner(
            half_win=self.params.half_win,
            remove_space_dis=self.params.remove_space_dis,
        )

        self.get_logger().info("=========================")
        self.get_logger().info("Parameters loaded:")
        self.get_logger().info(f"hz: {self.params.hz}")
        self.get_logger().info(f"half_win: {self.params.half_win}")
        self.get_logger().info(f"remove_space_dis: {self.params.remove_space_dis}")
        self.get_logger().info(f"depth_topic: {self.params.depth_topic}")
        self.get_logger().info(f"camera_info_topic: {self.params.camera_info_topic}")
        self.get_logger().info(f"bbox_topic: {self.params.bbox_topic}")
        self.get_logger().info(f"pantilt_topic: {self.params.pantilt_topic}")
        self.get_logger().info(f"vision_topic: {self.params.vision_topic}")
        self.get_logger().info(f"vision_feature_topic: {self.params.vision_feature_topic}")

        # ---------- TF ----------
        self.target_frame = self.params.base_link_frame
        self.source_frame = self.params.camera_frame

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self,)

        self.get_logger().info(f"TF source_frame: {self.source_frame}")
        self.get_logger().info(f"TF target_frame: {self.target_frame}")

        # ---------- Publishers ----------
        self.vision_pub = self.create_publisher(Robocupvision, self.params.vision_topic, 1,)
        self.vision_feature_pub = self.create_publisher(Robocupvisionfeature, self.params.vision_feature_topic, 1,)

        # ---------- Subscriptions ----------
        self.create_subscription(BoundingBox, self.params.bbox_topic, self.bbox_callback, 1,)
        self.create_subscription(Image, self.params.depth_topic, self.depth_callback, 1,)
        self.create_subscription(CameraInfo, self.params.camera_info_topic, self.camera_info_callback, 1,)

        # ---------- Locks ----------
        self.state_lock = threading.Lock()
        self.depth_lock = threading.Lock()

        # ---------- Camera intrinsics ----------
        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0
        self.width = 0
        self.height = 0
        self.have_caminfo = False

        # ---------- Depth storage ----------
        self.bridge = CvBridge()
        self.latest_depth_m = None

        # ---------- Detection caches ----------
        self.det_ball = []
        self.det_robot = []
        self.det_corner_line = []
        self.det_t_line = []
        self.det_cross_line = []
        self.det_goal_post = []

        # ---------- Pan motion state from TF ----------
        self.prev_tf_yaw_deg = None
        self.pan_moving = False
        self.pan_move_thresh_deg = self.params.pan_move_thresh_deg

        self.use_pan_move_flag = self.params.use_pan_move_flag

        # ---------- Timer ----------
        period = 1.0 / max(self.params.hz, 1e-6)
        self.timer = self.create_timer(period, self.bbox_processing)

        self.get_logger().info("RefinerNode initialized.")

    # ---------------- TF ----------------
    def transform_point_to_base(
        self,
        x: float,
        y: float,
        z: float,
    ) -> tuple[float, float, float]:
        """
        Convert point from zedm_left_camera_optical_frame to base_link.

        Input:
            x, y, z:
                3D point in camera optical frame [m]

        Output:
            Xt, Yt, Zt:
                3D point in base_link frame [m]

        base_link convention:
            x: forward
            y: left
            z: up
        """
        p = PointStamped()
        p.header.stamp = rclpy.time.Time().to_msg()
        p.header.frame_id = self.source_frame

        p.point.x = float(x)
        p.point.y = float(y)
        p.point.z = float(z)

        try:
            out = self.tf_buffer.transform(
                p,
                self.target_frame,
                timeout=Duration(seconds=0.05),
            )

            return (
                float(out.point.x),
                float(out.point.y),
                float(out.point.z),
            )

        except Exception as e:
            self.get_logger().warn(
                f"TF transform failed: "
                f"{self.source_frame} -> {self.target_frame}: {e}"
            )
            return 0.0, 0.0, -1.0
            
    def quat_to_yaw_deg(self, q):
        x = q.x
        y = q.y
        z = q.z
        w = q.w

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return math.degrees(yaw)
        
    def update_pan_motion_from_tf(self):
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.02),
            )

            yaw_deg = self.quat_to_yaw_deg(tf_msg.transform.rotation)

            if self.prev_tf_yaw_deg is None:
                self.prev_tf_yaw_deg = yaw_deg
                self.pan_moving = False
                return

            yaw_diff = abs(yaw_deg - self.prev_tf_yaw_deg)

            # 180도 경계 보정
            yaw_diff = min(yaw_diff, 360.0 - yaw_diff)

            if yaw_diff > self.pan_move_thresh_deg:
                self.pan_moving = True
            else:
                self.pan_moving = False

            self.prev_tf_yaw_deg = yaw_deg

        except Exception as e:
            self.get_logger().warn(f"Failed to update pan motion from TF: {e}")
            self.pan_moving = False

    # ---------------- Callbacks ----------------
    def camera_info_callback(self, msg: CameraInfo):
        k = msg.k

        with self.state_lock:
            self.fx = float(k[0])
            self.fy = float(k[4])
            self.cx = float(k[2])
            self.cy = float(k[5])
            self.width = int(msg.width)
            self.height = int(msg.height)
            self.have_caminfo = True

    def depth_callback(self, msg: Image):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding=msg.encoding,
            )

            if depth_image.dtype != np.float32:
                self.get_logger().warn(
                    f"depth dtype is {depth_image.dtype}, expected float32"
                )
                return

            with self.depth_lock:
                self.latest_depth_m = depth_image.copy()

        except Exception as e:
            self.get_logger().error(f"cv_bridge exception: {e}")

    def bbox_callback(self, msg: BoundingBox):
        det_ball = []
        det_robot = []
        det_corner_line = []
        det_t_line = []
        det_cross_line = []
        det_goal_post = []

        n = len(msg.class_ids)

        if not (
            len(msg.score) == n
            and len(msg.x1) == n
            and len(msg.y1) == n
            and len(msg.x2) == n
            and len(msg.y2) == n
        ):
            self.get_logger().warn("BoundingBox arrays size mismatch.")
            return

        for i in range(n):
            cls = int(msg.class_ids[i])
            score = float(msg.score[i])
            x1 = int(msg.x1[i])
            y1 = int(msg.y1[i])
            x2 = int(msg.x2[i])
            y2 = int(msg.y2[i])

            bbox = (score, x1, y1, x2, y2)

            if cls == 0:
                det_ball.append(bbox)
            elif cls == 1:
                det_robot.append(bbox)
            elif cls == 2:
                det_corner_line.append(bbox)
            elif cls == 3:
                det_t_line.append(bbox)
            elif cls == 4:
                det_cross_line.append(bbox)
            elif cls == 5:
                det_goal_post.append(bbox)
            
        with self.state_lock:
            self.det_ball = det_ball
            self.det_robot = det_robot
            self.det_corner_line = det_corner_line
            self.det_t_line = det_t_line
            self.det_cross_line = det_cross_line
            self.det_goal_post = det_goal_post

    # ---------------- Processing ----------------
    def bbox_processing(self):
        self.update_pan_motion_from_tf()

        with self.state_lock:
            if not self.have_caminfo:
                return

            fx = self.fx
            fy = self.fy
            cx = self.cx
            cy = self.cy

            if self.use_pan_move_flag:
                # Pan이 움직이는 동안은 선과 골대 검출 결과를 무시하여, 선과 골대가 일그러지는 현상을 완화
                if self.pan_moving:
                    corner_line = []
                    t_line = []
                    cross_line = []
                    goal_post = []
                else:
                    corner_line = list(self.det_corner_line)
                    t_line = list(self.det_t_line)
                    cross_line = list(self.det_cross_line)
                    goal_post = list(self.det_goal_post)

                detections = {
                    "ball": list(self.det_ball),
                    "robot": list(self.det_robot),
                    "corner_line": corner_line,
                    "t_line": t_line,
                    "cross_line": cross_line,
                    "goal_post": goal_post,
                }
            else:
                self.pan_moving = False

                detections = {
                    "ball": list(self.det_ball),
                    "robot": list(self.det_robot),
                    "corner_line": list(self.det_corner_line),
                    "t_line": list(self.det_t_line),
                    "cross_line": list(self.det_cross_line),
                    "goal_post": list(self.det_goal_post),
                }
            

        with self.depth_lock:
            if self.latest_depth_m is None:
                return

            depth_copy = self.latest_depth_m.copy()

        vision_msg, vision_feature_msg = self.refiner.refine(
            depth_m=depth_copy,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            detections=detections,
            transform_point=self.transform_point_to_base,
        )

        vision_msg.flag_pan = 1 if self.pan_moving else 0

        self.vision_pub.publish(vision_msg)
        self.vision_feature_pub.publish(vision_feature_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RefinerNode()

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