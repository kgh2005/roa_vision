import numpy as np
import math

from vision_interfaces.msg import Robocupvision, Robocupvisionfeature

from robocup_zedm.refiner.utils import pixel_to_cam_coords


class Refiner:
    def __init__(self, half_win: int, remove_space_dis: int):
        self.half_win = int(half_win)
        self.remove_space_dis = int(remove_space_dis)

    def refine(
        self,
        depth_m,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        detections: dict,
        transform_point,
    ):
        vision_msg = Robocupvision()
        vision_feature_msg = Robocupvisionfeature()

        if depth_m is None or depth_m.size == 0:
            self._set_ball_not_found(vision_msg)
            return vision_msg, vision_feature_msg

        h, w = depth_m.shape[:2]

        det_ball = detections.get("ball", [])
        det_robot = detections.get("robot", [])
        det_corner_line = detections.get("corner_line", [])
        det_t_line = detections.get("t_line", [])
        det_cross_line = detections.get("cross_line", [])
        det_goal_post = detections.get("goal_post", [])

        self._process_ball(
            vision_msg=vision_msg,
            depth_m=depth_m,
            det_ball=det_ball,
            w=w,
            h=h,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            transform_point=transform_point,
        )

        self._process_robots(
            vision_msg=vision_msg,
            depth_m=depth_m,
            det_robot=det_robot,
            w=w,
            h=h,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            transform_point=transform_point,
        )

        self._process_features(
            vision_msg=vision_msg,
            vision_feature_msg=vision_feature_msg,
            depth_m=depth_m,
            detections=det_corner_line,
            feature_type="corner",
            w=w,
            h=h,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            transform_point=transform_point,
        )

        self._process_features(
            vision_msg=vision_msg,
            vision_feature_msg=vision_feature_msg,
            depth_m=depth_m,
            detections=det_t_line,
            feature_type="t",
            w=w,
            h=h,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            transform_point=transform_point,
        )

        self._process_features(
            vision_msg=vision_msg,
            vision_feature_msg=vision_feature_msg,
            depth_m=depth_m,
            detections=det_cross_line,
            feature_type="cross",
            w=w,
            h=h,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            transform_point=transform_point,
        )

        self._process_goal_posts(
            vision_msg=vision_msg,
            depth_m=depth_m,
            detections=det_goal_post,
            w=w,
            h=h,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            transform_point=transform_point,
        )

        return vision_msg, vision_feature_msg

    def _bbox_center(self, x1: int, y1: int, x2: int, y2: int, w: int, h: int):
        u = x1 + (x2 - x1) // 2
        v = y1 + (y2 - y1) // 2

        u = int(np.clip(u, 0, w - 1))
        v = int(np.clip(v, 0, h - 1))

        return u, v

    def _project_and_transform(
        self,
        depth_m,
        u: int,
        v: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        transform_point,
    ):
        """
        1. Pixel + depth -> camera optical frame 3D point
        2. camera optical frame -> base_link frame using TF

        Returns:
            X, Y, Z:
                point in zedm_left_camera_optical_frame [m]

            Xt, Yt, Zt:
                point in base_link [m]
        """
        X, Y, Z = pixel_to_cam_coords(
            depth_m,
            u,
            v,
            fx,
            fy,
            cx,
            cy,
            self.half_win,
        )

        if Z <= 0.0:
            return X, Y, Z, 0.0, 0.0, -1.0

        Xt, Yt, Zt = transform_point(X, Y, Z)

        return X, Y, Z, Xt, Yt, Zt

    def _set_ball_not_found(self, vision_msg: Robocupvision):
        vision_msg.ball_x = 0
        vision_msg.ball_y = 0

        vision_msg.ball_cam_x = -999
        vision_msg.ball_cam_y = -999

        vision_msg.ball_2d_x = 0.0
        vision_msg.ball_2d_y = 0.0
        vision_msg.ball_d = 0.0

    def _process_ball(
        self,
        vision_msg: Robocupvision,
        depth_m,
        det_ball,
        w: int,
        h: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        transform_point,
    ):
        if len(det_ball) == 0:
            self._set_ball_not_found(vision_msg)
            return

        score, x1, y1, x2, y2 = det_ball[0]

        u, v = self._bbox_center(x1, y1, x2, y2, w, h)

        X, Y, Z, Xt, Yt, Zt = self._project_and_transform(
            depth_m=depth_m,
            u=u,
            v=v,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            transform_point=transform_point,
        )

        # base_link 기준:
        # x: 로봇 전방
        # y: 로봇 왼쪽
        # z: 로봇 위쪽
        #
        # 따라서 앞에 있는 물체인지 확인하려면 Xt를 봐야 함.
        if Xt <= 0.0:
            self._set_ball_not_found(vision_msg)
            return

        ball_dist_mm = Xt * 1000.0

        vision_msg.ball_x = int(u)
        vision_msg.ball_y = int(v)

        # 카메라 optical frame 기준 원본 좌표
        # X: 이미지 오른쪽
        # Y: 이미지 아래
        # Z: 카메라 정면
        vision_msg.ball_cam_x = int(X * 1000.0)
        vision_msg.ball_cam_y = int(Y * 1000.0)

        # base_link 기준 2D 좌표
        # 기존 메시지 의미를 유지한다고 가정:
        # ball_2d_x: 좌우 방향
        # ball_2d_y: 전방 방향
        #
        # base_link에서 좌우는 Yt, 전방은 Xt
        vision_msg.ball_2d_x = float(Yt * 1000.0)
        vision_msg.ball_2d_y = float(Xt * 1000.0)
        vision_msg.ball_d = float(ball_dist_mm)

    def _process_robots(
        self,
        vision_msg: Robocupvision,
        depth_m,
        det_robot,
        w: int,
        h: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        transform_point,
    ):
        for score, x1, y1, x2, y2 in det_robot:
            u, v = self._bbox_center(x1, y1, x2, y2, w, h)

            X, Y, Z, Xt, Yt, Zt = self._project_and_transform(
                depth_m=depth_m,
                u=u,
                v=v,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                transform_point=transform_point,
            )

            if Xt <= 0.0:
                continue

            # base_link 기준
            # x: 전방, y: 왼쪽
            vision_msg.robot_vec_x.append(float(Yt * 1000.0))
            vision_msg.robot_vec_y.append(float(Xt * 1000.0))

    def _process_features(
        self,
        vision_msg: Robocupvision,
        vision_feature_msg: Robocupvisionfeature,
        depth_m,
        detections,
        feature_type: str,
        w: int,
        h: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        transform_point,
    ):
        # 공이 정상적으로 잡혔는지 확인
        ball_found = vision_msg.ball_d > 0.0
        for score, x1, y1, x2, y2 in detections:
            u, v = self._bbox_center(x1, y1, x2, y2, w, h)

            X, Y, Z, Xt, Yt, Zt = self._project_and_transform(
                depth_m=depth_m,
                u=u,
                v=v,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                transform_point=transform_point,
            )

            if Xt <= 0.0:
                continue

            dist_mm = Xt * 1000.0

            if dist_mm >= float(self.remove_space_dis):
                continue

            # base_link 기준
            # point_x: 좌우
            # point_y: 전방
            point_x = float(Yt * 1000.0)
            point_y = float(Xt * 1000.0)

            if ball_found:
                # feature 좌표 - 공 좌표
                dx = point_x - float(vision_msg.ball_2d_x)  # 좌우 차이
                dy = point_y - float(vision_msg.ball_2d_y)  # 전방 차이

                # 공 기준 feature 방향각
                # 0도: 공 기준 전방
                # +각도: 공 기준 왼쪽
                # -각도: 공 기준 오른쪽
                feat_to_ball_angle = math.degrees(math.atan2(dx, dy))

                # 공과 feature 사이 거리 [mm]
                feat_to_ball_distance = math.sqrt(dx * dx + dy * dy)
            else:
                feat_to_ball_angle = 0.0
                feat_to_ball_distance = 0.0

            if feature_type == "corner":
                vision_feature_msg.corner_confidence.append(float(score))
                vision_feature_msg.corner_distance.append(float(dist_mm))
                vision_feature_msg.corner_point_vec_x.append(point_x)
                vision_feature_msg.corner_point_vec_y.append(point_y)
                vision_feature_msg.corner_ball_relative_angle.append(float(feat_to_ball_angle))
                vision_feature_msg.corner_ball_feature_distance.append(float(feat_to_ball_distance))

            elif feature_type == "t":
                vision_feature_msg.t_confidence.append(float(score))
                vision_feature_msg.t_distance.append(float(dist_mm))
                vision_feature_msg.t_point_vec_x.append(point_x)
                vision_feature_msg.t_point_vec_y.append(point_y)
                vision_feature_msg.t_ball_relative_angle.append(float(feat_to_ball_angle))
                vision_feature_msg.t_ball_feature_distance.append(float(feat_to_ball_distance))

            elif feature_type == "cross":
                vision_feature_msg.cross_confidence.append(float(score))
                vision_feature_msg.cross_distance.append(float(dist_mm))
                vision_feature_msg.cross_point_vec_x.append(point_x)
                vision_feature_msg.cross_point_vec_y.append(point_y)
                vision_feature_msg.cross_ball_relative_angle.append(float(feat_to_ball_angle))
                vision_feature_msg.cross_ball_feature_distance.append(float(feat_to_ball_distance))

    def _process_goal_posts(
        self,
        vision_msg: Robocupvision,
        depth_m,
        detections,
        w: int,
        h: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        transform_point,
    ):
        # 이전 값 누적 방지
        vision_msg.goal_post_pan_deg = []

        goal_pan_deg_list = []

        for score, x1, y1, x2, y2 in detections:
            u, v = self._bbox_center(x1, y1, x2, y2, w, h)

            X, Y, Z, Xt, Yt, Zt = self._project_and_transform(
                depth_m=depth_m,
                u=u,
                v=v,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                transform_point=transform_point,
            )

            # base_link 기준 x가 전방
            if Xt <= 0.0:
                continue

            # base_link 기준 골대 기둥 방향각
            # Xt: 전방, Yt: 왼쪽
            pan_deg = math.degrees(math.atan2(Yt, Xt))
            goal_pan_deg_list.append(float(pan_deg))

        # 골대 기둥이 정확히 2개 보일 때만 중심 방향각 publish
        if len(goal_pan_deg_list) == 2:
            avg_pan_deg = sum(goal_pan_deg_list) / 2.0
            vision_msg.goal_post_pan_deg.append(float(avg_pan_deg))