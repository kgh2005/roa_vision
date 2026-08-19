import math

from transitions import Machine

class PanTiltController:
    states = ["lost", "found"]

    def __init__(self, params):
        self.params = params

        self.rate_hz = params.rate_hz

        self.pan_max_deg = params.pan_max_deg
        self.pan_min_deg = params.pan_min_deg

        self.tilt_max_deg = params.tilt_max_deg
        self.tilt_min_deg = params.tilt_min_deg

        self.scan_pan_speed = params.scan_pan_speed_deg_s
        self.scan_tilt_speed = params.scan_tilt_speed_deg_s

        self.img_w = params.img_w
        self.img_h = params.img_h

        self.roi_hw = self.img_w // params.img_cut
        self.roi_hh = self.img_h // params.img_cut

        self.pan_dir = params.pan_dir
        self.tilt_dir = params.tilt_dir

        self.scan_points = [
            (self.pan_max_deg, self.tilt_min_deg),  # left bottom
            (0.0, self.tilt_min_deg),               # center bottom
            (self.pan_min_deg, self.tilt_min_deg),  # right bottom
            (self.pan_min_deg, self.tilt_max_deg),  # right top
            (0.0, self.tilt_max_deg),               # center top
            (self.pan_max_deg, self.tilt_max_deg),  # left top
        ]

        self.scan_i = 0
        self.scan_target_pan, self.scan_target_tilt = self.scan_points[self.scan_i]

        self.ball_seen = False

        self.ball_x = 0.0
        self.ball_y = 0.0

        self.ball_cam_x = 0.0
        self.ball_cam_y = 0.0

        self.pan_deg = 0.0
        self.tilt_deg = 0.0

        self.angle_deg = 0.0

        self.jump_mode = False
        self.jump_target_i = 0

        self.last_log = None

        self.lost_ball_right = False
        self.lost_ball_left = False

        self.machine = Machine(
            model=self,
            states=PanTiltController.states,
            initial="lost",
        )

        self.machine.add_transition("see_ball", "lost", "found")
        self.machine.add_transition("lose_ball", "found", "lost")

    def update_vision(
        self,
        ball_seen: bool,
        ball_x: int = 0.0,
        ball_y: int = 0.0,
        ball_cam_x: float = 0.0,
        ball_cam_y: float = 0.0,
    ):
        self.ball_seen = ball_seen

        if not ball_seen:
            return

        self.ball_x = float(ball_x)
        self.ball_y = float(ball_y)

        self.ball_cam_x = ball_cam_x
        self.ball_cam_y = ball_cam_y

    def tick(self):
        self.last_log = None

        if self.ball_seen and self.state == "lost":
            self.see_ball()

        elif (not self.ball_seen) and self.state == "found":
            self._start_lost_jump()
            self.lose_ball()

        dt = 1.0 / self.rate_hz
        
        match self.state:
            case"lost":
                if self.jump_mode:
                    arrived = self._move_to_target(dt)

                    if arrived:
                        self.jump_mode = False
                        self.last_log = (
                            f"lost: jump arrived, start scan at scan_i={self.scan_i}"
                        )
                else:
                    self._scan_update_constant_speed(dt)

            case "found":
                self._track_roi_px_simple(dt)

        return self.pan_deg, self.tilt_deg, self.lost_ball_right, self.lost_ball_left, self.last_log

    def _start_lost_jump(self):
        self.angle_deg = self.angle_deg_360(
            self.ball_cam_x,
            self.ball_cam_y,
        )

        angle_key = self.angle_to_scan_index(self.angle_deg)

        match angle_key:
            case "LB":
                self.lost_ball_right = False
                self.lost_ball_left = True
                self.jump_target_i = 0
            case "MB":
                self.lost_ball_right = False
                self.lost_ball_left = False
                self.jump_target_i = 1
            case "RB":
                self.lost_ball_right = True
                self.lost_ball_left = False
                self.jump_target_i = 2
            case "RT":
                self.lost_ball_right = True
                self.lost_ball_left = False
                self.jump_target_i = 3
            case "MT":
                self.lost_ball_right = False
                self.lost_ball_left = False
                self.jump_target_i = 4
            case "LT":
                self.lost_ball_right = False
                self.lost_ball_left = True
                self.jump_target_i = 5

        self.scan_i = self.jump_target_i
        self.scan_target_pan, self.scan_target_tilt = self.scan_points[self.scan_i]
        self.jump_mode = True

        self.last_log = (
            f"lost: jump start -> {angle_key} idx={self.jump_target_i} "
            f"target=({self.scan_target_pan},{self.scan_target_tilt})"
        )

    def _move_toward(self, cur, target, step):
        if cur < target:
            return min(cur + step, target)

        return max(cur - step, target)

    def _scan_update_constant_speed(self, dt):
        pan_step = abs(self.scan_pan_speed) * dt
        tilt_step = abs(self.scan_tilt_speed) * dt

        self.pan_deg = self._move_toward(
            self.pan_deg,
            self.scan_target_pan,
            pan_step,
        )

        self.tilt_deg = self._move_toward(
            self.tilt_deg,
            self.scan_target_tilt,
            tilt_step,
        )

        pan_arrived = abs(self.pan_deg - self.scan_target_pan) < 0.1
        tilt_arrived = abs(self.tilt_deg - self.scan_target_tilt) < 0.1

        if pan_arrived and tilt_arrived:
            self.scan_i = (self.scan_i + 1) % len(self.scan_points)
            self.scan_target_pan, self.scan_target_tilt = self.scan_points[
                self.scan_i
            ]

    def _track_roi_px_simple(self, dt):
        cx = self.img_w * 0.5
        cy = self.img_h * 0.5

        left = cx - self.roi_hw
        right = cx + self.roi_hw
        top = cy - self.roi_hh
        bottom = cy + self.roi_hh

        x = self.ball_x
        y = self.ball_y

        if left <= x <= right and top <= y <= bottom:
            return

        if x < left:
            self.pan_deg -= self.pan_dir * self.scan_pan_speed * dt
        elif x > right:
            self.pan_deg += self.pan_dir * self.scan_pan_speed * dt

        if y < top:
            self.tilt_deg += self.tilt_dir * self.scan_tilt_speed * dt
        elif y > bottom:
            self.tilt_deg -= self.tilt_dir * self.scan_tilt_speed * dt

        self.pan_deg = max(
            self.pan_min_deg,
            min(self.pan_max_deg, self.pan_deg),
        )

        self.tilt_deg = max(
            self.tilt_min_deg,
            min(self.tilt_max_deg, self.tilt_deg),
        )

    def angle_to_scan_index(self, angle_deg: float):
        a = angle_deg % 360.0

        if 0.0 <= a < 60.0:
            return "RT"
        elif 60.0 <= a < 120.0:
            return "MT"
        elif 120.0 <= a < 180.0:
            return "LT"
        elif 180.0 <= a < 240.0:
            return "LB"
        elif 240.0 <= a < 300.0:
            return "MB"

        return "RB"

    def _move_to_target(self, dt):
        pan_step = abs(self.scan_pan_speed) * dt
        tilt_step = abs(self.scan_tilt_speed) * dt

        self.pan_deg = self._move_toward(
            self.pan_deg,
            self.scan_target_pan,
            pan_step,
        )

        self.tilt_deg = self._move_toward(
            self.tilt_deg,
            self.scan_target_tilt,
            tilt_step,
        )

        pan_arrived = abs(self.pan_deg - self.scan_target_pan) < 0.1
        tilt_arrived = abs(self.tilt_deg - self.scan_target_tilt) < 0.1

        return pan_arrived and tilt_arrived

    def angle_deg_360(self, x, y):
        return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0