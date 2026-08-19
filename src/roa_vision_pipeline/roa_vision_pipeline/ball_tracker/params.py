from dataclasses import dataclass

@dataclass
class BallTrackerParams:
    rate_hz: float

    vision_topic: str
    pantilt_topic: str
    joint_state_topic: str
    ball_lost_topic: str
    master2vision_topic: str

    pan_id: int
    pan_max_deg: float
    pan_min_deg: float

    tilt_id: int
    tilt_max_deg: float
    tilt_min_deg: float

    scan_pan_speed_deg_s: float
    scan_tilt_speed_deg_s: float

    img_w: int
    img_h: int

    img_cut: int

    pan_dir: float
    tilt_dir: float

    pan_zero_flag: bool

    torso_joint_name: str
    pan_joint_name: str
    tilt_joint_name: str




def declare_ball_tracker_params(node):
    node.declare_parameter("rate_hz", 50.0)

    node.declare_parameter("topic.vision_topic", "vision")
    node.declare_parameter("topic.pantilt_topic", "pantilt_dxl")
    node.declare_parameter("topic.joint_state_topic", "/joint_states")
    node.declare_parameter("topic.ball_lost_topic", "/camera1/pan_tilt")
    node.declare_parameter("topic.master2vision_topic", "/master2vision")

    node.declare_parameter("motor.pan_id", 22)
    node.declare_parameter("motor.pan_max_deg", 70.0)
    node.declare_parameter("motor.pan_min_deg", -70.0)

    node.declare_parameter("motor.tilt_id", 23)
    node.declare_parameter("motor.tilt_max_deg", 0.0)
    node.declare_parameter("motor.tilt_min_deg", -30.0)

    node.declare_parameter("scan_pan_speed_deg_s", 35.0)
    node.declare_parameter("scan_tilt_speed_deg_s", 35.0)

    node.declare_parameter("img_w", 960)
    node.declare_parameter("img_h", 540)

    node.declare_parameter("img_cut", 4)

    node.declare_parameter("pan_dir", 1.0)
    node.declare_parameter("tilt_dir", 1.0)

    node.declare_parameter("pan_zero_flag", False)

    node.declare_parameter("tf_joint.torso_joint_name", "torso_yaw")
    node.declare_parameter("tf_joint.pan_joint_name", "pan")
    node.declare_parameter("tf_joint.tilt_joint_name", "tilt")


def load_ball_tracker_params(node) -> BallTrackerParams:
    return BallTrackerParams(
        rate_hz=float(node.get_parameter("rate_hz").value),

        vision_topic=str(node.get_parameter("topic.vision_topic").value),
        pantilt_topic=str(node.get_parameter("topic.pantilt_topic").value),
        joint_state_topic=str(node.get_parameter("topic.joint_state_topic").value),
        ball_lost_topic=str(node.get_parameter("topic.ball_lost_topic").value),
        master2vision_topic=str(node.get_parameter("topic.master2vision_topic").value),

        pan_id=int(node.get_parameter("motor.pan_id").value),
        pan_max_deg=float(node.get_parameter("motor.pan_max_deg").value),
        pan_min_deg=float(node.get_parameter("motor.pan_min_deg").value),

        tilt_id=int(node.get_parameter("motor.tilt_id").value),
        tilt_max_deg=float(node.get_parameter("motor.tilt_max_deg").value),
        tilt_min_deg=float(node.get_parameter("motor.tilt_min_deg").value),

        scan_pan_speed_deg_s=float(node.get_parameter("scan_pan_speed_deg_s").value),
        scan_tilt_speed_deg_s=float(node.get_parameter("scan_tilt_speed_deg_s").value),

        img_w=int(node.get_parameter("img_w").value),
        img_h=int(node.get_parameter("img_h").value),

        img_cut=int(node.get_parameter("img_cut").value),

        pan_dir=float(node.get_parameter("pan_dir").value),
        tilt_dir=float(node.get_parameter("tilt_dir").value),

        pan_zero_flag=bool(node.get_parameter("pan_zero_flag").value),

        torso_joint_name=str(node.get_parameter("tf_joint.torso_joint_name").value),
        pan_joint_name=str(node.get_parameter("tf_joint.pan_joint_name").value),
        tilt_joint_name=str(node.get_parameter("tf_joint.tilt_joint_name").value),
    )
