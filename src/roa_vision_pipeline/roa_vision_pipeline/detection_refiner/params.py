from dataclasses import dataclass

@dataclass
class DetectionRefinerParams:
    hz: float
    half_win: int
    remove_space_dis: int
    pan_move_thresh_deg: float
    use_pan_move_flag: bool

    depth_topic: str
    camera_info_topic: str
    bbox_topic: str
    pantilt_topic: str

    vision_topic: str
    vision_feature_topic: str

    base_link_frame: str
    camera_frame: str


def declare_detection_refiner_params(node):
    node.declare_parameter("hz", 15.0)
    node.declare_parameter("half_win", 1)
    node.declare_parameter("remove_space_dis", 3000)
    node.declare_parameter("pan_move_thresh_deg", 0.3)
    node.declare_parameter("use_pan_move_flag", True)

    node.declare_parameter("topic.depth_topic", "/zedm/zed_node/depth/depth_registered",)
    node.declare_parameter("topic.camera_info_topic", "/zedm/zed_node/rgb/color/rect/image/camera_info",)
    node.declare_parameter("topic.bbox_topic", "/Bounding_box")
    node.declare_parameter("topic.pantilt_topic", "pantilt_dxl")

    node.declare_parameter("topic.vision_topic", "vision")
    node.declare_parameter("topic.vision_feature_topic", "vision_feature")

    node.declare_parameter("tf_joint.base_link_frame", "base_link")
    node.declare_parameter("tf_joint.camera_frame", "zedm_left_camera_frame")


def load_detection_refiner_params(node) -> DetectionRefinerParams:
    half_win = int(node.get_parameter("half_win").value)
    half_win = max(0, min(6, half_win))

    params = DetectionRefinerParams(
        hz=float(node.get_parameter("hz").value),
        half_win=half_win,
        remove_space_dis=int(node.get_parameter("remove_space_dis").value),
        pan_move_thresh_deg=float(node.get_parameter("pan_move_thresh_deg").value),
        use_pan_move_flag=bool(node.get_parameter("use_pan_move_flag").value),
        
        depth_topic=str(node.get_parameter("topic.depth_topic").value),
        camera_info_topic=str(node.get_parameter("topic.camera_info_topic").value),
        bbox_topic=str(node.get_parameter("topic.bbox_topic").value),
        pantilt_topic=str(node.get_parameter("topic.pantilt_topic").value),

        vision_topic=str(node.get_parameter("topic.vision_topic").value),
        vision_feature_topic=str(node.get_parameter("topic.vision_feature_topic").value),

        base_link_frame=str(node.get_parameter("tf_joint.base_link_frame").value),
        camera_frame=str(node.get_parameter("tf_joint.camera_frame").value),
    )

    return params
