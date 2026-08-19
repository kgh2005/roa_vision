from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    model_path = PathJoinSubstitution(
        [FindPackageShare("vision_tensorrt"), "model", "best.engine"]
    )

    parameter_path = PathJoinSubstitution(
        [FindPackageShare("vision_tensorrt"), "config", "params.yaml"]
    )

    zed_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("zed_wrapper"), "launch", "zed_camera.launch.py"]
            )
        ),
        launch_arguments={
            "camera_model": "zedm",
            "camera_name": "zedm",
            "publish_tf": "false",
            "publish_map_tf": "false",
            "publish_imu_tf": "false",
            "use_sim_time": "false",
        }.items(),
    )

    detection_node = Node(
        package="vision_tensorrt",
        executable="vision_tensorrt_node",
        name="vision_tensorrt_node",
        output="screen",
        parameters=[
            parameter_path,
            {
                "model_path": model_path,
            },
        ],
    )

    return LaunchDescription([
        zed_launch,
        detection_node,
    ])
