from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    pipeline_params_path = PathJoinSubstitution(
        [FindPackageShare("roa_vision_pipeline"), "config", "vision_pipeline.yaml"]
    )

    vision_tensorrt_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("vision_tensorrt"),
                "launch",
                "vision_tensorrt.launch.py",
            ])
        )
    )

    roa_tf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("roa_vision_description"),
                "launch",
                "display.launch.py",
            ])
        ),
        launch_arguments={
            "use_sim_time": "false",
        }.items(),
    )

    detection_refiner_node = Node(
        package="roa_vision_pipeline",
        executable="detection_refiner_node",
        name="detection_refiner_node",
        output="screen",
        parameters=[pipeline_params_path],
    )

    return LaunchDescription([
        vision_tensorrt_launch,
        roa_tf_launch,
        detection_refiner_node,
    ])
