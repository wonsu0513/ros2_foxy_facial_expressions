import os
from ament_index_python.packages import get_package_share_directory , get_search_paths
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch import LaunchDescription, Action
import launch


def generate_launch_description():  
    ferplus_8_onnx_cpu = Node(
        package='ros2_foxy_facial_expressions', 
        executable='ferplus_8_onnx_node', 
        name='ferplus_8_onnx_node',
        namespace="Wonse_Jo_ZZANG",
        output='screen',
        parameters=[{'input_raw_camera': '/camera/color/image_raw', 
            'input_compressed_camera': '/camera/color/image_raw/compressed',
            'compressed_input_mode': True,
            'GPU_mode': False,
            }] 
        )         

    return LaunchDescription([ferplus_8_onnx_cpu])
