"""
Dynamic Window Approach Launch File

This launch file starts the Dynamic Window Approach node with its configuration parameters.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory
import os


def generate_launch_description():
    """
    Generate the launch description for the Dynamic Window Approach.

    Returns:
        LaunchDescription: The launch description containing the Dynamic Window Approach node
    """
    ld = LaunchDescription()

    # Get the path to the configuration file
    config_path = os.path.join(
        get_package_share_directory("overtaking"),
        "config",
        "dwa_config.yaml"
    )

    # Create the Dynamic Window Approach node
    dwa_node = Node(
        package='overtaking',
        executable='dwa_exe',
        name='dwa_ackermann_node',
        parameters=[config_path],
        output='screen',
        emulate_tty=True
    )

    ld.add_action(dwa_node)

    return ld
