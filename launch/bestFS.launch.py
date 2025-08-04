"""
Best First Search Path Planner Launch File

This launch file starts the Best First Search path planning node with its configuration parameters.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory
import os


def generate_launch_description():
    """
    Generate the launch description for the Best First Search path planner.

    Returns:
        LaunchDescription: The launch description containing the Best First Search planner node
    """
    ld = LaunchDescription()

    # Get the path to the configuration file
    config_path = os.path.join(
        get_package_share_directory("overtaking"),
        "config",
        "bestFS_config.yaml"
    )

    # Create the Best First Search planner node
    best_fs_node = Node(
        package='overtaking',
        executable='bestFS_exe',
        name='bestFS_node',
        parameters=[config_path],
        output='screen',
        emulate_tty=True
    )

    ld.add_action(best_fs_node)

    return ld
