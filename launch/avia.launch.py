#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get the package share directory
    pkg_share = get_package_share_directory('akf_lio')

    # Declare launch arguments
    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Whether to launch RViz'
    )

    # Config file path
    config_file = os.path.join(pkg_share, 'config', 'avia.yaml')
    rviz_config = os.path.join(pkg_share, 'rviz_cfg', 'loam.rviz')

    # AKF-LIO node
    akf_lio_node = Node(
        package='akf_lio',
        executable='run_mapping_online',
        name='laserMapping',
        output='screen',
        parameters=[
            config_file,
            {'runtime_pos_log_enable': False}
        ]
    )

    # RViz node (conditional)
    rviz_node = GroupAction(
        condition=IfCondition(LaunchConfiguration('rviz')),
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='ivox_rviz',
                arguments=['-d', rviz_config],
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        rviz_arg,
        akf_lio_node,
        rviz_node,
    ])
