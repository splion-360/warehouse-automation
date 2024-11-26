from launch import LaunchDescription
from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

import os

def generate_launch_description():
    package_name='seg'
    pkg_dir = os.path.join(os.getcwd(), 'src')

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        condition=IfCondition(LaunchConfiguration('rviz')),
        arguments=['-d', os.path.join(pkg_dir, package_name, 'rviz', 'seg.rviz')],
        )

    
    segnet_node = Node(
        package = package_name,
        executable='segment',
    )

    return LaunchDescription([ 
        segnet_node,
        DeclareLaunchArgument('rviz', default_value='true',
                              description='Open RViz'),
        rviz,
  
    ])