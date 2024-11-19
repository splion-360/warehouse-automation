from launch import LaunchDescription
from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, ExecuteProcess

import os

def generate_launch_description():
    package_name='bbox'
    pkg_dir = os.path.join(os.getcwd(), 'src')
   
    # video_node = Node(
    #     package = package_name,
    #     executable='video',
    # )
    
    rosbag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', os.path.join(pkg_dir, package_name, package_name, 'bag', 'internship_assignment_sample_bag_0.db3'), '--loop', '--rate', '1.0'],
    )


    rviz = Node(
        package='rviz2',
        executable='rviz2',
        condition=IfCondition(LaunchConfiguration('rviz')),
        arguments=['-d', os.path.join(pkg_dir, package_name, 'rviz', 'bbox.rviz')],
        )

    
    segnet_node = Node(
        package = package_name,
        executable='bounding_box',
    )

    return LaunchDescription([
        # video_node,
        rosbag_play, 
        segnet_node,
        DeclareLaunchArgument('rviz', default_value='true',
                              description='Open RViz'),
        rviz,
  
    ])