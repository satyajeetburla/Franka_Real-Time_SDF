cmake_minimum_required(VERSION 3.0.2)
project(sdf)

find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES sdf
#  CATKIN_DEPENDS orb_slam3_ros_wrapper rospy
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

