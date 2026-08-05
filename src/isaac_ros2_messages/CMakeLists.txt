cmake_minimum_required(VERSION 3.8)
project(isaac_ros2_messages)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetPrims.srv"
  "srv/GetPrimAttributes.srv"
  "srv/GetPrimAttribute.srv"
  "srv/SetPrimAttribute.srv"
)

ament_package()
