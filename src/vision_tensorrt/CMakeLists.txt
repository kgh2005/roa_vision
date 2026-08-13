cmake_minimum_required(VERSION 3.8)
project(vision_tensorrt)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
  )
endif()

# ==============================
# Dependencies
# ==============================

find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(vision_interfaces REQUIRED)

find_package(OpenCV REQUIRED COMPONENTS
  core
  dnn
  imgproc
  highgui
)

find_package(CUDAToolkit REQUIRED)

# ==============================
# TensorRT
# ==============================

find_path(TENSORRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS
    /usr/include
    /usr/local/include
    /usr/include/aarch64-linux-gnu
  PATH_SUFFIXES
    ${CMAKE_LIBRARY_ARCHITECTURE}
)

find_library(TENSORRT_LIBRARY
  NAMES nvinfer
  HINTS
    /usr/lib
    /usr/local/lib
    /usr/lib/aarch64-linux-gnu
  PATH_SUFFIXES
    ${CMAKE_LIBRARY_ARCHITECTURE}
)

if(NOT TENSORRT_INCLUDE_DIR)
  message(FATAL_ERROR "TensorRT header NvInfer.h was not found")
endif()

if(NOT TENSORRT_LIBRARY)
  message(FATAL_ERROR "TensorRT library libnvinfer.so was not found")
endif()

message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT library: ${TENSORRT_LIBRARY}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# ==============================
# Executable
# ==============================

add_executable(vision_tensorrt_node
  src/main.cpp
  src/node.cpp
  src/detector.cpp
  src/parameters.cpp
)

target_compile_features(
  vision_tensorrt_node
  PRIVATE
    cxx_std_17
)

target_include_directories(
  vision_tensorrt_node
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(
  vision_tensorrt_node
  rclcpp
  sensor_msgs
  cv_bridge
  vision_interfaces
)

target_link_libraries(
  vision_tensorrt_node
  CUDA::cudart
  ${TENSORRT_LIBRARY}
  ${OpenCV_LIBRARIES}
)

# ==============================
# Install
# ==============================

install(
  TARGETS vision_tensorrt_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY
    config
    launch
    model
  DESTINATION share/${PROJECT_NAME}
)

# ==============================
# Testing
# ==============================

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()
endif()

ament_package()