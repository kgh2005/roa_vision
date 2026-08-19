#pragma once

#include <string>
#include <vector>

#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "vision_tensorrt/type.hpp"

namespace vision_tensorrt
{

vision_msgs::msg::Detection2DArray to_detection2d_array(
  const std::vector<Detection> & detections,
  const std_msgs::msg::Header & header,
  const std::vector<std::string> & class_names);

}  // namespace vision_tensorrt
