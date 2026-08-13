#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

namespace vision_tensorrt
{
class Parameters
{
public:
  explicit Parameters(rclcpp::Node & node);
  
  double hz;

  std::string engine_path;
  std::string rgb_input_topic;

  std::vector<std::string> class_names;
  std::vector<double> confidence_thresholds;
  double nms_threshold;
};
} // namespace vision_tensorrt