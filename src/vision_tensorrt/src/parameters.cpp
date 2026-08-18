#include "vision_tensorrt/parameters.hpp"

#include <stdexcept>

namespace vision_tensorrt
{

Parameters::Parameters(rclcpp::Node & node)
{
  
  hz = node.declare_parameter<double>("hz", 15.0);

  engine_path = node.declare_parameter<std::string>(
    "model_path",
    "model/best.engine");

  rgb_input_topic = node.declare_parameter<std::string>(
    "topic.rgb_topic",
    "/camera/image_raw");
  
  class_names = node.declare_parameter<std::vector<std::string>>(
    "class_names",
    {"ball", "robot", "L", "T", "X", "goal"});
  
  confidence_thresholds = node.declare_parameter<std::vector<double>>(
    "confidence_thresholds",
    {0.5, 0.5, 0.5, 0.5, 0.5, 0.5});

  nms_threshold = node.declare_parameter<double>("nms_threshold", 0.4);

  single_detection_class_ids = node.declare_parameter<std::vector<int64_t>>(
    "single_detection_class_ids",
    std::vector<int64_t>{});

  if (hz <= 0.0) {
    throw std::invalid_argument("hz must be greater than 0");
  }

  if (engine_path.empty()) {
    throw std::invalid_argument("engine_path must not be empty");
  }

  if (rgb_input_topic.empty()) {
    throw std::invalid_argument("rgb_input_topic must not be empty");
  }

  if (class_names.size() != confidence_thresholds.size()) {
    throw std::invalid_argument("class_names and confidence_thresholds must have the same size");
  }

  if (nms_threshold < 0.0 || nms_threshold > 1.0) {
    throw std::invalid_argument("nms_threshold must be between 0 and 1");
  }

  for (const auto class_id : single_detection_class_ids) {
    if (class_id < 0 ||
      static_cast<std::size_t>(class_id) >= class_names.size())
    {
      throw std::invalid_argument(
        "single_detection_class_ids contains an invalid class ID");
    }
  }
}

} // namespace vision_tensorrt