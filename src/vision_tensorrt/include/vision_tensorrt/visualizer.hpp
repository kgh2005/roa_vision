#pragma once

#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <rclcpp/logger.hpp>

#include "vision_tensorrt/type.hpp"

namespace vision_tensorrt
{

class Visualizer
{
public:
  explicit Visualizer(
    const rclcpp::Logger & logger,
    const std::vector<std::string> & class_names,
    std::string window_name = "TensorRT Detection");

  void show(
    cv::Mat & image,
    const std::vector<Detection> & detections) const;

private:
  rclcpp::Logger logger_;
  std::vector<std::string> class_names_;
  std::vector<cv::Scalar> class_colors_;
  std::string window_name_;
};

}  // namespace vision_tensorrt