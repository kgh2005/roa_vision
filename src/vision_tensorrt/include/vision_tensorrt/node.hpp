#pragma once

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "vision_tensorrt/detector.hpp"
#include "vision_tensorrt/parameters.hpp"
#include "vision_tensorrt/type.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <mutex>

namespace vision_tensorrt
{
class Detector;

class TensoRTNode : public rclcpp::Node
{
public:
  explicit TensoRTNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~TensoRTNode() override;

private:
  // ===== 상태 =====
  cv::Mat bgr_image_;
  std::mutex image_mutex_;

  Parameters params_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::unique_ptr<Detector> detector_;

  // ===== ROS 통신 =====
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;

  // ===== Callback =====
  void timer_callback();
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg);
};
} // namespace vision_tensorrt