#pragma once

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/core/mat.hpp>

#include "vision_tensorrt/detector.hpp"
#include "vision_tensorrt/parameters.hpp"
#include "vision_tensorrt/type.hpp"
#include "vision_tensorrt/visualizer.hpp"
#include "vision_tensorrt/detection_filter.hpp"

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
  std_msgs::msg::Header image_header_;
  std::mutex image_mutex_;

  Parameters params_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::unique_ptr<Detector> detector_;
  DetectionFilter detection_filter_;
  Visualizer visualizer_;

  // ===== ROS 통신 =====
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detection_pub_;

  // ===== Callback =====
  void timer_callback();
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg);
};
} // namespace vision_tensorrt