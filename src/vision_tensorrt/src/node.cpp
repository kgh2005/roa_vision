#include <chrono>
#include <functional>
#include <utility>

#include "vision_tensorrt/node.hpp"

namespace vision_tensorrt
{

TensoRTNode::TensoRTNode(const rclcpp::NodeOptions & options) : Node("vision_tensorrt_node", options), params_(*this)
{
  detector_ = std::make_unique<Detector>(
    params_.engine_path,
    params_.confidence_thresholds,
    params_.nms_threshold,
    get_logger());

  if (!detector_->initialize()) {
    RCLCPP_FATAL( get_logger(), "TensorRT detector 초기화 실패");
    return;
  }

  image_sub_ = create_subscription<sensor_msgs::msg::Image>(params_.rgb_input_topic, rclcpp::SensorDataQoS(), std::bind(&TensoRTNode::image_callback, this, std::placeholders::_1));

  timer_ = create_wall_timer(
  std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(1.0 / params_.hz)), std::bind(&TensoRTNode::timer_callback, this));

  RCLCPP_INFO(get_logger(), "Vision TensorRT node started");
}

TensoRTNode::~TensoRTNode() = default;

void TensoRTNode::timer_callback()
{
  cv::Mat image;

  {
    std::lock_guard<std::mutex> lock(image_mutex_);

    if (bgr_image_.empty()) {
      return;
    }

    image = bgr_image_.clone();
  }

  const auto detections = detector_->infer(image);

  // 검출 결과 표시
  for (const auto & detection : detections) {
    cv::rectangle(
      image,
      cv::Point(detection.x1, detection.y1),
      cv::Point(detection.x2, detection.y2),
      cv::Scalar(0, 255, 0),
      2);

    const std::string label =
      "class: " + std::to_string(detection.class_id) +
      " conf: " +
      cv::format("%.2f", detection.confidence);

    cv::putText(
      image,
      label,
      cv::Point(
        detection.x1,
        std::max(detection.y1 - 5, 15)),
      cv::FONT_HERSHEY_SIMPLEX,
      0.5,
      cv::Scalar(0, 255, 0),
      1,
      cv::LINE_AA);
  }

  cv::imshow("TensorRT Detection", image);

  // imshow 창의 이벤트를 처리하려면 반드시 필요
  const int key = cv::waitKey(1);

  if (key == 27) {  // ESC
    RCLCPP_INFO(
      get_logger(),
      "ESC pressed, closing display window");

    cv::destroyWindow("TensorRT Detection");
  }
}

void TensoRTNode::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  try {
    cv::Mat image =
      cv_bridge::toCvShare(msg, "bgr8")->image.clone();

    {
      std::lock_guard<std::mutex> lock(image_mutex_);
      bgr_image_ = std::move(image);
    }
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR(
      get_logger(),
      "cv_bridge exception: %s",
      e.what());
    return;
  }
}

} // namespace vision_tensorrt