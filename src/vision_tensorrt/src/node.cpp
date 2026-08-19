#include <chrono>
#include <functional>
#include <utility>

#include "vision_tensorrt/detection_message_converter.hpp"
#include "vision_tensorrt/node.hpp"

namespace vision_tensorrt
{

TensoRTNode::TensoRTNode(const rclcpp::NodeOptions & options)
: Node("vision_tensorrt_node", options),
  params_(*this),
  detection_filter_(params_.single_detection_class_ids),
  visualizer_(get_logger(), params_.class_names)
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

  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
    params_.rgb_input_topic,
    rclcpp::SensorDataQoS().keep_last(1),
    std::bind(
      &TensoRTNode::image_callback,
      this,
      std::placeholders::_1));

  detection_pub_ = create_publisher<vision_msgs::msg::Detection2DArray>(
    params_.detection_output_topic,
    rclcpp::QoS(1));

  timer_ = create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(1.0 / params_.hz)),
    std::bind(&TensoRTNode::timer_callback, this));

  RCLCPP_INFO(get_logger(), "Vision TensorRT node started");
}

TensoRTNode::~TensoRTNode() = default;

void TensoRTNode::timer_callback()
{
  using Clock = std::chrono::steady_clock;

  const auto total_start = Clock::now();

  cv::Mat image;
  std_msgs::msg::Header image_header;

  {
    std::lock_guard<std::mutex> lock(image_mutex_);

    if (bgr_image_.empty()) {
      return;
    }

    image = bgr_image_.clone();
    image_header = image_header_;
  }

  const auto inference_start = Clock::now();

  auto detections = detector_->infer(image);

  const auto inference_end = Clock::now();

  detections =
    detection_filter_.apply(std::move(detections));

  detection_pub_->publish(
    to_detection2d_array(detections, image_header, params_.class_names));

  visualizer_.show(image, detections);

  const auto total_end = Clock::now();

  const double inference_ms =
    std::chrono::duration<double, std::milli>(
    inference_end - inference_start).count();

  const double total_ms =
    std::chrono::duration<double, std::milli>(
    total_end - total_start).count();

  RCLCPP_INFO_THROTTLE(
    get_logger(),
    *get_clock(),
    1000,
    "[PERF] inference=%.2f ms | total=%.2f ms | detections=%zu",
    inference_ms,
    total_ms,
    detections.size());
}

void TensoRTNode::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  try {
    cv::Mat image =
      cv_bridge::toCvShare(msg, "bgr8")->image.clone();

    {
      std::lock_guard<std::mutex> lock(image_mutex_);
      bgr_image_ = std::move(image);
      image_header_ = msg->header;
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