#include "vision_tensorrt/visualizer.hpp"

#include <algorithm>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/logging.hpp>

namespace vision_tensorrt
{

namespace
{

std::vector<cv::Scalar> create_class_colors(std::size_t class_count)
{
  std::vector<cv::Scalar> colors;
  colors.reserve(class_count);

  if (class_count == 0) {
    return colors;
  }

  for (std::size_t index = 0; index < class_count; index++) {
    const int hue = static_cast<int>(180 * index / class_count);

    cv::Mat hsv(
      1,
      1,
      CV_8UC3,
      cv::Scalar(hue, 220, 255));

    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

    const cv::Vec3b pixel = bgr.at<cv::Vec3b>(0, 0);
    colors.emplace_back(pixel[0], pixel[1], pixel[2]);
  }

  return colors;
}

}  // namespace

Visualizer::Visualizer(
  const rclcpp::Logger & logger,
  const std::vector<std::string> & class_names,
  std::string window_name)
: logger_(logger),
  class_names_(class_names),
  class_colors_(create_class_colors(class_names.size())),
  window_name_(std::move(window_name))
{
}

void Visualizer::show(cv::Mat & image, const std::vector<Detection> & detections) const
{
  for (const auto & detection : detections) {
    std::string class_name = "unknown";
    cv::Scalar color{255, 255, 255};

    if (detection.class_id >= 0) {
      const auto class_index = static_cast<std::size_t>(detection.class_id);

      if (class_index < class_names_.size()) {
        class_name = class_names_[class_index];
        color = class_colors_[class_index];
      }
    }

    cv::rectangle(
      image,
      cv::Point(detection.x1, detection.y1),
      cv::Point(detection.x2, detection.y2),
      color,
      2);

    const std::string label =
      class_name +
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
      color,
      1,
      cv::LINE_AA);
  }

  cv::imshow(window_name_, image);

  if (cv::waitKey(1) == 27) {
    RCLCPP_INFO(logger_, "ESC pressed, closing display window");
    cv::destroyWindow(window_name_);
  }
}

}  // namespace vision_tensorrt