#include "vision_tensorrt/detection_message_converter.hpp"

#include <cstddef>
#include <string>
#include <utility>

#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>

namespace vision_tensorrt
{

vision_msgs::msg::Detection2DArray to_detection2d_array(
  const std::vector<Detection> & detections,
  const std_msgs::msg::Header & header,
  const std::vector<std::string> & class_names)
{
  vision_msgs::msg::Detection2DArray message;
  message.header = header;
  message.detections.reserve(detections.size());

  for (const auto & detection : detections) {
    vision_msgs::msg::Detection2D detection_message;
    detection_message.header = header;

    detection_message.bbox.center.position.x =
      0.5 * static_cast<double>(detection.x1 + detection.x2);
    detection_message.bbox.center.position.y =
      0.5 * static_cast<double>(detection.y1 + detection.y2);
    detection_message.bbox.center.theta = 0.0;
    detection_message.bbox.size_x = static_cast<double>(detection.x2 - detection.x1);
    detection_message.bbox.size_y = static_cast<double>(detection.y2 - detection.y1);

    vision_msgs::msg::ObjectHypothesisWithPose result;
    if (detection.class_id >= 0 &&
      static_cast<std::size_t>(detection.class_id) < class_names.size())
    {
      result.hypothesis.class_id = class_names[detection.class_id];
    } else {
      result.hypothesis.class_id = std::to_string(detection.class_id);
    }
    result.hypothesis.score = static_cast<double>(detection.confidence);
    result.pose.pose.orientation.w = 1.0;

    detection_message.results.push_back(std::move(result));
    message.detections.push_back(std::move(detection_message));
  }

  return message;
}

}  // namespace vision_tensorrt
