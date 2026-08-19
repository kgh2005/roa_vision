#include "vision_tensorrt/detection_message_converter.hpp"

namespace vision_tensorrt
{

vision_interfaces::msg::BoundingBox to_bounding_box(
  const std::vector<Detection> & detections)
{
  vision_interfaces::msg::BoundingBox message;
  message.class_ids.reserve(detections.size());
  message.score.reserve(detections.size());
  message.x1.reserve(detections.size());
  message.y1.reserve(detections.size());
  message.x2.reserve(detections.size());
  message.y2.reserve(detections.size());

  for (const auto & detection : detections) {
    message.class_ids.push_back(detection.class_id);
    message.score.push_back(detection.confidence);
    message.x1.push_back(detection.x1);
    message.y1.push_back(detection.y1);
    message.x2.push_back(detection.x2);
    message.y2.push_back(detection.y2);
  }

  return message;
}

}  // namespace vision_tensorrt
