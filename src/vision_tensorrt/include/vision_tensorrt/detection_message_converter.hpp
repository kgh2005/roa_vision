#pragma once

#include <vector>

#include <vision_interfaces/msg/bounding_box.hpp>

#include "vision_tensorrt/type.hpp"

namespace vision_tensorrt
{

vision_interfaces::msg::BoundingBox to_bounding_box(
  const std::vector<Detection> & detections);

}  // namespace vision_tensorrt
