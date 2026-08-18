#pragma once

#include <cstdint>
#include <unordered_set>
#include <vector>

#include "vision_tensorrt/type.hpp"

namespace vision_tensorrt
{

class DetectionFilter
{
public:
  explicit DetectionFilter(
    const std::vector<int64_t> & single_detection_class_ids);

  std::vector<Detection> apply(
    std::vector<Detection> detections) const;

private:
  std::unordered_set<int> single_detection_class_ids_;
};

}  // namespace vision_tensorrt