#include "vision_tensorrt/detection_filter.hpp"

#include <cstddef>
#include <unordered_map>
#include <utility>

namespace vision_tensorrt
{

DetectionFilter::DetectionFilter(
  const std::vector<int64_t> & single_detection_class_ids)
{
  for (const auto class_id : single_detection_class_ids) {
    single_detection_class_ids_.insert(static_cast<int>(class_id));
  }
}

std::vector<Detection> DetectionFilter::apply(
  std::vector<Detection> detections) const
{
  if (single_detection_class_ids_.empty() || detections.size() < 2) {
    return detections;
  }

  std::unordered_map<int, std::size_t> best_indices;

  for (std::size_t index = 0; index < detections.size(); index++) {
    const auto & detection = detections[index];

    if (single_detection_class_ids_.find(detection.class_id) ==
      single_detection_class_ids_.end())
    {
      continue;
    }

    const auto iterator = best_indices.find(detection.class_id);

    if (iterator == best_indices.end() ||
      detection.confidence > detections[iterator->second].confidence)
    {
      best_indices[detection.class_id] = index;
    }
  }

  std::vector<Detection> filtered;
  filtered.reserve(detections.size());

  for (std::size_t index = 0; index < detections.size(); index++) {
    const auto & detection = detections[index];

    const bool is_single_detection_class =
      single_detection_class_ids_.find(detection.class_id) !=
      single_detection_class_ids_.end();

    if (!is_single_detection_class ||
      best_indices.at(detection.class_id) == index)
    {
      filtered.push_back(std::move(detections[index]));
    }
  }

  return filtered;
}

}  // namespace vision_tensorrt