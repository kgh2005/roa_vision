#pragma once

namespace vision_tensorrt
{

struct Detection
{
  int class_id{-1};
  float confidence{0.0F};

  int x1{0};
  int y1{0};
  int x2{0};
  int y2{0};
};

}  // namespace vision_tensorrt