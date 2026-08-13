#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "vision_tensorrt/node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<vision_tensorrt::TensoRTNode>();
  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}