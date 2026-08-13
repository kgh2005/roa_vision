#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <opencv2/core/mat.hpp>
#include <rclcpp/logger.hpp>

#include "vision_tensorrt/type.hpp"

namespace vision_tensorrt
{

class TrtLogger : public nvinfer1::ILogger
{
public:
  explicit TrtLogger(const rclcpp::Logger & logger);

  void log(
    Severity severity,
    const char * message) noexcept override;

private:
  rclcpp::Logger logger_;
};

class Detector
{
public:
  Detector(
    const std::string & engine_path,
    const std::vector<double> & confidence_thresholds,
    double nms_threshold,
    const rclcpp::Logger & logger);

  ~Detector();

  Detector(const Detector &) = delete;
  Detector & operator=(const Detector &) = delete;

  bool initialize();
  bool initialized() const;

  std::vector<Detection> infer(const cv::Mat & bgr_image);

private:
  bool load_engine();
  bool allocate_buffers();

  bool preprocess(const cv::Mat & bgr_image);
  bool execute();

  std::vector<Detection> postprocess(
    int original_width,
    int original_height) const;

  std::vector<Detection> apply_classwise_nms(
    std::vector<Detection> detections) const;

  void release();

  std::string engine_path_;
  std::vector<double> confidence_thresholds_;
  float nms_threshold_{0.4F};
  
  rclcpp::Logger ros_logger_;
  TrtLogger trt_logger_;

  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  std::string input_name_;
  std::string output_name_;

  int input_width_{0};
  int input_height_{0};

  std::size_t input_elements_{0};
  std::size_t output_elements_{0};
  std::size_t max_detections_{0};

  void * device_input_{nullptr};
  void * device_output_{nullptr};

  cudaStream_t stream_{nullptr};

  std::vector<float> host_input_;
  std::vector<float> host_output_;

  bool raw_yolo_output_{false};
  bool initialized_{false};
};

}  // namespace vision_tensorrt