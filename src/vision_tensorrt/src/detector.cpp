#include "vision_tensorrt/detector.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <utility>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>

namespace vision_tensorrt
{

TrtLogger::TrtLogger(
  const rclcpp::Logger & logger)
: logger_(logger)
{
}

void TrtLogger::log(Severity severity, const char * message) noexcept
{
  switch (severity) {
    case Severity::kINTERNAL_ERROR:
    case Severity::kERROR:
      RCLCPP_ERROR(logger_, "[TensorRT] %s", message);
      break;

    case Severity::kWARNING:
      RCLCPP_WARN(logger_, "[TensorRT] %s", message);
      break;

    default:
      RCLCPP_DEBUG(logger_, "[TensorRT] %s", message);
      break;
  }
}

Detector::Detector(
  const std::string & engine_path,
  const std::vector<double> & confidence_thresholds,
  double nms_threshold,
  const rclcpp::Logger & logger)
: engine_path_(engine_path),
  confidence_thresholds_(confidence_thresholds),
  nms_threshold_(static_cast<float>(nms_threshold)),
  ros_logger_(logger),
  trt_logger_(logger)
{
}

Detector::~Detector()
{
  release();
}

bool Detector::initialized() const
{
  return initialized_;
}

bool Detector::initialize()
{
  if (initialized_) {
    return true;
  }

  initialized_ = load_engine();
  return initialized_;
}

bool Detector::load_engine()
{
  std::ifstream file(
    engine_path_,
    std::ios::binary | std::ios::ate);

  if (!file) {
    RCLCPP_FATAL(ros_logger_, "엔진 파일을 열 수 없습니다: %s", engine_path_.c_str());
    return false;
  }

  const std::streamsize file_size = file.tellg();

  if (file_size <= 0) {
    RCLCPP_FATAL(ros_logger_, "엔진 파일이 비어 있습니다.");
    return false;
  }

  file.seekg(0, std::ios::beg);

  std::vector<char> engine_blob(
    static_cast<std::size_t>(file_size));

  if (!file.read(engine_blob.data(), file_size)) {
    RCLCPP_FATAL(ros_logger_, "엔진 파일 읽기 실패: %s", engine_path_.c_str());
    return false;
  }

  const char * serialized_engine = engine_blob.data();
  std::size_t serialized_engine_size = engine_blob.size();

  constexpr std::size_t metadata_length_size = sizeof(std::uint32_t);

  if (engine_blob.size() > metadata_length_size &&
    engine_blob[metadata_length_size] == '{')
  {
    std::uint32_t metadata_size = 0;
    std::memcpy(
      &metadata_size,
      engine_blob.data(),
      metadata_length_size);

    if (metadata_size > engine_blob.size() - metadata_length_size) {
      RCLCPP_FATAL(
        ros_logger_,
        "Ultralytics 엔진 메타데이터 헤더가 올바르지 않습니다: %s",
        engine_path_.c_str());
      return false;
    }

    const std::size_t engine_offset =
      metadata_length_size + static_cast<std::size_t>(metadata_size);

    constexpr char tensor_rt_magic[] = {'f', 't', 'r', 't'};

    if (engine_blob.size() - engine_offset < sizeof(tensor_rt_magic) ||
      std::memcmp(
        engine_blob.data() + engine_offset,
        tensor_rt_magic,
        sizeof(tensor_rt_magic)) != 0)
    {
      RCLCPP_FATAL(
        ros_logger_,
        "Ultralytics 메타데이터 뒤에 TensorRT plan이 없습니다: %s",
        engine_path_.c_str());
      return false;
    }

    serialized_engine = engine_blob.data() + engine_offset;
    serialized_engine_size = engine_blob.size() - engine_offset;

    RCLCPP_INFO(
      ros_logger_,
      "Ultralytics 엔진 메타데이터 %u바이트를 건너뜁니다.",
      metadata_size);
  }

  runtime_.reset(
    nvinfer1::createInferRuntime(trt_logger_));

  if (!runtime_) {
    RCLCPP_FATAL(ros_logger_, "TensorRT 런타임 생성 실패");
    return false;
  }

  engine_.reset(
    runtime_->deserializeCudaEngine(
      serialized_engine,
      serialized_engine_size));

  if (!engine_) {
    RCLCPP_FATAL(ros_logger_, "엔진 역직렬화 실패: %s", engine_path_.c_str());
    return false;
  }

  context_.reset(
    engine_->createExecutionContext());

  if (!context_) {
    RCLCPP_FATAL(ros_logger_, "실행 컨텍스트 생성 실패");
    return false;
  }

  for (int index = 0;
    index < engine_->getNbIOTensors();
    ++index)
  {
    const char * tensor_name =
      engine_->getIOTensorName(index);

    if (
      engine_->getTensorIOMode(tensor_name) ==
      nvinfer1::TensorIOMode::kINPUT)
    {
      input_name_ = tensor_name;
    } else {
      output_name_ = tensor_name;
    }
  }

  if (input_name_.empty() || output_name_.empty()) {
    RCLCPP_FATAL(ros_logger_, "입력 또는 출력 텐서를 찾지 못했습니다.");
    return false;
  }

  const nvinfer1::Dims input_dims =
    engine_->getTensorShape(input_name_.c_str());

  const nvinfer1::Dims output_dims =
    engine_->getTensorShape(output_name_.c_str());

  if (input_dims.nbDims != 4) {
    RCLCPP_FATAL(ros_logger_, "입력 텐서가 NCHW 형식이 아닙니다.");
    return false;
  }

  const int class_count =
    static_cast<int>(confidence_thresholds_.size());

  const bool end_to_end_output =
    output_dims.nbDims == 3 &&
    output_dims.d[0] == 1 &&
    output_dims.d[2] == 6;

  const bool raw_yolo_output =
    output_dims.nbDims == 3 &&
    output_dims.d[0] == 1 &&
    output_dims.d[1] == 4 + class_count;

  if (!end_to_end_output && !raw_yolo_output) {
    RCLCPP_FATAL(
      ros_logger_,
      "지원하지 않는 출력 텐서 형식입니다: nbDims=%d",
      output_dims.nbDims);
    return false;
  }

  raw_yolo_output_ = raw_yolo_output;

  input_height_ = input_dims.d[2];
  input_width_ = input_dims.d[3];

  input_elements_ = 1;

  for (int index = 0;
    index < input_dims.nbDims;
    ++index)
  {
    if (input_dims.d[index] <= 0) {
      RCLCPP_FATAL(ros_logger_, "동적 입력 shape은 지원하지 않습니다.");
      return false;
    }

    input_elements_ *= static_cast<std::size_t>(input_dims.d[index]);
  }

  output_elements_ = 1;

  for (int index = 0;
    index < output_dims.nbDims;
    ++index)
  {
    if (output_dims.d[index] <= 0) {
      RCLCPP_FATAL(ros_logger_, "동적 출력 shape은 지원하지 않습니다.");
      return false;
    }

    output_elements_ *= static_cast<std::size_t>(output_dims.d[index]);
  }

  max_detections_ = static_cast<std::size_t>(
    raw_yolo_output_ ? output_dims.d[2] : output_dims.d[1]);

  if (cudaStreamCreate(&stream_) != cudaSuccess) {
    RCLCPP_FATAL(ros_logger_, "CUDA stream 생성 실패");
    return false;
  }

  if (
    cudaMalloc(&device_input_, input_elements_ * sizeof(float)) != cudaSuccess)
  {
    RCLCPP_FATAL(ros_logger_, "CUDA 입력 버퍼 할당 실패");
    return false;
  }

  if (
    cudaMalloc(&device_output_, output_elements_ * sizeof(float)) != cudaSuccess)
  {
    RCLCPP_FATAL(
      ros_logger_,
      "CUDA 출력 버퍼 할당 실패");
    return false;
  }

  host_input_.resize(input_elements_);
  host_output_.resize(output_elements_);

  if (
    !context_->setTensorAddress(
      input_name_.c_str(),
      device_input_) ||
    !context_->setTensorAddress(
      output_name_.c_str(),
      device_output_))
  {
    RCLCPP_FATAL(
      ros_logger_,
      "TensorRT 입출력 버퍼 연결 실패");
    return false;
  }

  RCLCPP_INFO(
    ros_logger_,
    "엔진 로드 완료: %s (%dx%d)",
    engine_path_.c_str(),
    input_width_,
    input_height_);

  return true;
}

std::vector<Detection> Detector::infer(
  const cv::Mat & bgr_image)
{
  if (!initialized_ || bgr_image.empty()) {
    return {};
  }

  const int original_width = bgr_image.cols;
  const int original_height = bgr_image.rows;

  try {
    if (!preprocess(bgr_image)) {
      return {};
    }
  } catch (const cv::Exception & exception) {
    RCLCPP_ERROR(
      ros_logger_,
      "이미지 전처리 실패: %s",
      exception.what());
    return {};
  }

  if (!execute()) {
    return {};
  }

  return postprocess(
    original_width,
    original_height);
}

bool Detector::preprocess(
  const cv::Mat & input_image)
{
  if (input_image.empty()) {
    RCLCPP_ERROR(
      ros_logger_,
      "입력 이미지가 비어 있습니다.");
    return false;
  }

  cv::Mat gray_image;

  if (input_image.channels() == 1) {
    gray_image = input_image;
  } else if (input_image.channels() == 3) {
    cv::cvtColor(
      input_image,
      gray_image,
      cv::COLOR_BGR2GRAY);
  } else if (input_image.channels() == 4) {
    cv::cvtColor(
      input_image,
      gray_image,
      cv::COLOR_BGRA2GRAY);
  } else {
    RCLCPP_ERROR(
      ros_logger_,
      "지원하지 않는 이미지 채널 수: %d",
      input_image.channels());
    return false;
  }

  cv::Mat resized_image;

  cv::resize(
    gray_image,
    resized_image,
    cv::Size(input_width_, input_height_));

  cv::Mat normalized_image;

  resized_image.convertTo(
    normalized_image,
    CV_32FC1,
    1.0F / 255.0F);

  const std::size_t pixel_count =
    static_cast<std::size_t>(input_width_) *
    input_height_;

  if (host_input_.size() != pixel_count) {
    RCLCPP_ERROR(
      ros_logger_,
      "입력 버퍼 크기가 맞지 않습니다. "
      "buffer=%zu, expected=%zu",
      host_input_.size(),
      pixel_count);
    return false;
  }

  if (normalized_image.isContinuous()) {
    std::memcpy(
      host_input_.data(),
      normalized_image.ptr<float>(),
      pixel_count * sizeof(float));
  } else {
    for (int row = 0; row < input_height_; ++row) {
      std::memcpy(
        host_input_.data() +
        static_cast<std::size_t>(row) * input_width_,
        normalized_image.ptr<float>(row),
        static_cast<std::size_t>(input_width_) *
        sizeof(float));
    }
  }

  return true;
}

bool Detector::execute()
{
  if (
    cudaMemcpyAsync(
      device_input_,
      host_input_.data(),
      input_elements_ * sizeof(float),
      cudaMemcpyHostToDevice,
      stream_) != cudaSuccess)
  {
    RCLCPP_ERROR(
      ros_logger_,
      "입력 데이터 전송 실패");
    return false;
  }

  if (!context_->enqueueV3(stream_)) {
    RCLCPP_ERROR(
      ros_logger_,
      "TensorRT 추론 실행 실패");
    return false;
  }

  if (
    cudaMemcpyAsync(
      host_output_.data(),
      device_output_,
      output_elements_ * sizeof(float),
      cudaMemcpyDeviceToHost,
      stream_) != cudaSuccess)
  {
    RCLCPP_ERROR(
      ros_logger_,
      "출력 데이터 수신 실패");
    return false;
  }

  if (cudaStreamSynchronize(stream_) != cudaSuccess) {
    RCLCPP_ERROR(
      ros_logger_,
      "CUDA stream 동기화 실패");
    return false;
  }

  return true;
}

std::vector<Detection> Detector::postprocess(
  int original_width,
  int original_height) const
{
  std::vector<Detection> detections;

  if (raw_yolo_output_) {
    const std::size_t candidate_count = max_detections_;

    for (std::size_t index = 0; index < candidate_count; ++index) {
      const float center_x = host_output_[index];
      const float center_y = host_output_[candidate_count + index];
      const float width = host_output_[2 * candidate_count + index];
      const float height = host_output_[3 * candidate_count + index];

      int class_id = -1;
      float confidence = 0.0F;

      for (std::size_t class_index = 0;
        class_index < confidence_thresholds_.size();
        ++class_index)
      {
        const float class_confidence =
          host_output_[(4 + class_index) * candidate_count + index];

        if (class_confidence > confidence) {
          confidence = class_confidence;
          class_id = static_cast<int>(class_index);
        }
      }

      if (class_id < 0 ||
        confidence < confidence_thresholds_[static_cast<std::size_t>(class_id)])
      {
        continue;
      }

      int x1 = static_cast<int>(
        (center_x - width * 0.5F) / input_width_ * original_width);
      int y1 = static_cast<int>(
        (center_y - height * 0.5F) / input_height_ * original_height);
      int x2 = static_cast<int>(
        (center_x + width * 0.5F) / input_width_ * original_width);
      int y2 = static_cast<int>(
        (center_y + height * 0.5F) / input_height_ * original_height);

      x1 = std::clamp(x1, 0, original_width - 1);
      y1 = std::clamp(y1, 0, original_height - 1);
      x2 = std::clamp(x2, 0, original_width - 1);
      y2 = std::clamp(y2, 0, original_height - 1);

      if (x2 <= x1 || y2 <= y1) {
        continue;
      }

      detections.push_back(
        Detection{class_id, confidence, x1, y1, x2, y2});
    }

    return apply_classwise_nms(std::move(detections));
  }


  for (std::size_t index = 0;
    index < max_detections_;
    ++index)
  {
    const float * output =
      host_output_.data() + index * 6;

    const float confidence = output[4];
    const int class_id =
      static_cast<int>(output[5]);

    if (class_id < 0 ||
      static_cast<std::size_t>(class_id) >= confidence_thresholds_.size())
    {
      continue;
    }

    if (confidence < confidence_thresholds_[static_cast<std::size_t>(class_id)]) {
      continue;
    }

    int x1 = static_cast<int>(
      output[0] / input_width_ * original_width);

    int y1 = static_cast<int>(
      output[1] / input_height_ * original_height);

    int x2 = static_cast<int>(
      output[2] / input_width_ * original_width);

    int y2 = static_cast<int>(
      output[3] / input_height_ * original_height);

    x1 = std::clamp(x1, 0, original_width - 1);
    y1 = std::clamp(y1, 0, original_height - 1);
    x2 = std::clamp(x2, 0, original_width - 1);
    y2 = std::clamp(y2, 0, original_height - 1);

    if (x2 <= x1 || y2 <= y1) {
      continue;
    }


    detections.push_back(
      Detection{
        class_id,
        confidence,
        x1,
        y1,
        x2,
        y2
      });
  }

  return apply_classwise_nms(std::move(detections));
}

std::vector<Detection> Detector::apply_classwise_nms(
  std::vector<Detection> detections) const
{
  if (detections.empty()) {
    return detections;
  }

  std::vector<Detection> kept_detections;
  kept_detections.reserve(detections.size());

  for (std::size_t class_index = 0;
    class_index < confidence_thresholds_.size();
    ++class_index)
  {
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<std::size_t> detection_indices;

    for (std::size_t detection_index = 0;
      detection_index < detections.size();
      ++detection_index)
    {
      const auto & detection = detections[detection_index];

      if (detection.class_id != static_cast<int>(class_index)) {
        continue;
      }

      boxes.emplace_back(
        detection.x1,
        detection.y1,
        detection.x2 - detection.x1,
        detection.y2 - detection.y1);
      confidences.push_back(detection.confidence);
      detection_indices.push_back(detection_index);
    }

    std::vector<int> kept_indices;
    cv::dnn::NMSBoxes(
      boxes,
      confidences,
      0.0F,
      nms_threshold_,
      kept_indices);

    for (const int kept_index : kept_indices) {
      kept_detections.push_back(
        detections[detection_indices[static_cast<std::size_t>(kept_index)]]);
    }
  }

  std::sort(
    kept_detections.begin(),
    kept_detections.end(),
    [](const Detection & lhs, const Detection & rhs) {
      return lhs.confidence > rhs.confidence;
    });

  return kept_detections;
}

void Detector::release()
{
  initialized_ = false;

  if (device_input_ != nullptr) {
    cudaFree(device_input_);
    device_input_ = nullptr;
  }

  if (device_output_ != nullptr) {
    cudaFree(device_output_);
    device_output_ = nullptr;
  }

  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }

  context_.reset();
  engine_.reset();
  runtime_.reset();
}

}  // namespace vision_tensorrt