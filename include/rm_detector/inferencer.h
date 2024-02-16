// yolov5使用tensorrt进行部署的头文件

#ifndef INFERENCER_H
#define INFERENCER_H

#include "cuda_utils.h"
#include "logging.h"
#include "NvOnnxParser.h"

#include "preprocess.h"
#include "postprocess.h"

#include <iostream>
#include <chrono>
#include <cmath>

using namespace nvinfer1;

namespace rm_detector
{
class Inferencer
{
public:
  Inferencer();

  ~Inferencer();

  void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer,
                       float** cpu_output_buffer);

  void detect(std::vector<cv::Mat>& frame);

  void infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output, int batchsize);

  void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine,
                          IExecutionContext** context);

  void init(std::string& engine_path, Logger& logger, float conf_thresh, float nms_thresh, int input_h, int input_w);

  IRuntime* runtime_{ nullptr };
  ICudaEngine* engine_{ nullptr };
  IExecutionContext* context_{ nullptr };
  Logger* logger_{ nullptr };

  float* gpu_buffers_[2];
  float* cpu_output_buffer_ = nullptr;
  int input_h_;
  int input_w_;

  const static int K_OUTPUT_SIZE = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

  std::vector<std::vector<Detection>> target_objects_;
  // CudaProprecess cudaProprecess_;

  float conf_thresh_;
  float nms_thresh_;
};

class ArmorInferencer
{
public:
  ArmorInferencer();
  ~ArmorInferencer();
  void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer,
                       float** cpu_output_buffer);
  std::pair<int, float> detect(std::vector<cv::Mat>& frame);

  void infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output, int batchsize);

  void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine,
                          IExecutionContext** context);

  void init(std::string& engine_path, Logger& logger, float conf_thresh, float nms_thresh);

  IRuntime* runtime_{ nullptr };
  ICudaEngine* engine_{ nullptr };
  IExecutionContext* context_{ nullptr };
  Logger* logger_{ nullptr };

  float* gpu_buffers_[2];
  float* cpu_output_buffer_ = nullptr;

  const static int CLASS_OUTPUT_SIZE = 12;
  int input_h_ = 224;
  int input_w_ = 224;

  std::vector<std::vector<Detection>> target_objects_;
  // CudaProprecess cudaProprecess_;

  float output_prob_[12];
  float conf_thresh_;
  float nms_thresh_;
  void* buffers_[2];
};
}  // namespace rm_detector

#endif