// yzy
// yolov5进行tensorrt部署的源文件
#include "rm_detector/inferencer.h"
#include <ros/ros.h>

// 命名空间
using namespace cv;
using namespace nvinfer1;

namespace rm_detector
{
Inferencer::Inferencer()
{
  cudaSetDevice(kGpuId);
}

Inferencer::~Inferencer()
{
  // Release stream and buffers
  //        cudaStreamDestroy(stream_);
  CUDA_CHECK(cudaFree(gpu_buffers_[0]));
  CUDA_CHECK(cudaFree(gpu_buffers_[1]));
  delete[] cpu_output_buffer_;
  cuda_preprocess_destroy();
  // Destroy the engine
  context_->destroy();
  engine_->destroy();
  runtime_->destroy();
}
ArmorInferencer::~ArmorInferencer()
{
  // Release stream and buffers
  //        cudaStreamDestroy(stream_);
  CUDA_CHECK(cudaFree(gpu_buffers_[0]));
  CUDA_CHECK(cudaFree(gpu_buffers_[1]));
  CUDA_CHECK(cudaFree(buffers_[0]));
  CUDA_CHECK(cudaFree(buffers_[1]));
  delete[] cpu_output_buffer_;
  cuda_preprocess_destroy();
  // Destroy the engine
  context_->destroy();
  engine_->destroy();
  runtime_->destroy();
}

ArmorInferencer::ArmorInferencer()
{
  cudaSetDevice(kGpuId);
}

void Inferencer::init(std::string& engine_path, Logger& logger, float conf_thresh, float nms_thresh, int input_h,
                      int input_w)
{
  conf_thresh_ = conf_thresh;
  nms_thresh_ = nms_thresh;
  logger_ = &logger;
  this->input_h_ = input_h;
  this->input_w_ = input_w;

  std::cout << engine_path << std::endl;
  deserialize_engine(engine_path, &runtime_, &engine_, &context_);
  cuda_preprocess_init(kMaxInputImageSize);

  prepare_buffers(engine_, &gpu_buffers_[0], &gpu_buffers_[1], &cpu_output_buffer_);
}

void Inferencer::prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer,
                                 float** cpu_output_buffer)
{
  assert(engine->getNbBindings() == 2);
  // In order to bind the buffers, we need to know the names of the input and output tensors.
  // Note that indices are guaranteed to be less than IEngine::getNbBindings()
  const int inputIndex = engine->getBindingIndex(kInputTensorName);
  const int outputIndex = engine->getBindingIndex(kOutputTensorName);
  assert(inputIndex == 0);
  assert(outputIndex == 1);
  // Create GPU buffers on device
  CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * this->input_h_ * this->input_w_ * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer, kBatchSize * K_OUTPUT_SIZE * sizeof(float)));

  *cpu_output_buffer = new float[kBatchSize * K_OUTPUT_SIZE];
}

void Inferencer::detect(std::vector<Mat>& frame)
{
  target_objects_.clear();
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cuda_batch_preprocess(frame, gpu_buffers_[0], this->input_w_, this->input_h_, stream);

  infer(*context_, stream, (void**)gpu_buffers_, cpu_output_buffer_, kBatchSize);

  batch_nms(target_objects_, cpu_output_buffer_, frame.size(), 1000 * sizeof(Detection) / sizeof(float) + 1,
            conf_thresh_, nms_thresh_);
  cudaStreamDestroy(stream);
}

void Inferencer::infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output,
                       int batchsize)
{
  context.enqueue(batchsize, gpu_buffers, stream, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(output, gpu_buffers[1], batchsize * K_OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost,
                             stream));
  cudaStreamSynchronize(stream);
}

void ArmorInferencer::infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output,
                            int batchsize)
{
  context.enqueue(batchsize, buffers_, stream, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(output, buffers_[1], batchsize * CLASS_OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost,
                             stream));
  cudaStreamSynchronize(stream);
}

void Inferencer::deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine,
                                    IExecutionContext** context)
{
  std::ifstream file(engine_name, std::ios::binary);
  if (!file.good())
  {
    std::cerr << "read " << engine_name << " error!" << std::endl;
    assert(false);
  }
  size_t size = 0;
  file.seekg(0, file.end);
  size = file.tellg();
  file.seekg(0, file.beg);
  char* serialized_engine = new char[size];
  assert(serialized_engine);
  file.read(serialized_engine, size);
  file.close();

  *runtime = createInferRuntime(*logger_);
  assert(*runtime);
  *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
  assert(*engine);
  *context = (*engine)->createExecutionContext();
  assert(*context);
  delete[] serialized_engine;
}

void ArmorInferencer::deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine,
                                         IExecutionContext** context)
{
  std::ifstream file(engine_name, std::ios::binary);
  if (!file.good())
  {
    std::cerr << "read " << engine_name << " error!" << std::endl;
    assert(false);
  }
  size_t size = 0;
  file.seekg(0, file.end);
  size = file.tellg();
  file.seekg(0, file.beg);
  char* serialized_engine = new char[size];
  assert(serialized_engine);
  file.read(serialized_engine, size);
  file.close();

  *runtime = createInferRuntime(*logger_);
  assert(*runtime);
  *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
  assert(*engine);
  *context = (*engine)->createExecutionContext();
  assert(*context);
  delete[] serialized_engine;
}

void ArmorInferencer::init(std::string& engine_path, Logger& logger, float conf_thresh, float nms_thresh)
{
  conf_thresh_ = conf_thresh;
  nms_thresh_ = nms_thresh;
  logger_ = &logger;

  std::cout << engine_path << std::endl;
  deserialize_engine(engine_path, &runtime_, &engine_, &context_);
  cuda_preprocess_init(kMaxInputImageSize);

  prepare_buffers(engine_, &gpu_buffers_[0], &gpu_buffers_[1], &cpu_output_buffer_);
}

std::pair<int, float> ArmorInferencer::detect(std::vector<cv::Mat>& frame)
{
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  float mean[3] = { 0.485, 0.456, 0.406 };
  float std[3] = { 0.229, 0.224, 0.225 };
  cuda_batch_preprocess(frame, (float*)buffers_[0], 224, 224, mean, std, stream);
  // std::cout << frame[0].cols << std::endl;

  infer(*context_, stream, (void**)gpu_buffers_, cpu_output_buffer_, kBatchSize);
  // std::vector<float> softmax_output = softmax(cpu_output_buffer_);
  std::vector<float> softmax_output;
  std::cout << "----------------------------" << std::endl;
  for (int i = 0; i < 12; ++i)
  {
    std::cout << "value = " << cpu_output_buffer_[i] << std::endl;
    softmax_output.push_back(cpu_output_buffer_[i]);
  }
  auto max_prob_iter = std::max_element(softmax_output.begin(), softmax_output.end());
  std::cout << "max_prob_iter = " << *max_prob_iter << std::endl;
  int index = std::distance(softmax_output.begin(), max_prob_iter);
  std::cout << "index = " << index << std::endl;
  std::cout << "----------------------------" << std::endl;
  // get_output(output_prob_, cpu_output_buffer_, frame.size(), 12);
  cudaStreamDestroy(stream);
  return std::make_pair(index, *max_prob_iter);
}

void ArmorInferencer::prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer,
                                      float** cpu_output_buffer)
{
  assert(engine->getNbBindings() == 2);
  // In order to bind the buffers, we need to know the names of the input and output tensors.
  // Note that indices are guaranteed to be less than IEngine::getNbBindings()
  const int inputIndex = engine->getBindingIndex(kInputTensorName);
  const int outputIndex = engine->getBindingIndex(kOutputTensorName);
  assert(inputIndex == 0);
  assert(outputIndex == 1);
  CUDA_CHECK(cudaMalloc(&buffers_[inputIndex], kBatchSize * 3 * 224 * 224 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&buffers_[outputIndex], kBatchSize * CLASS_OUTPUT_SIZE * sizeof(float)));

  // Create GPU buffers on device
  CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * 224 * 224 * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer, kBatchSize * CLASS_OUTPUT_SIZE * sizeof(float)));

  *cpu_output_buffer = new float[kBatchSize * CLASS_OUTPUT_SIZE];
}
}  // namespace rm_detector