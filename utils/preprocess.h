#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <opencv2/opencv.hpp>

namespace rm_detector
{
class CudaProprecess
{
public:
  void cuda_preprocess_init(int max_image_size);

  void cuda_preprocess_destroy();

  void cuda_preprocess(uint8_t* src, int src_width, int src_height, float* dst, int dst_width, int dst_height,
                       cudaStream_t stream);

  void cuda_batch_preprocess(cv::Mat& img, float* dst, int dst_width, int dst_height, cudaStream_t stream);

  uint8_t* img_buffer_host_ = nullptr;
  uint8_t* img_buffer_device_ = nullptr;
};
}  // namespace rm_detector
