//
// Created by robotzero on 12/17/23.
//

#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "model.h"

#include <iostream>
#include <chrono>
#include <cmath>

using namespace nvinfer1;

static Logger gLogger;
const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

void serialize_engine(unsigned int max_batchsize, bool& is_p6, float& gd, float& gw, std::string& wts_name, std::string& engine_name) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine *engine = nullptr;
    if (is_p6) {
        engine = build_det_p6_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    } else {
        engine = build_det_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    }
    assert(engine != nullptr);

    // Serialize the engine
    IHostMemory* serialized_engine = engine->serialize();
    assert(serialized_engine != nullptr);

    // Save engine to file
    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
        std::cerr << "Could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

    // Close everything down
    engine->destroy();
    config->destroy();
    serialized_engine->destroy();
    builder->destroy();
}

int main(int argc, char** argv) {
    cudaSetDevice(kGpuId);

  std::string wts_name = "/media/robotzero/Steins_Gate/rm/tensorrtx/yolov5/docs/armor_det.wts";
    std::string engine_name = "/media/robotzero/Steins_Gate/rm/tensorrtx/yolov5/docs/armor_det.engine";
    bool is_p6 = false;
    float gd = 0.33f, gw = 0.5f;
    std::string img_dir = "/media/robotzero/Steins_Gate/rm/tensorrtx/yolov5/image_armor";

//  if (!parse_args(argc, argv, wts_name, engine_name, is_p6, gd, gw, img_dir)) {
//    std::cerr << "arguments not right!" << std::endl;
//    std::cerr << "./yolov5_det -s [.wts] [.engine] [n/s/m/l/x/n6/s6/m6/l6/x6 or c/c6 gd gw]  // serialize model to plan file" << std::endl;
//    std::cerr << "./yolov5_det -d [.engine] ../images  // deserialize plan file and run inference" << std::endl;
//    return -1;
//  }

    // Create a model using the API directly and serialize it to a file
  if (!wts_name.empty()) {
    serialize_engine(kBatchSize, is_p6, gd, gw, wts_name, engine_name);
    return 0;
  }
}