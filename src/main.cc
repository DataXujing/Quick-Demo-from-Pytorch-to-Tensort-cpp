#include <iostream>
#include "cuda_runtime_api.h"
#include "NvOnnxParser.h"
#include "NvInfer.h"

#define BATCH_SIZE 1
#define DATA_SHAPE 28

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
// #define 

class Logger : public nvinfer1::ILogger
{
public:
  Logger(Severity severity = Severity::kWARNING) : reportableSeverity(severity)
  {
  }

  void log(Severity severity, const char* msg) override
  {
    // suppress messages with severity enum value greater than the reportable
    if (severity > reportableSeverity)
      return;

    switch (severity)
    {
      case Severity::kINTERNAL_ERROR:
        std::cerr << "INTERNAL_ERROR: ";
        break;
      case Severity::kERROR:
        std::cerr << "ERROR: ";
        break;
      case Severity::kWARNING:
        std::cerr << "WARNING: ";
        break;
      case Severity::kINFO:
        std::cerr << "INFO: ";
        break;
      default:
        std::cerr << "UNKNOWN: ";
        break;
    }
    std::cerr << msg << std::endl;
  }

  Severity reportableSeverity;
};


void 
onnxTotrt(const std::string& model_file, // name of the onnx model
          nvinfer1::IHostMemory** trt_model_stream, // output buffer for the TensorRT model
          Logger g_logger_)
{

    int verbosity = static_cast<int>(nvinfer1::ILogger::Severity::kWARNING);
    
    // -- create the builder ------------------/
    const auto explicit_batch = static_cast<uint32_t>(BATCH_SIZE) 
        << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(g_logger_);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicit_batch);

    // --create the parser to load onnx file---/
    auto parser = nvonnxparser::createParser(*network,g_logger_);
    if (!parser->parseFromFile(model_file.c_str(),verbosity)){
        std::string msg("failed to parse onnx file");
        g_logger_.log(nvinfer1::ILogger::Severity::kERROR,msg.c_str());
        exit(EXIT_FAILURE);
    }
    
    // -- build the config for pass in specific parameters ---/
    builder->setMaxBatchSize(BATCH_SIZE);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize( 1 << 20);
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    // std::cout <<"engine bindings dimension" << engine->getNbBindings() << std::endl;
 
    

    // -- serialize the engine,then close everything down --/
    *trt_model_stream = engine->serialize();
    
    parser->destroy();
    engine->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
};


int main()
{
    
    // --initial a logger
    Logger g_logger_ ;
    nvinfer1::IHostMemory* trt_model_stream{ nullptr };
    std::string onnx_file = "../test.onnx";
    
    // --Pass the params recorded in ONNX_file to trt_model_stream --/
    onnxTotrt(onnx_file, &trt_model_stream,g_logger_);
    if (trt_model_stream == nullptr)
    {
        std::cerr << "Failed to load ONNX file " << std::endl;
    }

    // --deserialize the engine from the stream --- /
    nvinfer1::IRuntime* engine_runtime = nvinfer1::createInferRuntime(g_logger_);
    if (engine_runtime == nullptr)
    {
        std::cerr << "Failed to create TensorRT Runtime object." << std::endl;
    }

    // --load the infer engine -----/
    nvinfer1::ICudaEngine* engine_infer = engine_runtime->deserializeCudaEngine(trt_model_stream->data(),trt_model_stream->size(),nullptr);
    if (engine_infer == nullptr)
    {
        std::cerr << "Failed to create TensorRT Engine." << std::endl;
    }

    nvinfer1::IExecutionContext* engine_context = engine_infer->createExecutionContext();
    if (engine_context == nullptr)
    {
        std::cerr << "Failed to create TensorRT Execution Context." << std::endl;
    }

    // --destroy stream ---/.
    trt_model_stream->destroy();
    std::cout << "loaded trt model , do inference" << std::endl;

    
    ///////////////////////////////////////////////////////////////////
    // enqueue them up
    //////////////////////////////////////////////////////////////////

    // -- allocate host memory ------------/ 
    float h_input[DATA_SHAPE * DATA_SHAPE] = {0.f};
    float h_output[10];
    
    void* buffers[2];
    cudaMalloc(&buffers[0], DATA_SHAPE * DATA_SHAPE * sizeof(float));  //<- input
    cudaMalloc(&buffers[1],10 * sizeof(float)); //<- output

    

    cudaMemcpy(h_input, buffers[0],  DATA_SHAPE * DATA_SHAPE * sizeof(float), cudaMemcpyHostToDevice);

    // -- do execute --------///
    int32_t BATCH_SIZE_ = 1;
    engine_context->execute(BATCH_SIZE_, buffers);
    
    cudaMemcpy(&h_output, buffers[1],
                10 * sizeof(float),
               cudaMemcpyDeviceToHost);
    
    
    for (int i=0;i<10;i++)
    {
        std::cout << h_output[i] << " ";
    }
    std::cout << "\n";
    

    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    engine_runtime->destroy();
    engine_infer->destroy();
    // cudaStreamDestroy(stream);
    return 0;
}
