#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "pixel_shuffle/pixelShufflePlugin.hpp"
#include <iostream>
#include <memory>
#include <vector>
#include <assert.h>

// 错误检查宏
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

// 记录器类
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} gLogger;

using namespace nvinfer1;

// 构建引擎
nvinfer1::ICudaEngine* buildEngine(int batchSize, int channels, int height, int width, int upscaleFactor)
{
    auto builder = nvinfer1::createInferBuilder(gLogger);
    auto config = builder->createBuilderConfig();
    auto network = builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    auto input = network->addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{batchSize, channels, height, width});

    // 创建插件
    IPluginCreator* creator = getPluginRegistry()->getPluginCreator("PixelShufflePlugin", "1");

    PluginField pluginField[1] = { {"upscaleFactor", &upscaleFactor, PluginFieldType::kINT32, 1} };
    PluginFieldCollection pluginData{};
    pluginData.nbFields = 1;
    pluginData.fields = pluginField;

    auto* plugin = creator->createPlugin("pixel_shuffle", &pluginData);
    assert(plugin != nullptr);

    auto pixelShuffleLayer = network->addPluginV2(&input, 1, *plugin);
    pixelShuffleLayer->getOutput(0)->setName("output");
    network->markOutput(*pixelShuffleLayer->getOutput(0));

    config->setMaxWorkspaceSize(1 << 20);
    auto engine = builder->buildEngineWithConfig(*network, *config);

    plugin->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();

    return engine;
}

int main()
{
    // 设置输入参数
    int batchSize = 1;
    int channels = 4;
    int height = 2;
    int width = 2;
    int upscaleFactor = 2;

    initLibNvInferPlugins(&gLogger, "");

    auto engine = buildEngine(batchSize, channels, height, width, upscaleFactor);
    if (!engine) {
        std::cerr << "无法创建engine" << std::endl;
        return 1;
    }

    // 创建context
    auto context = engine->createExecutionContext();

    // 分配输入和输出缓冲区
    void* inputBuffer;
    void* outputBuffer;
    CHECK(cudaMalloc(&inputBuffer, batchSize * channels * height * width * sizeof(float)));
    CHECK(cudaMalloc(&outputBuffer, batchSize * channels / (upscaleFactor * upscaleFactor) * height * upscaleFactor * width * upscaleFactor * sizeof(float)));

    // 设置输入数据（示例数据）
    std::vector<float> inputData(batchSize * channels * height * width, 1.0f);
    CHECK(cudaMemcpy(inputBuffer, inputData.data(), inputData.size() * sizeof(float), cudaMemcpyHostToDevice));

    // 执行推理
    void* bindings[] = {inputBuffer, outputBuffer};
    context->executeV2(bindings);

    // 获取输出数据
    std::vector<float> outputData(batchSize * channels / (upscaleFactor * upscaleFactor) * height * upscaleFactor * width * upscaleFactor);
    CHECK(cudaMemcpy(outputData.data(), outputBuffer, outputData.size() * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "输出数据：" << std::endl;
    for (const auto& val : outputData) {
        std::cout << val << " ";
    }
    std::cout << std::endl;


    CHECK(cudaFree(inputBuffer));
    CHECK(cudaFree(outputBuffer));
    context->destroy();
    engine->destroy();

    return 0;
}
