#include "pixel_shuffle/pixelShufflePlugin.hpp"
#include <cassert>
#include <cstring>
#include <iostream>

using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace
{
const char* PIXEL_SHUFFLE_PLUGIN_VERSION{"1"};
const char* PIXEL_SHUFFLE_PLUGIN_NAME{"PixelShufflePlugin"};
} // namespace

// 静态类字段初始化
PluginFieldCollection PixelShufflePluginCreator::mFC{};
std::vector<PluginField> PixelShufflePluginCreator::mPluginAttributes;

template <typename scalar_t>
scalar_t readFromBuffer(const char*& buffer)
{
    scalar_t val = *reinterpret_cast<const scalar_t*>(buffer);
    buffer += sizeof(scalar_t);
    return val;
}

template <typename scalar_t>
void writeToBuffer(char*& buffer, const scalar_t& val)
{
    *reinterpret_cast<scalar_t*>(buffer) = val;
    buffer += sizeof(scalar_t);
}

PixelShufflePlugin::PixelShufflePlugin(int upscale_factor)
    : mUpscaleFactor(upscale_factor)
{
    assert(upscale_factor > 0);
}

PixelShufflePlugin::PixelShufflePlugin(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    const char* a = d;

    mUpscaleFactor = readFromBuffer<int>(d);

    assert(d == a + sizeof(int));
}

IPluginV2DynamicExt* PixelShufflePlugin::clone() const noexcept
{
    auto* plugin = new PixelShufflePlugin(mUpscaleFactor);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

DimsExprs PixelShufflePlugin::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{
    assert(nbInputs == 1);
    assert(outputIndex == 0);
    DimsExprs output;
    output.nbDims = 4;
    output.d[0] = inputs[0].d[0];
    output.d[1] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[0].d[1], *exprBuilder.constant(mUpscaleFactor * mUpscaleFactor));
    output.d[2] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[2], *exprBuilder.constant(mUpscaleFactor));
    output.d[3] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[3], *exprBuilder.constant(mUpscaleFactor));
    return output;
}

bool PixelShufflePlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    assert(nbInputs == 1 && nbOutputs == 1);
    const PluginTensorDesc& in = inOut[pos];
    if (pos == 0)
    {
        return (in.type == DataType::kFLOAT || in.type == DataType::kHALF) && (in.format == TensorFormat::kLINEAR);
    }
    const PluginTensorDesc& prev = inOut[pos - 1];
    return in.type == prev.type && in.format == prev.format;
}

void PixelShufflePlugin::configurePlugin(
    const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    // 配置插件，在这里可以进行一些必要的检查
}

size_t PixelShufflePlugin::getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int PixelShufflePlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // 获取输入张量的维度
    int batchSize = inputDesc[0].dims.d[0];
    int channels = inputDesc[0].dims.d[1];
    int height = inputDesc[0].dims.d[2];
    int width = inputDesc[0].dims.d[3];

    // 根据输入数据类型调用相应的核函数
    if (inputDesc[0].type == DataType::kFLOAT)
    {

        PixelShuffleForwardGpu(
            reinterpret_cast<const float*>(inputs[0]),
            reinterpret_cast<float*>(outputs[0]),
            batchSize,
            channels,
            height,
            width,
            mUpscaleFactor,
            stream
        );
    }
    else if (inputDesc[0].type == DataType::kHALF)
    {
        PixelShuffleForwardGpu(
                reinterpret_cast<const __half*>(inputs[0]),
                reinterpret_cast<__half*>(outputs[0]),
                batchSize,
                channels,
                height,
                width,
                mUpscaleFactor,
                stream
        );
    }
    else
    {
        // 不支持的数据类型
        return -1;
    }

    return 0;
}

DataType PixelShufflePlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    assert(index == 0);
    return inputTypes[0];
}

const char* PixelShufflePlugin::getPluginType() const noexcept
{
    return PIXEL_SHUFFLE_PLUGIN_NAME;
}

const char* PixelShufflePlugin::getPluginVersion() const noexcept
{
    return PIXEL_SHUFFLE_PLUGIN_VERSION;
}

int PixelShufflePlugin::getNbOutputs() const noexcept
{
    return 1;
}

int PixelShufflePlugin::initialize() noexcept
{
    return 0;
}

void PixelShufflePlugin::terminate() noexcept {}

size_t PixelShufflePlugin::getSerializationSize() const noexcept
{
    return sizeof(mUpscaleFactor);
}

void PixelShufflePlugin::serialize(void* buffer) const noexcept
{
    char* d = static_cast<char*>(buffer);
    char* a = d;

    writeToBuffer<int>(d, mUpscaleFactor);

    assert(d == a + getSerializationSize());
}

void PixelShufflePlugin::destroy() noexcept
{
    delete this;
}

void PixelShufflePlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

const char* PixelShufflePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

PixelShufflePlugin::~PixelShufflePlugin() {

}

PixelShufflePluginCreator::PixelShufflePluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("upscale_factor", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* PixelShufflePluginCreator::getPluginName() const noexcept
{
    return PIXEL_SHUFFLE_PLUGIN_NAME;
}

const char* PixelShufflePluginCreator::getPluginVersion() const noexcept
{
    return PIXEL_SHUFFLE_PLUGIN_VERSION;
}

const PluginFieldCollection* PixelShufflePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2DynamicExt* PixelShufflePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    int upscale_factor = 0;
    for (int i = 0; i < fc->nbFields; i++)
    {
        if (strcmp(fc->fields[i].name, "upscaleFactor") == 0)
        {
            upscale_factor = *reinterpret_cast<const int*>(fc->fields[i].data);
        }
    }

    std::cout << "upscale_factor: " << upscale_factor << std::endl;

    assert(upscale_factor > 0);
    auto* obj = new PixelShufflePlugin(upscale_factor);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2DynamicExt* PixelShufflePluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    auto* obj = new PixelShufflePlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

void PixelShufflePluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

const char* PixelShufflePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

REGISTER_TENSORRT_PLUGIN(PixelShufflePluginCreator);
