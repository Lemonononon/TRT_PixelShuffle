#ifndef TRT_PIXEL_SHUFFLE_PLUGIN_H
#define TRT_PIXEL_SHUFFLE_PLUGIN_H

#include "NvInferPlugin.h"
#include <string>
#include <vector>
#include <cuda_fp16.h>

namespace nvinfer1
{
namespace plugin
{

template<typename T>
void PixelShuffleForwardGpu(
        const T* input,
        T* output,
        int batch_size,
        int channels,
        int height,
        int width,
        int upscale_factor,
        cudaStream_t stream
);


class PixelShufflePlugin : public IPluginV2DynamicExt
{
public:
    PixelShufflePlugin() = delete;
    explicit PixelShufflePlugin(int upscale_factor);
    PixelShufflePlugin(const void* data, size_t length);
    ~PixelShufflePlugin() override;

    // IPluginV2DynamicExt 方法
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2Ext 方法
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    // IPluginV2 方法
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    int mUpscaleFactor;
    std::string mNamespace;
};

class PixelShufflePluginCreator : public IPluginCreator
{
public:
    PixelShufflePluginCreator();
    ~PixelShufflePluginCreator() override = default;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const PluginFieldCollection* getFieldNames() noexcept override;
    IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};


} // namespace plugin
} // namespace nvinfer1

#endif // TRT_PIXEL_SHUFFLE_PLUGIN_H
