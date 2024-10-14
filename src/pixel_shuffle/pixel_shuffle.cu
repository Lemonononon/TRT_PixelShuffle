#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "pixel_shuffle/pixelShufflePlugin.hpp"



namespace nvinfer1
{
namespace plugin
{


template <typename T>
__global__ void pixelShuffleKernel(
    const T* input,
    T* output,
    int batch_size,
    int channels,
    int height,
    int width,
    int upscale_factor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;

    if (idx < total_elements) {
        int pw = idx % width;
        int ph = (idx / width) % height;
        int c = (idx / (width * height)) % channels;
        int n = idx / (width * height * channels);

        int oc = c / (upscale_factor * upscale_factor);
        int oh = ph * upscale_factor + (c / upscale_factor) % upscale_factor;
        int ow = pw * upscale_factor + c % upscale_factor;

        int output_idx = ((n * channels / (upscale_factor * upscale_factor) + oc) * height * upscale_factor + oh) * width * upscale_factor + ow;
        output[output_idx] = input[idx];
    }
}

template <typename T>
void PixelShuffleForwardGpu(
        const T* input,
        T* output,
        int batch_size,
        int channels,
        int height,
        int width,
        int upscale_factor,
        cudaStream_t stream
){
    int total_elements = batch_size * channels * height * width;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    pixelShuffleKernel<T><<<grid_size, block_size, 0, stream>>>(
            input, output, batch_size, channels, height, width, upscale_factor
    );
}

// 显式实例化模板函数
template void PixelShuffleForwardGpu<float>(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    int upscale_factor,
    cudaStream_t stream
);

template void PixelShuffleForwardGpu<__half>(
    const __half* input,
    __half* output,
    int batch_size,
    int channels,
    int height,
    int width,
    int upscale_factor,
    cudaStream_t stream
);
}
}