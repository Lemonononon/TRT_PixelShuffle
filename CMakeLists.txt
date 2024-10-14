cmake_minimum_required(VERSION 3.16)
project(Real_ESRGAN_TRT)

set(CMAKE_CXX_STANDARD 17)


add_definitions(-std=c++17)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_BUILD_TYPE Debug)

SET(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")


INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/include)

# cuda
FIND_PACKAGE(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# <------------------------TensorRT Related------------------------->
#find_package(TensorRT REQUIRED)
#include_directories(${TensorRT_INCLUDE_DIRS})
include_directories(/home/xiaoying/code/dep/TensorRT-8.6.1.6/include)
link_directories(/home/xiaoying/code/dep/TensorRT-8.6.1.6/lib)


cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/src/pixel_shuffle/pixel_shuffle.cu
        ${PROJECT_SOURCE_DIR}/src/pixel_shuffle/pixelShufflePlugin.cpp
)
target_link_libraries(myplugins nvinfer cudart nvinfer_plugin)


# test
add_executable(test main.cpp src/pixel_shuffle/pixelShufflePlugin.cpp)
set_target_properties(test PROPERTIES CUDA_ARCHITECTURES "86")
TARGET_LINK_LIBRARIES(test nvinfer)
TARGET_LINK_LIBRARIES(test cudart)
TARGET_LINK_LIBRARIES(test myplugins)
TARGET_LINK_LIBRARIES(test nvinfer_plugin)
TARGET_LINK_LIBRARIES(test nvonnxparser)
