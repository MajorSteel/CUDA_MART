#include "mart.h"
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            return mart::Status::CUDA_ERROR; \
        } \
    } while (0)

namespace mart {

// 2D Kernel
__global__ void update2DKernel(float* image, const float* projections, int image_size, float update_factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < image_size) {
        float computedProjection = 0.0f;
        for (int j = 0; j < image_size; ++j) {
            computedProjection += image[i * image_size + j];
        }
        
        float error = projections[i] - computedProjection;

        for (int j = 0; j < image_size; ++j) {
            image[i * image_size + j] += update_factor * error / image_size;
        }
    }
}

// 3D Kernel
__global__ void update3DKernel(float* volume, const float* projections, int volume_size, float update_factor, float damping_factor, float max_correction) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < volume_size && j < volume_size && k < volume_size) {
        float estimatedProjection = 0.0f;
        for (int x = 0; x < volume_size; ++x) {
            for (int y = 0; y < volume_size; ++y) {
                estimatedProjection += volume[(x * volume_size + y) * volume_size + k];
            }
        }
        
        float correction = projections[k] / (estimatedProjection + 1e-6f);
        int idx = (i * volume_size + j) * volume_size + k;
        
        volume[idx] *= (1.0f + update_factor * correction);
        volume[idx] -= damping_factor * volume[idx];
        volume[idx] = fminf(fmaxf(volume[idx], 0.0f), max_correction);
    }
}

Status run_reconstruction_2d(
    const std::vector<float>& image_in,
    const std::vector<float>& projections_in,
    std::vector<float>& image_out,
    const Config& config
) {
    if (config.size <= 0 || config.iterations <= 0) return Status::INVALID_ARGUMENT;

    size_t image_bytes = image_in.size() * sizeof(float);
    size_t proj_bytes = projections_in.size() * sizeof(float);

    float* d_image = nullptr;
    float* d_projections = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_image, image_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_projections, proj_bytes));

    CUDA_CHECK(cudaMemcpy(d_image, image_in.data(), image_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_projections, projections_in.data(), proj_bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (config.size + blockSize - 1) / blockSize;

    for (int iter = 0; iter < config.iterations; ++iter) {
        update2DKernel<<<gridSize, blockSize>>>(d_image, d_projections, config.size, config.update_factor);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(image_out.data(), d_image, image_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_image);
    cudaFree(d_projections);

    return Status::SUCCESS;
}

Status run_reconstruction_3d(
    const std::vector<float>& volume_in,
    const std::vector<float>& projections_in,
    std::vector<float>& volume_out,
    const Config& config
) {
    if (config.size <= 0 || config.iterations <= 0) return Status::INVALID_ARGUMENT;

    size_t volume_bytes = volume_in.size() * sizeof(float);
    size_t proj_bytes = projections_in.size() * sizeof(float);

    float* d_volume = nullptr;
    float* d_projections = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_volume, volume_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_projections, proj_bytes));

    CUDA_CHECK(cudaMemcpy(d_volume, volume_in.data(), volume_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_projections, projections_in.data(), proj_bytes, cudaMemcpyHostToDevice));

    dim3 blockSize(8, 8, 8);
    dim3 gridSize(
        (config.size + blockSize.x - 1) / blockSize.x,
        (config.size + blockSize.y - 1) / blockSize.y,
        (config.size + blockSize.z - 1) / blockSize.z
    );

    for (int iter = 0; iter < config.iterations; ++iter) {
        update3DKernel<<<gridSize, blockSize>>>(d_volume, d_projections, config.size, config.update_factor, config.damping_factor, config.max_correction);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(volume_out.data(), d_volume, volume_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_volume);
    cudaFree(d_projections);

    return Status::SUCCESS;
}

} // namespace mart
