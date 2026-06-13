#pragma once

#include <vector>

namespace mart {

    // Status codes
    enum class Status {
        SUCCESS,
        CUDA_ERROR,
        INVALID_ARGUMENT
    };

    // Configuration parameters for the MART algorithm
    struct Config {
        int mode = 1; // 1 = 2D, 2 = 3D
        int size = 8;
        int iterations = 10;
        float update_factor = 0.1f;
        float damping_factor = 0.02f;
        float max_correction = 1000.0f;
    };

    // Main API functions
    Status run_reconstruction_2d(
        const std::vector<float>& image_in,
        const std::vector<float>& projections_in,
        std::vector<float>& image_out,
        const Config& config
    );

    Status run_reconstruction_3d(
        const std::vector<float>& volume_in,
        const std::vector<float>& projections_in,
        std::vector<float>& volume_out,
        const Config& config
    );

} // namespace mart
