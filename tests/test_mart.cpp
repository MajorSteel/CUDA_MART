#include "mart.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

bool almost_equal(float a, float b, float epsilon = 1e-4) {
    return std::abs(a - b) < epsilon;
}

void test_2d_reconstruction() {
    std::cout << "Running test_2d_reconstruction...\n";
    mart::Config config;
    config.mode = 1;
    config.size = 2;
    config.iterations = 5;

    std::vector<float> image = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> projections = {2.0f, 2.0f};
    std::vector<float> output(4);

    mart::Status status = mart::run_reconstruction_2d(image, projections, output, config);
    assert(status == mart::Status::SUCCESS);

    // The output values should be non-zero after iterations
    for (float val : output) {
        assert(val > 0.0f);
    }
    std::cout << "test_2d_reconstruction passed!\n";
}

int main() {
    test_2d_reconstruction();
    std::cout << "All tests passed successfully.\n";
    return 0;
}
