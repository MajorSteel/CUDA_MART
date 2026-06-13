#include "mart.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>

void print_help() {
    std::cout << "CUDA MART Reconstruction Application\n"
              << "Usage: mart_app [options]\n"
              << "Options:\n"
              << "  --mode <1|2>       1 for 2D, 2 for 3D (default: 1)\n"
              << "  --size <N>         Size of the grid (default: 8)\n"
              << "  --iters <N>        Number of iterations (default: 10)\n"
              << "  --out <file>       Output file for the result (optional)\n"
              << "  --help             Show this help message\n";
}

int main(int argc, char** argv) {
    mart::Config config;
    std::string out_file = "";

    // Basic arg parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            print_help();
            return 0;
        } else if (arg == "--mode" && i + 1 < argc) {
            config.mode = std::stoi(argv[++i]);
        } else if (arg == "--size" && i + 1 < argc) {
            config.size = std::stoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            config.iterations = std::stoi(argv[++i]);
        } else if (arg == "--out" && i + 1 < argc) {
            out_file = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_help();
            return 1;
        }
    }

    std::cout << "Starting CUDA MART Reconstruction...\n";
    std::cout << "Mode: " << (config.mode == 1 ? "2D" : "3D") << "\n";
    std::cout << "Size: " << config.size << "\n";
    std::cout << "Iterations: " << config.iterations << "\n\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    if (config.mode == 1) {
        size_t total_elements = config.size * config.size;
        std::vector<float> image(total_elements, 0.0f);
        std::vector<float> projections(config.size, 1.0f); // Dummy projections
        std::vector<float> output(total_elements);

        mart::Status status = mart::run_reconstruction_2d(image, projections, output, config);
        
        if (status != mart::Status::SUCCESS) {
            std::cerr << "Reconstruction failed.\n";
            return 1;
        }

        if (!out_file.empty()) {
            std::ofstream ofs(out_file, std::ios::binary);
            ofs.write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(float));
            std::cout << "Saved 2D result to " << out_file << "\n";
        }
        
    } else if (config.mode == 2) {
        size_t total_elements = config.size * config.size * config.size;
        std::vector<float> volume(total_elements, 1.0f);
        std::vector<float> projections(config.size, 10.0f); // Dummy projections
        std::vector<float> output(total_elements);

        mart::Status status = mart::run_reconstruction_3d(volume, projections, output, config);
        
        if (status != mart::Status::SUCCESS) {
            std::cerr << "Reconstruction failed.\n";
            return 1;
        }

        if (!out_file.empty()) {
            std::ofstream ofs(out_file, std::ios::binary);
            ofs.write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(float));
            std::cout << "Saved 3D result to " << out_file << "\n";
        }
    } else {
        std::cerr << "Invalid mode selected.\n";
        return 1;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Reconstruction completed in " << elapsed.count() << " seconds.\n";
    return 0;
}
