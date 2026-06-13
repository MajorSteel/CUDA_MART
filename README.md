# CUDA-MART ⚡

[![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C++-17-00599C?style=for-the-badge&logo=c%2B%2B)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-Build-064F8C?style=for-the-badge&logo=cmake)](https://cmake.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

A high-performance C++ and NVIDIA CUDA implementation of the **Multiplicative Algebraic Reconstruction Technique (MART)**.

MART is an iterative algorithm primarily used in computed tomography (CT) and other imaging systems to reconstruct 2D images or 3D volumes from projection data. By leveraging NVIDIA GPUs, this implementation significantly accelerates the massive matrix multiplications and iterative updates required for high-resolution reconstructions.

## 🚀 Features

* **GPU Accelerated:** Implements highly parallelized `__global__` CUDA kernels.
* **2D & 3D Support:** Capable of reconstructing both flat 2D image grids and complex 3D volumes.
* **Modern C++ Architecture:** Uses `CMake` and C++17 for a robust, maintainable, and separable compilation pipeline.
* **Dynamic Configuration:** Adjust grid sizes, iterations, and modes dynamically via the Command Line Interface (CLI).
* **Unit Tested:** Includes foundational unit tests to ensure mathematical correctness of the iterative updates.

## ⚙️ Architecture

The algorithm operates iteratively to correct voxel values based on the error between real captured projections and computed estimates.

1. **Initialization**: The 3D Volume or 2D Image is initialized on the device.
2. **Estimation**: For each iteration, rays are simulated passing through the current volume state to compute an *estimated* projection.
3. **Correction**: The ratio (or difference) between the *actual* projection and the *estimated* projection forms a correction factor.
4. **Update**: A relaxation/damping parameter is applied, and the GPU parallelizes the multiplicative update across all voxels simultaneously.

## 🛠️ Prerequisites

To compile and run this project, you will need:

* **NVIDIA GPU** with CUDA support.
* [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (`nvcc`).
* [CMake](https://cmake.org/download/) (v3.18 or higher).
* C++17 compatible compiler (e.g., `MSVC` for Windows, `GCC/Clang` for Linux).

## 🏗️ Building from Source

Clone the repository and build using CMake:

```bash
git clone https://github.com/MajorSteel/CUDA_MART.git
cd CUDA_MART
mkdir build
cd build

# Configure the project
cmake ..

# Build the executable
cmake --build . --config Release
```

## 💻 Usage

Run the compiled executable with standard command-line flags.

**For a 3D Reconstruction:**
```bash
./bin/Release/mart_app --mode 2 --size 128 --iters 50 --out result_3d.bin
```

**For a 2D Reconstruction:**
```bash
./bin/Release/mart_app --mode 1 --size 256 --iters 20 --out result_2d.bin
```

### Arguments

| Flag | Description | Default |
| --- | --- | --- |
| `--mode <1\|2>` | `1` for 2D, `2` for 3D | `1` |
| `--size <N>` | The grid size (N x N or N x N x N) | `8` |
| `--iters <N>` | Number of correction iterations | `10` |
| `--out <file>` | Path to save the resulting binary output | *(none)* |

## 🧪 Running Tests

The project includes mathematical unit tests configured via CTest.

```bash
cd build
ctest -C Release -V
```

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
