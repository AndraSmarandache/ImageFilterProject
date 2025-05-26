# Gaussian Blur Image Processor

A comprehensive implementation of Gaussian filtering for image processing, featuring sequential, parallel MPI, and GPU-accelerated OpenCL versions with detailed performance analysis.

## 🎯 Project Overview

This project implements and analyzes the performance of Gaussian blur algorithms across multiple computing paradigms:
- **Sequential Implementation**: C# with OpenCvSharp
- **Parallel CPU Implementation**: Python with MPI4Py 
- **GPU Implementation**: C++ with OpenCL

The project processes images ranging from 128×128 to 32768×32768 pixels using a Gaussian filter with radius=30 and σ=10, providing comprehensive performance metrics and analysis.

## 🏗️ Project Structure

```
ImageFilteringProject/
├── Sequential/           # C# sequential implementation
├── gaussian_blur_mpi/    # Python MPI parallel implementation  
├── OPENCL/              # C++ OpenCL GPU implementation
└── README.md
```

## 📊 Performance Results

### Execution Times (milliseconds)

| Image Size | Sequential | MPI 4 Processes | MPI 8 Processes | OpenCL (GPU) |
|------------|------------|-----------------|-----------------|--------------|
| 128×128    | 4.71       | 1.45           | 1.47           | 0.12         |
| 256×256    | 6.45       | 2.81           | 2.32           | 0.26         |
| 512×512    | 15.79      | 6.37           | 5.65           | 0.83         |
| 1024×1024  | 30.18      | 17.63          | 11.95          | 3.16         |
| 2048×2048  | 97.91      | 43.74          | 50.61          | 11.64        |
| 4096×4096  | 406.92     | 146.05         | 111.65         | 46.24        |
| 8192×8192  | 1285.24    | 515.54         | 349.52         | 182.41       |
| 16384×16384| 4531.74    | 2013.94        | 1524.81        | 1142.22      |
| 32768×32768| 19887.24   | 10622.58       | 8176.18        | Memory limit |

### Speedup Analysis

| Image Size | MPI 4 Proc | MPI 8 Proc | OpenCL |
|------------|------------|------------|--------|
| 128×128    | 3.25×      | 3.20×      | 39.95× |
| 256×256    | 2.30×      | 2.78×      | 25.05× |
| 512×512    | 2.48×      | 2.79×      | 18.96× |
| 1024×1024  | 1.71×      | 2.53×      | 9.54×  |
| 4096×4096  | 2.79×      | 3.64×      | 8.80×  |
| 8192×8192  | 2.49×      | 3.68×      | 7.05×  |

**Maximum achieved speedup**: 39.95× (OpenCL), 3.68× (MPI)

## 🚀 Getting Started

### Sequential Implementation (C#)

**Requirements:**
- .NET 9.0
- Visual Studio 2022
- OpenCvSharp package

**How to run:**
1. Open `FilterImages.sln` in Visual Studio
2. Press F5 to debug or use `Debug > Start Without Debugging`

### Parallel MPI Implementation (Python)

**Requirements:**
- Python 3.9+
- MPI4Py
- OpenCV for Python
- NumPy

**Installation:**
```bash
pip install mpi4py opencv-python numpy
```

**Execution:**
```bash
mpiexec -n <number_of_processes> python gaussian_filter_mpi.py
```

Example with 8 processes:
```bash
mpiexec -n 8 python gaussian_filter_mpi.py
```

### OpenCL GPU Implementation (C++)

**Requirements:**
- Visual Studio 2019/2022
- OpenCL SDK
- Compatible GPU (Intel Iris Xe or NVIDIA GeForce recommended)

**How to run:**
1. Open the OpenCL project in Visual Studio
2. Ensure OpenCL SDK is properly configured
3. Build and run the solution

## 🔬 Technical Implementation Details

### Sequential Algorithm
- Uses OpenCvSharp's Filter2D function
- Processes each color channel independently
- Manual Gaussian kernel generation with normalization

### MPI Parallel Strategy
- **Horizontal decomposition**: Image divided into strips
- **Overlap handling**: Manages border regions between processes
- **Load balancing**: Distributes work evenly across processes
- **Communication pattern**: Scatter-gather with border exchange

### OpenCL GPU Implementation
- **Separable filtering**: 2D convolution split into two 1D passes
- **Memory optimization**: Uses local memory for kernel caching
- **Thread-level parallelism**: One thread per pixel
- **Buffer reuse**: Optimizes memory allocation for multiple images

## 📈 Performance Analysis

### Key Findings

1. **OpenCL Dominance**: Achieves up to 39.95× speedup for pure computation
2. **MPI Scalability**: Best performance with 8 processes for large images
3. **Overhead Impact**: OpenCL overhead significant for small images
4. **Sweet Spot**: Medium-sized images (4096×4096) optimal for MPI efficiency

### Amdahl's Law Analysis

- **MPI Implementation**: ~16% sequential fraction
- **OpenCL Implementation**: ~2.5% sequential fraction (pure computation)

### Efficiency Metrics

The parallel efficiency decreases with more processes due to:
- Communication overhead
- Synchronization costs
- Memory bandwidth limitations
- Load balancing challenges

## 🎛️ Hardware Configuration

**Test Platform:**
- **CPU**: Intel Core i7-1255U (1.7 GHz, 10 cores, 12 threads)
- **RAM**: 16.5 GB
- **GPU**: Intel Iris Xe Graphics + NVIDIA GeForce MX550
- **OS**: Windows 11 Home (Build 26100)

## 📋 Algorithm Specifications

- **Filter Type**: Gaussian Blur
- **Kernel Radius**: 30 pixels
- **Sigma Value**: 10.0
- **Supported Formats**: JPEG images
- **Output Naming**: `filtered_[original_name].jpg`
- **Size Range**: 128×128 to 32768×32768 pixels

## 🔍 Use Cases

- **Research**: Parallel computing performance analysis
- **Education**: Understanding parallelization trade-offs
- **Production**: High-performance image processing pipelines
- **Benchmarking**: Comparing different parallel approaches

## 📚 Detailed Documentation

For comprehensive algorithm analysis, experimental methodology, and detailed results, see the [complete documentation](https://docs.google.com/document/d/1Lvr94UeL-PmCsR4VaBGzNiWaocePwUWuNJOjV7AGTTY/edit?usp=sharing).

## 🏆 Key Achievements

- Successfully implemented three different parallelization approaches
- Achieved significant speedups across all implementations
- Provided comprehensive performance analysis following Amdahl's Law
- Demonstrated trade-offs between different parallel computing paradigms

## 📄 References

- [MPI Image Processing Case Study](https://github.com/example/mpi-image-processing)
- [Parallel Image Processing with MPI](https://example.com/parallel-processing)
- [Image Filtering using OpenCL](https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy)
---

*Developed by Smarandache Andra-Maria, CR 3.1B*
