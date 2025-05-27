#include <opencv2/opencv.hpp>
#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <vector>

using namespace cv;
using namespace std;

void checkError(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        cerr << "Error during " << operation << ": " << err << endl;
        exit(1);
    }
}

vector<float> createGaussianKernel1D(int radius, float sigma) {
    int size = 2 * radius + 1;
    vector<float> kernel(size);
    float sum = 0.0f;

    for (int i = 0; i < size; i++) {
        float diff = i - radius;
        kernel[i] = exp(-(diff * diff) / (2.0f * sigma * sigma));
        sum += kernel[i];
    }

    for (int i = 0; i < size; i++) {
        kernel[i] /= sum;
    }

    return kernel;
}

class OpenCLGaussianFilter {
private:
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel horizontalKernel;
    cl_kernel verticalKernel;
    bool initialized;

    cl_mem inputBuffer;
    cl_mem tempBuffer;
    cl_mem outputBuffer;
    cl_mem kernelBuffer;
    size_t currentImageSize;

public:
    OpenCLGaussianFilter() : initialized(false), currentImageSize(0) {}

    bool initialize() {
        if (initialized) return true;

        cl_int err = clGetPlatformIDs(1, &platform, NULL);
        if (err != CL_SUCCESS) return false;

        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        if (err != CL_SUCCESS) return false;

        const cl_context_properties props[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0
        };
        context = clCreateContext(props, 1, &device, NULL, NULL, &err);
        if (err != CL_SUCCESS) return false;

        queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
        if (err != CL_SUCCESS) return false;

        string kernelSource = R"(
__kernel void gaussianHorizontal(
    __global const uchar* input,
    __global uchar* output,
    __constant float* filter,
    const int width,
    const int height,
    const int filterSize,
    const int channels)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    const int radius = filterSize / 2;
    const int rowStart = y * width * channels;
    
    __local float localFilter[64];
    if (get_local_id(0) == 0 && get_local_id(1) == 0) {
        for (int i = 0; i < filterSize && i < 64; i++) {
            localFilter[i] = filter[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        #pragma unroll 8
        for (int i = -radius; i <= radius; i++) {
            int sampleX = clamp(x + i, 0, width - 1);
            int idx = rowStart + sampleX * channels + c;
            sum += input[idx] * localFilter[i + radius];
        }
        
        output[rowStart + x * channels + c] = convert_uchar_sat(sum);
    }
}

__kernel void gaussianVertical(
    __global const uchar* input,
    __global uchar* output,
    __constant float* filter,
    const int width,
    const int height,
    const int filterSize,
    const int channels)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    const int radius = filterSize / 2;
    
    __local float localFilter[64];
    if (get_local_id(0) == 0 && get_local_id(1) == 0) {
        for (int i = 0; i < filterSize && i < 64; i++) {
            localFilter[i] = filter[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        #pragma unroll 8
        for (int i = -radius; i <= radius; i++) {
            int sampleY = clamp(y + i, 0, height - 1);
            int idx = sampleY * width * channels + x * channels + c;
            sum += input[idx] * localFilter[i + radius];
        }
        
        output[y * width * channels + x * channels + c] = convert_uchar_sat(sum);
    }
}
)";

        const char* source = kernelSource.c_str();
        size_t sourceSize = kernelSource.size();

        program = clCreateProgramWithSource(context, 1, &source, &sourceSize, &err);
        if (err != CL_SUCCESS) return false;

        err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math -cl-mad-enable -cl-no-signed-zeros", NULL, NULL);
        if (err != CL_SUCCESS) return false;

        horizontalKernel = clCreateKernel(program, "gaussianHorizontal", &err);
        if (err != CL_SUCCESS) return false;

        verticalKernel = clCreateKernel(program, "gaussianVertical", &err);
        if (err != CL_SUCCESS) return false;

        initialized = true;
        return true;
    }

    bool processImage(const Mat& input, Mat& output, const vector<float>& kernel1D) {
        if (!initialized && !initialize()) {
            return false;
        }

        int width = input.cols;
        int height = input.rows;
        int channels = input.channels();
        size_t imageSize = width * height * channels * sizeof(unsigned char);
        int kernelSize = kernel1D.size();

        if (imageSize != currentImageSize) {
            if (currentImageSize > 0) {
                clReleaseMemObject(inputBuffer);
                clReleaseMemObject(tempBuffer);
                clReleaseMemObject(outputBuffer);
            }

            cl_int err;
            inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, imageSize, NULL, &err);
            if (err != CL_SUCCESS) return false;

            tempBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, imageSize, NULL, &err);
            if (err != CL_SUCCESS) return false;

            outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, imageSize, NULL, &err);
            if (err != CL_SUCCESS) return false;

            currentImageSize = imageSize;
        }

        cl_int err;
        kernelBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            kernelSize * sizeof(float), (void*)kernel1D.data(), &err);
        if (err != CL_SUCCESS) return false;

        void* mappedInput = clEnqueueMapBuffer(queue, inputBuffer, CL_TRUE, CL_MAP_WRITE,
            0, imageSize, 0, NULL, NULL, &err);
        if (err != CL_SUCCESS) return false;

        memcpy(mappedInput, input.data, imageSize);
        clEnqueueUnmapMemObject(queue, inputBuffer, mappedInput, 0, NULL, NULL);

        size_t maxWorkGroupSize;
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);

        size_t localWorkSize[2];
        if (maxWorkGroupSize >= 256) {
            localWorkSize[0] = 32;
            localWorkSize[1] = 8;
        }
        else {
            localWorkSize[0] = 16;
            localWorkSize[1] = 8;
        }

        size_t globalWorkSize[2] = {
            ((width + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0],
            ((height + localWorkSize[1] - 1) / localWorkSize[1]) * localWorkSize[1]
        };

        auto computeStart = chrono::high_resolution_clock::now();

        err = clSetKernelArg(horizontalKernel, 0, sizeof(cl_mem), &inputBuffer);
        err |= clSetKernelArg(horizontalKernel, 1, sizeof(cl_mem), &tempBuffer);
        err |= clSetKernelArg(horizontalKernel, 2, sizeof(cl_mem), &kernelBuffer);
        err |= clSetKernelArg(horizontalKernel, 3, sizeof(int), &width);
        err |= clSetKernelArg(horizontalKernel, 4, sizeof(int), &height);
        err |= clSetKernelArg(horizontalKernel, 5, sizeof(int), &kernelSize);
        err |= clSetKernelArg(horizontalKernel, 6, sizeof(int), &channels);
        if (err != CL_SUCCESS) return false;

        err = clEnqueueNDRangeKernel(queue, horizontalKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        if (err != CL_SUCCESS) return false;

        err = clSetKernelArg(verticalKernel, 0, sizeof(cl_mem), &tempBuffer);
        err |= clSetKernelArg(verticalKernel, 1, sizeof(cl_mem), &outputBuffer);
        err |= clSetKernelArg(verticalKernel, 2, sizeof(cl_mem), &kernelBuffer);
        err |= clSetKernelArg(verticalKernel, 3, sizeof(int), &width);
        err |= clSetKernelArg(verticalKernel, 4, sizeof(int), &height);
        err |= clSetKernelArg(verticalKernel, 5, sizeof(int), &kernelSize);
        err |= clSetKernelArg(verticalKernel, 6, sizeof(int), &channels);
        if (err != CL_SUCCESS) return false;

        err = clEnqueueNDRangeKernel(queue, verticalKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        if (err != CL_SUCCESS) return false;

        clFinish(queue);

        output.create(height, width, input.type());
        void* mappedOutput = clEnqueueMapBuffer(queue, outputBuffer, CL_TRUE, CL_MAP_READ,
            0, imageSize, 0, NULL, NULL, &err);
        if (err != CL_SUCCESS) return false;

        memcpy(output.data, mappedOutput, imageSize);
        clEnqueueUnmapMemObject(queue, outputBuffer, mappedOutput, 0, NULL, NULL);

        clReleaseMemObject(kernelBuffer);
        auto computeEnd = chrono::high_resolution_clock::now();
        cout << "Computation time: " << chrono::duration<double, milli>(computeEnd - computeStart).count() << endl;
        return true;
    }

    ~OpenCLGaussianFilter() {
        if (initialized) {
            if (currentImageSize > 0) {
                clReleaseMemObject(inputBuffer);
                clReleaseMemObject(tempBuffer);
                clReleaseMemObject(outputBuffer);
            }
            clReleaseKernel(horizontalKernel);
            clReleaseKernel(verticalKernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
        }
    }
};

int main(int argc, char** argv) {
    int radius = 30;
    float sigma = radius / 3.0f;
    string inputPath = "images/resized_image_16384x16384.jpg";
    string outputPath = "images/filtered_image_16384x16384_opencl_optimized.jpg";

    Mat src = imread(inputPath, IMREAD_COLOR);
    if (src.empty()) {
        cerr << "Failed to load image: " << inputPath << endl;
        return -1;
    }

    vector<float> kernel1D = createGaussianKernel1D(radius, sigma);
    OpenCLGaussianFilter filter;
    Mat result;

    if (!filter.processImage(src, result, kernel1D)) {
        cerr << "OpenCL processing failed!" << endl;
        return -1;
    }

    imwrite(outputPath, result);
    return 0;
}
