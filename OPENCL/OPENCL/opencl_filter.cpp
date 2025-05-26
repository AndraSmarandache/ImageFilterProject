//#include <CL/cl.h>
//#include <stdio.h>
//
//int main() {
//    cl_platform_id platform;
//    cl_device_id device;
//    cl_int err;
//
//    // Get platform
//    err = clGetPlatformIDs(1, &platform, NULL);
//    if (err != CL_SUCCESS) {
//        printf("Error getting platform: %d\n", err);
//        return 1;
//    }
//
//    // Get GPU device
//    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
//    if (err != CL_SUCCESS) {
//        printf("Error getting device: %d\n", err);
//        return 1;
//    }
//
//    // Create context
//    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
//    if (err != CL_SUCCESS) {
//        printf("Error creating context: %d\n", err);
//        return 1;
//    }
//
//    printf("OpenCL initialized successfully!\n");
//
//    // Clean up
//    clReleaseContext(context);
//
//    return 0;
//}