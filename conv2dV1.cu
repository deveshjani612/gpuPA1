#include <iostream>
#include <cuda.h>
#include <random>
#include <fstream>
#include <iomanip>

#define BLOCK_SIZE 16  // CUDA block size

// CUDA Kernel for 2D Convolution
__global__ void conv2d_kernel(float* input, float* filter, float* output, int H, int W, int R) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int pad = R / 2;  // Padding to maintain "same" output size

    if (x < W && y < H) {
        float sum = 0.0f;

        for (int i = 0; i < R; i++) {
            for (int j = 0; j < R; j++) {
                int img_x = x + j - pad;
                int img_y = y + i - pad;

                if (img_x >= 0 && img_x < W && img_y >= 0 && img_y < H) {
                    sum += input[img_y * W + img_x] * filter[i * R + j];
                }
            }
        }

        output[y * W + x] = sum;
    }
}

int main(int argc, char *argv[]) {

    // Read the inputs from command line
    if (argc != 3) {
        std::cerr << "Usage: ./conv2dV1 input.txt filter.txt" << std::endl;
        return 1;
    }

    // Open input image file
    std::ifstream inputFile(argv[1]);
    int H, W, R;
    inputFile >> H >> W;

    // Allocate host memory for image and filter
    float* h_input = (float*)malloc(H * W * sizeof(float));
    for (int i = 0; i < H * W; i++) {
        inputFile >> h_input[i];
    }
    inputFile.close();

    // Open filter file
    std::ifstream filterFile(argv[2]);
    filterFile >> R;
    float* h_filter = (float*)malloc(R * R * sizeof(float));
    for (int i = 0; i < R * R; i++) {
        filterFile >> h_filter[i];
    }
    filterFile.close();

    // Allocate host memory for output
    float* h_output = (float*)malloc(H * W * sizeof(float));

    // Allocate/move data using cudaMalloc and cudaMemCpy
    float *d_input, *d_filter, *d_output;
    cudaMalloc(&d_input, H * W * sizeof(float));
    cudaMalloc(&d_filter, R * R * sizeof(float));
    cudaMalloc(&d_output, H * W * sizeof(float));

    cudaMemcpy(d_input, h_input, H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, R * R * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((W + BLOCK_SIZE - 1) / BLOCK_SIZE, (H + BLOCK_SIZE - 1) / BLOCK_SIZE);


    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Launch the kernel
    conv2d_kernel<<<grid, block>>>(d_input, d_filter, d_output, H, W, R);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;
    // Copy output from device to host
    cudaMemcpy(h_output, d_output, H * W * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the output
    for (int i = 0; i < H * W; i++) {
        std::cout << std::fixed << std::setprecision(3) << h_output[i] << std::endl;
    }

    // Clean up the memory
    free(h_input);
    free(h_filter);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);

    return 0;
}
