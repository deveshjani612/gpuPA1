#include <iostream>
#include <cuda.h>
#include <random>
#include <fstream>
#include <iomanip>

#define BLOCK_SIZE 16  // CUDA block size

// The CUDA kernel
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
        std::cerr << "Usage: ./conv2dV2 input.txt filter.txt" << std::endl;
        return 1;
    }

    // Open input image file
    std::ifstream inputFile(argv[1]);
    int H, W, R;
    inputFile >> H >> W;

    // Allocate data using cudaMallocManaged -- no need for cudaMemCpy
    float *input, *filter, *output;
    cudaMallocManaged(&input, H * W * sizeof(float));
    cudaMallocManaged(&output, H * W * sizeof(float));

    for (int i = 0; i < H * W; i++) {
        inputFile >> input[i];
    }
    inputFile.close();

    // Open filter file
    std::ifstream filterFile(argv[2]);
    filterFile >> R;
    cudaMallocManaged(&filter, R * R * sizeof(float));

    for (int i = 0; i < R * R; i++) {
        filterFile >> filter[i];
    }
    filterFile.close();

    // Define grid and block sizes
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((W + BLOCK_SIZE - 1) / BLOCK_SIZE, (H + BLOCK_SIZE - 1) / BLOCK_SIZE);


    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start);
    // Launch the kernel
    conv2d_kernel<<<grid, block>>>(input, filter, output, H, W, R);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;
    cudaDeviceSynchronize();

    // Print the output
    for (int i = 0; i < H * W; i++) {
        std::cout << std::fixed << std::setprecision(3) << output[i] << std::endl;
    }

    // Clean up the memory
    cudaFree(input);
    cudaFree(filter);
    cudaFree(output);

    return 0;
}
