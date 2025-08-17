#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>

#define MASK_RADIUS 2
#define MASK_DIM ((MASK_RADIUS)*2 + 1)

#define IN_TILE_DIM 16
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2 * (MASK_RADIUS))

// Allocate constant memory on the GPU
// Use cudaMemcpyToSymbol() in the host
__constant__ float mask_c[MASK_DIM][MASK_DIM];


inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "Cuda Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

__global__ void tiled_convolution_kernel(float *input, float *output, int width, int height) {
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - MASK_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - MASK_RADIUS;

    // Load input tile into shared memory
    __shared__ float input_s[IN_TILE_DIM][IN_TILE_DIM];
    if (row >= 0 && row < height && col >= 0 && col < width) {
        input_s[threadIdx.y][threadIdx.x] = input[row * width + col];
    }
    else {
        input_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Calculating output elements
    int tileCol = threadIdx.x - MASK_RADIUS;
    int tileRow = threadIdx.y - MASK_RADIUS;

    // Turning off the threads at the edges of the block
    if (col >= 0 && col < width && row >= 0 && row < height) {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM) {
            float outputValue = 0.0f;
            for (int fRow = 0; fRow < 2 * MASK_RADIUS+1; fRow++) {
                for (int fCol = 0; fCol < 2 * MASK_RADIUS+1; fCol++) {
                    outputValue += mask_c[fRow][fCol] * input_s[tileRow+fRow][tileCol+fCol];
                }
            }
            output[row * width + col] = outputValue;
        }
    }
}


// Parallelization approach: 
// Assign one thread to compute each output element 
// by iterating over input elements and mask weights
__global__ void naive_convolution_kernel(float *input, float *output, unsigned int width, unsigned int height) {

    // Output thread index
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
 
    if (outRow < height && outCol < width) {

        float sum = 0.0f;
        
        // Loop through the mask
        for (int maskRow = 0; maskRow < MASK_DIM; maskRow++) {
            for (int maskCol = 0; maskCol < MASK_DIM; maskCol++) {
                
                // Find what elements to work on the input
                int inRow = outRow - MASK_RADIUS + maskRow;
                int inCol = outCol - MASK_RADIUS + maskCol;

                if ((inRow < height && inRow >= 0) && (inCol < width && inCol >= 0)) {
                    sum += mask_c[maskRow][maskCol] * input[inRow * width + inCol];
                }
            }
        }
        output[outRow * width + outCol] = sum;
    }
}

void convolution_cpu(float *input, float *output, float *mask, int width, int height) {
    for (int outRow = 0; outRow < height; outRow++) {
        for (int outCol = 0; outCol < width; outCol++) {
            float sum = 0.0f;
            for (int maskRow = 0; maskRow < MASK_DIM; maskRow++) {
                for (int maskCol = 0; maskCol < MASK_DIM; maskCol++) {
                    int inRow = outRow - MASK_RADIUS + maskRow;
                    int inCol = outCol - MASK_RADIUS + maskCol;
                    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                        sum += mask[maskRow * MASK_DIM + maskCol] * input[inRow * width + inCol];
                    }
                }
            }
            output[outRow * width + outCol] = sum;
        }
    }
}


void tiled_convolution(float *input_h, float *output_h, float *mask_h, int width) {
    float *input_d, *output_d;

    // Allocate GPU memory
    checkCuda( cudaMalloc((void**)&input_d, width * width * sizeof(float)) );
    checkCuda( cudaMalloc((void**)&output_d, width * width * sizeof(float)) );

    // Transfer data host -> device
    checkCuda( cudaMemcpy(input_d, input_h, width * width * sizeof(float), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpyToSymbol(mask_c, mask_h, MASK_DIM * MASK_DIM * sizeof(float)) );

    // Launch Kernel
    dim3 threadsPerBlock(IN_TILE_DIM, IN_TILE_DIM);
    dim3 numBlocks((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                   (width + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    tiled_convolution_kernel<<<numBlocks, threadsPerBlock>>>(input_d, output_d, width, width);
    checkCuda( cudaGetLastError() );
    checkCuda( cudaDeviceSynchronize() );

    // Transfer data device -> host
    checkCuda( cudaMemcpy(output_h, output_d, width * width * sizeof(float), cudaMemcpyDeviceToHost) );

    // Free GPU memory
    checkCuda( cudaFree(input_d) );
    checkCuda( cudaFree(output_d) );
}

void naive_convolution(float *input_h, float *output_h, float *mask_h, int width) {
    float *input_d, *output_d;

    // Allocate GPU memory
    checkCuda( cudaMalloc((void**)&input_d, width * width * sizeof(float)) );
    checkCuda( cudaMalloc((void**)&output_d, width * width * sizeof(float)) );

    // Transfer data host -> device
    checkCuda( cudaMemcpy(input_d, input_h, width * width * sizeof(float), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpyToSymbol(mask_c, mask_h, MASK_DIM * MASK_DIM * sizeof(float)) );

    // Launch Kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (width + threadsPerBlock.y - 1) / threadsPerBlock.y);


    naive_convolution_kernel<<<numBlocks, threadsPerBlock>>>(input_d, output_d, width, width);
    checkCuda( cudaGetLastError() );
    checkCuda( cudaDeviceSynchronize() );

    // Transfer data device -> host
    checkCuda( cudaMemcpy(output_h, output_d, width * width * sizeof(float), cudaMemcpyDeviceToHost) );

    // Free GPU memory
    checkCuda( cudaFree(input_d) );
    checkCuda( cudaFree(output_d) );
}

int main(void) {
    srand(time(NULL));

    // N x N matrix 
    // 1 << 10 = 1024x1024
    // 1 << 11 = 2048x2048
    // 1 << 12 = 4096x4096
    // 1 << 13 = 8192x8192
    // 1 << 14 = 16384x16384 (max)
    int n = 1 << 14;

    float *input  = (float*) malloc(n * n * sizeof(float));
    float *output_gpu_naive = (float*) malloc(n * n * sizeof(float));
    float *output_gpu_tiled = (float*) malloc(n * n * sizeof(float));
    float *output_cpu = (float*) malloc(n * n * sizeof(float));
    float *mask   = (float*) malloc(MASK_DIM * MASK_DIM * sizeof(float));

    if (!input || !output_gpu_naive || !output_gpu_tiled || !output_cpu || !mask) {
        fprintf(stderr, "Malloc Error.\n");
        return 1;
    }

    // Initilized input with random values
    for (int i = 0; i < n * n; i++) 
        input[i] = rand() / (float)RAND_MAX;

    // Mask with ones (sums the neighborhood)
     for (int i = 0; i < MASK_DIM * MASK_DIM; i++)
        mask[i] = 1.0f;

    // Run GPU naive
    naive_convolution(input, output_gpu_naive, mask, n);

    // Run GPU tiled
    tiled_convolution(input, output_gpu_tiled, mask, n);

    // Run CPU convolution and benchmark
    clock_t start = clock();
    convolution_cpu(input, output_cpu, mask, n, n);
    clock_t end = clock();
    double cpu_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("CPU convolution time: %f seconds\n", cpu_time);

    // Check result
    printf("Naive GPU Output[0] = %f\n", output_gpu_naive[0]);
    printf("Tiled GPU Output[0] = %f\n", output_gpu_tiled[0]);
    printf("CPU Output[0] = %f\n", output_cpu[0]);

    free(input);
    free(output_gpu_naive);
    free(output_gpu_tiled);
    free(output_cpu);
    free(mask);

    return 0;
}

