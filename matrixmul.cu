#include <stdio.h>

#define ROW_A 2
#define COL_A 3
#define COL_B 2

// CUDA kernel for matrix multiplication
__global__ void matrixMul(int* a, int* b, int* c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;
    for (int k = 0; k < COL_A; k++) {
        sum += a[row * COL_A + k] * b[k * COL_B + col];
    }

    c[row * COL_B + col] = sum;
}

int main() {
    int a[ROW_A][COL_A];
    int b[COL_A][COL_B];
    int c[ROW_A][COL_B];  // Output matrix

    printf("Enter the elements of matrix A (%dx%d):\n", ROW_A, COL_A);
    for (int i = 0; i < ROW_A; i++) {
        for (int j = 0; j < COL_A; j++) {
            scanf("%d", &a[i][j]);
        }
    }

    printf("Enter the elements of matrix B (%dx%d):\n", COL_A, COL_B);
    for (int i = 0; i < COL_A; i++) {
        for (int j = 0; j < COL_B; j++) {
            scanf("%d", &b[i][j]);
        }
    }

    int *dev_a, *dev_b, *dev_c;  // Device copies of input and output matrices

    // Allocate memory on the device
    cudaMalloc((void**)&dev_a, ROW_A * COL_A * sizeof(int));
    cudaMalloc((void**)&dev_b, COL_A * COL_B * sizeof(int));
    cudaMalloc((void**)&dev_c, ROW_A * COL_B * sizeof(int));

    // Copy input matrices from host to device
    cudaMemcpy(dev_a, a, ROW_A * COL_A * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, COL_A * COL_B * sizeof(int), cudaMemcpyHostToDevice);

    // Set grid and block dimensions
    dim3 blockDim(COL_B, ROW_A);
    dim3 gridDim(1, 1);

    // Launch kernel
    matrixMul<<<gridDim, blockDim>>>(dev_a, dev_b, dev_c);

    // Copy output matrix from device to host
    cudaMemcpy(c, dev_c, ROW_A * COL_B * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the output matrix
    printf("Resultant matrix C (%dx%d):\n", ROW_A, COL_B);
    for (int i = 0; i < ROW_A; i++) {
        for (int j = 0; j < COL_B; j++) {
            printf("%d\t", c[i][j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
