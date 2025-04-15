/*
 * Rectangular matrix multiplication
 * A[M][K] * B[k][N] = C[M][N]
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/timeb.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono> 
#include <sys/time.h>

double read_timer() {
    struct timeval tm;
    gettimeofday(&tm, NULL);
    return (double) tm.tv_sec + (double) tm.tv_usec / 1000000.0;
}

double read_timer_ms() {
    struct timeval tm;
    gettimeofday(&tm, NULL);
    return (double) tm.tv_sec * 1000.0 + (double) tm.tv_usec / 1000.0;
}

#define REAL float
#define BLOCK_SIZE 16

void init(int M, int N, REAL * A) {
    int i, j;

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            A[i*N+j] = (REAL) drand48();
        }
    }
}

double maxerror(int M, int N, REAL * A, REAL *B) {
    int i, j;
    double error = 0.0;

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            double diff = (A[i*N+j] - B[i*N+j]) / A[i*N+j];
            if (diff < 0)
                diff = -diff;
            if (diff > error)
                error = diff;
        }
    }
    return error;
}

void matmul_base(int N, REAL *A, REAL * B, REAL *C);
void matmul_openmp(int N, REAL *A, REAL *B, REAL *C, int num_tasks);
void matmul_cuda_v1_vanilla(int N, REAL *A, REAL *B, REAL *C);
void matmul_cuda_v2_shmem(int N, REAL *A, REAL *B, REAL *C);
void matmul_cuda_v3_cublas(int N, REAL *A, REAL *B, REAL *C);

int main(int argc, char *argv[]) {
    int N;
    int num_tasks = 5; 
    double elapsed_base, elapsed_openmp, elapsed_cuda_v1, elapsed_cuda_v2, elapsed_cuda_v3;
    if (argc < 2) {
        fprintf(stderr, "Usage: matmul <n> [<#tasks(%d)>]\n", num_tasks);
        exit(1);
    }
    N = atoi(argv[1]);
    if (argc > 2) num_tasks = atoi(argv[2]);
    REAL * heap_buffer = (REAL*)malloc(sizeof(REAL)*N*N*6); 
 
    REAL *A = heap_buffer;
    REAL *B = &heap_buffer[N*N];
    REAL *C_base = &heap_buffer[2*N*N];
    REAL *C_openmp = &heap_buffer[3*N*N];
    REAL *C_cuda_v1 = &heap_buffer[4*N*N];
    REAL *C_cuda_v2 = &heap_buffer[5*N*N];

    srand48((1 << 12));
    init(N, N, A);
    init(N, N, B);

    elapsed_base = read_timer();
    matmul_base(N, A, B, C_base);
    elapsed_base = (read_timer() - elapsed_base);

    elapsed_openmp = read_timer();
    matmul_openmp(N, A, B, C_openmp, num_tasks);
    elapsed_openmp = (read_timer() - elapsed_openmp);

    cudaSetDevice(0); 
    
    elapsed_cuda_v1 = read_timer();
    matmul_cuda_v1_vanilla(N, A, B, C_cuda_v1);
    elapsed_cuda_v1 = (read_timer() - elapsed_cuda_v1);

    elapsed_cuda_v2 = read_timer();
    matmul_cuda_v2_shmem(N, A, B, C_cuda_v2);
    elapsed_cuda_v2 = (read_timer() - elapsed_cuda_v2);

    elapsed_cuda_v3 = read_timer();
    matmul_cuda_v3_cublas(N, A, B, C_base);
    elapsed_cuda_v3 = (read_timer() - elapsed_cuda_v3);

    printf("======================================================================================================\n");
    printf("Matrix Multiplication: A[M][K] * B[k][N] = C[M][N], M=K=N=%d, %d threads/tasks\n", N, num_tasks);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS \t\tError (compared to base)\n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("matmul_base:\t\t%4f\t%4f \t\t%g\n", elapsed_base * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_base)), maxerror(N, N, C_base, C_base));
    printf("matmul_openmp:\t\t%4f\t%4f \t\t%g\n", elapsed_openmp * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_openmp)), maxerror(N, N, C_base, C_openmp));
    printf("matmul_cuda_v1:\t\t%4f\t%4f \t\t%g\n", elapsed_cuda_v1 * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_cuda_v1)), maxerror(N, N, C_base, C_cuda_v1));
    printf("matmul_cuda_v2:\t\t%4f\t%4f \t\t%g\n", elapsed_cuda_v2 * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_cuda_v2)), maxerror(N, N, C_base, C_cuda_v2));
    printf("matmul_cuda_v3:\t\t%4f\t%4f \t\t%g\n", elapsed_cuda_v3 * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_cuda_v3)), maxerror(N, N, C_base, C_base));
    
    free(heap_buffer);
    return 0;
}

void matmul_base(int N, REAL *A, REAL * B, REAL *C) {
    int i, j, k;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            REAL temp = 0.0;
            for (k = 0; k < N; k++) {
                temp += A[i*N+k] * B[k*N+j];
            }
            C[i*N+j] = temp;
        }
    }
}

void matmul_openmp(int N, REAL *A, REAL *B, REAL *C, int num_tasks) {
    int i, j, k;
#pragma omp parallel for shared(N,A,B,C,num_tasks) private(i,j,k) num_threads(num_tasks)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            REAL temp = 0.0;
            for (k = 0; k < N; k++) {
                temp += A[i*N+k] * B[k*N+j];
            }
            C[i*N+j] = temp;
        }
    }
}

__global__ void matmul_kernel_vanilla(int N, REAL *A, REAL *B, REAL *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        REAL sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void matmul_cuda_v1_vanilla(int N, REAL *A, REAL *B, REAL *C) {
    REAL *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(REAL);

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    matmul_kernel_vanilla<<<gridSize, blockSize>>>(N, d_A, d_B, d_C);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

__global__ void matmul_kernel_shmem(int N, REAL *A, REAL *B, REAL *C) {
    __shared__ REAL As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ REAL Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    REAL sum = 0.0f;
    
    for (int m = 0; m < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {

        if (row < N && (m * BLOCK_SIZE + tx) < N) {
            As[ty][tx] = A[row * N + m * BLOCK_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if ((m * BLOCK_SIZE + ty) < N && col < N) {
            Bs[ty][tx] = B[(m * BLOCK_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void matmul_cuda_v2_shmem(int N, REAL *A, REAL *B, REAL *C) {
    REAL *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(REAL);

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    matmul_kernel_shmem<<<gridSize, blockSize>>>(N, d_A, d_B, d_C);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void matmul_cuda_v3_cublas(int N, REAL *A, REAL *B, REAL *C) {
    REAL *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(REAL);

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);    

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    
    REAL alpha = 1.0f;
    REAL beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                N, N, N, 
                &alpha, 
                d_A, N, 
                d_B, N, 
                &beta, 
                d_C, N);
    
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
