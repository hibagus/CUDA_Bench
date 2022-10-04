// Matrix-matrix Multiplication using cuBLAS
// (C) 2022 Bagus Hanindhito

#include <CUDA_Bench/gemm/gemm_cublas.cuh>
#include <CUDA_Bench/gemm/gemm_util.cuh>
#include <CUDA_Bench/util/gpuinfo.cuh>
#include <CUDA_Bench/util/gpucheck.cuh>
#include <CUDA_Bench/util/precision_select.cuh>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <iostream>


int gemm_cublas(int dim_M, int dim_N, int dim_K, Precision precision, int num_iter, bool print_result, bool tensor_cores)
{
    // Detect Available CUDA Devices
    int nDevices;
    gpuErrchk(cudaGetDeviceCount(&nDevices));
    print_cuda_device_info(nDevices);
    if(nDevices>0) {std::cout << "[WARN] This program does not currently support Multi-GPU run.\n";}

    // Initialize cuBLAS
	cublasHandle_t handle;	// CUBLAS context
    gpuErrchk(cublasCreate(&handle));

    // Matrices on device
    half* dev_matA;
    half* dev_matB;
    half* dev_matC;
    half alpha = 1.0;
    half beta = 0.0;

    gpuErrchk(cudaMalloc((void**)&dev_matA, dim_M * dim_K * sizeof(half)));
    gpuErrchk(cudaMalloc((void**)&dev_matB, dim_K * dim_N * sizeof(half)));
    gpuErrchk(cudaMalloc((void**)&dev_matC, dim_M * dim_N * sizeof(half)));

    initialize_matrix<<<((dim_M*dim_K)+512-1)/512,512>>>(dev_matA, dim_M, dim_K, 1.0);
    initialize_matrix<<<((dim_K*dim_N)+512-1)/512,512>>>(dev_matB, dim_K, dim_N, 1.0);
    initialize_matrix<<<((dim_M*dim_N)+512-1)/512,512>>>(dev_matC, dim_M, dim_N, 0.0);

    // Invoke cuBLASGemmEX to do C = (alpha)x(AxB) + (beta)xC
    cudaProfilerStart();
    gpuErrchk(cublasGemmEx(handle,                       // handle to cuBLAS library context
                           CUBLAS_OP_N,                  // CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C
                           CUBLAS_OP_N,                  // CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C
                           dim_M,                        // dimension M 
                           dim_N,                        // dimension N
                           dim_K,                        // dimension K
                           &alpha,                       // Scaling factor alpha where (alpha)x(AxB)
                           dev_matA,                     // Pointer to Matrix A on Device
                           CUDA_R_16F,                   // Data type of Matrix A
                           dim_M,                        // Leading Dimension of Matrix A
                           dev_matB,                     // Pointer to Matrix B on Device
                           CUDA_R_16F,                   // Data Type of Matrix B
                           dim_K,                        // Leading Dimension of Matrix B
                           &beta,                        // Scaling factor beta where (beta)xC
                           dev_matC,                     // Pointer to Matrix C on Device
                           CUDA_R_16F,                   // Data Type of Matrix C
                           dim_M,                        // Leading Dimension of Matrix C
                           CUBLAS_COMPUTE_16F,           // Computation Type
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP // Computation Algorithm
    ));
    cudaProfilerStop();



    //std::cout << "Matrix A: " << std::endl;
    //view_matrix<<<1,1>>>(dev_matA, dim_M, dim_K);
    //gpuErrchk(cudaDeviceSynchronize());
    //std::cout << "Matrix B: " << std::endl;
    //view_matrix<<<1,1>>>(dev_matB, dim_K, dim_N);
    //gpuErrchk(cudaDeviceSynchronize());
    //std::cout << "Matrix C: " << std::endl;
    //view_matrix<<<1,1>>>(dev_matC, dim_M, dim_N);
    //gpuErrchk(cudaDeviceSynchronize());



    gpuErrchk(cudaFree(dev_matA));
    gpuErrchk(cudaFree(dev_matB));
    gpuErrchk(cudaFree(dev_matC));

    return 0;


}


__global__
void initialize_matrix(half* matrix, int n_rows, int n_cols, half val)
{
    int workerID = blockIdx.x*blockDim.x + threadIdx.x;
    int n_elements = n_rows * n_cols;
    if(workerID<n_elements)
    {
        matrix[workerID] = val;
    }
}

__global__
void view_matrix(half* matrix, int n_rows, int n_cols)
{
    for(int col=0; col<n_cols; col++)
    {
        for(int row=0; row<n_rows; row++)
        {
            float temp = matrix[col*n_rows+row];
            printf("%f ", temp);
        }
        printf("\n");
    }
}