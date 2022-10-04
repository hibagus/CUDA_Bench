#pragma once
#include <CUDA_Bench/util/precision_select.cuh>
#include <CUDA_Bench/util/gpucheck.cuh>
#include <CUDA_Bench/gemm/gemm_util.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>

template<typename S, typename M, typename A>
static inline int gemm_cublas_launch_int(int dim_M, int dim_N, int dim_K, cudaDataType_t mulDataType, cudaDataType_t accDataType, cublasComputeType_t computeType, cublasGemmAlgo_t algoType, int num_iter, bool print_result, bool tensor_cores, bool profiling)
{
    // Initialize cuBLAS
	cublasHandle_t handle;	// CUBLAS context
    gpuErrchk(cublasCreate(&handle));

    // Device Memory Allocation
    M* dev_matA;
    M* dev_matB;
    A* dev_matC;
    S alpha = 1;
    S beta  = 0;

    gpuErrchk(cudaMalloc((void**)&dev_matA, dim_M * dim_K * sizeof(M)));
    gpuErrchk(cudaMalloc((void**)&dev_matB, dim_K * dim_N * sizeof(M)));
    gpuErrchk(cudaMalloc((void**)&dev_matC, dim_M * dim_N * sizeof(A)));

    // Initialize Matrix
    if(tensor_cores) {initialize_rownegpos_matrix<M><<<((dim_M*dim_K)+512-1)/512,512>>>(dev_matA, dim_M, dim_K, 1);}
    else             {initialize_colnegpos_matrix<M><<<((dim_M*dim_K)+512-1)/512,512>>>(dev_matA, dim_M, dim_K, 1);}
    initialize_colposneg_matrix<M><<<((dim_K*dim_N)+512-1)/512,512>>>(dev_matB, dim_K, dim_N, 1);
    initialize_matrix<A><<<((dim_M*dim_N)+512-1)/512,512>>>(dev_matC, dim_M, dim_N, 0);
    gpuErrchk(cudaDeviceSynchronize());

    
    if(tensor_cores)
    {
        // Start Multiplication
        cudaProfilerStart();
        for(int iter=0;iter<num_iter;iter++)
        {
            gpuErrchk(cublasGemmEx(handle,                       // handle to cuBLAS library context
                                   CUBLAS_OP_T,                  // CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C
                                   CUBLAS_OP_N,                  // CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C
                                   dim_M,                        // dimension M 
                                   dim_N,                        // dimension N
                                   dim_K,                        // dimension K
                                   &alpha,                       // Scaling factor alpha where (alpha)x(AxB)
                                   dev_matA,                     // Pointer to Matrix A on Device
                                   mulDataType,                  // Data type of Matrix A
                                   dim_K,                        // Leading Dimension of Matrix A
                                   dev_matB,                     // Pointer to Matrix B on Device
                                   mulDataType,                  // Data Type of Matrix B
                                   dim_K,                        // Leading Dimension of Matrix B
                                   &beta,                        // Scaling factor beta where (beta)xC
                                   dev_matC,                     // Pointer to Matrix C on Device
                                   accDataType,                  // Data Type of Matrix C
                                   dim_M,                        // Leading Dimension of Matrix C
                                   computeType,                  // Computation Type
                                   algoType                      // Computation Algorithm
            ));
        }
        cudaProfilerStop();
    }
    else
    {
        // Start Multiplication
        cudaProfilerStart();
        for(int iter=0;iter<num_iter;iter++)
        {
            gpuErrchk(cublasGemmEx(handle,                       // handle to cuBLAS library context
                                   CUBLAS_OP_N,                  // CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C
                                   CUBLAS_OP_N,                  // CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C
                                   dim_M,                        // dimension M 
                                   dim_N,                        // dimension N
                                   dim_K,                        // dimension K
                                   &alpha,                       // Scaling factor alpha where (alpha)x(AxB)
                                   dev_matA,                     // Pointer to Matrix A on Device
                                   mulDataType,                  // Data type of Matrix A
                                   dim_M,                        // Leading Dimension of Matrix A
                                   dev_matB,                     // Pointer to Matrix B on Device
                                   mulDataType,                  // Data Type of Matrix B
                                   dim_K,                        // Leading Dimension of Matrix B
                                   &beta,                        // Scaling factor beta where (beta)xC
                                   dev_matC,                     // Pointer to Matrix C on Device
                                   accDataType,                  // Data Type of Matrix C
                                   dim_M,                        // Leading Dimension of Matrix C
                                   computeType,                  // Computation Type
                                   algoType                      // Computation Algorithm
            ));
        }
        cudaProfilerStop();
    }
    


    if(print_result)
    {
        std::cout << "Matrix A: " << std::endl;
        view_matrix_int<M><<<1,1>>>(dev_matA, dim_M, dim_K);
        gpuErrchk(cudaDeviceSynchronize());
        std::cout << "Matrix B: " << std::endl;
        view_matrix_int<M><<<1,1>>>(dev_matB, dim_K, dim_N);
        gpuErrchk(cudaDeviceSynchronize());
        std::cout << "Matrix C: " << std::endl;
        view_matrix_int<A><<<1,1>>>(dev_matC, dim_M, dim_N);
        gpuErrchk(cudaDeviceSynchronize());
    }


    gpuErrchk(cudaFree(dev_matA));
    gpuErrchk(cudaFree(dev_matB));
    gpuErrchk(cudaFree(dev_matC));
    gpuErrchk(cublasDestroy(handle));
    return 0;
}