#pragma once
#include <CUDA_Bench/util/precision_select.cuh>
#include <CUDA_Bench/util/gpucheck.cuh>
#include <CUDA_Bench/gemm/gemm_util.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>

template<typename T, typename S>
int gemm_cublas_launch(int dim_M, int dim_N, int dim_K, Precision mulprecision, Precision accprecision, int num_iter, bool print_result, bool tensor_cores)
{
    // Initialize cuBLAS
	cublasHandle_t handle;	// CUBLAS context
    gpuErrchk(cublasCreate(&handle));

    // Device Memory Allocation
    T* dev_matA;
    T* dev_matB;
    S* dev_matC;
    S alpha = 1.0;
    S beta = 0.0;

    gpuErrchk(cudaMalloc((void**)&dev_matA, dim_M * dim_K * sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&dev_matB, dim_K * dim_N * sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&dev_matC, dim_M * dim_N * sizeof(S)));

    // Initialize Matrix
    initialize_colnegpos_matrix<T><<<((dim_M*dim_K)+512-1)/512,512>>>(dev_matA, dim_M, dim_K, 1.0);
    initialize_colposneg_matrix<T><<<((dim_K*dim_N)+512-1)/512,512>>>(dev_matB, dim_K, dim_N, 1.0);
    initialize_matrix<S><<<((dim_M*dim_N)+512-1)/512,512>>>(dev_matC, dim_M, dim_N, 0.0);
    gpuErrchk(cudaDeviceSynchronize());

    // Set-up cuBLAS Parameter
    cudaDataType_t typeA, typeB, typeC;
    cublasComputeType_t computeType;
    cublasGemmAlgo_t algoType;

    switch(mulprecision)
    {
        case PRECISION_FP64: {typeA = CUDA_R_64F; typeB = CUDA_R_64F; break;}
        case PRECISION_FP32: {typeA = CUDA_R_32F; typeB = CUDA_R_32F; break;}
        case PRECISION_FP16: {typeA = CUDA_R_16F; typeB = CUDA_R_16F; break;}
        case PRECISION_INT8: {typeA = CUDA_R_8I;  typeB = CUDA_R_8I;  break;}
        case PRECISION_INT4: {typeA = CUDA_R_4I;  typeB = CUDA_R_4I;  break;}
    }

    switch(accprecision)
    {
        case PRECISION_FP64: {typeC = CUDA_R_64F; break;}
        case PRECISION_FP32: {typeC = CUDA_R_32F; break;}
        case PRECISION_FP16: {typeC = CUDA_R_16F; break;}
        case PRECISION_INT8: {typeC = CUDA_R_8I;  break;}
        case PRECISION_INT4: {typeC = CUDA_R_4I;  break;}
    }

    if(typeA==CUDA_R_16F && typeB==CUDA_R_16F && typeC==CUDA_R_16F)
    {
        if(tensor_cores) {computeType = CUBLAS_COMPUTE_16F;          algoType = CUBLAS_GEMM_DEFAULT_TENSOR_OP;}
        else             {computeType = CUBLAS_COMPUTE_16F_PEDANTIC; algoType = CUBLAS_GEMM_DEFAULT;}
    }
    else if(typeC==CUDA_R_32F)
    {
        
    }


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
                               typeA,                        // Data type of Matrix A
                               dim_M,                        // Leading Dimension of Matrix A
                               dev_matB,                     // Pointer to Matrix B on Device
                               typeB,                        // Data Type of Matrix B
                               dim_K,                        // Leading Dimension of Matrix B
                               &beta,                        // Scaling factor beta where (beta)xC
                               dev_matC,                     // Pointer to Matrix C on Device
                               typeC,                        // Data Type of Matrix C
                               dim_M,                        // Leading Dimension of Matrix C
                               computeType,                  // Computation Type
                               algoType                      // Computation Algorithm
        ));
    }
    cudaProfilerStop();


    if(print_result)
    {
        std::cout << "Matrix A: " << std::endl;
        view_matrix<T><<<1,1>>>(dev_matA, dim_M, dim_K);
        gpuErrchk(cudaDeviceSynchronize());
        std::cout << "Matrix B: " << std::endl;
        view_matrix<T><<<1,1>>>(dev_matB, dim_K, dim_N);
        gpuErrchk(cudaDeviceSynchronize());
        std::cout << "Matrix C: " << std::endl;
        view_matrix<S><<<1,1>>>(dev_matC, dim_M, dim_N);
        gpuErrchk(cudaDeviceSynchronize());
    }


    gpuErrchk(cudaFree(dev_matA));
    gpuErrchk(cudaFree(dev_matB));
    gpuErrchk(cudaFree(dev_matC));
    gpuErrchk(cublasDestroy(handle));
    return 0;
}