#pragma once
#include <CUDA_Bench/util/precision_select.cuh>
#include <CUDA_Bench/util/gpucheck.cuh>
#include <CUDA_Bench/gemm/gemm_util.cuh>
#include <CUDA_Bench/gemm/gemm_global.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>

template<typename S, typename M, typename A>
static inline int gemm_cublas_launch_int()
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

    gpuErrchk(cudaMalloc((void**)&dev_matA, gdim_M * gdim_K * sizeof(M)));
    gpuErrchk(cudaMalloc((void**)&dev_matB, gdim_K * gdim_N * sizeof(M)));
    gpuErrchk(cudaMalloc((void**)&dev_matC, gdim_M * gdim_N * sizeof(A)));

    // Initialize Matrix
    if(gtensor_cores) {initialize_rownegpos_matrix<M><<<((gdim_M*gdim_K)+512-1)/512,512>>>(dev_matA, gdim_M, gdim_K, 1);}
    else             {initialize_colnegpos_matrix<M><<<((gdim_M*gdim_K)+512-1)/512,512>>>(dev_matA, gdim_M, gdim_K, 1);}
    initialize_colposneg_matrix<M><<<((gdim_K*gdim_N)+512-1)/512,512>>>(dev_matB, gdim_K, gdim_N, 1);
    initialize_matrix<A><<<((gdim_M*gdim_N)+512-1)/512,512>>>(dev_matC, gdim_M, gdim_N, 0);
    gpuErrchk(cudaDeviceSynchronize());


    cudaDataType_t mulDataType, accDataType;
    cublasComputeType_t computeType;
    cublasGemmAlgo_t algoType;
    cublasOperation_t matA_op;
    cublasOperation_t matB_op;

    switch(gmulprecision)
    {
        case PRECISION_INT8: {mulDataType = CUDA_R_8I;  break;}
        case PRECISION_INT4: {mulDataType = CUDA_R_4I;  break;}
    }

    switch(gaccprecision)
    {
        case PRECISION_INT8: {accDataType = CUDA_R_8I;  break;}
        case PRECISION_INT4: {accDataType = CUDA_R_4I;  break;}
    }

    if (gmulprecision==PRECISION_INT8 && gaccprecision==PRECISION_INT8)
    {
        std::cout << "[WARN] Promoting accumulation precision to int32 to maintain compability\n";  
        if(gtensor_cores) 
        {
            computeType = CUBLAS_COMPUTE_32I;          
            algoType    = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
            matA_op     = CUBLAS_OP_T;
            matB_op     = CUBLAS_OP_N;
        }
        else             
        {
           
            computeType = CUBLAS_COMPUTE_32I_PEDANTIC;          
            algoType    = CUBLAS_GEMM_DEFAULT; 
            matA_op     = CUBLAS_OP_N;
            matB_op     = CUBLAS_OP_N;
        }
    }
    else
    {
        std::cerr <<"[ERR!] Precision combination is not supported\n\n\n";
        std::exit(1);
    } 

    
    cudaProfilerStart();
    for(int iter=0;iter<gnum_iter;iter++)
    {
        gpuErrchk(cublasGemmEx(handle,                       // handle to cuBLAS library context
                               matA_op,                  // CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C
                               matB_op,                  // CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C
                               gdim_M,                        // dimension M 
                               gdim_N,                        // dimension N
                               gdim_K,                        // dimension K
                               &alpha,                       // Scaling factor alpha where (alpha)x(AxB)
                               dev_matA,                     // Pointer to Matrix A on Device
                               mulDataType,                  // Data type of Matrix A
                               gdim_K,                        // Leading Dimension of Matrix A
                               dev_matB,                     // Pointer to Matrix B on Device
                               mulDataType,                  // Data Type of Matrix B
                               gdim_K,                        // Leading Dimension of Matrix B
                               &beta,                        // Scaling factor beta where (beta)xC
                               dev_matC,                     // Pointer to Matrix C on Device
                               accDataType,                  // Data Type of Matrix C
                               gdim_M,                        // Leading Dimension of Matrix C
                               computeType,                  // Computation Type
                               algoType                      // Computation Algorithm
        ));
    }
    cudaProfilerStop();
    gpuErrchk(cudaDeviceSynchronize());
    

    if(gprint_result)
    {
        std::cout << "Matrix A: " << std::endl;
        view_matrix_int<M><<<1,1>>>(dev_matA, gdim_M, gdim_K);
        gpuErrchk(cudaDeviceSynchronize());
        std::cout << "Matrix B: " << std::endl;
        view_matrix_int<M><<<1,1>>>(dev_matB, gdim_K, gdim_N);
        gpuErrchk(cudaDeviceSynchronize());
        std::cout << "Matrix C: " << std::endl;
        view_matrix_int<A><<<1,1>>>(dev_matC, gdim_M, gdim_N);
        gpuErrchk(cudaDeviceSynchronize());
    }


    gpuErrchk(cudaFree(dev_matA));
    gpuErrchk(cudaFree(dev_matB));
    gpuErrchk(cudaFree(dev_matC));
    gpuErrchk(cublasDestroy(handle));
    return 0;
}