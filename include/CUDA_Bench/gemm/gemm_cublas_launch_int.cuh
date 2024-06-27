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
    cudaEvent_t time_start, time_stop;
    // Initialize cuBLAS
	cublasHandle_t handle;	// CUBLAS context
    gpuErrchk(cublasCreate(&handle));

    // Device Memory Allocation
    M* dev_matA;
    M* dev_matB;
    A* dev_matC;
    S alpha = (S) galpha;
    S beta  = (S) gbeta;

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
        case PRECISION_INT8: {accDataType = CUDA_R_32I;  break;}
        case PRECISION_INT4: {accDataType = CUDA_R_32I;  break;}
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

    cudaEventCreate(&time_start);
    cudaEventCreate(&time_stop);
    cudaEventRecord(time_start,0);
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
    cudaEventRecord(time_stop, 0);
    cudaEventSynchronize(time_stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, time_start, time_stop);
    cudaEventDestroy(time_start);
    cudaEventDestroy(time_stop);

    long long total_flops = (long long) 2 * gdim_M * gdim_N * gdim_K;
    if (galpha != 1.0) // non-identity matrix multiplication scaling
    {
        std::cout << "[INFO] Non-Identity matrix multiplication scaling factor Alpha: " << galpha << std::endl; 
        total_flops = (long long) total_flops + gdim_M * gdim_N;
    }     
    if (gbeta != 0.0)  // non-zero matrix accumulation scaling
    {
        std::cout << "[INFO] Non-Zero matrix accumulation scaling factor Beta: " << gbeta << std::endl; 
        total_flops = (long long) total_flops + 2 * gdim_M * gdim_N;
    } 
    float gflops_per_second = (total_flops / ((elapsedTime/gnum_iter)*0.001)) / 1000000000;
    std::cout << "[INFO] Execution Time: " << elapsedTime << "ms for " << gnum_iter << " iterations (" << elapsedTime/gnum_iter << "ms/iteration)" << std::endl;
    std::cout << "[INFO] Total FLOPs per iteration: " << total_flops << std::endl;
    std::cout << "[INFO] Average Throughput: " << gflops_per_second <<" GFLOP/s" << std::endl;
    

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



int gemm_cublas_launch_fp_int8_int8_int8();