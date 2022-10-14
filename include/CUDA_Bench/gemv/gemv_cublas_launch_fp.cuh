#pragma once
#include <CUDA_Bench/util/precision_select.cuh>
#include <CUDA_Bench/util/gpucheck.cuh>
#include <CUDA_Bench/gemv/gemv_util.cuh>
#include <CUDA_Bench/gemv/gemv_global.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <nvbench/nvbench.cuh>

template<typename S, typename M, typename A>
int gemv_cublas_launch_fp()
{
    cudaEvent_t time_start, time_stop;
    // Initialize cuBLAS
	cublasHandle_t handle;	// CUBLAS context
    gpuErrchk(cublasCreate(&handle));

    // Device Memory Allocation
    M* dev_matA;
    M* dev_matB;
    A* dev_matC;
    S alpha = 1.0;
    S beta  = 0.0;

    gpuErrchk(cudaMalloc((void**)&dev_matA, gdim_M * gdim_K * sizeof(M)));
    gpuErrchk(cudaMalloc((void**)&dev_matB, gdim_K * 1 * sizeof(M)));
    gpuErrchk(cudaMalloc((void**)&dev_matC, gdim_M * 1 * sizeof(A)));

    // Initialize Matrix
    initialize_colnegpos_matrix<M><<<((gdim_M*gdim_K)+512-1)/512,512>>>(dev_matA, gdim_M, gdim_K, 1.0);
    initialize_colposneg_matrix<M><<<((gdim_K*1)+512-1)/512,512>>>(dev_matB, gdim_K, 1, 1.0);
    initialize_matrix<A><<<((gdim_M*1)+512-1)/512,512>>>(dev_matC, gdim_M, 1, 0.0);
    gpuErrchk(cudaDeviceSynchronize());

    cudaDataType_t mulDataType, accDataType;
    cublasComputeType_t computeType;
    cublasGemmAlgo_t algoType;
    cublasOperation_t matA_op;
    cublasOperation_t matB_op;

    switch(gmulprecision)
    {
        case PRECISION_FP64: {mulDataType = CUDA_R_64F; break;}
        case PRECISION_FP32: {mulDataType = CUDA_R_32F; break;}
        case PRECISION_FP16: {mulDataType = CUDA_R_16F; break;}
    }

    switch(gaccprecision)
    {
        case PRECISION_FP64: {accDataType = CUDA_R_64F; break;}
        case PRECISION_FP32: {accDataType = CUDA_R_32F; break;}
        case PRECISION_FP16: {accDataType = CUDA_R_16F; break;}
    }

    if      (mulDataType==CUDA_R_64F && accDataType==CUDA_R_64F)
    {
        if(gtensor_cores) 
        {
            // verified
            computeType = CUBLAS_COMPUTE_64F;          
            algoType    = CUBLAS_GEMM_DEFAULT_TENSOR_OP; // will fallback to CUBLAS_GEMM_DEFAULT
            matA_op     = CUBLAS_OP_N;
            matB_op     = CUBLAS_OP_N;
        } 
        else             
        {
            // verified
            computeType =CUBLAS_COMPUTE_64F_PEDANTIC;          
            algoType    =CUBLAS_GEMM_DEFAULT; 
            matA_op     = CUBLAS_OP_N;
            matB_op     = CUBLAS_OP_N;   
        }
    }
    else if (mulDataType==CUDA_R_32F && accDataType==CUDA_R_32F)
    {
        if(gtensor_cores) 
        {
            // verified
            computeType = CUBLAS_COMPUTE_32F_FAST_16F; 
            algoType    = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
            matA_op     = CUBLAS_OP_N;
            matB_op     = CUBLAS_OP_N;
            std::cout << "[WARN] Currently Tensor Cores are not supporting FP32 multiplication and accumulation, and thus lossy precision is used\n";
        }
        else             
        {
            // verified
            computeType = CUBLAS_COMPUTE_32F_PEDANTIC;          
            algoType    = CUBLAS_GEMM_DEFAULT; 
            matA_op     = CUBLAS_OP_N;
            matB_op     = CUBLAS_OP_N;      
        }
    }
    else if ((mulDataType==CUDA_R_16F) && accDataType==CUDA_R_32F)
    {
        if(gtensor_cores) 
        {
            // verified
            computeType = CUBLAS_COMPUTE_32F;          
            algoType    = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
            matA_op     = CUBLAS_OP_N;
            matB_op     = CUBLAS_OP_N;
        }    
        else             
        {
            // verified
            computeType = CUBLAS_COMPUTE_32F_PEDANTIC;          
            algoType    = CUBLAS_GEMM_DEFAULT;
            matA_op     = CUBLAS_OP_N;
            matB_op     = CUBLAS_OP_N;          
        }
    }
    else if (mulDataType==CUDA_R_16F && accDataType==CUDA_R_16F)
    {
        if(gtensor_cores) 
        {
            // verified
            computeType = CUBLAS_COMPUTE_16F;          
            algoType    = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
            matA_op     = CUBLAS_OP_N;
            matB_op     = CUBLAS_OP_N;
        }
        else             
        {   
            // verified
            computeType = CUBLAS_COMPUTE_16F_PEDANTIC;          
            algoType    = CUBLAS_GEMM_DEFAULT;
            matA_op     = CUBLAS_OP_N;
            matB_op     = CUBLAS_OP_N;     
        }
    }

    // Start Multiplication
    cudaEventCreate(&time_start);
    cudaEventCreate(&time_stop);
    cudaEventRecord(time_start,0);
    cudaProfilerStart();
    for(int iter=0;iter<gnum_iter;iter++)
    {
        gpuErrchk(cublasGemmEx(handle,                       // handle to cuBLAS library context
                               matA_op,                      // CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C
                               matB_op,                      // CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C
                               gdim_M,                       // dimension M 
                               1,                       // dimension N
                               gdim_K,                       // dimension K
                               &alpha,                       // Scaling factor alpha where (alpha)x(AxB)
                               dev_matA,                     // Pointer to Matrix A on Device
                               mulDataType,                  // Data type of Matrix A
                               gdim_M,                        // Leading Dimension of Matrix A
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
    std::cout << "[INFO] Execution Time: " << elapsedTime << "ms for " << gnum_iter << " iterations (" << elapsedTime/gnum_iter << "ms/iteration)" << std::endl;


    if(gprint_result)
    {
        std::cout << "Matrix A: " << std::endl;
        view_matrix_fp<M><<<1,1>>>(dev_matA, gdim_M, gdim_K);
        gpuErrchk(cudaDeviceSynchronize());
        std::cout << "Matrix B: " << std::endl;
        view_matrix_fp<M><<<1,1>>>(dev_matB, gdim_K, 1);
        gpuErrchk(cudaDeviceSynchronize());
        std::cout << "Matrix C: " << std::endl;
        view_matrix_fp<A><<<1,1>>>(dev_matC, gdim_M, 1);
        gpuErrchk(cudaDeviceSynchronize());
    }


    gpuErrchk(cudaFree(dev_matA));
    gpuErrchk(cudaFree(dev_matB));
    gpuErrchk(cudaFree(dev_matC));
    gpuErrchk(cublasDestroy(handle));
    return 0;
}


template<typename S, typename M, typename A>
int gemv_cublas_launch_fp(nvbench::state& state)
{
    // Initialize cuBLAS
	cublasHandle_t handle;	// CUBLAS context
    gpuErrchk(cublasCreate(&handle));

    // Device Memory Allocation
    M* dev_matA;
    M* dev_matB;
    A* dev_matC;
    S alpha = 1.0;
    S beta  = 0.0;

    gpuErrchk(cudaMalloc((void**)&dev_matA, gdim_M * gdim_K * sizeof(M)));
    gpuErrchk(cudaMalloc((void**)&dev_matB, gdim_K * 1 * sizeof(M)));
    gpuErrchk(cudaMalloc((void**)&dev_matC, gdim_M * 1 * sizeof(A)));

    // Initialize Matrix
    initialize_colnegpos_matrix<M><<<((gdim_M*gdim_K)+512-1)/512,512>>>(dev_matA, gdim_M, gdim_K, 1.0);
    initialize_colposneg_matrix<M><<<((gdim_K*1)+512-1)/512,512>>>(dev_matB, gdim_K, 1, 1.0);
    initialize_matrix<A><<<((gdim_M*1)+512-1)/512,512>>>(dev_matC, gdim_M, 1, 0.0);
    gpuErrchk(cudaDeviceSynchronize());

    cudaDataType_t mulDataType, accDataType;
    cublasComputeType_t computeType;
    cublasGemmAlgo_t algoType;
    cublasOperation_t matA_op;
    cublasOperation_t matB_op;

    switch(gmulprecision)
    {
        case PRECISION_FP64: {mulDataType = CUDA_R_64F; break;}
        case PRECISION_FP32: {mulDataType = CUDA_R_32F; break;}
        case PRECISION_FP16: {mulDataType = CUDA_R_16F; break;}
    }

    switch(gaccprecision)
    {
        case PRECISION_FP64: {accDataType = CUDA_R_64F; break;}
        case PRECISION_FP32: {accDataType = CUDA_R_32F; break;}
        case PRECISION_FP16: {accDataType = CUDA_R_16F; break;}
    }

    if      (mulDataType==CUDA_R_64F && accDataType==CUDA_R_64F)
    {
        if(gtensor_cores) 
        {
            // verified
            computeType = CUBLAS_COMPUTE_64F;          
            algoType    = CUBLAS_GEMM_DEFAULT_TENSOR_OP; // will fallback to CUBLAS_GEMM_DEFAULT
            matA_op     = CUBLAS_OP_N;
            matB_op     = CUBLAS_OP_N;
        } 
        else             
        {
            // verified
            computeType =CUBLAS_COMPUTE_64F_PEDANTIC;          
            algoType    =CUBLAS_GEMM_DEFAULT; 
            matA_op     = CUBLAS_OP_N;
            matB_op     = CUBLAS_OP_N;   
        }
    }
    else if (mulDataType==CUDA_R_32F && accDataType==CUDA_R_32F)
    {
        if(gtensor_cores) 
        {
            // verified
            computeType = CUBLAS_COMPUTE_32F_FAST_16F; 
            algoType    = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
            matA_op     = CUBLAS_OP_N;
            matB_op     = CUBLAS_OP_N;
            std::cout << "[WARN] Currently Tensor Cores are not supporting FP32 multiplication and accumulation, and thus lossy precision is used\n";
        }
        else             
        {
            // verified
            computeType = CUBLAS_COMPUTE_32F_PEDANTIC;          
            algoType    = CUBLAS_GEMM_DEFAULT; 
            matA_op     = CUBLAS_OP_N;
            matB_op     = CUBLAS_OP_N;      
        }
    }
    else if ((mulDataType==CUDA_R_16F) && accDataType==CUDA_R_32F)
    {
        if(gtensor_cores) 
        {
            // verified
            computeType = CUBLAS_COMPUTE_32F;          
            algoType    = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
            matA_op     = CUBLAS_OP_N;
            matB_op     = CUBLAS_OP_N;
        }    
        else             
        {
            // verified
            computeType = CUBLAS_COMPUTE_32F_PEDANTIC;          
            algoType    = CUBLAS_GEMM_DEFAULT;
            matA_op     = CUBLAS_OP_N;
            matB_op     = CUBLAS_OP_N;          
        }
    }
    else if (mulDataType==CUDA_R_16F && accDataType==CUDA_R_16F)
    {
        if(gtensor_cores) 
        {
            // verified
            computeType = CUBLAS_COMPUTE_16F;          
            algoType    = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
            matA_op     = CUBLAS_OP_N;
            matB_op     = CUBLAS_OP_N;
        }
        else             
        {   
            // verified
            computeType = CUBLAS_COMPUTE_16F_PEDANTIC;          
            algoType    = CUBLAS_GEMM_DEFAULT;
            matA_op     = CUBLAS_OP_N;
            matB_op     = CUBLAS_OP_N;     
        }
    }

    // Start Multiplication
    if(gprofiling) // NVBench
    {
        // Initialize Profiling Metrics
        state.collect_dram_throughput();
        state.collect_l1_hit_rates();
        state.collect_l2_hit_rates();
        state.collect_loads_efficiency();
        state.collect_stores_efficiency();
        state.collect_cupti_metrics();
        cudaProfilerStart();
        for(int iter=0;iter<gnum_iter;iter++)
        {
            state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) 
            {
                gpuErrchk(cublasGemmEx(handle,                       // handle to cuBLAS library context
                                   matA_op,                      // CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C
                                   matB_op,                      // CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C
                                   gdim_M,                       // dimension M 
                                   1,                       // dimension N
                                   gdim_K,                       // dimension K
                                   &alpha,                       // Scaling factor alpha where (alpha)x(AxB)
                                   dev_matA,                     // Pointer to Matrix A on Device
                                   mulDataType,                  // Data type of Matrix A
                                   gdim_M,                        // Leading Dimension of Matrix A
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
            });
        }
        cudaProfilerStop();
        gpuErrchk(cudaDeviceSynchronize());
    }
    else // No NVBench
    {
        cudaProfilerStart();
        for(int iter=0;iter<gnum_iter;iter++)
        {
            gpuErrchk(cublasGemmEx(handle,                       // handle to cuBLAS library context
                                   matA_op,                      // CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C
                                   matB_op,                      // CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C
                                   gdim_M,                       // dimension M 
                                   1,                       // dimension N
                                   gdim_K,                       // dimension K
                                   &alpha,                       // Scaling factor alpha where (alpha)x(AxB)
                                   dev_matA,                     // Pointer to Matrix A on Device
                                   mulDataType,                  // Data type of Matrix A
                                   gdim_M,                        // Leading Dimension of Matrix A
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
    }

    if(gprint_result)
    {
        std::cout << "Matrix A: " << std::endl;
        view_matrix_fp<M><<<1,1>>>(dev_matA, gdim_M, gdim_K);
        gpuErrchk(cudaDeviceSynchronize());
        std::cout << "Matrix B: " << std::endl;
        view_matrix_fp<M><<<1,1>>>(dev_matB, gdim_K, 1);
        gpuErrchk(cudaDeviceSynchronize());
        std::cout << "Matrix C: " << std::endl;
        view_matrix_fp<A><<<1,1>>>(dev_matC, gdim_M, 1);
        gpuErrchk(cudaDeviceSynchronize());
    }


    gpuErrchk(cudaFree(dev_matA));
    gpuErrchk(cudaFree(dev_matB));
    gpuErrchk(cudaFree(dev_matC));
    gpuErrchk(cublasDestroy(handle));
    return 0;
}

int gemv_cublas_launch_fp_double_double_double();
int gemv_cublas_launch_fp_double_double_double(nvbench::state& state);
int gemv_cublas_launch_fp_float_float_float();
int gemv_cublas_launch_fp_float_float_float(nvbench::state& state);
int gemv_cublas_launch_fp_float_half_float();
int gemv_cublas_launch_fp_float_half_float(nvbench::state& state);
int gemv_cublas_launch_fp_half_half_half();
int gemv_cublas_launch_fp_half_half_half(nvbench::state& state);


