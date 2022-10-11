
#include <CUDA_Bench/util/precision_select.cuh>
#include <CUDA_Bench/util/gpucheck.cuh>
#include <CUDA_Bench/gemm/gemm_util.cuh>
#include <CUDA_Bench/gemm/gemm_global.cuh>
#include <CUDA_Bench/gemm/gemm_cublas_launch_fp.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <nvbench/nvbench.cuh>

void gemm_cublas_launch_fp_Bench(nvbench::state& state)
{
    // Initialize cuBLAS
	//cublasHandle_t handle;	// CUBLAS context
    //gpuErrchk(cublasCreate(&handle));
    printf("TEST\n");
     state.collect_dram_throughput();
  state.collect_l1_hit_rates();
  state.collect_l2_hit_rates();
  state.collect_loads_efficiency();
  state.collect_stores_efficiency();

    // Device Memory Allocation
    half* dev_matA;
    half* dev_matB;
    half* dev_matC;
    half alpha = 1.0;
    half beta  = 0.0;

    gpuErrchk(cudaMalloc((void**)&dev_matA, gdim_M * gdim_K * sizeof(half)));
    gpuErrchk(cudaMalloc((void**)&dev_matB, gdim_K * gdim_N * sizeof(half)));
    gpuErrchk(cudaMalloc((void**)&dev_matC, gdim_M * gdim_N * sizeof(half)));

    // Initialize Matrix
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) 
    {
    initialize_colnegpos_matrix<half><<<((gdim_M*gdim_K)+512-1)/512,512>>>(dev_matA, gdim_M, gdim_K, 1.0);
        });
    initialize_colposneg_matrix<half><<<((gdim_K*gdim_N)+512-1)/512,512>>>(dev_matB, gdim_K, gdim_N, 1.0);
    initialize_matrix<half><<<((gdim_M*gdim_N)+512-1)/512,512>>>(dev_matC, gdim_M, gdim_N, 0.0);
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
    //cudaProfilerStart();
       /* gpuErrchk(cublasGemmEx(handle,                       // handle to cuBLAS library context
                               matA_op,                      // CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C
                               matB_op,                      // CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C
                               gdim_M,                       // dimension M 
                               gdim_N,                       // dimension N
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
        */
    //cudaProfilerStop();


    if(gprint_result)
    {
        std::cout << "Matrix A: " << std::endl;
        view_matrix_fp<half><<<1,1>>>(dev_matA, gdim_M, gdim_K);
        gpuErrchk(cudaDeviceSynchronize());
        std::cout << "Matrix B: " << std::endl;
        view_matrix_fp<half><<<1,1>>>(dev_matB, gdim_K, gdim_N);
        gpuErrchk(cudaDeviceSynchronize());
        std::cout << "Matrix C: " << std::endl;
        view_matrix_fp<half><<<1,1>>>(dev_matC, gdim_M, gdim_N);
        gpuErrchk(cudaDeviceSynchronize());
    }


    gpuErrchk(cudaFree(dev_matA));
    gpuErrchk(cudaFree(dev_matB));
    gpuErrchk(cudaFree(dev_matC));
    //gpuErrchk(cublasDestroy(handle));
}