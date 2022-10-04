// Matrix-matrix Multiplication using cuBLAS
// (C) 2022 Bagus Hanindhito

#include <CUDA_Bench/gemm/gemm_cublas.cuh>
#include <CUDA_Bench/gemm/gemm_cublas_launch_fp.cuh>
#include <CUDA_Bench/gemm/gemm_cublas_launch_int.cuh>
#include <CUDA_Bench/gemm/gemm_util.cuh>
#include <CUDA_Bench/util/gpuinfo.cuh>
#include <CUDA_Bench/util/gpucheck.cuh>
#include <CUDA_Bench/util/precision_select.cuh>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <iostream>

int gemm_cublas(int dim_M, int dim_N, int dim_K, Precision mulprecision, Precision accprecision, int num_iter, bool print_result, bool tensor_cores, bool profiling)
{
    // Detect Available CUDA Devices
    int nDevices;
    gpuErrchk(cudaGetDeviceCount(&nDevices));
    print_cuda_device_info(nDevices);
    if(nDevices>0) {std::cout << "[WARN] This program does not currently support Multi-GPU run.\n";}

    // Precision Compability Check
    cudaDataType_t mulDataType, accDataType;
    cublasComputeType_t computeType;
    cublasGemmAlgo_t algoType;

    switch(mulprecision)
    {
        case PRECISION_FP64: {mulDataType = CUDA_R_64F; break;}
        case PRECISION_FP32: {mulDataType = CUDA_R_32F; break;}
        case PRECISION_FP16: {mulDataType = CUDA_R_16F; break;}
        case PRECISION_INT8: {mulDataType = CUDA_R_8I;  break;}
        case PRECISION_INT4: {mulDataType = CUDA_R_4I;  break;}
    }

    switch(accprecision)
    {
        case PRECISION_FP64: {accDataType = CUDA_R_64F; break;}
        case PRECISION_FP32: {accDataType = CUDA_R_32F; break;}
        case PRECISION_FP16: {accDataType = CUDA_R_16F; break;}
        case PRECISION_INT8: {accDataType = CUDA_R_8I;  break;}
        case PRECISION_INT4: {accDataType = CUDA_R_4I;  break;}
    }

    // Call cuBLAS Launcher
    if      (mulDataType==CUDA_R_64F && accDataType==CUDA_R_64F)
    {
        if(tensor_cores) 
        {
            // verified
            computeType = CUBLAS_COMPUTE_64F;          
            algoType    = CUBLAS_GEMM_DEFAULT_TENSOR_OP; // will fallback to CUBLAS_GEMM_DEFAULT
            std::cout << "[WARN] Currently Tensor Cores are not supporting FP64 multiplication and accumulation\n";
            gemm_cublas_launch_fp<double, double, double>(dim_M, dim_N, dim_K, mulDataType, accDataType, computeType, algoType, num_iter, print_result, profiling);
        } 
        else             
        {
            // verified
            computeType =CUBLAS_COMPUTE_64F_PEDANTIC;          
            algoType    =CUBLAS_GEMM_DEFAULT; 
            gemm_cublas_launch_fp<double, double, double>(dim_M, dim_N, dim_K, mulDataType, accDataType, computeType, algoType, num_iter, print_result, profiling);         
        }
    }
    else if (mulDataType==CUDA_R_32F && accDataType==CUDA_R_32F)
    {
        if(tensor_cores) 
        {
            // verified
            computeType = CUBLAS_COMPUTE_32F_FAST_16F; 
            algoType    = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
            gemm_cublas_launch_fp<float, float, float>(dim_M, dim_N, dim_K, mulDataType, accDataType, computeType, algoType, num_iter, print_result, profiling);
        }
        else             
        {
            // verified
            computeType = CUBLAS_COMPUTE_32F_PEDANTIC;          
            algoType    = CUBLAS_GEMM_DEFAULT;
            gemm_cublas_launch_fp<float, float, float>(dim_M, dim_N, dim_K, mulDataType, accDataType, computeType, algoType, num_iter, print_result, profiling);          
        }
    }
    else if ((mulDataType==CUDA_R_16F) && accDataType==CUDA_R_32F)
    {
        if(tensor_cores) 
        {
            // verified
            computeType = CUBLAS_COMPUTE_32F;          
            algoType    = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
            gemm_cublas_launch_fp<float, half, float>(dim_M, dim_N, dim_K, mulDataType, accDataType, computeType, algoType, num_iter, print_result, profiling);
        }    
        else             
        {
            // verified
            computeType = CUBLAS_COMPUTE_32F_PEDANTIC;          
            algoType    = CUBLAS_GEMM_DEFAULT;          
            gemm_cublas_launch_fp<float, half, float>(dim_M, dim_N, dim_K, mulDataType, accDataType, computeType, algoType, num_iter, print_result, profiling);
        }
    }
    else if (mulDataType==CUDA_R_16F && accDataType==CUDA_R_16F)
    {
        if(tensor_cores) 
        {
            // verified
            computeType = CUBLAS_COMPUTE_16F;          
            algoType    = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
            gemm_cublas_launch_fp<half, half, half>(dim_M, dim_N, dim_K, mulDataType, accDataType, computeType, algoType, num_iter, print_result, profiling);
        }
        else             
        {   
            // verified
            computeType = CUBLAS_COMPUTE_16F_PEDANTIC;          
            algoType    = CUBLAS_GEMM_DEFAULT;     
            gemm_cublas_launch_fp<half, half, half>(dim_M, dim_N, dim_K, mulDataType, accDataType, computeType, algoType, num_iter, print_result, profiling);     
        }
    }
    else if (mulDataType==CUDA_R_8I && accDataType==CUDA_R_8I)
    {
        if(tensor_cores) 
        {
            computeType = CUBLAS_COMPUTE_32I;          
            algoType    = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
            std::cout << "[WARN] Promoting accumulation precision to int32 to maintain compability\n";
            gemm_cublas_launch_int<int, int8_t, int>(dim_M, dim_N, dim_K, mulDataType, CUDA_R_32I, computeType, algoType, num_iter, print_result, tensor_cores, profiling);
        }
        else             
        {
            std::cout << "[WARN] Promoting accumulation precision to int32 to maintain compability\n";  
            computeType = CUBLAS_COMPUTE_32I_PEDANTIC;          
            algoType    = CUBLAS_GEMM_DEFAULT; 
            gemm_cublas_launch_int<int, int8_t, int>(dim_M, dim_N, dim_K, mulDataType, CUDA_R_32I, computeType, algoType, num_iter, print_result, tensor_cores, profiling);     
        }
    }
    else
    {
        std::cerr <<"[ERR!] Precision combination is not supported\n\n\n";
        std::exit(1);
    } 
    return 0;
}