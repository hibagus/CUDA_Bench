// Matrix-matrix Multiplication using cuBLAS
// (C) 2022 Bagus Hanindhito

#include <CUDA_Bench/gemm/gemm_cublas.cuh>
#include <CUDA_Bench/gemm/gemm_cublas_bench.cuh>
#include <CUDA_Bench/gemm/gemm_cublas_launch_fp.cuh>
#include <CUDA_Bench/gemm/gemm_cublas_launch_int.cuh>
#include <CUDA_Bench/gemm/gemm_util.cuh>
#include <CUDA_Bench/gemm/gemm_global.cuh>
#include <CUDA_Bench/util/gpuinfo.cuh>
#include <CUDA_Bench/util/gpucheck.cuh>
#include <CUDA_Bench/util/precision_select.cuh>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <iostream>

#include <nvbench/nvbench.cuh>

int gemm_cublas()
{
    // Detect Available CUDA Devices
    int nDevices;
    gpuErrchk(cudaGetDeviceCount(&nDevices));
    print_cuda_device_info(nDevices);
    if(nDevices>0) {std::cout << "[WARN] This program does not currently support Multi-GPU run.\n";}

    // Call cuBLAS Launcher
    if      (gmulprecision==PRECISION_FP64 && gaccprecision==PRECISION_FP64)
    {
        gemm_cublas_launch_fp<double, double, double>();
    }
    else if (gmulprecision==PRECISION_FP32 && gaccprecision==PRECISION_FP32)
    {
        gemm_cublas_launch_fp<float, float, float>();
    }
    else if ((gmulprecision==PRECISION_FP16) && gaccprecision==PRECISION_FP32)
    {
        gemm_cublas_launch_fp<float, half, float>();
    }
    else if (gmulprecision==PRECISION_FP16 && gaccprecision==PRECISION_FP16)
    {
        if(gprofiling) 
        {
            printf("TESTbefore\n");
            NVBENCH_BENCH(gemm_cublas_launch_fp_Bench);
        }
        else
        {
            gemm_cublas_launch_fp<half, half, half>();
        }
        
    }
    else if (gmulprecision==PRECISION_INT8 && gaccprecision==PRECISION_INT8)
    {
        gemm_cublas_launch_int<int, int8_t, int>();
    }
    else
    {
        std::cerr <<"[ERR!] Precision combination is not supported\n\n\n";
        std::exit(1);
    } 
    return 0;
}