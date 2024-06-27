// Matrix-matrix Multiplication using cuBLAS
// (C) 2022 Bagus Hanindhito

#include <CUDA_Bench/gemm/gemm_cublas.cuh>
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

int gemm_cublas()
{
    // Detect Available CUDA Devices
    int nDevices;
    gpuErrchk(cudaGetDeviceCount(&nDevices));
    print_cuda_device_info(nDevices);
    if(nDevices>0) {std::cout << "[WARN] This program does not currently support Multi-GPU run.\n";}

    if(gdevice<nDevices)
    {
        std::cout << "[INFO] Using GPU index " << gdevice << " to run the benchmark." << std::endl;
        gpuErrchk(cudaSetDevice(gdevice));
    }
    else
    {
        std::cout << "[ERR!] Invalid GPU index " << gdevice << "!" << std::endl;
        exit(1);
    }

    // Call cuBLAS Launcher
    if      (gmulprecision==PRECISION_FP64 && gaccprecision==PRECISION_FP64)
    {
        gemm_cublas_launch_fp_double_double_double();
    }
    else if (gmulprecision==PRECISION_FP32 && gaccprecision==PRECISION_FP32)
    {
        gemm_cublas_launch_fp_float_float_float();
    }
    else if (gmulprecision==PRECISION_TF32 && gaccprecision==PRECISION_TF32)
    {
        gemm_cublas_launch_fp_float_float_float();
    }
    else if (gmulprecision==PRECISION_BF16 && gaccprecision==PRECISION_BF16)
    {
        gemm_cublas_launch_fp_float_float_float();
    }
    else if ((gmulprecision==PRECISION_FP16) && gaccprecision==PRECISION_FP32)
    {
        gemm_cublas_launch_fp_float_half_float();
    }
    else if (gmulprecision==PRECISION_FP16 && gaccprecision==PRECISION_FP16)
    {
       gemm_cublas_launch_fp_half_half_half();
    }
    else if (gmulprecision==PRECISION_INT8 && gaccprecision==PRECISION_INT8)
    {
        gemm_cublas_launch_fp_int8_int8_int8();
    }
    else
    {
        std::cerr <<"[ERR!] Precision combination is not supported\n\n\n";
        std::exit(1);
    } 
    return 0;
}