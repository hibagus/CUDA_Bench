#include <CUDA_Bench/util/precision_select.cuh>
#include <CUDA_Bench/util/gpucheck.cuh>
#include <CUDA_Bench/gemm/gemm_util.cuh>
#include <CUDA_Bench/gemm/gemm_global.cuh>
#include <CUDA_Bench/gemm/gemm_cublas_launch_fp.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>

int gemm_cublas_launch_fp_double_double_double()
{
    gemm_cublas_launch_fp<double, double, double>();
    return 0;
}

int gemm_cublas_launch_fp_float_float_float()
{
    gemm_cublas_launch_fp<float, float, float>();
    return 0;
}

int gemm_cublas_launch_fp_float_half_float()
{
    gemm_cublas_launch_fp<float, half, float>();
    return 0;
}

int gemm_cublas_launch_fp_half_half_half()
{
    gemm_cublas_launch_fp<half, half, half>();
    return 0;
}