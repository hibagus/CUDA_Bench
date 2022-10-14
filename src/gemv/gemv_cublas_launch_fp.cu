#include <CUDA_Bench/util/precision_select.cuh>
#include <CUDA_Bench/util/gpucheck.cuh>
#include <CUDA_Bench/gemv/gemv_util.cuh>
#include <CUDA_Bench/gemv/gemv_global.cuh>
#include <CUDA_Bench/gemv/gemv_cublas_launch_fp.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <nvbench/nvbench.cuh>

int gemv_cublas_launch_fp_double_double_double()
{
    gemv_cublas_launch_fp<double, double, double>();
    return 0;
}

int gemv_cublas_launch_fp_double_double_double(nvbench::state& state)
{
    gemv_cublas_launch_fp<double, double, double>(state);
    return 0;
}

int gemv_cublas_launch_fp_float_float_float()
{
    gemv_cublas_launch_fp<float, float, float>();
    return 0;
}

int gemv_cublas_launch_fp_float_float_float(nvbench::state& state)
{
    gemv_cublas_launch_fp<float, float, float>(state);
    return 0;
}

int gemv_cublas_launch_fp_float_half_float()
{
    gemv_cublas_launch_fp<float, half, float>();
    return 0;
}

int gemv_cublas_launch_fp_float_half_float(nvbench::state& state)
{
    gemv_cublas_launch_fp<float, half, float>(state);
    return 0;
}

int gemv_cublas_launch_fp_half_half_half()
{
    gemv_cublas_launch_fp<half, half, half>();
    return 0;
}

int gemv_cublas_launch_fp_half_half_half(nvbench::state& state)
{
    gemv_cublas_launch_fp<half, half, half>(state);
    return 0;
}