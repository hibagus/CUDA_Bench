#include <CUDA_Bench/util/precision_select.cuh>
#include <CUDA_Bench/util/gpucheck.cuh>
#include <CUDA_Bench/gemv/gemv_util.cuh>
#include <CUDA_Bench/gemv/gemv_global.cuh>
#include <CUDA_Bench/gemv/gemv_cublas_launch_int.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <nvbench/nvbench.cuh>

int gemv_cublas_launch_fp_int8_int8_int8()
{
    gemv_cublas_launch_int<int, int8_t, int>();
    return 0;
}

int gemv_cublas_launch_fp_int8_int8_int8(nvbench::state& state)
{
    gemv_cublas_launch_int<int, int8_t, int>(state);
    return 0;
}


