#include <CUDA_Bench/util/precision_select.cuh>
#include <CUDA_Bench/util/gpucheck.cuh>
#include <CUDA_Bench/gemm/gemm_util.cuh>
#include <CUDA_Bench/gemm/gemm_global.cuh>
#include <CUDA_Bench/gemm/gemm_cublas_launch_int.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>

int gemm_cublas_launch_fp_int8_int8_int8()
{
    gemm_cublas_launch_int<int, int8_t, int>();
    return 0;
}


