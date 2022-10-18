#include <CUDA_Bench/util/precision_select.cuh>
#include <CUDA_Bench/util/gpucheck.cuh>
#include <CUDA_Bench/fir/fir_util.cuh>
#include <CUDA_Bench/fir/fir_global.cuh>
#include <CUDA_Bench/fir/fir_cublas_launch_int.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <nvbench/nvbench.cuh>

int fir_cublas_launch_fp_int8_int8_int8()
{
    fir_cublas_launch_int<int, int8_t, int>();
    return 0;
}

int fir_cublas_launch_fp_int8_int8_int8(nvbench::state& state)
{
    fir_cublas_launch_int<int, int8_t, int>(state);
    return 0;
}


