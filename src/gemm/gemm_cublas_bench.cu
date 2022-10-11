#include <nvbench/nvbench.cuh>
#include <CUDA_Bench/gemm/gemm_cublas_bench.cuh>
#include <CUDA_Bench/gemm/gemm_global.cuh>
#include <CUDA_Bench/gemm/gemm_cublas_launch_fp.cuh>
#include <CUDA_Bench/gemm/gemm_cublas_launch_int.cuh>

void gemm_cublas_bench(nvbench::state &state)
{
  cudaStream_t default_stream = 0;
  state.set_cuda_stream(nvbench::make_cuda_stream_view(default_stream));
  state.exec([](nvbench::launch&) {
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
        gemm_cublas_launch_fp<half, half, half>();
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
  });


}