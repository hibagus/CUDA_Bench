#pragma once
#include <CUDA_Bench/util/precision_select.cuh>

int gemm_cublas(int dim_M, int dim_N, int dim_K, Precision precision, bool print_result);

