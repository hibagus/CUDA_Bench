#pragma once

#include <CUDA_Bench/util/gpucheck.cuh>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

int gemm_cutlass_launch_turing_int32_int8_int32_tc(int dim_M, int dim_N, int dim_K, int num_iter, bool print_result, bool profiling);
int gemm_cutlass_launch_ampere_int32_int8_int32_tc(int dim_M, int dim_N, int dim_K, int num_iter, bool print_result, bool profiling);
int gemm_cutlass_launch_turing_int32_int4_int32_tc(int dim_M, int dim_N, int dim_K, int num_iter, bool print_result, bool profiling);
int gemm_cutlass_launch_ampere_int32_int4_int32_tc(int dim_M, int dim_N, int dim_K, int num_iter, bool print_result, bool profiling);