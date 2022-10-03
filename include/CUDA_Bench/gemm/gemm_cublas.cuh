#pragma once

#include <cuda.h>
#include <cuda_fp16.h>

int gemm_cublas(int dim_M, int dim_N, int dim_K);

__global__ 
void initialize_matrix(half* matrix, int n_rows, int n_cols, half val);

__global__
void view_matrix(half* matrix, int n_rows, int n_cols);