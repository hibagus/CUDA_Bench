#pragma once

#include <cuda.h>
#include <cuda_fp16.h>

__global__ 
void initialize_matrix(half* matrix, int n_rows, int n_cols, half val);

__global__
void view_matrix(half* matrix, int n_rows, int n_cols);