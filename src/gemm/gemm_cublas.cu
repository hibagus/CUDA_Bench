// Matrix-matrix Multiplication using cuBLAS
// (C) 2022 Bagus Hanindhito

#include <CUDA_Bench/gemm/gemm_cublas.cuh>
#include <CUDA_Bench/util/gpucheck.cuh>
#include <cuda.h>
#include <cublas_v2.h>

int gemm_cublas(int dim_M, int dim_N, int dim_K)
{
    cublasStatus_t stat;    // CUBLAS functions status
	cublasHandle_t handle;	// CUBLAS context

    gpuErrchk(cublasCreate(&handle));

}