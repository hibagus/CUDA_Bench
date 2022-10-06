#pragma once

#include <CUDA_Bench/util/gpucheck.cuh>
#include <CUDA_Bench/gemm/gemm_util.cuh>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>

#include <cuda_profiler_api.h>

template<typename Gemm, typename scalePrecision, typename mulPrecision, typename accPrecision>
int gemm_cutlass_launch_fp(int dim_M, int dim_N, int dim_K, int num_iter, bool print_result, bool profiling)
{
    // Problem Size
    cutlass::gemm::GemmCoord problem_dim(dim_M, dim_N, dim_K);

    // Allocate Matrices on Device Memory
    mulPrecision* dev_matA;
    mulPrecision* dev_matB;
    accPrecision* dev_matC;
    scalePrecision alpha = scalePrecision(1.0);
    scalePrecision beta  = scalePrecision(0.0);

    gpuErrchk(cudaMalloc((void**)&dev_matA, dim_M * dim_K * sizeof(mulPrecision)));
    gpuErrchk(cudaMalloc((void**)&dev_matB, dim_K * dim_N * sizeof(mulPrecision)));
    gpuErrchk(cudaMalloc((void**)&dev_matC, dim_M * dim_N * sizeof(accPrecision)));

    // Fill Matrices with Dummy Data
    initialize_colposneg_matrix<mulPrecision><<<((dim_M*dim_K)+512-1)/512,512>>>(dev_matA, dim_M, dim_K, mulPrecision(1.0));
    initialize_rownegpos_matrix<mulPrecision><<<((dim_K*dim_N)+512-1)/512,512>>>(dev_matB, dim_K, dim_N, mulPrecision(1.0));
    initialize_matrix<accPrecision><<<((dim_M*dim_N)+512-1)/512,512>>>(dev_matC, dim_M, dim_N, accPrecision(0.0));
    gpuErrchk(cudaDeviceSynchronize());
    
    // Prepare launch arguments and extra device memory for matrix multiplication
    typename Gemm::Arguments arguments{problem_dim,  // <- problem size of matrix multiplication
                                       {dev_matA, dim_K},  // <- reference to matrix A on device //MxK
                                       {dev_matB, dim_K},  // <- reference to matrix B on device //KxN
                                       {dev_matC, dim_M},  // <- reference to matrix C on device //MxN
                                       {dev_matC, dim_M},  // <- reference to matrix D on device
                                       {alpha, beta}, // <- tuple of alpha and beta
                                       1};            // <- k-dimension split factor

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm_op;

    // Check the problem size is supported or not 
    gpuErrchk(gemm_op.can_implement(arguments));

    // Initialize CUTLASS kernel with arguments and workspace pointer
    gpuErrchk(gemm_op.initialize(arguments, workspace.get()));

    // Launch initialized CUTLASS kernel
    cudaProfilerStart();
    for(int iter=0;iter<num_iter;iter++)
    {
        gpuErrchk(gemm_op());
    }
    cudaProfilerStop();
    gpuErrchk(cudaDeviceSynchronize());

    if(print_result)
    {
        std::cout << "Matrix A: " << std::endl;
        view_matrix_fp<mulPrecision><<<1,1>>>(dev_matA, dim_M, dim_K);
        gpuErrchk(cudaDeviceSynchronize());
        std::cout << "Matrix B: " << std::endl;
        view_matrix_fp<mulPrecision><<<1,1>>>(dev_matB, dim_K, dim_N);
        gpuErrchk(cudaDeviceSynchronize());
        std::cout << "Matrix C: " << std::endl;
        view_matrix_fp<accPrecision><<<1,1>>>(dev_matC, dim_M, dim_N);
        gpuErrchk(cudaDeviceSynchronize());
    }

    gpuErrchk(cudaFree(dev_matA));
    gpuErrchk(cudaFree(dev_matB));
    gpuErrchk(cudaFree(dev_matC));

    return 0;
}

int gemm_cutlass_launch_volta_fp16_fp16_fp16_tc(int dim_M, int dim_N, int dim_K, int num_iter, bool print_result, bool profiling);
int gemm_cutlass_launch_volta_fp16_fp16_fp16_ntc(int dim_M, int dim_N, int dim_K, int num_iter, bool print_result, bool profiling);
int gemm_cutlass_launch_turing_fp16_fp16_fp16_tc(int dim_M, int dim_N, int dim_K, int num_iter, bool print_result, bool profiling);
int gemm_cutlass_launch_turing_fp16_fp16_fp16_ntc(int dim_M, int dim_N, int dim_K, int num_iter, bool print_result, bool profiling);
int gemm_cutlass_launch_ampere_fp16_fp16_fp16_tc(int dim_M, int dim_N, int dim_K, int num_iter, bool print_result, bool profiling);
int gemm_cutlass_launch_ampere_fp16_fp16_fp16_ntc(int dim_M, int dim_N, int dim_K, int num_iter, bool print_result, bool profiling);
int gemm_cutlass_launch_volta_fp32_fp16_fp32_tc(int dim_M, int dim_N, int dim_K, int num_iter, bool print_result, bool profiling);
int gemm_cutlass_launch_volta_fp32_fp16_fp32_ntc(int dim_M, int dim_N, int dim_K, int num_iter, bool print_result, bool profiling);
int gemm_cutlass_launch_turing_fp32_fp16_fp32_tc(int dim_M, int dim_N, int dim_K, int num_iter, bool print_result, bool profiling);
int gemm_cutlass_launch_turing_fp32_fp16_fp32_ntc(int dim_M, int dim_N, int dim_K, int num_iter, bool print_result, bool profiling);
int gemm_cutlass_launch_ampere_fp32_fp16_fp32_tc(int dim_M, int dim_N, int dim_K, int num_iter, bool print_result, bool profiling);
int gemm_cutlass_launch_ampere_fp32_fp16_fp32_ntc(int dim_M, int dim_N, int dim_K, int num_iter, bool print_result, bool profiling);

int gemm_cutlass_launch_volta_fp32_fp32_fp32_ntc(int dim_M, int dim_N, int dim_K, int num_iter, bool print_result, bool profiling);
int gemm_cutlass_launch_turing_fp32_fp32_fp32_ntc(int dim_M, int dim_N, int dim_K, int num_iter, bool print_result, bool profiling);
int gemm_cutlass_launch_ampere_fp32_fp32_fp32_ntc(int dim_M, int dim_N, int dim_K, int num_iter, bool print_result, bool profiling);

int gemm_cutlass_launch_volta_fp64_fp64_fp64_ntc(int dim_M, int dim_N, int dim_K, int num_iter, bool print_result, bool profiling);
int gemm_cutlass_launch_turing_fp64_fp64_fp64_ntc(int dim_M, int dim_N, int dim_K, int num_iter, bool print_result, bool profiling);
int gemm_cutlass_launch_ampere_fp64_fp64_fp64_ntc(int dim_M, int dim_N, int dim_K, int num_iter, bool print_result, bool profiling);
