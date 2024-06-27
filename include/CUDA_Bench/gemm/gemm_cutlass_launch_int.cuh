#pragma once

#include <CUDA_Bench/util/gpucheck.cuh>
#include <CUDA_Bench/gemm/gemm_util.cuh>
#include <CUDA_Bench/gemm/gemm_global.cuh>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>

#include <cuda_profiler_api.h>


template<typename Gemm, typename scalePrecision, typename mulPrecision, typename accPrecision>
int gemm_cutlass_launch_int()
{
    cudaEvent_t time_start, time_stop;
    // Problem Size
    cutlass::gemm::GemmCoord problem_dim(gdim_M, gdim_N, gdim_K);

    // Allocate Matrices on Device Memory
    mulPrecision* dev_matA;
    mulPrecision* dev_matB;
    accPrecision* dev_matC;
    scalePrecision alpha = scalePrecision(galpha);
    scalePrecision beta  = scalePrecision(gbeta);

    gpuErrchk(cudaMalloc((void**)&dev_matA, gdim_M * gdim_K * sizeof(mulPrecision)));
    gpuErrchk(cudaMalloc((void**)&dev_matB, gdim_K * gdim_N * sizeof(mulPrecision)));
    gpuErrchk(cudaMalloc((void**)&dev_matC, gdim_M * gdim_N * sizeof(accPrecision)));

    // Fill Matrices with Dummy Data
    initialize_colposneg_matrix<mulPrecision><<<((gdim_M*gdim_K)+512-1)/512,512>>>(dev_matA, gdim_M, gdim_K, mulPrecision(1));
    initialize_rownegpos_matrix<mulPrecision><<<((gdim_K*gdim_N)+512-1)/512,512>>>(dev_matB, gdim_K, gdim_N, mulPrecision(1));
    initialize_matrix<accPrecision><<<((gdim_M*gdim_N)+512-1)/512,512>>>(dev_matC, gdim_M, gdim_N, accPrecision(0));
    gpuErrchk(cudaDeviceSynchronize());
    
    // Prepare launch arguments and extra device memory for matrix multiplication
    typename Gemm::Arguments arguments{problem_dim,  // <- problem size of matrix multiplication
                                       {dev_matA, gdim_K},  // <- reference to matrix A on device //MxK
                                       {dev_matB, gdim_K},  // <- reference to matrix B on device //KxN
                                       {dev_matC, gdim_M},  // <- reference to matrix C on device //MxN
                                       {dev_matC, gdim_M},  // <- reference to matrix D on device
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
    cudaEventCreate(&time_start);
    cudaEventCreate(&time_stop);
    cudaEventRecord(time_start,0);
    cudaProfilerStart();
    for(int iter=0;iter<gnum_iter;iter++)
    {
        gpuErrchk(gemm_op());
    }
    cudaProfilerStop();
    gpuErrchk(cudaDeviceSynchronize());
    cudaEventRecord(time_stop, 0);
    cudaEventSynchronize(time_stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, time_start, time_stop);
    cudaEventDestroy(time_start);
    cudaEventDestroy(time_stop);

    long long total_flops = (long long) 2 * gdim_M * gdim_N * gdim_K;
    if (galpha != 1.0) // non-identity matrix multiplication scaling
    {
        std::cout << "[INFO] Non-Identity matrix multiplication scaling factor Alpha: " << galpha << std::endl; 
        total_flops = (long long) total_flops + gdim_M * gdim_N;
    }     
    if (gbeta != 0.0)  // non-zero matrix accumulation scaling
    {
        std::cout << "[INFO] Non-Zero matrix accumulation scaling factor Beta: " << gbeta << std::endl; 
        total_flops = (long long) total_flops + 2 * gdim_M * gdim_N;
    } 
    float gflops_per_second = (total_flops / ((elapsedTime/gnum_iter)*0.001)) / 1000000000;
    std::cout << "[INFO] Execution Time: " << elapsedTime << "ms for " << gnum_iter << " iterations (" << elapsedTime/gnum_iter << "ms/iteration)" << std::endl;
    std::cout << "[INFO] Total FLOPs per iteration: " << total_flops << std::endl;
    std::cout << "[INFO] Average Throughput: " << gflops_per_second <<" GFLOP/s" << std::endl;

    if(gprint_result)
    {
        std::cout << "Matrix A: " << std::endl;
        view_matrix_int<mulPrecision><<<1,1>>>(dev_matA, gdim_M, gdim_K);
        gpuErrchk(cudaDeviceSynchronize());
        std::cout << "Matrix B: " << std::endl;
        view_matrix_int<mulPrecision><<<1,1>>>(dev_matB, gdim_K, gdim_N);
        gpuErrchk(cudaDeviceSynchronize());
        std::cout << "Matrix C: " << std::endl;
        view_matrix_int<accPrecision><<<1,1>>>(dev_matC, gdim_M, gdim_N);
        gpuErrchk(cudaDeviceSynchronize());
    }

    gpuErrchk(cudaFree(dev_matA));
    gpuErrchk(cudaFree(dev_matB));
    gpuErrchk(cudaFree(dev_matC));

    return 0;
}

int gemm_cutlass_launch_volta_int32_int8_int32_ntc();

int gemm_cutlass_launch_turing_int32_int8_int32_ntc();

int gemm_cutlass_launch_turing_int32_int8_int32_tc();

int gemm_cutlass_launch_ampere_int32_int8_int32_ntc();

int gemm_cutlass_launch_ampere_int32_int8_int32_tc();

int gemm_cutlass_launch_turing_int32_int4_int32_tc();

int gemm_cutlass_launch_ampere_int32_int4_int32_tc();


