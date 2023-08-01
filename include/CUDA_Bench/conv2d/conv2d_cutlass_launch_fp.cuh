#pragma once

#include <CUDA_Bench/util/gpucheck.cuh>
#include <CUDA_Bench/conv2d/conv2d_util.cuh>
#include <CUDA_Bench/conv2d/conv2d_global.cuh>

#include <cutlass/cutlass.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/gemm/device/gemm.h>
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include <cutlass/util/host_tensor.h>


#include <cuda_profiler_api.h>
#include <nvbench/nvbench.cuh>


template<typename ImplicitGemm, typename scalePrecision, typename mulPrecision, typename accPrecision, typename layout_matA, typename layout_matB, typename layout_matC>
int conv2d_cutlass_launch_fp()
{
    cudaEvent_t time_start, time_stop;

     // Problem Size
    cutlass::Tensor4DCoord input_size(ginput_N, ginput_H, ginput_W, ginput_C);
    cutlass::Tensor4DCoord filter_size(gfilter_K, gfilter_R, gfilter_S, ginput_C);

    // Calculate Padding and Verify Padding
    cutlass::Tensor4DCoord padding; // Should I pad?
    padding.n() = filter_size.h() / 2;
    padding.h() = filter_size.h() / 2;
    padding.w() = filter_size.w() / 2;
    padding.c() = filter_size.w() / 2;

    if ((padding.h() != filter_size.h() / 2) || (padding.w() != filter_size.w() / 2)) 
    {
      std::cout << "[ERR!] Invalid padding size" << std::endl;
      return false;
    }

    // FIR experiment
    //padding.n() = 0;
    //padding.h() = 0;
    padding.w() = filter_size.w()-1;
    padding.c() = 0;

    cutlass::MatrixCoord conv_stride(gstride_V,gstride_H);
    cutlass::MatrixCoord dilation(1,1);

    // Determining output size based on input size, filter size, padding, and stride
    cutlass::Tensor4DCoord output_size(
        input_size.n(), // output batch size depends on number of batch of the input
        (input_size.h() + padding.n() + padding.h() - filter_size.h()) / conv_stride.row() + 1,
        (input_size.w() + padding.w() + padding.c() - filter_size.w()) / conv_stride.column() + 1,
        filter_size.n() // output channel depends on number of channel of the filter.
    );
/*
    cutlass::Tensor4DCoord output_size(
        input_size.n(), // output batch size depends on number of batch of the input
        (input_size.h() + padding.n() + padding.h() - filter_size.h()) / conv_stride.row() + 1,
        (input_size.w() + padding.w() + padding.c() - filter_size.w()) / conv_stride.column() + 1,
        filter_size.n() // output channel depends on number of channel of the filter.
    );
*/
    // Allocate Tensors on Device Memory
    mulPrecision* dev_matA;
    mulPrecision* dev_matB;
    accPrecision* dev_matC;
    scalePrecision alpha = scalePrecision(1);
    scalePrecision beta  = scalePrecision(0);

    gpuErrchk(cudaMalloc((void**)&dev_matA, input_size.product() * sizeof(mulPrecision)));
    gpuErrchk(cudaMalloc((void**)&dev_matB, filter_size.product() * sizeof(mulPrecision)));
    gpuErrchk(cudaMalloc((void**)&dev_matC, output_size.product() * sizeof(accPrecision)));

    // Initialize Input Tensor with Data
    for(int batch = 0; batch<input_size.n(); batch++)
    {
        for(int channel = 0; channel<input_size.c(); channel++)
        {
            //if(channel==0){ // Test for FIR
            //T* matrix, long n_batches, long n_rows, long n_cols, long n_channels, long batch, long channel, T val
            initialize_nhwc_matrix<mulPrecision><<<((input_size.h()*input_size.w())+512-1)/512,512>>>(dev_matA, input_size.n(), input_size.h(), input_size.w(), input_size.c(), batch, channel, mulPrecision(1));
            //}
        }
    }

    // Initialize Filter Tensor with Data
    for(int batch = 0; batch<filter_size.n(); batch++)
    {
        for(int channel = 0; channel<filter_size.c(); channel++)
        {
            //if(channel==0){ // Test for FIR
            //T* matrix, long n_batches, long n_rows, long n_cols, long n_channels, long batch, long channel, T val
            initialize_nhwc_matrix<mulPrecision><<<((filter_size.h()*filter_size.w())+512-1)/512,512>>>(dev_matB, filter_size.n(), filter_size.h(), filter_size.w(), filter_size.c(), batch, channel, mulPrecision(1));
            //}
        }
    }



    // Tensor Ref for Cutlass
    cutlass::TensorRef<mulPrecision, layout_matA> dev_matA_ref(dev_matA, layout_matA::packed(input_size));
    cutlass::TensorRef<mulPrecision, layout_matB> dev_matB_ref(dev_matB, layout_matB::packed(filter_size));
    cutlass::TensorRef<accPrecision, layout_matC> dev_matC_ref(dev_matC, layout_matB::packed(output_size));

    // Prepare launch arguments and extra device memory for matrix multiplication
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;
    int split_k_slices = 1;

    // Construct Conv2dProblemSize with user defined output size
    cutlass::conv::Conv2dProblemSize problem_size(input_size,
                                                  filter_size,
                                                  padding,
                                                  conv_stride,
                                                  dilation,
                                                  output_size,
                                                  mode,
                                                  split_k_slices);

    // Construct ImplicitGemm::Argument structure with conv2d 
    // problem size, data pointers, and epilogue values
    typename ImplicitGemm::Arguments arguments{problem_size,
                                               dev_matA_ref,
                                               dev_matB_ref,
                                               dev_matC_ref,
                                               dev_matC_ref,
                                               {alpha, beta},};

    // Initialize CUTLASS Convolution
    ImplicitGemm implicit_gemm_op;

    // Allocate workspace memory
    size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Check the problem size is supported or not 
    gpuErrchk(implicit_gemm_op.can_implement(arguments));

    // Initialize CUTLASS kernel with arguments and workspace pointer
    gpuErrchk(implicit_gemm_op.initialize(arguments, workspace.get()));

    cudaEventCreate(&time_start);
    cudaEventCreate(&time_stop);
    cudaEventRecord(time_start,0);
    cudaProfilerStart();
    for(int iter=0;iter<gnum_iter;iter++)
    {
        gpuErrchk(implicit_gemm_op());
    }
    cudaProfilerStop();
    gpuErrchk(cudaDeviceSynchronize());
    cudaEventRecord(time_stop, 0);
    cudaEventSynchronize(time_stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, time_start, time_stop);
    cudaEventDestroy(time_start);
    cudaEventDestroy(time_stop);
    std::cout << "[INFO] Execution Time: " << elapsedTime << "ms for " << gnum_iter << " iterations (" << elapsedTime/gnum_iter << "ms/iteration)" << std::endl;
    
    if(gprint_result)
    {
        // Print Input Tensor
        std::cout << "Matrix A: " << std::endl;
        for(int batch = 0; batch<input_size.n(); batch++)
        {
            for(int channel = 0; channel<input_size.c(); channel++)
            {
                std::cout << "Batch: " << batch << ", Channel: " << channel << std::endl;
                view_nhwc_matrix<mulPrecision><<<1,1>>>(dev_matA, input_size.n(), input_size.h(), input_size.w(), input_size.c(), batch, channel);
                gpuErrchk(cudaDeviceSynchronize());
            }
        }
        std::cout << std::endl << std::endl;
    
        // Print Input Tensor
        std::cout << "Matrix B: " << std::endl;
        for(int batch = 0; batch<filter_size.n(); batch++)
        {
            for(int channel = 0; channel<filter_size.c(); channel++)
            {
                std::cout << "Batch: " << batch << ", Channel: " << channel << std::endl;
                view_nhwc_matrix<mulPrecision><<<1,1>>>(dev_matB, filter_size.n(), filter_size.h(), filter_size.w(), filter_size.c(), batch, channel);
                gpuErrchk(cudaDeviceSynchronize());
            }
        }
        std::cout << std::endl << std::endl;

        // Print Output Tensor
        std::cout << "Matrix C: " << std::endl;
        for(int batch = 0; batch<output_size.n(); batch++)
        {
            for(int channel = 0; channel<output_size.c(); channel++)
            {
                std::cout << "Batch: " << batch << ", Channel: " << channel << std::endl;
                view_nhwc_matrix<accPrecision><<<1,1>>>(dev_matC, output_size.n(), output_size.h(), output_size.w(), output_size.c(), batch, channel);
                gpuErrchk(cudaDeviceSynchronize());
            }
        }
        std::cout << std::endl << std::endl;
    }

    gpuErrchk(cudaFree(dev_matA));
    gpuErrchk(cudaFree(dev_matB));
    gpuErrchk(cudaFree(dev_matC));

    return 0;
}

int conv2d_cutlass_launch_ampere_fp16_fp16_fp16_tc();
int conv2d_cutlass_launch_ampere_fp16_fp16_fp16_ntc();
int conv2d_cutlass_launch_ampere_fp32_fp32_fp32_ntc();
