#include <CUDA_Bench/conv2d/conv2d_cutlass_launch_fp.cuh>
#include <CUDA_Bench/conv2d/conv2d_util.cuh>
#include <CUDA_Bench/conv2d/conv2d_global.cuh>
#include <CUDA_Bench/util/gpucheck.cuh>

#include <cutlass/cutlass.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/gemm/device/gemm.h>
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include <cutlass/util/host_tensor.h>

#include <cuda_profiler_api.h>
#include <nvbench/nvbench.cuh>

int conv2d_cutlass_launch_ampere_fp16_fp16_fp16_tc()
{
    // Launch cutlass for NVIDIA Ampere, scale precision fp16, multiplication precision fp16, accumulation precision fp16
    // Declare the operation precision
    using mulPrecision   = cutlass::half_t;   // multiplication precision
    using accPrecision   = cutlass::half_t;            // accumulation precision
    using scalePrecision = cutlass::half_t;            // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::TensorNHWC;
    using layout_matB = cutlass::layout::TensorNHWC;
    using layout_matC = cutlass::layout::TensorNHWC;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassTensorOp;         // use Tensor Cores
    using SmArch              = cutlass::arch::Sm80;                    // Ampere SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 64>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<64, 64, 64>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<16, 8, 16>;    // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombination<accPrecision, 128/cutlass::sizeof_bits<accPrecision>::value, accPrecision, accPrecision>;
    
    // This code section describe iterator algorithm selected is Analytic or Optimized
    static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kOptimized;

    // Instantiate CUTLASS CONV2D
    using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<mulPrecision, 
                                                                                 layout_matA,
                                                                                 mulPrecision, 
                                                                                 layout_matB,
                                                                                 accPrecision, 
                                                                                 layout_matC,
                                                                                 accPrecision,
                                                                                 MMAOp,
                                                                                 SmArch,
                                                                                 ShapeMMAThreadBlock,
                                                                                 ShapeMMAWarp,
                                                                                 ShapeMMAOp,
                                                                                 EpilogueOutputOp,
                                                                                 SwizzleThreadBlock,
                                                                                 3,
                                                                                 cutlass::arch::OpMultiplyAdd,
                                                                                 IteratorAlgorithm>::Kernel;
    
    using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

    conv2d_cutlass_launch_fp<ImplicitGemm, scalePrecision, mulPrecision, accPrecision, layout_matA, layout_matB, layout_matC>(); 

    return 0;
}


int conv2d_cutlass_launch_ampere_fp16_fp16_fp16_ntc()
{
    // Launch cutlass for NVIDIA Ampere, scale precision fp16, multiplication precision fp16, accumulation precision fp16
    // Declare the operation precision
    using mulPrecision   = cutlass::half_t;   // multiplication precision
    using accPrecision   = cutlass::half_t;            // accumulation precision
    using scalePrecision = cutlass::half_t;            // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::TensorNHWC;
    using layout_matB = cutlass::layout::TensorNHWC;
    using layout_matC = cutlass::layout::TensorNHWC;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassSimt;         // use Tensor Cores
    using SmArch              = cutlass::arch::Sm80;                    // Ampere SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 8>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<32, 64, 8>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<1, 1, 1>;    // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombination<accPrecision, 1, accPrecision, accPrecision>;
    
    // This code section describe iterator algorithm selected is Analytic or Optimized
    static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kOptimized;

    // Instantiate CUTLASS CONV2D
    using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<mulPrecision, 
                                                                                 layout_matA,
                                                                                 mulPrecision, 
                                                                                 layout_matB,
                                                                                 accPrecision, 
                                                                                 layout_matC,
                                                                                 accPrecision,
                                                                                 MMAOp,
                                                                                 SmArch,
                                                                                 ShapeMMAThreadBlock,
                                                                                 ShapeMMAWarp,
                                                                                 ShapeMMAOp,
                                                                                 EpilogueOutputOp,
                                                                                 SwizzleThreadBlock,
                                                                                 2,
                                                                                 cutlass::arch::OpMultiplyAdd,
                                                                                 IteratorAlgorithm>::Kernel;
    
    using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

    conv2d_cutlass_launch_fp<ImplicitGemm, scalePrecision, mulPrecision, accPrecision, layout_matA, layout_matB, layout_matC>(); 

    return 0;
    
}


int conv2d_cutlass_launch_ampere_fp32_fp32_fp32_ntc()
{
    // Launch cutlass for NVIDIA Ampere, scale precision fp16, multiplication precision fp16, accumulation precision fp16
    // Declare the operation precision
    using mulPrecision   = float;   // multiplication precision
    using accPrecision   = float;            // accumulation precision
    using scalePrecision = float;            // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::TensorNHWC;
    using layout_matB = cutlass::layout::TensorNHWC;
    using layout_matC = cutlass::layout::TensorNHWC;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassSimt;         // use Tensor Cores
    using SmArch              = cutlass::arch::Sm80;                    // Ampere SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 8>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<32, 64, 8>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<1, 1, 1>;    // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombination<accPrecision, 1, accPrecision, accPrecision>;
    
    // This code section describe iterator algorithm selected is Analytic or Optimized
    static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kOptimized;

    // Instantiate CUTLASS CONV2D
    using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<mulPrecision, 
                                                                                 layout_matA,
                                                                                 mulPrecision, 
                                                                                 layout_matB,
                                                                                 accPrecision, 
                                                                                 layout_matC,
                                                                                 accPrecision,
                                                                                 MMAOp,
                                                                                 SmArch,
                                                                                 ShapeMMAThreadBlock,
                                                                                 ShapeMMAWarp,
                                                                                 ShapeMMAOp,
                                                                                 EpilogueOutputOp,
                                                                                 SwizzleThreadBlock,
                                                                                 2,
                                                                                 cutlass::arch::OpMultiplyAdd,
                                                                                 IteratorAlgorithm>::Kernel;
    
    using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

    conv2d_cutlass_launch_fp<ImplicitGemm, scalePrecision, mulPrecision, accPrecision, layout_matA, layout_matB, layout_matC>(); 

    return 0;
}