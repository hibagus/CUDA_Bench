#include <CUDA_Bench/gemm/gemm_cutlass_launch_int.cuh>
#include <CUDA_Bench/gemm/gemm_util.cuh>
#include <CUDA_Bench/gemm/gemm_global.cuh>
#include <CUDA_Bench/util/gpucheck.cuh>

#include <cutlass/cutlass.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>

int gemm_cutlass_launch_volta_int32_int8_int32_ntc()
{
    // Launch cutlass for NVIDIA Volta, scale precision int32, multiplication precision int8, accumulation precision int32    
    // Declare the operation precision
    using mulPrecision   = int8_t;   // multiplication precision
    using accPrecision   = int32_t;  // accumulation precision
    using scalePrecision = int32_t;  // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::RowMajor;    layout_matA _layout_matA;
    using layout_matB = cutlass::layout::ColumnMajor; layout_matB _layout_matB;
    using layout_matC = cutlass::layout::ColumnMajor; layout_matC _layout_matC;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassSimt;             // Use CUDA Cores
    using SmArch              = cutlass::arch::Sm70;                    // Volta SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<32, 64, 32>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<1, 1, 4>;      // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombinationClamp<accPrecision, 1, accPrecision, accPrecision>;

    // Instatiate CUTLASS GEMM 
    using Gemm = cutlass::gemm::device::Gemm<mulPrecision,              // matrix A precision
                                             layout_matA,               // matrix A layout
                                             mulPrecision,              // matrix B precision
                                             layout_matB,               // matrix B layout
                                             accPrecision,              // matrix C precision
                                             layout_matC,               // matrix C layout
                                             accPrecision,              // matrix C precision (output)
                                             MMAOp,                     
                                             SmArch,
                                             ShapeMMAThreadBlock,
                                             ShapeMMAWarp,
                                             ShapeMMAOp,
                                             EpilogueOutputOp,
                                             SwizzleThreadBlock,
                                             2>;

    gemm_cutlass_launch_int<Gemm, scalePrecision, mulPrecision, accPrecision>();
    return 0;
}

int gemm_cutlass_launch_turing_int32_int8_int32_ntc()
{
    // Launch cutlass for NVIDIA Turing, scale precision int32, multiplication precision int8, accumulation precision int32    
    // Declare the operation precision
    using mulPrecision   = int8_t;   // multiplication precision
    using accPrecision   = int32_t;  // accumulation precision
    using scalePrecision = int32_t;  // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::RowMajor;    layout_matA _layout_matA;
    using layout_matB = cutlass::layout::ColumnMajor; layout_matB _layout_matB;
    using layout_matC = cutlass::layout::ColumnMajor; layout_matC _layout_matC;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassSimt;             // Use CUDA Cores
    using SmArch              = cutlass::arch::Sm75;                    // Turing SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<32, 64, 32>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<1, 1, 4>;      // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombinationClamp<accPrecision, 1, accPrecision, accPrecision>;

    // Instatiate CUTLASS GEMM 
    using Gemm = cutlass::gemm::device::Gemm<mulPrecision,              // matrix A precision
                                             layout_matA,               // matrix A layout
                                             mulPrecision,              // matrix B precision
                                             layout_matB,               // matrix B layout
                                             accPrecision,              // matrix C precision
                                             layout_matC,               // matrix C layout
                                             accPrecision,              // matrix C precision (output)
                                             MMAOp,                     
                                             SmArch,
                                             ShapeMMAThreadBlock,
                                             ShapeMMAWarp,
                                             ShapeMMAOp,
                                             EpilogueOutputOp,
                                             SwizzleThreadBlock,
                                             2>;

    gemm_cutlass_launch_int<Gemm, scalePrecision, mulPrecision, accPrecision>();
    return 0;
}

int gemm_cutlass_launch_turing_int32_int8_int32_tc()
{
    // Launch cutlass for NVIDIA Turing, scale precision int32, multiplication precision int8, accumulation precision int32
    // Declare the operation precision
    using mulPrecision   = int8_t;   // multiplication precision
    using accPrecision   = int32_t;  // accumulation precision
    using scalePrecision = int32_t;  // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::RowMajor;    layout_matA _layout_matA;
    using layout_matB = cutlass::layout::ColumnMajor; layout_matB _layout_matB;
    using layout_matC = cutlass::layout::ColumnMajor; layout_matC _layout_matC;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassTensorOp;         // use Tensor Cores
    using SmArch              = cutlass::arch::Sm75;                    // Turing SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 64>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<64, 64, 64>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<8, 8, 16>;    // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombinationClamp<accPrecision, 128/cutlass::sizeof_bits<accPrecision>::value, accPrecision, accPrecision>;

    // Instatiate CUTLASS GEMM 
    using Gemm = cutlass::gemm::device::Gemm<mulPrecision,              // matrix A precision
                                             layout_matA,               // matrix A layout
                                             mulPrecision,              // matrix B precision
                                             layout_matB,               // matrix B layout
                                             accPrecision,              // matrix C precision
                                             layout_matC,               // matrix C layout
                                             accPrecision,              // matrix C precision (output)
                                             MMAOp,                     
                                             SmArch,
                                             ShapeMMAThreadBlock,
                                             ShapeMMAWarp,
                                             ShapeMMAOp,
                                             EpilogueOutputOp,
                                             SwizzleThreadBlock,
                                             2>;

    gemm_cutlass_launch_int<Gemm, scalePrecision, mulPrecision, accPrecision>();
    return 0;
}

int gemm_cutlass_launch_ampere_int32_int8_int32_ntc()
{
    // Launch cutlass for NVIDIA Ampere, scale precision int32, multiplication precision int8, accumulation precision int32    
    // Declare the operation precision
    using mulPrecision   = int8_t;   // multiplication precision
    using accPrecision   = int32_t;  // accumulation precision
    using scalePrecision = int32_t;  // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::RowMajor;    layout_matA _layout_matA;
    using layout_matB = cutlass::layout::ColumnMajor; layout_matB _layout_matB;
    using layout_matC = cutlass::layout::ColumnMajor; layout_matC _layout_matC;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassSimt;             // use CUDA Cores
    using SmArch              = cutlass::arch::Sm80;                    // Ampere SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<32, 64, 32>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<1, 1, 4>;      // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombinationClamp<accPrecision, 1, accPrecision, accPrecision>;

    // Instatiate CUTLASS GEMM 
    using Gemm = cutlass::gemm::device::Gemm<mulPrecision,              // matrix A precision
                                             layout_matA,               // matrix A layout
                                             mulPrecision,              // matrix B precision
                                             layout_matB,               // matrix B layout
                                             accPrecision,              // matrix C precision
                                             layout_matC,               // matrix C layout
                                             accPrecision,              // matrix C precision (output)
                                             MMAOp,                     
                                             SmArch,
                                             ShapeMMAThreadBlock,
                                             ShapeMMAWarp,
                                             ShapeMMAOp,
                                             EpilogueOutputOp,
                                             SwizzleThreadBlock,
                                             2>;

    gemm_cutlass_launch_int<Gemm, scalePrecision, mulPrecision, accPrecision>();
    return 0;
}

int gemm_cutlass_launch_ampere_int32_int8_int32_tc()
{
    // Launch cutlass for NVIDIA Ampere, scale precision int32, multiplication precision int8, accumulation precision int32    
    // Declare the operation precision
    using mulPrecision   = int8_t;   // multiplication precision
    using accPrecision   = int32_t;  // accumulation precision
    using scalePrecision = int32_t;  // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::RowMajor;    layout_matA _layout_matA;
    using layout_matB = cutlass::layout::ColumnMajor; layout_matB _layout_matB;
    using layout_matC = cutlass::layout::ColumnMajor; layout_matC _layout_matC;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassTensorOp;         // use Tensor Cores
    using SmArch              = cutlass::arch::Sm80;                    // Ampere SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 64>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<64, 64, 64>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<16, 8, 32>;    // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombinationClamp<accPrecision, 128/cutlass::sizeof_bits<accPrecision>::value, accPrecision, accPrecision>;

    // Instatiate CUTLASS GEMM 
    using Gemm = cutlass::gemm::device::Gemm<mulPrecision,              // matrix A precision
                                             layout_matA,               // matrix A layout
                                             mulPrecision,              // matrix B precision
                                             layout_matB,               // matrix B layout
                                             accPrecision,              // matrix C precision
                                             layout_matC,               // matrix C layout
                                             accPrecision,              // matrix C precision (output)
                                             MMAOp,                     
                                             SmArch,
                                             ShapeMMAThreadBlock,
                                             ShapeMMAWarp,
                                             ShapeMMAOp,
                                             EpilogueOutputOp,
                                             SwizzleThreadBlock,
                                             3>;

    gemm_cutlass_launch_int<Gemm, scalePrecision, mulPrecision, accPrecision>();
    return 0;
}

int gemm_cutlass_launch_turing_int32_int4_int32_tc()
{
    // Launch cutlass for NVIDIA Turing, scale precision int32, multiplication precision int4, accumulation precision int32
    // Declare the operation precision
    using mulPrecision   = cutlass::int4b_t;   // multiplication precision
    using accPrecision   = int32_t;            // accumulation precision
    using scalePrecision = int32_t;            // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::RowMajor;    layout_matA _layout_matA;
    using layout_matB = cutlass::layout::ColumnMajor; layout_matB _layout_matB;
    using layout_matC = cutlass::layout::ColumnMajor; layout_matC _layout_matC;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassTensorOp;          // use Tensor Cores
    using SmArch              = cutlass::arch::Sm75;                     // Turing SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 128>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<64, 64, 128>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<8, 8, 32>;      // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombinationClamp<accPrecision, 128/cutlass::sizeof_bits<accPrecision>::value, accPrecision, accPrecision>;

    // Instatiate CUTLASS GEMM 
    using Gemm = cutlass::gemm::device::Gemm<mulPrecision,              // matrix A precision
                                             layout_matA,               // matrix A layout
                                             mulPrecision,              // matrix B precision
                                             layout_matB,               // matrix B layout
                                             accPrecision,              // matrix C precision
                                             layout_matC,               // matrix C layout
                                             accPrecision,              // matrix C precision (output)
                                             MMAOp,                     
                                             SmArch,
                                             ShapeMMAThreadBlock,
                                             ShapeMMAWarp,
                                             ShapeMMAOp,
                                             EpilogueOutputOp,
                                             SwizzleThreadBlock,
                                             2>;

    gemm_cutlass_launch_int<Gemm, scalePrecision, mulPrecision, accPrecision>();
    return 0;
}

int gemm_cutlass_launch_ampere_int32_int4_int32_tc()
{
    // Launch cutlass for NVIDIA Ampere, scale precision int32, multiplication precision int4, accumulation precision int32
    // Declare the operation precision
    using mulPrecision   = cutlass::int4b_t;   // multiplication precision
    using accPrecision   = int32_t;            // accumulation precision
    using scalePrecision = int32_t;            // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::RowMajor;    layout_matA _layout_matA;
    using layout_matB = cutlass::layout::ColumnMajor; layout_matB _layout_matB;
    using layout_matC = cutlass::layout::ColumnMajor; layout_matC _layout_matC;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassTensorOp;         // use Tensor Cores
    using SmArch              = cutlass::arch::Sm80;                    // Ampere SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 128>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<64, 64, 128>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<16, 8, 64>;    // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombinationClamp<accPrecision, 128/cutlass::sizeof_bits<accPrecision>::value, accPrecision, accPrecision>;

    // Instatiate CUTLASS GEMM 
    using Gemm = cutlass::gemm::device::Gemm<mulPrecision,              // matrix A precision
                                             layout_matA,               // matrix A layout
                                             mulPrecision,              // matrix B precision
                                             layout_matB,               // matrix B layout
                                             accPrecision,              // matrix C precision
                                             layout_matC,               // matrix C layout
                                             accPrecision,              // matrix C precision (output)
                                             MMAOp,                     
                                             SmArch,
                                             ShapeMMAThreadBlock,
                                             ShapeMMAWarp,
                                             ShapeMMAOp,
                                             EpilogueOutputOp,
                                             SwizzleThreadBlock,
                                             3>;
    
    gemm_cutlass_launch_int<Gemm, scalePrecision, mulPrecision, accPrecision>();
    return 0;
}

