#include <CUDA_Bench/gemm/gemm_cutlass_launch_int.cuh>
#include <CUDA_Bench/gemm/gemm_util.cuh>
#include <CUDA_Bench/gemm/gemm_global.cuh>
#include <CUDA_Bench/util/gpucheck.cuh>

#include <cutlass/cutlass.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>

#include <nvbench/nvbench.cuh>

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

int gemm_cutlass_launch_volta_int32_int8_int32_ntc(nvbench::state& state)
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

    gemm_cutlass_launch_int<Gemm, scalePrecision, mulPrecision, accPrecision>(state);
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

int gemm_cutlass_launch_turing_int32_int8_int32_ntc(nvbench::state& state)
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

    gemm_cutlass_launch_int<Gemm, scalePrecision, mulPrecision, accPrecision>(state);
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

int gemm_cutlass_launch_turing_int32_int8_int32_tc(nvbench::state& state)
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

    gemm_cutlass_launch_int<Gemm, scalePrecision, mulPrecision, accPrecision>(state);
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

int gemm_cutlass_launch_ampere_int32_int8_int32_ntc(nvbench::state& state)
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

    gemm_cutlass_launch_int<Gemm, scalePrecision, mulPrecision, accPrecision>(state);
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

int gemm_cutlass_launch_ampere_int32_int8_int32_tc(nvbench::state& state)
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

    gemm_cutlass_launch_int<Gemm, scalePrecision, mulPrecision, accPrecision>(state);
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

int gemm_cutlass_launch_turing_int32_int4_int32_tc(nvbench::state& state)
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

    gemm_cutlass_launch_int<Gemm, scalePrecision, mulPrecision, accPrecision>(state);
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

int gemm_cutlass_launch_ampere_int32_int4_int32_tc(nvbench::state& state)
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
    gemm_cutlass_launch_int<Gemm, scalePrecision, mulPrecision, accPrecision>(state);
    return 0;
}

/*
int gemm_cutlass_launch_int32_int8_int32(int dim_M, int dim_N, int dim_K, int num_iter, bool print_result, bool tensor_cores, bool profiling)
{
    // The code section below describes datatype for input, output matrices and computation between
    // elements in input matrices.
    using ElementAccumulator = int32_t;                 // <- data type of accumulator
    using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
    using ElementInputA = int8_t;                       // <- data type of elements in input matrix A
    using ElementInputB = int8_t;                       // <- data type of elements in input matrix B
    using ElementOutput = int32_t;                      // <- data type of elements in output matrix D
    
    // The code section below describes matrix layout of input and output matrices. Column Major for
    // Matrix A, Row Major for Matrix B and Row Major for Matrix C
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;
    
    // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
    using MMAOp = cutlass::arch::OpClassTensorOp;
    
    // This code section describes CUDA SM architecture number
    using SmArch = cutlass::arch::Sm75;
    
    // This code section describes the tile size a thread block will compute
    using ShapeMMAThreadBlock =
        cutlass::gemm::GemmShape<128, 256, 64>;  // <- threadblock tile M = 128, N = 256, K = 64
    // This code section describes tile size a warp will compute
    using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 64>;  // <- warp tile M = 64, N = 64, K = 64 
    // This code section describes the size of MMA op
    using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 16>;  // <- MMA Op tile M = 8, N = 8, K = 16
    
    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??
    
    // This code section describes the epilogue part of the kernel
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                     // <- data type of output matrix
        128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                           // memory access. For a byte, it's 16
                                                           // elements. This becomes the vector width of
                                                           // math instructions in the epilogue too
        ElementAccumulator,                                // <- data type of accumulator
        ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function
    
    // Number of pipelines you want to use
    constexpr int NumStages = 2;
    
    using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                             LayoutInputA,
                                             ElementInputB,
                                             LayoutInputB,
                                             ElementOutput,
                                             LayoutOutput,
                                             ElementAccumulator,
                                             MMAOp,
                                             SmArch,
                                             ShapeMMAThreadBlock,
                                             ShapeMMAWarp,
                                             ShapeMMAOp,
                                             EpilogueOp,
                                             SwizzleThreadBlock,
                                             NumStages>;

    const int length_m = 1024;
    const int length_n = 1024;
    const int length_k = 1024;

    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

    // Initialize tensors using CUTLASS helper functions
    cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
        problem_size.mk());  // <- Create matrix A with dimensions M x K
    cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
        problem_size.kn());  // <- Create matrix B with dimensions K x N
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
        problem_size.mn());  // <- Create matrix C with dimensions M x N
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
        problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                             // CUTLASS kernel
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
        problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                             // reference kernel

    // Fill input and output matrices on host using CUTLASS helper functions
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_a.host_view(),
        1,
        ElementInputA(4),
        ElementInputA(-4),
        0);  // <- Fill matrix A on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_b.host_view(),
        1,
        ElementInputB(4),
        ElementInputB(-4),
        0);  // <- Fill matrix B on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_c.host_view(),
        1,
        ElementOutput(4),
        ElementOutput(-4),
        0);  // <- Fill matrix C on host with uniform-distribution random data
    cutlass::reference::host::TensorFill(
        tensor_d.host_view());  // <- fill matrix D on host with zeros
    cutlass::reference::host::TensorFill(
        tensor_ref_d.host_view());  // <- fill matrix D for reference on host with zeros

    // Copy data from host to GPU
    tensor_a.sync_device();
    tensor_b.sync_device();
    tensor_c.sync_device();
    tensor_d.sync_device();
    tensor_ref_d.sync_device();

    // Initialize alpha and beta for dot product computation
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0);

    // Split K dimension into 1 partitions
    int split_k_slices = 1;

    // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
    // instantiated CUTLASS kernel
    typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                       tensor_a.device_ref(),  // <- reference to matrix A on device
                                       tensor_b.device_ref(),  // <- reference to matrix B on device
                                       tensor_c.device_ref(),  // <- reference to matrix C on device
                                       tensor_d.device_ref(),  // <- reference to matrix D on device
                                       {alpha, beta},          // <- tuple of alpha and beta
                                       split_k_slices};        // <- k-dimension split factor

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm_op;

    // Check the problem size is supported or not 
    cutlass::Status status = gemm_op.can_implement(arguments);
    gpuErrchk(status);

    // Initialize CUTLASS kernel with arguments and workspace pointer
    status = gemm_op.initialize(arguments, workspace.get());
    gpuErrchk(status);

    // Launch initialized CUTLASS kernel
    status = gemm_op();
    gpuErrchk(status);

    // Create instantiation for device reference gemm kernel
    cutlass::reference::device::Gemm<ElementInputA,
                                     LayoutInputA,
                                     ElementInputB,
                                     LayoutInputB,
                                     ElementOutput,
                                     LayoutOutput,
                                     ElementComputeEpilogue,
                                     ElementComputeEpilogue>
        gemm_device;

    // Launch device reference gemm kernel
    gemm_device(problem_size,
                alpha,
                tensor_a.device_ref(),
                tensor_b.device_ref(),
                beta,
                tensor_c.device_ref(),
                tensor_ref_d.device_ref());

    // Wait for kernels to finish
    cudaDeviceSynchronize();

    // Copy output data from CUTLASS and reference kernel to host for comparison
    tensor_d.sync_host();
    tensor_ref_d.sync_host();

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool passed = cutlass::reference::host::TensorEquals(
      tensor_d.host_view(),
      tensor_ref_d.host_view());

    std::cout << (passed ? "Passed" : "Failed") << std::endl;                                         
    return 0;
}

int gemm_cutlass_launch_int32_int4_int32(int dim_M, int dim_N, int dim_K, int num_iter, bool print_result, bool tensor_cores, bool profiling)
{
    using ElementOutput      = int32_t;
    using ElementAccumulator = int32_t;
    using ElementCompute     = int32_t;

    // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
    using MMAOp = cutlass::arch::OpClassTensorOp;
    
    // This code section describes CUDA SM architecture number
    using SmArch = cutlass::arch::Sm80;

    // This code section describes the tile size a thread block will compute
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 128>;  // <- threadblock tile M = 128, N = 256, K = 64
    // This code section describes tile size a warp will compute
    using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 128>;           // <- warp tile M = 64, N = 64, K = 64 
    // This code section describes the size of MMA op
    using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 64>;               // <- MMA Op tile M = 8, N = 8, K = 16

    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    // This code section describes the epilogue part of the kernel
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<  ElementOutput,                                     // <- data type of output matrix
                                                                      128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                                                                                         // memory access. For a byte, it's 16
                                                                                                                         // elements. This becomes the vector width of
                                                                                                                         // math instructions in the epilogue too
                                                                      int32_t,                                // <- data type of accumulator
                                                                      int32_t>;                                   // <- data type for alpha/beta in linear combination function

    // Instantiate GEMM Operator
    using Gemm = cutlass::gemm::device::Gemm<cutlass::int4b_t,
                                             cutlass::layout::RowMajor,
                                             cutlass::int4b_t,
                                             cutlass::layout::ColumnMajor,
                                             ElementOutput,
                                             cutlass::layout::ColumnMajor,
                                             ElementAccumulator,
                                             MMAOp,
                                             SmArch,
                                             ShapeMMAThreadBlock,
                                             ShapeMMAWarp,
                                             ShapeMMAOp,
                                             EpilogueOp,
                                             SwizzleThreadBlock,
                                             3>;

    const int length_m = 1024;
    const int length_n = 1024;
    const int length_k = 1024;

    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

    // Initialize tensors using CUTLASS helper functions
    cutlass::HostTensor<cutlass::int4b_t, cutlass::layout::RowMajor> tensor_a(
        problem_size.mk());  // <- Create matrix A with dimensions M x K
    cutlass::HostTensor<cutlass::int4b_t, cutlass::layout::ColumnMajor> tensor_b(
        problem_size.kn());  // <- Create matrix B with dimensions K x N
    cutlass::HostTensor<ElementOutput, cutlass::layout::ColumnMajor> tensor_c(
        problem_size.mn());  // <- Create matrix C with dimensions M x N
    cutlass::HostTensor<ElementOutput, cutlass::layout::ColumnMajor> tensor_d(
        problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                             // CUTLASS kernel
    cutlass::HostTensor<ElementOutput, cutlass::layout::ColumnMajor> tensor_ref_d(
        problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                             // reference kernel
    // Fill input and output matrices on host using CUTLASS helper functions
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_a.host_view(),
        1,
        1,
        -1,
        0);  // <- Fill matrix A on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_b.host_view(),
        1,
        1,
        -1,
        0);  // <- Fill matrix B on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_c.host_view(),
        1,
        0,
        0,
        0);  // <- Fill matrix C on host with uniform-distribution random data
    cutlass::reference::host::TensorFill(
        tensor_d.host_view());  // <- fill matrix D on host with zeros
    cutlass::reference::host::TensorFill(
        tensor_ref_d.host_view());  // <- fill matrix D for reference on host with zeros

    // Copy data from host to GPU
    tensor_a.sync_device();
    tensor_b.sync_device();
    tensor_c.sync_device();
    tensor_d.sync_device();
    tensor_ref_d.sync_device();

    // Initialize alpha and beta for dot product computation
    ElementCompute alpha= 1;
    ElementCompute beta = 0;

    // Split K dimension into 1 partitions
    int split_k_slices = 1;

    // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
    // instantiated CUTLASS kernel
    typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                       tensor_a.device_ref(),  // <- reference to matrix A on device
                                       tensor_b.device_ref(),  // <- reference to matrix B on device
                                       tensor_c.device_ref(),  // <- reference to matrix C on device
                                       tensor_d.device_ref(),  // <- reference to matrix D on device
                                       {alpha, beta},          // <- tuple of alpha and beta
                                       split_k_slices};        // <- k-dimension split factor

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm_op;

    // Check the problem size is supported or not 
    cutlass::Status status = gemm_op.can_implement(arguments);
    gpuErrchk(status);

    // Initialize CUTLASS kernel with arguments and workspace pointer
    status = gemm_op.initialize(arguments, workspace.get());
    gpuErrchk(status);

    // Launch initialized CUTLASS kernel
    status = gemm_op();
    gpuErrchk(status);

    // Create instantiation for device reference gemm kernel
    cutlass::reference::device::Gemm<cutlass::int4b_t,
                                     cutlass::layout::RowMajor,
                                     cutlass::int4b_t,
                                     cutlass::layout::ColumnMajor,
                                     ElementOutput,
                                     cutlass::layout::ColumnMajor,
                                     ElementCompute,
                                     ElementCompute>
        gemm_device;

    // Launch device reference gemm kernel
    gemm_device(problem_size,
                alpha,
                tensor_a.device_ref(),
                tensor_b.device_ref(),
                beta,
                tensor_c.device_ref(),
                tensor_ref_d.device_ref());

    // Wait for kernels to finish
    cudaDeviceSynchronize();

    // Copy output data from CUTLASS and reference kernel to host for comparison
    tensor_d.sync_host();
    tensor_ref_d.sync_host();

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool passed = cutlass::reference::host::TensorEquals(
      tensor_d.host_view(),
      tensor_ref_d.host_view());

    std::cout << (passed ? "Passed" : "Failed") << std::endl;         
                         
    return 0;

}
*/


/*
// Launch cutlass for NVIDIA Ampere, scale precision int32, multiplication precision int8, accumulation precision int32
    cutlass::gemm::GemmCoord problem_dim(dim_M, dim_N, dim_K);
    
    // Declare the operation precision
    using mulPrecision   = int8_t;   // multiplication precision
    using accPrecision   = int32_t;  // accumulation precision
    using scalePrecision = int32_t;  // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::RowMajor;
    using layout_matB = cutlass::layout::ColumnMajor;
    using layout_matC = cutlass::layout::ColumnMajor; 

    // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
    using MMAOp = cutlass::arch::OpClassTensorOp;
    
    // This code section describes CUDA SM architecture number
    using SmArch = cutlass::arch::Sm80;

    // This code section describes the tile size a thread block will compute
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 128>;  // <- threadblock tile M = 128, N = 256, K = 64
    // This code section describes tile size a warp will compute
    using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 128>;           // <- warp tile M = 64, N = 64, K = 64 
    // This code section describes the size of MMA op
    using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 64>;               // <- MMA Op tile M = 8, N = 8, K = 16

    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    // This code section describes the epilogue part of the kernel
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<  accPrecision,                                     // <- data type of output matrix
                                                                      128 / cutlass::sizeof_bits<accPrecision>::value,  // <- the number of elements per vectorized
                                                                                                                         // memory access. For a byte, it's 16
                                                                                                                         // elements. This becomes the vector width of
                                                                                                                         // math instructions in the epilogue too
                                                                      int32_t,                                // <- data type of accumulator
                                                                      int32_t>;                                   // <- data type for alpha/beta in linear combination function

    // Instantiate GEMM Operator
    using Gemm = cutlass::gemm::device::Gemm<cutlass::int4b_t,
                                             layout_matA,
                                             cutlass::int4b_t,
                                             layout_matB,
                                             accPrecision,
                                             layout_matC,
                                             accPrecision,
                                             MMAOp,
                                             SmArch,
                                             ShapeMMAThreadBlock,
                                             ShapeMMAWarp,
                                             ShapeMMAOp,
                                             EpilogueOp,
                                             SwizzleThreadBlock,
                                             3>;

    const int length_m = 1024;
    const int length_n = 1024;
    const int length_k = 1024;

    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

    // Initialize tensors using CUTLASS helper functions
    cutlass::HostTensor<cutlass::int4b_t, layout_matA> tensor_a(
        problem_size.mk());  // <- Create matrix A with dimensions M x K
    cutlass::HostTensor<cutlass::int4b_t, layout_matB> tensor_b(
        problem_size.kn());  // <- Create matrix B with dimensions K x N
    cutlass::HostTensor<accPrecision, layout_matC> tensor_c(
        problem_size.mn());  // <- Create matrix C with dimensions M x N
    cutlass::HostTensor<accPrecision, layout_matC> tensor_d(
        problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                             // CUTLASS kernel
    cutlass::HostTensor<accPrecision, layout_matC> tensor_ref_d(
        problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                             // reference kernel
    // Fill input and output matrices on host using CUTLASS helper functions
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_a.host_view(),
        1,
        1,
        -1,
        0);  // <- Fill matrix A on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_b.host_view(),
        1,
        1,
        -1,
        0);  // <- Fill matrix B on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_c.host_view(),
        1,
        0,
        0,
        0);  // <- Fill matrix C on host with uniform-distribution random data
    cutlass::reference::host::TensorFill(
        tensor_d.host_view());  // <- fill matrix D on host with zeros
    cutlass::reference::host::TensorFill(
        tensor_ref_d.host_view());  // <- fill matrix D for reference on host with zeros

    // Copy data from host to GPU
    tensor_a.sync_device();
    tensor_b.sync_device();
    tensor_c.sync_device();
    tensor_d.sync_device();
    tensor_ref_d.sync_device();

    // Initialize alpha and beta for dot product computation
    scalePrecision alpha= 1;
    scalePrecision beta = 0;

    // Split K dimension into 1 partitions
    int split_k_slices = 1;

    // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
    // instantiated CUTLASS kernel
    typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                       tensor_a.device_ref(),  // <- reference to matrix A on device
                                       tensor_b.device_ref(),  // <- reference to matrix B on device
                                       tensor_c.device_ref(),  // <- reference to matrix C on device
                                       tensor_d.device_ref(),  // <- reference to matrix D on device
                                       {alpha, beta},          // <- tuple of alpha and beta
                                       split_k_slices};        // <- k-dimension split factor

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm_op;

    // Check the problem size is supported or not 
    cutlass::Status status = gemm_op.can_implement(arguments);
    gpuErrchk(status);

    // Initialize CUTLASS kernel with arguments and workspace pointer
    status = gemm_op.initialize(arguments, workspace.get());
    gpuErrchk(status);

    // Launch initialized CUTLASS kernel
    status = gemm_op();
    gpuErrchk(status);

    cudaDeviceSynchronize();

    // Copy output data from CUTLASS and reference kernel to host for comparison
    tensor_d.sync_host();
    tensor_ref_d.sync_host();

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool passed = cutlass::reference::host::TensorEquals(
      tensor_d.host_view(),
      tensor_ref_d.host_view());

    std::cout << (passed ? "Passed" : "Failed") << std::endl;         
                         
    return 0;    
*/

/*
// Launch cutlass for NVIDIA Ampere, scale precision int32, multiplication precision int8, accumulation precision int32
    cutlass::gemm::GemmCoord problem_dim(dim_M, dim_N, dim_K);
    
    // Declare the operation precision
    using mulPrecision   = int8_t;   // multiplication precision
    using accPrecision   = int32_t;  // accumulation precision
    using scalePrecision = int32_t;  // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::RowMajor;
    using layout_matB = cutlass::layout::RowMajor;
    using layout_matC = cutlass::layout::RowMajor;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassTensorOp;         // use Tensor Cores
    using SmArch              = cutlass::arch::Sm80;                    // Ampere SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 64>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<64, 64, 64>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<16, 8, 32>;    // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombinationClamp<accPrecision, 128/cutlass::sizeof_bits<accPrecision>::value, accPrecision, accPrecision>;

    constexpr int NumStages = 3;
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
                                             NumStages>;

    // Allocate Matrices on Host Memory
    cutlass::HostTensor<mulPrecision, layout_matA> matA(problem_dim.mk());  // <- Create matrix A with dimensions M x K
    cutlass::HostTensor<mulPrecision, layout_matB> matB(problem_dim.kn());  // <- Create matrix B with dimensions K x N
    cutlass::HostTensor<accPrecision, layout_matC> matC(problem_dim.mn());  // <- Create matrix C with dimensions M x N
    scalePrecision alpha = scalePrecision(1);
    scalePrecision beta  = scalePrecision(0);

    // Fill Matrices with Dummy Data
    cutlass::reference::host::TensorFillRandomUniform(matA.host_view(), 1, mulPrecision(1), mulPrecision(1), 0); // Fill with random
    cutlass::reference::host::TensorFillIdentity(matB.host_view());          // Initialize identity Matrices
    cutlass::reference::host::TensorFill(matC.host_view(), accPrecision(0)); // Zero Initialization for C

    // Copy data from host to GPU
    matA.sync_device();
    matB.sync_device();
    matC.sync_device();

    int split_k_slices = 1;
    
    // Prepare launch arguments and extra memory for matrix multiplication computation
    typename Gemm::Arguments arguments{problem_dim,  // <- problem size of matrix multiplication
                                       matA.device_ref(),  // <- reference to matrix A on device
                                       matB.device_ref(),  // <- reference to matrix B on device
                                       matC.device_ref(),  // <- reference to matrix C on device
                                       matC.device_ref(),  // <- reference to matrix D on device
                                       {alpha, beta}, // <- tuple of alpha and beta
                                       split_k_slices};            // <- k-dimension split factor

    //size_t workspace_size = Gemm::get_workspace_size(arguments);
    //cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Instantiate CUTLASS kernel depending on templates
    //Gemm gemm_op;

    // Check the problem size is supported or not 
    //gpuErrchk(gemm_op.can_implement(arguments));

    // Initialize CUTLASS kernel with arguments and workspace pointer
    //gpuErrchk(gemm_op.initialize(arguments, workspace.get()));

    // Launch initialized CUTLASS kernel
    //gpuErrchk(gemm_op());

    //gpuErrchk(cudaDeviceSynchronize());

    // Copy back the result
    //matC.sync_host();

    // Verify the result
    //bool passed = cutlass::reference::host::TensorEquals(matC.host_view(), matA.host_view());
    //
    //std::cout << (passed ? "[INFO] Both matrices are equal, test passed." : "[ERR!] Both matrices are not matched, test failed") << std:endl;
*/

/*
    // Launch cutlass for NVIDIA Ampere, scale precision int32, multiplication precision int8, accumulation precision int32
    cutlass::gemm::GemmCoord problem_dim(dim_M, dim_N, dim_K);
    
    // Declare the operation precision
    using mulPrecision   = int8_t;   // multiplication precision
    using accPrecision   = int32_t;  // accumulation precision
    using scalePrecision = int32_t;  // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::RowMajor; layout_matA _layout_matA;
    using layout_matB = cutlass::layout::RowMajor; layout_matB _layout_matB;
    using layout_matC = cutlass::layout::RowMajor; layout_matC _layout_matC;

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

    // Allocate Matrices on Device Memory
    mulPrecision* dev_matA;
    mulPrecision* dev_matB;
    accPrecision* dev_matC;
    scalePrecision alpha = scalePrecision(1);
    scalePrecision beta  = scalePrecision(0);

    gpuErrchk(cudaMalloc((void**)&dev_matA, dim_M * dim_K * sizeof(mulPrecision)));
    gpuErrchk(cudaMalloc((void**)&dev_matB, dim_K * dim_N * sizeof(mulPrecision)));
    gpuErrchk(cudaMalloc((void**)&dev_matC, dim_M * dim_N * sizeof(accPrecision)));

    // Get Pointer to Tensor on Device Memory and wrap it on TensorRef and TensorView
    cutlass::TensorRef<mulPrecision, layout_matA> ref_dev_matA (dev_matA, _layout_matA);
    cutlass::TensorRef<mulPrecision, layout_matB> ref_dev_matB (dev_matB, _layout_matB);
    cutlass::TensorRef<accPrecision, layout_matC> ref_dev_matC (dev_matC, _layout_matC);

    cutlass::TensorView<mulPrecision, layout_matA> view_dev_matA (ref_dev_matA, problem_dim.mk());
    cutlass::TensorView<mulPrecision, layout_matB> view_dev_matB (ref_dev_matB, problem_dim.kn());
    cutlass::TensorView<accPrecision, layout_matC> view_dev_matC (ref_dev_matC, problem_dim.mn());

    // Fill Matrices with Dummy Data
    cutlass::reference::device::TensorFillRandomUniform(view_dev_matA, 1, mulPrecision(1), mulPrecision(1), 0);
    cutlass::reference::device::TensorFillIdentity(view_dev_matB);
    cutlass::reference::device::TensorFill(view_dev_matC, accPrecision(0));
    
    // Prepare launch arguments
     typename Gemm::Arguments arguments{problem_dim,  // <- problem size of matrix multiplication
                                       ref_dev_matA,  // <- reference to matrix A on device
                                       ref_dev_matB,  // <- reference to matrix B on device
                                       ref_dev_matC,  // <- reference to matrix C on device
                                       ref_dev_matC,  // <- reference to matrix D on device
                                       {alpha, beta}, // <- tuple of alpha and beta
                                       1};            // <- k-dimension split factor

    //cutlass::HostTensor<ElementInputA, LayoutInputA>


    gpuErrchk(cudaFree(dev_matA));
    gpuErrchk(cudaFree(dev_matB));
    gpuErrchk(cudaFree(dev_matC));
*/