#include <CUDA_Bench/gemm/gemm_cutlass_launch_fp.cuh>
#include <CUDA_Bench/gemm/gemm_util.cuh>
#include <CUDA_Bench/util/gpucheck.cuh>

#include <cutlass/cutlass.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>

int gemm_cutlass_launch_volta_fp16_fp16_fp16_tc()
{
    // Launch cutlass for NVIDIA Volta, scale precision fp16, multiplication precision fp16, accumulation precision fp16

    // Declare the operation precision
    using mulPrecision   = cutlass::half_t;   // multiplication precision
    using accPrecision   = cutlass::half_t;   // accumulation precision
    using scalePrecision = cutlass::half_t;   // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::RowMajor;    layout_matA _layout_matA;
    using layout_matB = cutlass::layout::ColumnMajor; layout_matB _layout_matB;
    using layout_matC = cutlass::layout::ColumnMajor; layout_matC _layout_matC;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassTensorOp;         // use Tensor Cores
    using SmArch              = cutlass::arch::Sm70;                    // Volta SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<64, 64, 32>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<8, 8, 4>;    // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombination<accPrecision, 128/cutlass::sizeof_bits<accPrecision>::value, accPrecision, accPrecision>;
    
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

    gemm_cutlass_launch_fp<Gemm, scalePrecision, mulPrecision, accPrecision>();
    return 0;
}


int gemm_cutlass_launch_volta_fp16_fp16_fp16_ntc()
{
    // Launch cutlass for NVIDIA Volta, scale precision fp16, multiplication precision fp16, accumulation precision fp16

    // Declare the operation precision
    using mulPrecision   = cutlass::half_t;   // multiplication precision
    using accPrecision   = cutlass::half_t;   // accumulation precision
    using scalePrecision = cutlass::half_t;   // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::RowMajor;    layout_matA _layout_matA;
    using layout_matB = cutlass::layout::ColumnMajor; layout_matB _layout_matB;
    using layout_matC = cutlass::layout::ColumnMajor; layout_matC _layout_matC;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassSimt;         // use CUDA Cores
    using SmArch              = cutlass::arch::Sm70;                    // Volta SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 8>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<32, 64, 8>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<1, 1, 1>;    // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombination<accPrecision, 1, accPrecision, accPrecision>;
    
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

    gemm_cutlass_launch_fp<Gemm, scalePrecision, mulPrecision, accPrecision>();
    return 0;
}


int gemm_cutlass_launch_turing_fp16_fp16_fp16_tc()
{
    // Launch cutlass for NVIDIA Turing, scale precision fp16, multiplication precision fp16, accumulation precision fp16

    // Declare the operation precision
    using mulPrecision   = cutlass::half_t;   // multiplication precision
    using accPrecision   = cutlass::half_t;   // accumulation precision
    using scalePrecision = cutlass::half_t;   // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::RowMajor;    layout_matA _layout_matA;
    using layout_matB = cutlass::layout::ColumnMajor; layout_matB _layout_matB;
    using layout_matC = cutlass::layout::ColumnMajor; layout_matC _layout_matC;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassTensorOp;         // use Tensor Cores
    using SmArch              = cutlass::arch::Sm75;                    // Turing SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<64, 64, 32>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<16, 8, 8>;    // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombination<accPrecision, 128/cutlass::sizeof_bits<accPrecision>::value, accPrecision, accPrecision>;
    
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

    gemm_cutlass_launch_fp<Gemm, scalePrecision, mulPrecision, accPrecision>();
    return 0;
}


int gemm_cutlass_launch_turing_fp16_fp16_fp16_ntc()
{
    // Launch cutlass for NVIDIA Turing, scale precision fp16, multiplication precision fp16, accumulation precision fp16

    // Declare the operation precision
    using mulPrecision   = cutlass::half_t;   // multiplication precision
    using accPrecision   = cutlass::half_t;   // accumulation precision
    using scalePrecision = cutlass::half_t;   // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::RowMajor;    layout_matA _layout_matA;
    using layout_matB = cutlass::layout::ColumnMajor; layout_matB _layout_matB;
    using layout_matC = cutlass::layout::ColumnMajor; layout_matC _layout_matC;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassSimt;         // use CUDA Cores
    using SmArch              = cutlass::arch::Sm75;                    // Turing SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 8>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<32, 64, 8>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<1, 1, 1>;    // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombination<accPrecision, 1, accPrecision, accPrecision>;
    
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

    gemm_cutlass_launch_fp<Gemm, scalePrecision, mulPrecision, accPrecision>();
    return 0;
}


int gemm_cutlass_launch_ampere_fp16_fp16_fp16_tc()
{
    // Launch cutlass for NVIDIA Ampere, scale precision fp16, multiplication precision fp16, accumulation precision fp16

    // Declare the operation precision
    using mulPrecision   = cutlass::half_t;   // multiplication precision
    using accPrecision   = cutlass::half_t;   // accumulation precision
    using scalePrecision = cutlass::half_t;   // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::RowMajor;    layout_matA _layout_matA;
    using layout_matB = cutlass::layout::ColumnMajor; layout_matB _layout_matB;
    using layout_matC = cutlass::layout::ColumnMajor; layout_matC _layout_matC;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassTensorOp;         // use Tensor Cores
    using SmArch              = cutlass::arch::Sm80;                    // Ampere SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 64>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<64, 64, 64>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<16, 8, 16>;    // Instruction Shape
    //using ShapeMMAOp          = cutlass::gemm::GemmShape<16, 8, 8>;    // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombination<accPrecision, 128/cutlass::sizeof_bits<accPrecision>::value, accPrecision, accPrecision>;
    
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

    gemm_cutlass_launch_fp<Gemm, scalePrecision, mulPrecision, accPrecision>();
    return 0;
}


int gemm_cutlass_launch_ampere_fp16_fp16_fp16_ntc()
{
    // Launch cutlass for NVIDIA Ampere, scale precision fp16, multiplication precision fp16, accumulation precision fp16

    // Declare the operation precision
    using mulPrecision   = cutlass::half_t;   // multiplication precision
    using accPrecision   = cutlass::half_t;   // accumulation precision
    using scalePrecision = cutlass::half_t;   // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::RowMajor;    layout_matA _layout_matA;
    using layout_matB = cutlass::layout::ColumnMajor; layout_matB _layout_matB;
    using layout_matC = cutlass::layout::ColumnMajor; layout_matC _layout_matC;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassSimt;         // use CUDA Cores
    using SmArch              = cutlass::arch::Sm80;                    // Ampere SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 8>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<32, 64, 8>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<1, 1, 1>;    // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombination<accPrecision, 1, accPrecision, accPrecision>;
    
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

    gemm_cutlass_launch_fp<Gemm, scalePrecision, mulPrecision, accPrecision>();
    return 0;
}

int gemm_cutlass_launch_volta_fp32_fp16_fp32_tc()
{
    // Launch cutlass for NVIDIA Volta, scale precision fp32, multiplication precision fp16, accumulation precision fp32
    
    // Declare the operation precision
    using mulPrecision   = cutlass::half_t;   // multiplication precision
    using accPrecision   = float;             // accumulation precision
    using scalePrecision = float;             // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::RowMajor;    layout_matA _layout_matA;
    using layout_matB = cutlass::layout::ColumnMajor; layout_matB _layout_matB;
    using layout_matC = cutlass::layout::ColumnMajor; layout_matC _layout_matC;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassTensorOp;         // use Tensor Cores
    using SmArch              = cutlass::arch::Sm70;                    // Volta SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<64, 64, 32>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<8, 8, 4>;    // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombination<accPrecision, 128/cutlass::sizeof_bits<accPrecision>::value, accPrecision, accPrecision>;
    
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

    gemm_cutlass_launch_fp<Gemm, scalePrecision, mulPrecision, accPrecision>();
    return 0;
}


int gemm_cutlass_launch_volta_fp32_fp16_fp32_ntc()
{
    // Launch cutlass for NVIDIA Volta, scale precision fp32, multiplication precision fp16, accumulation precision fp32

    // Declare the operation precision
    using mulPrecision   = cutlass::half_t;   // multiplication precision
    using accPrecision   = float;             // accumulation precision
    using scalePrecision = float;             // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::RowMajor;    layout_matA _layout_matA;
    using layout_matB = cutlass::layout::ColumnMajor; layout_matB _layout_matB;
    using layout_matC = cutlass::layout::ColumnMajor; layout_matC _layout_matC;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassSimt;         // use CUDA Cores
    using SmArch              = cutlass::arch::Sm70;                    // Volta SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 8>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<32, 64, 8>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<1, 1, 1>;    // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombination<accPrecision, 1, accPrecision, accPrecision>;
    
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

    gemm_cutlass_launch_fp<Gemm, scalePrecision, mulPrecision, accPrecision>();
    return 0;
}

int gemm_cutlass_launch_turing_fp32_fp16_fp32_tc()
{
    // Launch cutlass for NVIDIA Turing, scale precision fp32, multiplication precision fp16, accumulation precision fp32
    
    // Declare the operation precision
    using mulPrecision   = cutlass::half_t;   // multiplication precision
    using accPrecision   = float;             // accumulation precision
    using scalePrecision = float;             // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::RowMajor;    layout_matA _layout_matA;
    using layout_matB = cutlass::layout::ColumnMajor; layout_matB _layout_matB;
    using layout_matC = cutlass::layout::ColumnMajor; layout_matC _layout_matC;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassTensorOp;         // use Tensor Cores
    using SmArch              = cutlass::arch::Sm75;                    // Turing SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<64, 64, 32>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<16, 8, 8>;    // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombination<accPrecision, 128/cutlass::sizeof_bits<accPrecision>::value, accPrecision, accPrecision>;
    
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

    gemm_cutlass_launch_fp<Gemm, scalePrecision, mulPrecision, accPrecision>();
    return 0;
}

int gemm_cutlass_launch_turing_fp32_fp16_fp32_ntc()
{
    // Launch cutlass for NVIDIA Turing, scale precision fp32, multiplication precision fp16, accumulation precision fp32

    // Declare the operation precision
    using mulPrecision   = cutlass::half_t;   // multiplication precision
    using accPrecision   = float;             // accumulation precision
    using scalePrecision = float;             // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::RowMajor;    layout_matA _layout_matA;
    using layout_matB = cutlass::layout::ColumnMajor; layout_matB _layout_matB;
    using layout_matC = cutlass::layout::ColumnMajor; layout_matC _layout_matC;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassSimt;         // use CUDA Cores
    using SmArch              = cutlass::arch::Sm75;                    // Turing SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 8>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<32, 64, 8>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<1, 1, 1>;    // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombination<accPrecision, 1, accPrecision, accPrecision>;
    
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

    gemm_cutlass_launch_fp<Gemm, scalePrecision, mulPrecision, accPrecision>();
    return 0;
}

int gemm_cutlass_launch_ampere_fp32_fp16_fp32_tc()
{
    // Launch cutlass for NVIDIA Ampere, scale precision fp32, multiplication precision fp16, accumulation precision fp32
    
    // Declare the operation precision
    using mulPrecision   = cutlass::half_t;   // multiplication precision
    using accPrecision   = float;             // accumulation precision
    using scalePrecision = float;             // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::ColumnMajor;  layout_matA _layout_matA;
    using layout_matB = cutlass::layout::RowMajor;     layout_matB _layout_matB;
    using layout_matC = cutlass::layout::RowMajor;     layout_matC _layout_matC;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassTensorOp;         // use Tensor Cores
    using SmArch              = cutlass::arch::Sm80;                    // Ampere SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 64>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<64, 64, 64>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<16, 8, 16>;    // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombination<accPrecision, 128/cutlass::sizeof_bits<accPrecision>::value, accPrecision, accPrecision>;
    
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

    gemm_cutlass_launch_fp<Gemm, scalePrecision, mulPrecision, accPrecision>();
    return 0;
}

int gemm_cutlass_launch_ampere_fp32_fp16_fp32_ntc()
{
    // Launch cutlass for NVIDIA Ampere, scale precision fp32, multiplication precision fp16, accumulation precision fp32

    // Declare the operation precision
    using mulPrecision   = cutlass::half_t;   // multiplication precision
    using accPrecision   = float;             // accumulation precision
    using scalePrecision = float;             // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::RowMajor;    layout_matA _layout_matA;
    using layout_matB = cutlass::layout::ColumnMajor; layout_matB _layout_matB;
    using layout_matC = cutlass::layout::ColumnMajor; layout_matC _layout_matC;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassSimt;         // use CUDA Cores
    using SmArch              = cutlass::arch::Sm80;                    // Ampere SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 8>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<32, 64, 8>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<1, 1, 1>;    // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombination<accPrecision, 1, accPrecision, accPrecision>;
    
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

    gemm_cutlass_launch_fp<Gemm, scalePrecision, mulPrecision, accPrecision>();
    return 0;
}

int gemm_cutlass_launch_volta_fp32_fp32_fp32_ntc()
{
    // Launch cutlass for NVIDIA Volta, scale precision fp32, multiplication precision fp32, accumulation precision fp32

    // Declare the operation precision
    using mulPrecision   = float;   // multiplication precision
    using accPrecision   = float;             // accumulation precision
    using scalePrecision = float;             // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::RowMajor;    layout_matA _layout_matA;
    using layout_matB = cutlass::layout::ColumnMajor; layout_matB _layout_matB;
    using layout_matC = cutlass::layout::ColumnMajor; layout_matC _layout_matC;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassSimt;         // use CUDA Cores
    using SmArch              = cutlass::arch::Sm70;                    // Volta SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 8>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<32, 64, 8>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<1, 1, 1>;    // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombination<accPrecision, 1, accPrecision, accPrecision>;
    
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

    gemm_cutlass_launch_fp<Gemm, scalePrecision, mulPrecision, accPrecision>();
    return 0;
}

int gemm_cutlass_launch_turing_fp32_fp32_fp32_ntc()
{
    // Launch cutlass for NVIDIA Turing, scale precision fp32, multiplication precision fp32, accumulation precision fp32

    // Declare the operation precision
    using mulPrecision   = float;   // multiplication precision
    using accPrecision   = float;             // accumulation precision
    using scalePrecision = float;             // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::RowMajor;    layout_matA _layout_matA;
    using layout_matB = cutlass::layout::ColumnMajor; layout_matB _layout_matB;
    using layout_matC = cutlass::layout::ColumnMajor; layout_matC _layout_matC;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassSimt;         // use CUDA Cores
    using SmArch              = cutlass::arch::Sm75;                    // Turing SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 8>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<32, 64, 8>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<1, 1, 1>;    // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombination<accPrecision, 1, accPrecision, accPrecision>;
    
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

    gemm_cutlass_launch_fp<Gemm, scalePrecision, mulPrecision, accPrecision>();
    return 0;
}

int gemm_cutlass_launch_ampere_fp32_fp32_fp32_ntc()
{
    // Launch cutlass for NVIDIA Ampere, scale precision fp32, multiplication precision fp32, accumulation precision fp32

    // Declare the operation precision
    using mulPrecision   = float;   // multiplication precision
    using accPrecision   = float;             // accumulation precision
    using scalePrecision = float;             // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::RowMajor;    layout_matA _layout_matA;
    using layout_matB = cutlass::layout::ColumnMajor; layout_matB _layout_matB;
    using layout_matC = cutlass::layout::ColumnMajor; layout_matC _layout_matC;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassSimt;         // use CUDA Cores
    using SmArch              = cutlass::arch::Sm80;                    // Ampere SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 8>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<32, 64, 8>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<1, 1, 1>;    // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombination<accPrecision, 1, accPrecision, accPrecision>;
    
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

    gemm_cutlass_launch_fp<Gemm, scalePrecision, mulPrecision, accPrecision>();
    return 0;
}

int gemm_cutlass_launch_volta_fp64_fp64_fp64_ntc()
{
    // Launch cutlass for NVIDIA Volta, scale precision fp64, multiplication precision fp64, accumulation precision fp64

    // Declare the operation precision
    using mulPrecision   = double;   // multiplication precision
    using accPrecision   = double;             // accumulation precision
    using scalePrecision = double;             // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::RowMajor;    layout_matA _layout_matA;
    using layout_matB = cutlass::layout::ColumnMajor; layout_matB _layout_matB;
    using layout_matC = cutlass::layout::ColumnMajor; layout_matC _layout_matC;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassSimt;         // use CUDA Cores
    using SmArch              = cutlass::arch::Sm70;                    // Volta SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 8>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<32, 64, 8>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<1, 1, 1>;    // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombination<accPrecision, 1, accPrecision, accPrecision>;
    
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

    gemm_cutlass_launch_fp<Gemm, scalePrecision, mulPrecision, accPrecision>();
    return 0;
}

int gemm_cutlass_launch_turing_fp64_fp64_fp64_ntc()
{
    // Launch cutlass for NVIDIA Turing, scale precision fp64, multiplication precision fp64, accumulation precision fp64

    // Declare the operation precision
    using mulPrecision   = double;   // multiplication precision
    using accPrecision   = double;             // accumulation precision
    using scalePrecision = double;             // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::RowMajor;    layout_matA _layout_matA;
    using layout_matB = cutlass::layout::ColumnMajor; layout_matB _layout_matB;
    using layout_matC = cutlass::layout::ColumnMajor; layout_matC _layout_matC;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassSimt;         // use CUDA Cores
    using SmArch              = cutlass::arch::Sm75;                    // Turing SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 8>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<32, 64, 8>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<1, 1, 1>;    // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombination<accPrecision, 1, accPrecision, accPrecision>;
    
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

    gemm_cutlass_launch_fp<Gemm, scalePrecision, mulPrecision, accPrecision>();
    return 0;
}

int gemm_cutlass_launch_ampere_fp64_fp64_fp64_ntc()
{
    // Launch cutlass for NVIDIA Ampere, scale precision fp64, multiplication precision fp64, accumulation precision fp64

    // Declare the operation precision
    using mulPrecision   = double;   // multiplication precision
    using accPrecision   = double;             // accumulation precision
    using scalePrecision = double;             // scaling precision

    // Define Layout
    using layout_matA = cutlass::layout::RowMajor;    layout_matA _layout_matA;
    using layout_matB = cutlass::layout::ColumnMajor; layout_matB _layout_matB;
    using layout_matC = cutlass::layout::ColumnMajor; layout_matC _layout_matC;

    // Device-Related Kernel Settings
    using MMAOp               = cutlass::arch::OpClassSimt;         // use CUDA Cores
    using SmArch              = cutlass::arch::Sm80;                    // Ampere SM
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 8>; // Thread Block Shape 
    using ShapeMMAWarp        = cutlass::gemm::GemmShape<32, 64, 8>;   // Warp Shape
    using ShapeMMAOp          = cutlass::gemm::GemmShape<1, 1, 1>;    // Instruction Shape

    using SwizzleThreadBlock  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // default
    using EpilogueOutputOp    = cutlass::epilogue::thread::LinearCombination<accPrecision, 1, accPrecision, accPrecision>;
    
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

    gemm_cutlass_launch_fp<Gemm, scalePrecision, mulPrecision, accPrecision>();
    return 0;
}