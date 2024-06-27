#include <argparse/argparse.hpp>
#include <CUDA_Bench/gemm/gemm_cublas.cuh>
#include <CUDA_Bench/gemm/gemm_cutlass.cuh>
#include <CUDA_Bench/util/precision_select.cuh>
#include <CUDA_Bench/gemm/gemm_global.cuh>

int gdim_M;               // Global dimension of M
int gdim_N;               // Global dimension of N
int gdim_K;               // Global dimension of K

double galpha;            // Scaling factor (Alpha) of matrix multiplication
double gbeta;             // Scaling factor (Beta) of matrix accumulation

int gdevice;               // Device select

int gnum_iter;            // Global number of iteration
Precision gmulprecision;  // Global multiplication precision
Precision gaccprecision;  // Global accumulation precision
bool gprint_result;       // Global print result
bool gtensor_cores;       // Global tensor cores
bool guse_cublas;         // Global use cublas

int main(int argc, char *argv[])
{
    // Program Title
    std::cout << "[INFO] CUDA Bench - General Matrix-Matrix Multiplication (GEMM) \n";
    std::cout << "[INFO] Version 1.1.0 (C)2024 Bagus Hanindhito \n";
    std::cout << "[INFO] Matrix-Matrix multiplication follows equation: C = (alpha)x(AxB) + (beta)xC\n";
    std::cout << "[INFO] where A[MxK], B[KxN], and C[MxN] are matrices.";
    std::cout << "[INFO] Both Alpha and Beta are configurable and set by default to 1.00 and 0.00, respectively. \n\n\n";

    // Arguments Parser
    argparse::ArgumentParser program(argv[0], "1.1.0", argparse::default_arguments::help);
        program.add_argument("dim_M")
            .help("Positive integer that describes M dimension of the matrices A(MxK) and C(MxN)")
            .scan<'i', int>();
        program.add_argument("dim_N")
            .help("Positive integer that describes N dimension of the matrices B(KxN) and C(MxN)")
            .scan<'i', int>();
        program.add_argument("dim_K")
            .help("Positive integer that describes K dimension of the matrices A(MxK) and B(KxN)")
            .scan<'i', int>();
        program.add_argument("-R", "--result")
            .help("Show result at the end of program")
            .default_value(false)
            .implicit_value(true)
            .metavar("RESULT");
        program.add_argument("-C", "--cudacoresonly")
            .help("Use CUDA Cores only and do not use Tensor Cores")
            .default_value(false)
            .implicit_value(true)
            .metavar("CUDACORES");
        program.add_argument("-B", "--usecublas")
            .help("Use NVIDIA CUBLAS library instead of NVIDIA CUTLASS for GEMM")
            .default_value(false)
            .implicit_value(true)
            .metavar("CUBLAS");
        program.add_argument("-M", "--mulprecision")
            .help("Select matrix multiplication precision: fp64, tf32, fp32, bf16, fp16, int8, or int4")
            .default_value(std::string("fp16"))
            .metavar("MULPREC");
        program.add_argument("-A", "--accprecision")
            .help("Select matrix accumulation precision: fp64, tf32, fp32, bf16, fp16, int8, or int4")
            .default_value(std::string("fp16"))
            .metavar("ACCPREC");
        program.add_argument("-P", "--alpha")
            .help("The value of alpha for scaling factor of matrix multiplication.")
            .scan<'g', double>()
            .default_value(1.00)
            .metavar("ALPHA");
        program.add_argument("-Q", "--beta")
            .help("The value of beta for scaling factor of matrix accumulation.")
            .scan<'g', double>()
            .default_value(0.00)
            .metavar("BETA");
        program.add_argument("-D", "--device")
            .help("Select device (GPU) where the benchmark is run according to the index number provided by nvidia-smi.")
            .scan<'i', int>()
            .default_value(0)
            .metavar("DEVICE");
        program.add_argument("-I", "--iterations")
            .help("Number of iterations, useful for performance profiling")
            .scan<'i', int>()
            .default_value(1)
            .metavar("ITER");
    try 
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) 
    {
        std::cerr << "[ERR!] Argument parsing error: " << err.what() << std::endl << std::endl << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    // Argument Processing
    gdim_M = program.get<int>("dim_M");
    gdim_N = program.get<int>("dim_N");
    gdim_K = program.get<int>("dim_K");

    galpha = program.get<double>("--alpha");
    gbeta = program.get<double>("--beta");

    gdevice = program.get<int>("--device");

    gnum_iter = program.get<int>("--iterations");

    std::string str_mulprecision   = program.get<std::string>("--mulprecision");
    std::string str_accprecision   = program.get<std::string>("--accprecision");

    gprint_result = program.get<bool>("--result");
    gtensor_cores = !(program.get<bool>("--cudacoresonly"));
    guse_cublas   = program.get<bool>("--usecublas");

    // Argument Validation
    if(gdim_M<=0 || gdim_N<=0 || gdim_K<=0)
    {
        std::cerr <<"[ERR!] Argument parsing error: Matrices' dimensions must be positive integers\n\n\n";
        std::cerr << program;
        std::exit(1);
    }
    
    if(gnum_iter<=0)
    {
        std::cerr <<"[ERR!] Argument parsing error: Number of iterations must be positive integers\n\n\n";
        std::cerr << program;
        std::exit(1);
    }

    if      (str_mulprecision=="fp64") {gmulprecision=PRECISION_FP64;}
    else if (str_mulprecision=="fp32") {gmulprecision=PRECISION_FP32;}
    else if (str_mulprecision=="tf32") {gmulprecision=PRECISION_TF32;}
    else if (str_mulprecision=="fp16") {gmulprecision=PRECISION_FP16;}
    else if (str_mulprecision=="bf16") {gmulprecision=PRECISION_BF16;}
    else if (str_mulprecision=="int8") {gmulprecision=PRECISION_INT8;}
    else if (str_mulprecision=="int4") {gmulprecision=PRECISION_INT4;}
//  else if (str_mulprecision=="int1") {gmulprecision=PRECISION_INT1;}
    else
    {
        std::cerr <<"[ERR!] Argument parsing error: Unsupported matrix multiplication precision\n\n\n";
        std::cerr << program;
        std::exit(1);
    }

    if      (str_accprecision=="fp64") {gaccprecision=PRECISION_FP64;}
    else if (str_accprecision=="fp32") {gaccprecision=PRECISION_FP32;}
    else if (str_accprecision=="tf32") {gaccprecision=PRECISION_TF32;}
    else if (str_accprecision=="fp16") {gaccprecision=PRECISION_FP16;}
    else if (str_accprecision=="bf16") {gaccprecision=PRECISION_BF16;}
    else if (str_accprecision=="int8") {gaccprecision=PRECISION_INT8;}
    else if (str_accprecision=="int4") {gaccprecision=PRECISION_INT4;}
//  else if (str_accprecision=="int1") {gaccprecision=PRECISION_INT1;}
    else
    {
        std::cerr <<"[ERR!] Argument parsing error: Unsupported matrix accumulation precision\n\n\n";
        std::cerr << program;
        std::exit(1);
    }

    if(guse_cublas)
    {
        if(gmulprecision==PRECISION_INT4 || gaccprecision==PRECISION_INT4)
        {
            std::cerr <<"[ERR!] CUBLAS GEMM implementation currently only supports fp64, fp32, fp16, and int8\n\n\n";
            std::exit(1);
        }
        else
        {
            std::cout <<"[INFO] Program is using NVIDIA CUBLAS library for GEMM\n";
            gemm_cublas();
        }
    }
    else
    {
        std::cout <<"[INFO] Program is using NVIDIA CUTLASS library for GEMM\n";
        gemm_cutlass();
    }    
    return 0;
}