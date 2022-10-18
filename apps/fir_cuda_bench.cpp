#include <argparse/argparse.hpp>
#include <CUDA_Bench/fir/fir_cublas.cuh>
#include <CUDA_Bench/fir/fir_cutlass.cuh>
#include <CUDA_Bench/util/precision_select.cuh>
#include <CUDA_Bench/fir/fir_global.cuh>

// This is global variables needed by NVBench :(
long gdim_S;              // Global Padded Signal Dimension 
long gdim_M;              // Global dimension of M
long gdim_K;              // Global dimension of K
int gnum_iter;            // Global number of iteration
Precision gmulprecision;  // Global multiplication precision
Precision gaccprecision;  // Global accumulation precision
bool gprint_result;       // Global print result
bool gtensor_cores;       // Global tensor cores
bool guse_cublas;         // Global use cublas
bool gprofiling;          // Global profiling
const int  gargc_nvbench = 3;
const char *gargv_nvbench[] = {"fir_cuda_bench", "--devices", "0"};

int main(int argc, char *argv[])
{
    // Program Title
    std::cout << "[INFO] CUDA Bench - Finite Impulse Response Filter (FIR) \n";
    std::cout << "[INFO] Version 1.0.0 (C)2022 Bagus Hanindhito \n";
    std::cout << "[INFO] FIR filter follows equation: y[n] = f0*x[n] + f1*x[n-1] + ... + fK*x[n-K] \n";
    std::cout << "[INFO] where f0..fK is filter coefficient in the form of vector F[K], \n";
    std::cout << "[INFO] x[n]..x[n-K] is part of input signal in the form of vector X[M], \n";
    std::cout << "[INFO] and produce an output vector Y[M] \n\n\n";


    // Arguments Parser
    argparse::ArgumentParser program(argv[0], "1.0.0", argparse::default_arguments::help);
        program.add_argument("dim_M")
            .help("Positive integer that describes the length of input signal vector, X[M].")
            .scan<'i', int>();
        program.add_argument("dim_K")
            .help("Positive integer that describes the length of filter vector, F[K].")
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
            .help("Use NVIDIA CUBLAS library instead of NVIDIA CUTLASS for FIR")
            .default_value(false)
            .implicit_value(true)
            .metavar("CUBLAS");
        program.add_argument("-P", "--profile")
            .help("Enable built-in kernel profiling with NVBench")
            .default_value(false)
            .implicit_value(true)
            .metavar("PROFILE");
        program.add_argument("-M", "--mulprecision")
            .help("Select matrix multiplication precision: fp64, fp32, fp16, int8, or int4")
            .default_value(std::string("fp16"))
            .metavar("MULPREC");
        program.add_argument("-A", "--accprecision")
            .help("Select matrix accumulation precision: fp64, fp32, fp16, int8, or int4")
            .default_value(std::string("fp16"))
            .metavar("ACCPREC");
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
    gdim_K = program.get<int>("dim_K");

    gnum_iter = program.get<int>("--iterations");

    std::string str_mulprecision   = program.get<std::string>("--mulprecision");
    std::string str_accprecision   = program.get<std::string>("--accprecision");

    gprint_result = program.get<bool>("--result");
    gtensor_cores = !(program.get<bool>("--cudacoresonly"));
    gprofiling    = program.get<bool>("--profile");
    guse_cublas   = program.get<bool>("--usecublas");

    // Argument Validation
    if(gdim_M<=0 || gdim_K<=0)
    {
        std::cerr <<"[ERR!] Argument parsing error: Input/Filter dimensions must be positive integers\n\n\n";
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
//  else if (str_mulprecision=="tf32") {gmulprecision=PRECISION_TF32;}
    else if (str_mulprecision=="fp16") {gmulprecision=PRECISION_FP16;}
//  else if (str_mulprecision=="bf16") {gmulprecision=PRECISION_BF16;}
    else if (str_mulprecision=="int8") {gmulprecision=PRECISION_INT8;}
    else if (str_mulprecision=="int4") {gmulprecision=PRECISION_INT4;}
//  else if (str_mulprecision=="int1") {gmulprecision=PRECISION_INT1;}
    else
    {
        std::cerr <<"[ERR!] Argument parsing error: Unsupported matrix/vector multiplication precision\n\n\n";
        std::cerr << program;
        std::exit(1);
    }

    if      (str_accprecision=="fp64") {gaccprecision=PRECISION_FP64;}
    else if (str_accprecision=="fp32") {gaccprecision=PRECISION_FP32;}
//  else if (str_accprecision=="tf32") {gaccprecision=PRECISION_TF32;}
    else if (str_accprecision=="fp16") {gaccprecision=PRECISION_FP16;}
//  else if (str_accprecision=="bf16") {gaccprecision=PRECISION_BF16;}
    else if (str_accprecision=="int8") {gaccprecision=PRECISION_INT8;}
    else if (str_accprecision=="int4") {gaccprecision=PRECISION_INT4;}
//  else if (str_accprecision=="int1") {gaccprecision=PRECISION_INT1;}
    else
    {
        std::cerr <<"[ERR!] Argument parsing error: Unsupported matrix/vector accumulation precision\n\n\n";
        std::cerr << program;
        std::exit(1);
    }

    // Calculate matrices size
    int padding = gdim_K - (gdim_M % gdim_K); //total input signal length modulo the filter length
    gdim_S = gdim_M + padding;

    if(guse_cublas)
    {
        if(gmulprecision==PRECISION_INT4 || gaccprecision==PRECISION_INT4)
        {
            std::cerr <<"[ERR!] CUBLAS FIR implementation currently only supports fp64, fp32, fp16, and int8\n\n\n";
            std::exit(1);
        }
        else
        {
            std::cout <<"[INFO] Program is using NVIDIA CUBLAS library for FIR\n";
            fir_cublas();
        }
    }
    else
    {
        std::cout <<"[INFO] Program is using NVIDIA CUTLASS library for FIR\n";
        fir_cutlass();
    }    
    return 0;
}