#include <argparse/argparse.hpp>
#include <CUDA_Bench/conv2d/conv2d_cutlass.cuh>
#include <CUDA_Bench/util/precision_select.cuh>
#include <CUDA_Bench/conv2d/conv2d_global.cuh>

// This is global variables needed by NVBench :(
long ginput_N;            // Global Input Batch
long ginput_H;            // Global Input Height
long ginput_W;            // Global Input Width
long ginput_C;            // Global Input Channel
long gfilter_K;           // Global Filter Channel
long gfilter_R;           // Global Filter Height
long gfilter_S;           // Global Filter Width
long gstride_H;           // Global Stride Horizontal
long gstride_V;           // Global Stride Vertical

int gnum_iter;            // Global number of iteration
Precision gmulprecision;  // Global multiplication precision
Precision gaccprecision;  // Global accumulation precision
bool gprint_result;       // Global print result
bool gtensor_cores;       // Global tensor cores

bool gprofiling;          // Global profiling
const int  gargc_nvbench = 3;
const char *gargv_nvbench[] = {"conv2d_cuda_bench", "--devices", "0"};

int main(int argc, char *argv[])
{
    // Program Title
    std::cout << "[INFO] CUDA Bench - Two Dimensional Convolution (CONV2D) \n";
    std::cout << "[INFO] Version 1.0.0 (C)2022 Bagus Hanindhito \n";
    std::cout << "[INFO] Convolution 2D perform convolution of 4D input tensor with 4D filter tensor\n";
    std::cout << "[INFO] where input tensor has dimension N (batch), H (height), W (width), and C (channel) \n";
    std::cout << "[INFO] while filter tensor has dimension K (channel), R (height), and S (width) with \n";
    std::cout << "[INFO] specific vertical and horizontal stride. \n\n\n";


    // Arguments Parser
    argparse::ArgumentParser program(argv[0], "1.0.0", argparse::default_arguments::help);
        program.add_argument("input_N")
            .help("Positive integer that describes the batch size of input tensor.")
            .scan<'i', int>();
        program.add_argument("input_H")
            .help("Positive integer that describes the height of input tensor.")
            .scan<'i', int>();
        program.add_argument("input_W")
            .help("Positive integer that describes the width of input tensor.")
            .scan<'i', int>();
        program.add_argument("input_C")
            .help("Positive integer that describes the number of channel of input tensor.")
            .scan<'i', int>();
        program.add_argument("filter_K")
            .help("Positive integer that describes the number of channel of filter tensor.")
            .scan<'i', int>();
        program.add_argument("filter_R")
            .help("Positive integer that describes the height of filter tensor.")
            .scan<'i', int>();
        program.add_argument("filter_S")
            .help("Positive integer that describes the width of filter tensor.")
            .scan<'i', int>();
        program.add_argument("stride_H")
            .help("Positive integer that describes the horizontal stride.")
            .scan<'i', int>();
        program.add_argument("stride_V")
            .help("Positive integer that describes the vertical stride.")
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
    ginput_N  = program.get<int>("input_N");  
    ginput_H  = program.get<int>("input_H"); 
    ginput_W  = program.get<int>("input_W"); 
    ginput_C  = program.get<int>("input_C"); 
    gfilter_K = program.get<int>("filter_K");
    gfilter_R = program.get<int>("filter_R");
    gfilter_S = program.get<int>("filter_S");
    gstride_H = program.get<int>("stride_H");
    gstride_V = program.get<int>("stride_V");

    gnum_iter = program.get<int>("--iterations");

    std::string str_mulprecision   = program.get<std::string>("--mulprecision");
    std::string str_accprecision   = program.get<std::string>("--accprecision");

    gprint_result = program.get<bool>("--result");
    gtensor_cores = !(program.get<bool>("--cudacoresonly"));
    gprofiling    = program.get<bool>("--profile");

    // Argument Validation
    if(ginput_N<=0 || ginput_H<=0 || ginput_W<=0 || ginput_C<=0 || gfilter_K<=0
     || gfilter_R<=0 || gfilter_S<=0 || gstride_H<=0 || gstride_V<=0)
    {
        std::cerr <<"[ERR!] Argument parsing error: Input/Filter dimensions and strides must be positive integers\n\n\n";
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

    std::cout <<"[INFO] Program is using NVIDIA CUTLASS library for 2D Convolution\n";
    conv2d_cutlass();

    
    return 0;
}