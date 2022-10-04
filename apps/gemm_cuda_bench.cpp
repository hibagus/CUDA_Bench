#include <argparse/argparse.hpp>
#include <CUDA_Bench/gemm/gemm_cublas.cuh>
#include <CUDA_Bench/util/precision_select.cuh>

int main(int argc, char *argv[])
{
    // Program Title
    std::cout << "[INFO] CUDA Bench - General Matrix-Matrix Multiplication (GEMM) \n";
    std::cout << "[INFO] Version 1.0.0 (C)2022 Bagus Hanindhito \n";
    std::cout << "[INFO] Matrix multiplication follows equation: C = (alpha)x(AxB) + (beta)xC\n";
    std::cout << "[INFO] where alpha=1.00, beta=0.00, and A[MxK], B[KxN], and C[MxN] are matrices \n\n\n";

    // Arguments Parser
    argparse::ArgumentParser program(argv[0], "1.0.0", argparse::default_arguments::help);
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
        program.add_argument("-P", "--precision")
            .help("Select matrix multiplication precision: fp64, fp32, fp16, int8, or int4")
            .default_value(std::string("fp16"))
            .metavar("PREC");
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
    int dim_M = program.get<int>("dim_M");
    int dim_N = program.get<int>("dim_N");
    int dim_K = program.get<int>("dim_K");
    int num_iter = program.get<int>("--iterations");
    std::string str_precision = program.get<std::string>("--precision");
    bool print_result = program.get<bool>("--result");
    bool tensor_cores = ~(program.get<bool>("--cudacoresonly"));

    // Argument Validation
    if(dim_M<=0 || dim_N<=0 || dim_K<=0)
    {
        std::cerr <<"[ERR!] Argument parsing error: Matrices' dimensions must be positive integers\n\n\n";
        std::cerr << program;
        std::exit(1);
    }
    
    if(num_iter<=0)
    {
        std::cerr <<"[ERR!] Argument parsing error: Number of iterations must be positive integers\n\n\n";
        std::cerr << program;
        std::exit(1);
    }

    Precision precision;
    if (str_precision=="fp64") {precision=PRECISION_FP64;}
    else if (str_precision=="fp32") {precision=PRECISION_FP32;}
    //else if (str_precision=="tf32") {precision=PRECISION_TF32;}
    else if (str_precision=="fp16") {precision=PRECISION_FP16;}
    //else if (str_precision=="bf16") {precision=PRECISION_BF16;}
    else if (str_precision=="int8") {precision=PRECISION_INT8;}
    else if (str_precision=="int4") {precision=PRECISION_INT4;}
    //else if (str_precision=="int1") {precision=PRECISION_INT1;}
    else
    {
        std::cerr <<"[ERR!] Argument parsing error: Unsupported matrix multiplication precision\n\n\n";
        std::cerr << program;
        std::exit(1);
    }
    
    // Call cuBlas
    gemm_cublas(dim_M, dim_N, dim_K, precision, num_iter, print_result, tensor_cores);
    return 0;
}