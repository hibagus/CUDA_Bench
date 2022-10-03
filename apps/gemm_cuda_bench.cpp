#include <argparse/argparse.hpp>
#include <CUDA_Bench/gemm/gemm_cublas.cuh>
//#include <CUDA_Bench/util/gpuinfo.cuh>
//#include <CUDA_Bench/util/gpucheck.cuh>

//int argparse(char *argv);
//void help();

int main(int argc, char *argv[])
{
    // Program Title
    std::cout << "[INFO] CUDA Bench - General Matrix-Matrix Multiplication (GEMM) \n";
    std::cout << "[INFO] Version 1.0.0 (C)2022 Bagus Hanindhito \n\n\n";

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
        program.add_argument("-P", "--precision")
            .help("Select matrix multiplication precision")
            .default_value(std::string("fp16"))
            .metavar("PREC");

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
    std::string precision = program.get<std::string>("--precision");

    // Argument Validation
    if(dim_M<=0 || dim_N<=0 || dim_K<=0)
    {
        std::cerr <<"[ERR!] Argument parsing error: Matrices' dimensions must be positive integers\n\n\n";
        std::cerr << program;
        std::exit(1);
    }
    
    // Call cuBlas
    gemm_cublas(dim_M, dim_N, dim_K);
    return 0;
}