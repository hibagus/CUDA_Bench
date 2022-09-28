#include <cuda.h>
#include <CUDA_Bench/gemm/gemm_cublas.cuh>
#include <CUDA_Bench/util/gpuinfo.cuh>
#include <CUDA_Bench/util/gpucheck.cuh>

int argparse(char *argv);
void help();

int main(int argc, char *argv[])
{
    // Program Title
    std::cout << "[INFO] CUDA Bench - General Matrix-Matrix Multiplication (GEMM) \n";
    std::cout << "[INFO] (C)2022 Bagus Hanindhito \n";

    // Detect Available CUDA Devices
    int nDevices;
    gpuErrchk(cudaGetDeviceCount(&nDevices));
    print_cuda_device_info(nDevices);
    if(nDevices>0) {std::cout << "[WARN] This program does not currently support Multi-GPU run.\n";}

    // Matrices Size
    int dim_M;
    int dim_N;
    int dim_K;

    // Input Validation
    if(argc>=4)
    {
        dim_M = argparse(argv[1]);
        dim_N = argparse(argv[2]);
        dim_K = argparse(argv[3]);
        if(dim_M<=0 || dim_N<=0 || dim_K<=0)
        {
            std::cerr << "[ERR!] Matrix dimension must be positive integers!\n";
            help();
            exit(1);
        }
    }
    else
    {
        std::cerr << "[ERR!] This program requires arguments to set matrices size!\n";
        help();
        exit(1);
    }

    // Call Function
    int test = gemm_cublas(dim_M, dim_N, dim_K);


    return 0;
}


void help()
{
    std::cerr << "[ERR!] To launch program: ./gemm_cuda_bench M N K\n";
    std::cerr << "[ERR!] M, N, K are positive integers representing the dimension of matrices A, B, and C in matrix multiplication C = AxB\n";
    std::cerr << "[ERR!] Matrices Dimension: A (MxK) | B (KxN) | C (MxN) \n";
}

int argparse(char *argv)
{
    std::string arg = argv;
    try 
    {
        std::size_t pos;
        int x = std::stoi(arg, &pos);
        if (pos < arg.size()) 
        {
            std::cerr << "[ERR!] Trailing characters after number: " << arg << '\n';
            help();
            exit(1);
        }
        return x;
    } 
    catch (std::invalid_argument const &ex) 
    {
        std::cerr << "[ERR!] Invalid number: " << arg << '\n';
        help();
        exit(1);
    } 
    catch (std::out_of_range const &ex) 
    {
        std::cerr << "[ERR!] Number out of range: " << arg << '\n';
        help();
        exit(1);
    }
    return 0;
}