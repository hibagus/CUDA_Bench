# FIR benchmark
#add_executable(fir_cuda_bench src/fir_cuda_bench.cu)
#target_link_libraries(fir_cuda_bench nvbench::main)

# GEMM benchmark
add_executable(gemm_cuda_bench gemm_cuda_bench.cu)