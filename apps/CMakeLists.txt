set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
# FIR benchmark
#add_executable(fir_cuda_bench src/fir_cuda_bench.cu)
#target_link_libraries(fir_cuda_bench nvbench::main)

# GEMM benchmark
set(GEMM_BENCH_EXECUTABLE_NAME "gemm_cuda_bench")
add_executable(${GEMM_BENCH_EXECUTABLE_NAME} gemm_cuda_bench.cu)
set_target_properties(${GEMM_BENCH_EXECUTABLE_NAME}
                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON 
                        CUDA_ARCHITECTURES "${GPU_ARCHITECTURE_SUPPORT}")
  
target_link_libraries(${GEMM_BENCH_EXECUTABLE_NAME}
                      PRIVATE ${PROJECT_GEMM_LIB})