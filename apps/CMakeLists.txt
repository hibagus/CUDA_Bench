set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
# FIR benchmark
#add_executable(fir_cuda_bench src/fir_cuda_bench.cu)
#target_link_libraries(fir_cuda_bench nvbench::main)

# GEMM benchmark
set(GEMM_BENCH_EXECUTABLE_NAME "gemm_cuda_bench")
add_executable(${GEMM_BENCH_EXECUTABLE_NAME} gemm_cuda_bench.cpp)
add_dependencies(${GEMM_BENCH_EXECUTABLE_NAME} argparse nvbench)
set_target_properties(${GEMM_BENCH_EXECUTABLE_NAME}
                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON 
                        CUDA_ARCHITECTURES "${GPU_ARCHITECTURE_SUPPORT}"
                        CXX_STANDARD 17)
  
target_link_libraries(${GEMM_BENCH_EXECUTABLE_NAME}
                      PRIVATE ${PROJECT_GEMM_LIB} external_argparse external_nvbench -lcuda)

# GEMV benchmark
set(GEMV_BENCH_EXECUTABLE_NAME "gemv_cuda_bench")
add_executable(${GEMV_BENCH_EXECUTABLE_NAME} gemv_cuda_bench.cpp)
add_dependencies(${GEMV_BENCH_EXECUTABLE_NAME} argparse nvbench)
set_target_properties(${GEMV_BENCH_EXECUTABLE_NAME}
                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON 
                        CUDA_ARCHITECTURES "${GPU_ARCHITECTURE_SUPPORT}"
                        CXX_STANDARD 17)
  
target_link_libraries(${GEMV_BENCH_EXECUTABLE_NAME}
                      PRIVATE ${PROJECT_GEMV_LIB} external_argparse external_nvbench -lcuda)
