set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

# GEMM
set(GEMM_BENCH_EXECUTABLE_NAME "gemm_cuda_bench")
add_executable(${GEMM_BENCH_EXECUTABLE_NAME} gemm_cuda_bench.cpp)
add_dependencies(${GEMM_BENCH_EXECUTABLE_NAME} argparse)
set_target_properties(${GEMM_BENCH_EXECUTABLE_NAME}
                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON 
                        CUDA_ARCHITECTURES "${GPU_ARCHITECTURE_SUPPORT}"
                        CXX_STANDARD 17)
  
target_link_libraries(${GEMM_BENCH_EXECUTABLE_NAME}
                      PRIVATE ${PROJECT_GEMM_LIB} external_argparse -lcuda)
