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

# VectorAdd benchmark
set(VectorAdd_BENCH_EXECUTABLE_NAME "vectoradd_cuda_bench")
add_executable(${VectorAdd_BENCH_EXECUTABLE_NAME} vectoradd_cuda_bench.cpp)
add_dependencies(${VectorAdd_BENCH_EXECUTABLE_NAME} argparse nvbench)
set_target_properties(${VectorAdd_BENCH_EXECUTABLE_NAME}
                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON 
                        CUDA_ARCHITECTURES "${GPU_ARCHITECTURE_SUPPORT}"
                        CXX_STANDARD 17)
  
target_link_libraries(${VectorAdd_BENCH_EXECUTABLE_NAME}
                      PRIVATE ${PROJECT_VectorAdd_LIB} external_argparse external_nvbench -lcuda)

# FIR benchmark
set(FIR_BENCH_EXECUTABLE_NAME "fir_cuda_bench")
add_executable(${FIR_BENCH_EXECUTABLE_NAME} fir_cuda_bench.cpp)
add_dependencies(${FIR_BENCH_EXECUTABLE_NAME} argparse nvbench)
set_target_properties(${FIR_BENCH_EXECUTABLE_NAME}
                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON 
                        CUDA_ARCHITECTURES "${GPU_ARCHITECTURE_SUPPORT}"
                        CXX_STANDARD 17)
  
target_link_libraries(${FIR_BENCH_EXECUTABLE_NAME}
                      PRIVATE ${PROJECT_FIR_LIB} external_argparse external_nvbench -lcuda)

# Conv2D benchmark
set(CONV2D_BENCH_EXECUTABLE_NAME "conv2d_cuda_bench")
add_executable(${CONV2D_BENCH_EXECUTABLE_NAME} conv2d_cuda_bench.cpp)
add_dependencies(${CONV2D_BENCH_EXECUTABLE_NAME} argparse nvbench)
set_target_properties(${CONV2D_BENCH_EXECUTABLE_NAME}
                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON 
                        CUDA_ARCHITECTURES "${GPU_ARCHITECTURE_SUPPORT}"
                        CXX_STANDARD 17)
  
target_link_libraries(${CONV2D_BENCH_EXECUTABLE_NAME}
                      PRIVATE ${PROJECT_CONV2D_LIB} external_argparse external_nvbench -lcuda)
