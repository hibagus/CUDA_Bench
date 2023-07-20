macro (add_sources sourcelist)
    file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")

    foreach (_src ${ARGN})
        if (_relPath)
            list (APPEND SOURCES "${CMAKE_SOURCE_DIR}/${_relPath}/${_src}")
        else()
            list (APPEND SOURCES "${CMAKE_SOURCE_DIR}/${_src}")
        endif()
    endforeach()
    if (_relPath)
        # propagate SOURCES to parent directory
        set (${sourcelist} ${SOURCES} PARENT_SCOPE)
    endif()
endmacro()


# Iterate through source files
add_subdirectory(gemm)
add_subdirectory(gemv)
add_subdirectory(vectoradd)
add_subdirectory(fir)
add_subdirectory(conv2d)


# Set Library
## PROJECT_GEMM_LIB
add_library(${PROJECT_GEMM_LIB} STATIC ${PROJECT_GEMM_SRC})
add_dependencies(${PROJECT_GEMM_LIB} nvbench cutlass)
set_target_properties(${PROJECT_GEMM_LIB}
                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                        CUDA_ARCHITECTURES ${GPU_ARCHITECTURE_SUPPORT}
                        CXX_STANDARD 17)
target_link_libraries(${PROJECT_GEMM_LIB}
                      PRIVATE -lcublas external_cutlass external_nvbench)

## PROJECT_GEMV_LIB
add_library(${PROJECT_GEMV_LIB} STATIC ${PROJECT_GEMV_SRC})
add_dependencies(${PROJECT_GEMV_LIB} nvbench cutlass)
set_target_properties(${PROJECT_GEMV_LIB}
                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                        CUDA_ARCHITECTURES ${GPU_ARCHITECTURE_SUPPORT}
                        CXX_STANDARD 17)
target_link_libraries(${PROJECT_GEMV_LIB}
                      PRIVATE -lcublas external_cutlass external_nvbench)

## PROJECT_VectorAdd_LIB
add_library(${PROJECT_VectorAdd_LIB} STATIC ${PROJECT_VectorAdd_SRC})
add_dependencies(${PROJECT_VectorAdd_LIB} nvbench cutlass)
set_target_properties(${PROJECT_VectorAdd_LIB}
                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                        CUDA_ARCHITECTURES ${GPU_ARCHITECTURE_SUPPORT}
                        CXX_STANDARD 17)
target_link_libraries(${PROJECT_VectorAdd_LIB}
                      PRIVATE -lcublas external_cutlass external_nvbench)

## PROJECT_FIR_LIB
add_library(${PROJECT_FIR_LIB} STATIC ${PROJECT_FIR_SRC})
add_dependencies(${PROJECT_FIR_LIB} nvbench cutlass)
set_target_properties(${PROJECT_FIR_LIB}
                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                        CUDA_ARCHITECTURES ${GPU_ARCHITECTURE_SUPPORT}
                        CXX_STANDARD 17)
target_link_libraries(${PROJECT_FIR_LIB}
                      PRIVATE -lcublas external_cutlass external_nvbench)

## PROJECT_FIR_LIB
add_library(${PROJECT_CONV2D_LIB} STATIC ${PROJECT_CONV2D_SRC})
add_dependencies(${PROJECT_CONV2D_LIB} nvbench cutlass)
set_target_properties(${PROJECT_CONV2D_LIB}
                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                        CUDA_ARCHITECTURES ${GPU_ARCHITECTURE_SUPPORT}
                        CXX_STANDARD 17)
target_link_libraries(${PROJECT_CONV2D_LIB}
                      PRIVATE -lcublas external_cutlass external_nvbench)