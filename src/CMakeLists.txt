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


# Set Library
## PROJECT_GEMM_LIB
add_library(${PROJECT_GEMM_LIB} STATIC ${PROJECT_GEMM_SRC})
add_dependencies(${PROJECT_GEMM_LIB} nvbench cutlass)
set_target_properties(${PROJECT_GEMM_LIB}
                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                        CUDA_ARCHITECTURES ${GPU_ARCHITECTURE_SUPPORT})
target_link_libraries(${PROJECT_GEMM_LIB}
                      PRIVATE -lcublas external_cutlass)