include (ExternalProject)
include (GNUInstallDirs)

# argparse
add_library(external_argparse INTERFACE)
ExternalProject_Add(argparse
    PREFIX ${CMAKE_SOURCE_DIR}/build/argparse/build
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/3rdParty/argparse/
    CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CMAKE_SOURCE_DIR}/libs/argparse
)
target_include_directories(external_argparse
    INTERFACE ${CMAKE_SOURCE_DIR}/libs/argparse/include)

# cutlass
add_library(external_cutlass INTERFACE)
ExternalProject_Add(cutlass
    PREFIX ${CMAKE_SOURCE_DIR}/build/cutlass/build
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/3rdParty/cutlass/
    CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CMAKE_SOURCE_DIR}/libs/cutlass
    -DCUTLASS_NVCC_ARCHS=${GPU_ARCHITECTURE_SUPPORT}
    -DCUTLASS_ENABLE_TOOLS_INIT=ON
    -DCUTLASS_ENABLE_TOOLS=ON
    -DCUTLASS_ENABLE_LIBRARY_INIT=OFF
    -DCUTLASS_ENABLE_LIBRARY=OFF
    -DCUTLASS_ENABLE_EXAMPLES_INIT=OFF
    -DCUTLASS_ENABLE_EXAMPLES=OFF
    -DCUTLASS_ENABLE_TESTS_INIT=OFF
    -DCUTLASS_ENABLE_TESTS=OFF
    -DCUTLASS_ENABLE_PROFILER=OFF

)
target_include_directories(external_cutlass
    INTERFACE ${CMAKE_SOURCE_DIR}/libs/cutlass/include)