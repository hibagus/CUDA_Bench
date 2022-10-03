include (ExternalProject)

# nvBench
add_library(external_nvbench INTERFACE)
ExternalProject_Add(nvbench
    PREFIX ${CMAKE_SOURCE_DIR}/3rdParty/nvbench/build
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/3rdParty/nvbench/
    CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CMAKE_SOURCE_DIR}/libs/nvbench
    -DCMAKE_CUDA_ARCHITECTURES=${GPU_ARCHITECTURE_SUPPORT}
)
target_link_libraries(external_nvbench 
    INTERFACE ${CMAKE_SOURCE_DIR}/libs/nvbench/lib/libnvbench.so)
target_include_directories(external_nvbench
    INTERFACE ${CMAKE_SOURCE_DIR}/libs/nvbench/include)



# argparse
add_library(external_argparse INTERFACE)
ExternalProject_Add(argparse
    PREFIX ${CMAKE_SOURCE_DIR}/3rdParty/argparse/build
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/3rdParty/argparse/
    CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CMAKE_SOURCE_DIR}/libs/argparse
)
target_include_directories(external_argparse
    INTERFACE ${CMAKE_SOURCE_DIR}/libs/argparse/include)