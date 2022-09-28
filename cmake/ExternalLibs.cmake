include (ExternalProject)
# NVBench
ExternalProject_Add(NVBench
    PREFIX ${CMAKE_SOURCE_DIR}/3rdParty/nvbench/build
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/3rdParty/nvbench/
    CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CMAKE_SOURCE_DIR}/libs/nvbench
    -DCMAKE_CUDA_ARCHITECTURES=${GPU_ARCHITECTURE_SUPPORT}
)