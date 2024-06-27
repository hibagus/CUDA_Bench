cmake_minimum_required(VERSION 3.20.1)

# Declare Project Name and Project Type
set(PROJECT_NAME "CUDA_Bench")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++14")
project(${PROJECT_NAME} C CXX CUDA)

# prevent in source builds
if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
   message(SEND_ERROR "In-source builds are not allowed.")
endif()

#find_package(CUDA REQUIRED) # Disable FindCUDA since it is deprecated by CMake
# set to Release to enable optimization
#        Debug for debugging
if(${BUILD_MODE} STREQUAL Debug)
    set(CMAKE_BUILD_TYPE Debug)
elseif(${BUILD_MODE} STREQUAL Release)
    set(CMAKE_BUILD_TYPE Release)
else()
    message(FATAL_ERROR "Unrecognized build mode")
endif()

include(cmake/CudaSettings.cmake)

# interface library for external library
include(cmake/ExternalLibs.cmake)

# add project include dirs for everything
include_directories(${CMAKE_SOURCE_DIR}/include)

# add project library
set(PROJECT_GEMM_LIB "gemmbenchlib" )

# Add Source Code and Apps
add_subdirectory(src)
add_subdirectory(apps)

