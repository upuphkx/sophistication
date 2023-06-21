cmake_minimum_required(VERSION 3.18)
project(register_base_code LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
message("-- CUDA INCLUIDE DIR: " ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

include_directories(${CMAKE_SOURCE_DIR}/include
                ${CMAKE_SOURCE_DIR}/src/include
                ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})                  

link_libraries("/usr/local/cuda-10.0/lib64/stubs/")

file(GLOB_RECURSE REGISTER_LIB_SRC  ${CMAKE_SOURCE_DIR}/src/*.cpp 
                                ${CMAKE_SOURCE_DIR}/register_app/*.cpp
                                ${CMAKE_SOURCE_DIR}/register_app/*.cu)

add_library(register SHARED ${REGISTER_LIB_SRC})

target_link_libraries(register PRIVATE CUDA::cufft CUDA::cudart CUDA::nvrtc cuda)

add_subdirectory(${CMAKE_SOURCE_DIR}/test)

install(TARGETS register
        DESTINATION ${CMAKE_BINARY_DIR}/lib)

install(TARGETS main
        DESTINATION ${CMAKE_BINARY_DIR}/bin)