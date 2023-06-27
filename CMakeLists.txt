cmake_minimum_required(VERSION 3.18)
project(register_base_code LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
# find_package(GTest)

if(${GTest_FOUND})
else()
if(NOT EXISTS ${CMAKE_SOURCE_DIR}/third_party)
  execute_process(COMMAND mkdir third_party
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

if(NOT EXISTS ${CMAKE_SOURCE_DIR}/third_party/googletest-main)
  
  execute_process(COMMAND git clone https://github.com/google/googletest.git
                 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party)

endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/googletest-main)

set(GTEST_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/third_party/googletest-main/googletest/include)

set(GTEST_BOTH_LIBRARIES gtest gtest_main)

endif()

message(STATUS "CUDA INCLUDE DIR: " ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

include_directories(${CMAKE_SOURCE_DIR}/include
                ${CMAKE_SOURCE_DIR}/register_app
                ${CMAKE_SOURCE_DIR}/src/include
                ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})                  


file(GLOB_RECURSE REGISTER_LIB_SRC ${CMAKE_SOURCE_DIR}/src/*.cpp 
                                ${CMAKE_SOURCE_DIR}/register_app/*.cpp
                                ${CMAKE_SOURCE_DIR}/register_app/*.cu)

add_library(register SHARED ${REGISTER_LIB_SRC})

target_link_libraries(register PRIVATE CUDA::cufft CUDA::cudart CUDA::nvrtc cuda)

add_subdirectory(${CMAKE_SOURCE_DIR}/test)

install(TARGETS register
        DESTINATION ${CMAKE_BINARY_DIR}/lib)