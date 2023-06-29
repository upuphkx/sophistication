cmake_minimum_required(VERSION 3.18)
project(register_base_code LANGUAGES CXX)

if(DEFINED ${GTEST_INCLUDE_DIRS})

include_directories(${GTEST_INCLUDE_DIRS})

endif()

file(GLOB_RECURSE test_src ./*.cpp)
add_executable(main ${test_src})

add_dependencies(main register)

target_link_libraries(main PRIVATE register ${GTEST_BOTH_LIBRARIES} 
                                                ${CMAKE_THREAD_LIBS_INIT})

install(TARGETS main
        DESTINATION ${CMAKE_BINARY_DIR}/bin
        COMPONENT base)        