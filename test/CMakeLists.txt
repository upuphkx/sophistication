cmake_minimum_required(VERSION 3.18)
project(register_base_code LANGUAGES CXX)

file(GLOB_RECURSE test_src ./*.cpp)
add_executable(main ${test_src})

add_dependencies(main register)

target_link_libraries(main PRIVATE register)

install(TARGETS main
        DESTINATION ${CMAKE_BINARY_DIR}/bin)        