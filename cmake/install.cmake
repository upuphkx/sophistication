add_custom_target(install-base
                COMMAND "${CMAKE_COMMAND}"
                        -DCMAKE_INSTALL_COMPONENT="base"
                        -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
                USES_TERMINAL)

add_custom_target(install-tests
                COMMAND "${CMAKE_COMMAND}"
                        -DCMAKE_INSTALL_COMPONENT="tests"
                        -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
                USES_TERMINAL)                