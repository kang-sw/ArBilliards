cmake_minimum_required(VERSION 3.1)

project("CoreModel")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/third/pipepp/third/nana/include")
add_subdirectory(third/pipepp)
add_subdirectory(Recognition)

add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")