cmake_minimum_required(VERSION 3.1)

project("CoreModel")

set(MSVC_IDE 1)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/third/pipepp/third/nana/include")
add_subdirectory(third/pipepp)
add_subdirectory(third/gSLICr)

add_subdirectory(Recognition)