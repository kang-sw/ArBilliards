cmake_minimum_required(VERSION 3.1)

project("CoreModel")

set(MSVC_IDE 1)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/third/pipepp/third/nana/include")
add_subdirectory(third/templates)
add_subdirectory(third/pipepp/third/fmt)
add_subdirectory(third/pipepp/third/json)
add_subdirectory(third/pipepp/third/nana)
add_subdirectory(third/pipepp/core)
add_subdirectory(third/pipepp/gui)

add_subdirectory(Recognition)