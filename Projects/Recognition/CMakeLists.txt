# CMakeList.txt : CMake project for Recognition, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("Recognition")

find_package(OpenCV REQUIRED)

# Add source to this project's executable.
add_executable (Recognition "Recognition.cpp" "Recognition.h")

target_compile_features(Recognition PUBLIC cxx_std_17)
target_link_libraries(Recognition PUBLIC ${OpenCV_LIBS})
target_include_directories(Recognition PUBLIC ${OpenCV_INCLUDE_DIRS})


# TODO: Add tests and install targets if needed.

