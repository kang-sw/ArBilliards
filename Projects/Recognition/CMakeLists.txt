# CMakeList.txt : CMake project for Recognition, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("Recognition")

find_package(OpenCV REQUIRED)
find_package(ZED REQUIRED)

find_package(CUDA REQUIRED)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

# Add source to this project's executable.
add_executable (Recognition "src/entry.cpp"  "src/recognition.hpp" "src/thread_pool.hxx")

target_compile_features(Recognition PUBLIC cxx_std_17)
target_link_directories(Recognition PUBLIC ${ZED_LIBRARY_DIR} ${CUDA_DEP_LIBRARIES_ZED})
target_link_libraries(Recognition PUBLIC ${OpenCV_LIBS} ${ZED_LIBRARIES})
target_include_directories(Recognition PUBLIC ${OpenCV_INCLUDE_DIRS} ${ZED_INCLUDE_DIRS})


# TODO: Add tests and install targets if needed.

