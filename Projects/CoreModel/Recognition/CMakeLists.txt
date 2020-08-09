# CMakeList.txt : CMake project for Recognition, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("Recognition")

find_package(OpenCV 4 REQUIRED)
find_package(ZED 2 REQUIRED)

find_package(CUDA REQUIRED)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

set(RECOGNITION_SOURCE_DIR "src")
set(RECOGNITION_EXAMPLE_DIR "example")

file(GLOB_RECURSE RECOGNITION_SOURCES "${RECOGNITION_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE RECOGNITION_EXAMPLE_SOURCES "${RECOGNITION_EXAMPLE_DIR}/*.cpp")

# Add source to this project's executable.
add_library(Recognition STATIC ${RECOGNITION_SOURCES})

target_compile_features(Recognition PUBLIC cxx_std_17)
target_link_directories(Recognition PUBLIC ${ZED_LIBRARY_DIR} ${CUDA_DEP_LIBRARIES_ZED})
target_link_libraries(Recognition PUBLIC ${OpenCV_LIBS} ${ZED_LIBRARIES})
target_include_directories(Recognition PUBLIC "include/" ${OpenCV_INCLUDE_DIRS} ${ZED_INCLUDE_DIRS})

add_executable(RecognitionTest ${RECOGNITION_EXAMPLE_SOURCES})
add_dependencies(RecognitionTest Recognition)
target_link_libraries(RecognitionTest PUBLIC Recognition)


message("SHOW: ${OpenCV_LIBS}")
