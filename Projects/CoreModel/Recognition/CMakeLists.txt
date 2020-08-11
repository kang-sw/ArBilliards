# CMakeList.txt : CMake project for Recognition, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("Recognition")

# Third-party libraries
add_subdirectory(third/socketpp)

find_package(OpenCV 4 REQUIRED)
find_package(ZED 2 REQUIRED)

find_package(CUDA REQUIRED)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

set(RECOGNITION_SOURCE_DIR "src") 
file(GLOB_RECURSE RECOGNITION_SOURCES "${RECOGNITION_SOURCE_DIR}/*.cpp")

#target_compile_features(Recognition PUBLIC cxx_std_17)
#target_link_directories(Recognition PUBLIC ${ZED_LIBRARY_DIR} ${CUDA_DEP_LIBRARIES_ZED})
#target_link_libraries(Recognition PUBLIC ${OpenCV_LIBS} ${ZED_LIBRARIES})
#target_include_directories(Recognition PUBLIC "third/nlohmann/include/" "include/" ${OpenCV_INCLUDE_DIRS} ${ZED_INCLUDE_DIRS})

add_executable(
	Recognition 
	${RECOGNITION_SOURCES}
)

target_compile_features(
	Recognition
	PRIVATE
		cxx_std_17
)

add_dependencies(
	Recognition 
		sockpp-static
)

target_link_libraries(
	Recognition
	PRIVATE
		sockpp-static
		${OpenCV_LIBS} 
		${ZED_LIBRARIES}
)

target_include_directories(
	Recognition
	PRIVATE
		"include/"
		$<TARGET_PROPERTY:sockpp-static,INTERFACE_INCLUDE_DIRECTORIES>
		"third/nlohmann/include/" 
		${OpenCV_INCLUDE_DIRS} 
		${ZED_INCLUDE_DIRS}
)

