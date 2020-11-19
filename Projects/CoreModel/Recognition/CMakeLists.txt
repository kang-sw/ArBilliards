# CMakeList.txt : CMake project for Recognition, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("Recognition")

# Third-party libraries 
find_package(Boost REQUIRED COMPONENTS system thread regex) 
find_package(OpenCV 4 REQUIRED)
find_package(ZED 2 REQUIRED)

find_package(CUDA REQUIRED)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

# add_subdirectory(third/nana)

# project settings 
set(RECOGNITION_SOURCE_DIR "src") 
file(GLOB_RECURSE RECOGNITION_SOURCES "${RECOGNITION_SOURCE_DIR}/*.cpp" "${RECOGNITION_SOURCE_DIR}/*.hpp" "${RECOGNITION_SOURCE_DIR}/*.hxx" "${RECOGNITION_SOURCE_DIR}/*.c" )

add_executable(
	Recognition 
	${RECOGNITION_SOURCES}
	"third/cpp-base64/base64.cpp"
)

target_compile_features(
	Recognition
	PRIVATE
		cxx_std_20
)

add_dependencies(
	Recognition
		nana
		pipepp_gui
		fmt
)

target_link_libraries(
	Recognition
	PUBLIC
		${OpenCV_LIBS} 
		${ZED_LIBRARIES}
		${Boost_LIBRARIES}
		nana
		pipepp_gui
		nlohmann_json
		fmt
)

target_include_directories(
	Recognition
	PRIVATE
		"third/cpp-base64/" 
		"third/cvui"
		${OpenCV_INCLUDE_DIRS} 
		${ZED_INCLUDE_DIRS}
		${Boost_INCLUDE_DIRS}
)