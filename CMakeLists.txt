cmake_minimum_required(VERSION 3.20)

project(eKYC_Project)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

execute_process(COMMAND brew --prefix opencv
	OUTPUT_VARIABLE OPENCV_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)

set(ONNXRUNTIME_PREFIX "${CMAKE_SOURCE_DIR}/onnxruntime_dist")

find_package(OpenCV CONFIG REQUIRED
	PATHS "${OPENCV_PREFIX}/lib/cmake/opencv5"
	NO_DEFAULT_PATH)

find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
	PATHS "${ONNXRUNTIME_PREFIX}/lib"
	NO_DEFAULT_PATH
	REQUIRED)

add_executable(ekyc_app main.cpp)

target_compile_definitions(ekyc_app PRIVATE
	PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

target_include_directories(ekyc_app PRIVATE
	${OpenCV_INCLUDE_DIRS}
	"${ONNXRUNTIME_PREFIX}/include"
)
target_link_libraries(ekyc_app PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARY})
set_target_properties(ekyc_app PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_PREFIX}/lib"
)
