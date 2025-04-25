cmake_minimum_required(VERSION 3.15)
project(ImagePlugin)

set(CMAKE_CXX_STANDARD 17)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include headers
include_directories(${OpenCV_INCLUDE_DIRS} include)

# Build dynamic library
add_library(ImagePlugin SHARED src/ImageProcessor.cpp)

# Link OpenCV
target_link_libraries(ImagePlugin ${OpenCV_LIBS})

# macOS-specific: set output name
set_target_properties(ImagePlugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    OUTPUT_NAME "ImagePlugin"
    SUFFIX ".dylib"
)

