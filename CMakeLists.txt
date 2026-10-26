cmake_minimum_required(VERSION 3.16)
project(OpenCVExample CXX)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV 4 REQUIRED)

file(GLOB SOURCES ${CMAKE_SOURCE_DIR}/*.cpp ${CMAKE_SOURCE_DIR}/src/*.cpp)

add_executable(OpenCVExample ${SOURCES}
        src/ImageLoader.h
        src/ImageLoader.cpp)
target_include_directories(OpenCVExample PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})
target_link_libraries(OpenCVExample PRIVATE ${OpenCV_LIBS})
target_compile_options(OpenCVExample PRIVATE -Wno-write-strings -Wno-deprecated-declarations)