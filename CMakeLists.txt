cmake_minimum_required(VERSION 3.10)

project(camera_playground)

set(CMAKE_CXX_STANDARD 14)

find_package(SDL3 REQUIRED)
find_package(SDL3_TTF REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${SDL3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${SDL3_ttf_INCLUDE_DIRS} )

add_executable(camera_playground main.cpp)
target_link_libraries(camera_playground ${SDL3_LIBRARIES} ${OpenCV_LIBS} SDL3_ttf::SDL3_ttf-shared)
