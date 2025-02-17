cmake_minimum_required(VERSION 3.12)

# set the project name
project(VST)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-D__STDC_CONSTANT_MACROS")
SET(OpenCV_DIR /apdcephfs_cq2/share_1615605/xiaqiangdai/ft_local/opencv-4.6.0/build/)
find_package(OpenCV REQUIRED )

#find_package(Eigen3 REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})
#include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(VST SHARED
        ${PROJECT_SOURCE_DIR}/camera.cpp
        ${PROJECT_SOURCE_DIR}/tinyxml2.cpp
        )

add_executable(main main.cpp)
target_link_libraries(main
        ${OpenCV_LIBS}
        VST
        )


