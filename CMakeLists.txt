cmake_minimum_required(VERSION 3.10)
project(rm_detector)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70 75 80)
endif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# TODO(Call for PR): make cmake compatible with Windows
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        dynamic_reconfigure
        std_msgs
        cv_bridge
        nodelet
        pluginlib
        )
#find_package(OpenCV REQUIRED)
find_package(OpenCV 4 REQUIRED)


include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

include_directories(/opt/TensorRT-8.4.0.6/include/)
link_directories(/opt/TensorRT-8.4.0.6/targets/x86_64-linux-gnu/lib/)

include_directories(./utils/)
include_directories(./plugin/)
file(GLOB_RECURSE SRCS ./utils/*.cpp ${PROJECT_SOURCE_DIR}/utils/*.cu)
file(GLOB_RECURSE PLUGIN_SRCS ./plugin/*.cu)

add_library(myplugins SHARED ${PLUGIN_SRCS})
target_link_libraries(myplugins nvinfer cudart)

generate_dynamic_reconfigure_options(
        cfg/dynamic.cfg
)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

link_directories(${OpenCV_LIBRARY_DIRS})

catkin_package(
        INCLUDE_DIRS
        include
        LIBRARIES
        CATKIN_DEPENDS
        roscpp
        nodelet
        dynamic_reconfigure
        std_msgs
        cv_bridge
        LIBRARIES ${PROJECT_NAME}
)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        /usr/local/cuda-11.6/include
)
## Declare a C++ library
add_library(${PROJECT_NAME} src/detector.cpp
        src/inferencer.cpp
        ${SRCS}
        src/get_engine.cpp
)

target_link_libraries(${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${InferenceEngine_LIBRARIES}
        ${NGRAPH_LIBRARIES}
        ${OpenCV_LIBS}
        ${TENSORRT_LIB}
        ${CUDA_LIB}
        ${glog_LIB}
        )


target_link_libraries(rm_detector nvinfer)
target_link_libraries(rm_detector cudart)
target_link_libraries(rm_detector myplugins)
target_link_libraries(rm_detector ${OpenCV_LIBS})

add_executable(get_engine src/get_engine.cpp)
target_link_libraries(get_engine nvinfer)
target_link_libraries(get_engine cudart)
target_link_libraries(get_engine myplugins)
target_link_libraries(get_engine ${OpenCV_LIBS})

add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg)
