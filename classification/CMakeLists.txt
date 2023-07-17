cmake_minimum_required(VERSION 3.16)
project(test_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package(Threads)

set(OPENCV_INCLUDE /usr/include/opencv4/)
include_directories(${OPENCV_INCLUDE})

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/common SRCLIST_COMMON)
add_library(common SHARED ${SRCLIST_COMMON})

add_executable(test_classification_accuracy test_classification_accuracy.cpp)
add_executable(test_classification_fps test_classification_fps.cpp)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/classification SRCLIST_CLASSIFICATION)
add_library(classification_lib SHARED ${SRCLIST_CLASSIFICATION})

target_link_libraries(test_classification_accuracy
    classification_lib
    common 
    vitis_ai_library-classification
    vitis_ai_library-dpu_task
    vitis_ai_library-xnnpp
    vitis_ai_library-model_config
    vitis_ai_library-math
    vart-util
    xir
    pthread
    json-c 
    glog
    opencv_core
    opencv_videoio
    opencv_imgproc
    opencv_imgcodecs
    opencv_highgui  
)

target_link_libraries(test_classification_fps
    classification_lib
    common 
    vitis_ai_library-classification
    vitis_ai_library-dpu_task
    vitis_ai_library-xnnpp
    vitis_ai_library-model_config
    vitis_ai_library-math
    vart-util
    xir
    pthread
    json-c 
    glog
    opencv_core
    opencv_videoio
    opencv_imgproc
    opencv_imgcodecs
    opencv_highgui  
)

