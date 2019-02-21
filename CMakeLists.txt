# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
PROJECT( Eye_Tracking )


# *************** OpenCV *****************
FIND_PACKAGE( OpenCV REQUIRED )
message(STATUS "**OpenCV library status:")
message(STATUS "--version: ${OpenCV_VERSION}")
message(STATUS "--libraries: ${OpenCV_LIBS}")
message(STATUS "--include path: ${OpenCV_INCLUDE_DIRS}")

if(CMAKE_VERSION VERSION_LESS "2.8.11")
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()


# **************** Eigen *****************
SET( EIGEN3_INCLUDE_DIR "/usr/local/share/eigen3/cmake" )
IF( NOT EIGEN3_INCLUDE_DIR )
    MESSAGE( FATAL_ERROR "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
ENDIF()
INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )


# ************** Usr Def ****************
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    )
add_library(${PROJECT_NAME} SHARED
#    src/findEyeCenter.cpp
#    src/findEyeCorner.cpp
    src/findPupils.cpp
    src/optEyePose.cpp
#    src/helpers.cpp
    )
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    )


# ************** cmake ******************
cmake_minimum_required(VERSION 2.8)


# ************** path *******************
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_BINARY_DIR ${PROJECT_BINARY_DIR}/bin)


# Declare the executable target built from your sources
add_executable(eyetracking src/main.cpp)


# Link your application with OpenCV libraries
target_link_libraries(eyetracking ${OpenCV_LIBS})
target_link_libraries(eyetracking ${PROJECT_NAME})
