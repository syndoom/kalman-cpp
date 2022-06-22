cmake_minimum_required(VERSION 2.8.4)
project(kalman)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})



option(KALMANTEST "Flag to test execution" OFF)

if (KALMANTEST)
    set(SOURCE_FILES kalman.cpp kalman-test2.cpp)
    add_executable(kalman-test2 ${SOURCE_FILES})

else()
    add_library(onlinekalman STATIC
    kalman.cpp kalman.hpp
    )
    target_include_directories(onlinekalman INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
endif()
