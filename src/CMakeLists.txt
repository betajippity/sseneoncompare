cmake_minimum_required(VERSION 3.0.0)

project(bbox4)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64")
endif()

set(SOURCE_FILES "sseneoncompare.cpp")

set(CMAKE_CXX_FLAGS "-O3")
add_library(bbox4 ${SOURCE_FILES})
