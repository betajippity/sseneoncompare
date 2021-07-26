cmake_minimum_required(VERSION 3.0.0)

project(sseneoncompare)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SOURCE_FILES "src/sseneoncompare.cpp")

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64")
endif()

add_executable(sseneoncompare ${SOURCE_FILES})
