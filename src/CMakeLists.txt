cmake_minimum_required(VERSION 3.0.0)

project(bbox4 C CXX ISPC)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(${ISPC_HEADER_DIRECTORY})

set(SOURCE_FILES "sseneoncompare.cpp")
set(SOURCE_FILES ${SOURCE_FILES} "bbox4.ispc")

set(CMAKE_CXX_FLAGS "-O3")

EXECUTE_PROCESS( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )
if( ${ARCHITECTURE} MATCHES "arm64" OR ${ARCHITECTURE} MATCHES "aarch64" )
    set(CMAKE_ISPC_FLAGS "--target=neon -O3")
else()
    set(CMAKE_ISPC_FLAGS "--target=sse4 -O3")
endif()

add_library(bbox4 ${SOURCE_FILES})
