cmake_minimum_required(VERSION 3.19.0)

project(bbox4_autovec C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SOURCE_FILES "../sseneoncompare_autovec.cpp")

set(CMAKE_CXX_FLAGS "-O3 -ffast-math")

EXECUTE_PROCESS( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    if(${CMAKE_OSX_ARCHITECTURES} MATCHES "arm64")
        set(ARCHITECTURE "arm64")
    elseif(${CMAKE_OSX_ARCHITECTURES} MATCHES "x86_64")
        set(ARCHITECTURE "x86_64")
    endif()
endif()
if( ${ARCHITECTURE} MATCHES "arm64" OR ${ARCHITECTURE} MATCHES "aarch64" )
    set(CMAKE_ISPC_FLAGS "--target=neon -O3")
else()
    set(CMAKE_ISPC_FLAGS "--target=sse4 -O3")
endif()

add_library(bbox4_autovec ${SOURCE_FILES})
