cmake_minimum_required(VERSION 3.19.0)

project(sseneoncompare)

add_subdirectory(src)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SOURCE_FILES "src/main.cpp")

set(CMAKE_CXX_FLAGS "-O0")
add_executable(sseneoncompare ${SOURCE_FILES})
target_link_libraries(sseneoncompare bbox4)
