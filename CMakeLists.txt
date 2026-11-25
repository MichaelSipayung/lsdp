cmake_minimum_required(VERSION 3.20)
project(lsdp LANGUAGES CXX)

# Exact multiobjective DP for the lot sizing model of Kania et al. (CIE 2022)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(LSDP_NATIVE "Optimise for the build machine (-march=native)" ON)

find_package(OpenMP REQUIRED)

add_executable(lsdp src/lsdp.cpp)
target_link_libraries(lsdp PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(lsdp PRIVATE -Wall -Wextra $<$<CONFIG:Release>:-O3>)
if(LSDP_NATIVE)
  target_compile_options(lsdp PRIVATE -march=native)
endif()
