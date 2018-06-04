cmake_minimum_required(VERSION 3.10)
project(hpc_pohlig)
set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS "-lntl -fopenmp")

set(SOURCE_FILES lenstra_ecm.cpp)
add_executable(hpc_lenstra ${SOURCE_FILES})